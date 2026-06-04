"""
QuantaAlpha Backend API
FastAPI-based REST + WebSocket API for factor mining and backtesting.

Integrates with the core QuantaAlpha CLI to launch experiments
and reads factor library JSON for the factor browsing API.
"""

import asyncio
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Resolve project root (two levels up from this file: frontend-v2/backend/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Ensure import quantaalpha is available (when backend is started from frontend-v2 directory, repo root is not in sys.path)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DOTENV_PATH = PROJECT_ROOT / ".env"

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="QuantaAlpha API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:3001", "http://127.0.0.1:3001",
        "http://localhost:3003", "http://127.0.0.1:3003",
        "http://10.16.110.109:3003", "http://10.16.110.109:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================== Pydantic Models ==========================


class MiningStartRequest(BaseModel):
    """Request to start a factor mining experiment."""
    direction: str = Field(..., description="Research direction, e.g. 'price-volume factor mining'")
    numDirections: Optional[int] = Field(2, description="Parallel exploration directions")
    maxRounds: Optional[int] = Field(5, description="Evolution rounds")
    maxLoops: Optional[int] = Field(3, description="Iterations per direction")
    factorsPerHypothesis: Optional[int] = Field(3, description="Factors per hypothesis")
    librarySuffix: Optional[str] = Field(None, description="Factor library file suffix")
    qualityGateEnabled: Optional[bool] = Field(True, description="Enable quality gate checks")
    parallelEnabled: Optional[bool] = Field(None, description="Enable parallel execution within evolution phases")


class BacktestStartRequest(BaseModel):
    """Request to start an independent backtest."""
    factorJson: str = Field(..., description="Path to factor library JSON")
    factorSource: str = Field("custom", description="custom | combined")
    configPath: Optional[str] = Field(None, description="Path to backtest config")


class SystemConfigUpdate(BaseModel):
    """Partial update to system configuration (.env)."""
    QLIB_DATA_DIR: Optional[str] = None
    DATA_RESULTS_DIR: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    CHAT_MODEL: Optional[str] = None
    REASONING_MODEL: Optional[str] = None
    USE_SOVA: Optional[str] = None
    ENABLE_STOCK_QLIB_BACKTEST: Optional[str] = None
    ENABLE_GREY_FOREX_BACKTEST: Optional[str] = None
    FOREX_DATA_TYPE: Optional[str] = None
    FOREX_ASSET: Optional[str] = None


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


# ========================== In-Memory State ==========================

tasks: Dict[str, Dict[str, Any]] = {}
ws_connections: Dict[str, List[WebSocket]] = {}  # task_id -> list of WS


# ========================== Grey-Compatible Quanta/SOVA Routes ==========================


def _repo_root_from_here() -> Path:
    # This file lives under QuantaAlpha/frontend-v2/backend/; repo root is one level above PROJECT_ROOT.
    return PROJECT_ROOT.parent


def _quanta_jobs_root() -> Path:
    root = PROJECT_ROOT / "tmp" / "quanta_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _quanta_job_dir(job_id: str) -> Path:
    d = _quanta_jobs_root() / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _quanta_status_path(job_id: str) -> Path:
    return _quanta_job_dir(job_id) / "status.json"


def _quanta_log_path(job_id: str) -> Path:
    return _quanta_job_dir(job_id) / "run.log"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _tail_text(path: Path, max_lines: int) -> str:
    if max_lines <= 0:
        return ""
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
        lines = data.splitlines()[-max_lines:]
        return b"\n".join(lines).decode("utf-8", errors="replace")
    except Exception:
        return ""


_quanta_running: Dict[str, subprocess.Popen] = {}
_quanta_running_lock = threading.Lock()


class QuantaRunRequest(BaseModel):
    """Run QuantaAlpha dual-flow smoke test (stock+forex) in background."""

    max_rounds: int = Field(2, ge=1, le=20)
    stock_attempts: int = Field(3, ge=1, le=20)

    # Optional env overrides (kept minimal; add more only as needed).
    verify_beam: Optional[int] = Field(None, ge=1, le=64)
    forex_mdd_cap: Optional[float] = Field(None, ge=0.01, le=0.50)


@app.post("/api/ai/quanta/run")
async def quanta_run(request: QuantaRunRequest):
    repo_root = _repo_root_from_here()
    job_id = uuid.uuid4().hex

    job_dir = _quanta_job_dir(job_id)
    log_path = _quanta_log_path(job_id)
    status_path = _quanta_status_path(job_id)

    started_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    output_dir = repo_root / "data" / "results" / f"sova_dual_flow_api_{job_id}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # Enforce real backtests in this integration.
    env.setdefault("SOVA_REQUIRE_REAL_BACKTEST", "1")
    env.setdefault("SOVA_REFLECTION", "1")
    env.setdefault("SOVA_GENERALIZATION_GATES", "1")
    env.setdefault("SOVA_GENERALIZATION_REQUIRE_ON_READY", "1")

    if request.verify_beam is not None:
        env["SOVA_VERIFY_BEAM"] = str(int(request.verify_beam))
    if request.forex_mdd_cap is not None:
        env["SOVA_FOREX_MDD_CAP"] = str(float(request.forex_mdd_cap))

    cmd = [
        sys.executable,
        "-u",
        str((repo_root / "QuantaAlpha" / "run_dual_flow_smoketest.py").resolve()),
        "--max-rounds",
        str(int(request.max_rounds)),
        "--stock-attempts",
        str(int(request.stock_attempts)),
        "--output-dir",
        str(output_dir),
    ]

    status_payload: Dict[str, Any] = {
        "job_id": job_id,
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "exit_code": None,
        "pid": None,
        "cmd": cmd,
        "output_dir": str(output_dir),
        "error": None,
    }
    _write_json_atomic(status_path, status_payload)

    try:
        log_f = log_path.open("ab", buffering=0)
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

        status_payload["pid"] = proc.pid
        _write_json_atomic(status_path, status_payload)

        with _quanta_running_lock:
            _quanta_running[job_id] = proc

        def _watch() -> None:
            try:
                rc = proc.wait()
                ended_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                payload = _read_json(status_path)
                payload.update(
                    {
                        "status": "completed" if rc == 0 else "failed",
                        "ended_at": ended_at,
                        "exit_code": rc,
                    }
                )

                summary_path = Path(payload.get("output_dir", "")) / "summary.json"
                if summary_path.exists():
                    payload["summary_path"] = str(summary_path)
                _write_json_atomic(status_path, payload)
            finally:
                with _quanta_running_lock:
                    _quanta_running.pop(job_id, None)

        threading.Thread(target=_watch, daemon=True).start()

    except Exception as e:
        status_payload.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        _write_json_atomic(status_path, status_payload)
        raise HTTPException(status_code=500, detail=status_payload["error"])

    return {
        "success": True,
        "job_id": job_id,
        "status_url": f"/api/ai/quanta/progress/{job_id}",
        "logs_url": f"/api/ai/quanta/logs/{job_id}",
        "result_url": f"/api/ai/quanta/result/{job_id}",
    }


@app.get("/api/ai/quanta/progress/{job_id}")
async def quanta_progress(job_id: str):
    status_path = _quanta_status_path(job_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="job_id not found")
    payload = _read_json(status_path)
    return {"success": True, "data": payload}


@app.get("/api/ai/quanta/logs/{job_id}")
async def quanta_logs(job_id: str, tail: int = Query(200, ge=1, le=5000)):
    lp = _quanta_log_path(job_id)
    if not lp.exists():
        raise HTTPException(status_code=404, detail="log not found")
    return {"success": True, "job_id": job_id, "tail": tail, "log": _tail_text(lp, tail)}


@app.get("/api/ai/quanta/result/{job_id}")
async def quanta_result(job_id: str):
    status_path = _quanta_status_path(job_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="job_id not found")

    payload = _read_json(status_path)
    output_dir = payload.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="output_dir not available")

    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="summary.json not ready")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse summary.json: {e}")

    return {"success": True, "job_id": job_id, "output_dir": output_dir, "summary": summary}


@app.post("/api/ai/quanta/stop/{job_id}")
async def quanta_stop(job_id: str):
    status_path = _quanta_status_path(job_id)
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="job_id not found")

    payload = _read_json(status_path)
    pid = payload.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="pid not available")

    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop process: {e}")

    payload.update(
        {
            "status": "stopping",
            "stop_requested_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
    )
    _write_json_atomic(status_path, payload)
    return {"success": True, "job_id": job_id, "message": "SIGTERM sent"}


# ========================== Utility Helpers ==========================

def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


def _now() -> str:
    return datetime.now().isoformat()


def _load_dotenv_dict() -> Dict[str, str]:
    """Parse the .env file into a dict (simple key=value, ignoring comments)."""
    env: Dict[str, str] = {}
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                key, _, val = stripped.partition("=")
                env[key.strip()] = val.strip()
    return env


def _find_factor_jsons() -> List[str]:
    """Find all factor library JSON files in data/factorlib/."""
    factorlib_dir = PROJECT_ROOT / "data" / "factorlib"
    pattern = str(factorlib_dir / "all_factors_library*.json")
    results = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    old_pattern = str(PROJECT_ROOT / "all_factors_library*.json")
    old_results = sorted(glob.glob(old_pattern), key=os.path.getmtime, reverse=True)

    seen = set(results)
    for r in old_results:
        if r not in seen:
            results.append(r)
    return results


def _discover_grey_market_options() -> Dict[str, Any]:
    """Discover available Grey market data folders/assets from filesystem."""
    data_root = PROJECT_ROOT.parent / "Grey" / "data"
    options: Dict[str, Any] = {
        "greyDataRoot": str(data_root),
        "forexDataTypes": [],
        "forexAssetsByType": {},
        "forexAssets": [],
    }

    if not data_root.exists() or not data_root.is_dir():
        return options

    all_assets: set[str] = set()
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        # Hide backup folders from user-facing selectors (e.g. OKX_backup).
        if child.name.lower().endswith("_backup"):
            continue

        assets: set[str] = set()
        for pkl in sorted(child.glob("*.pkl")):
            # Normalize names like BTCUSDT_backup_1769401583 -> BTCUSDT
            stem = re.sub(r"_backup_\d+$", "", pkl.stem)
            if not stem:
                continue
            assets.add(stem.upper())

        if assets:
            assets_sorted = sorted(assets)
            options["forexDataTypes"].append(child.name)
            options["forexAssetsByType"][child.name] = assets_sorted
            all_assets.update(assets)

    options["forexDataTypes"] = sorted(options["forexDataTypes"])
    options["forexAssets"] = sorted(all_assets)
    return options


def _load_factor_library(path: str) -> Dict[str, Any]:
    """Load and parse a factor library JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort cast to float for mixed numeric payloads."""
    try:
        if value is None:
            return default
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return default
            if text.endswith("%"):
                return float(text[:-1].strip()) / 100.0
            text = text.replace(",", "")
            return float(text)
        return float(value)
    except Exception:
        return default


def _translate_factor_description_to_vi(text: str) -> str:
    """Best-effort deterministic EN→VI translation for factor descriptions (display only)."""
    if not isinstance(text, str) or not text.strip():
        return text

    lang = os.environ.get("FACTOR_DESCRIPTION_LANG", "vi").strip().lower()
    if not lang.startswith("vi"):
        return text

    out = text
    replacements = [
        ("This factor applies a ", "Nhân tố này áp dụng chuỗi biến đổi "),
        (" transformation pipeline ", " "),
        ("transformation pipeline", "chuỗi biến đổi"),
        ("daily high", "giá cao nhất trong ngày"),
        ("daily low", "giá thấp nhất trong ngày"),
        ("closing price", "giá đóng cửa"),
        ("opening price", "giá mở cửa"),
        ("trading volume", "khối lượng giao dịch"),
        ("daily return", "lợi suất ngày"),
        ("Cross-sectional", "Theo lát cắt ngang (cross-sectional)"),
        ("market-neutral", "trung tính thị trường"),
        ("Economic mechanism:", "Cơ chế kinh tế:"),
        ("Mathematical construction:", "Cấu trúc toán học:"),
        ("Empirical note:", "Ghi chú thực nghiệm:"),
        ("Strategy fit:", "Chiến lược phù hợp:"),
        ("overbought", "quá mua"),
        ("oversold", "quá bán"),
        ("volatility", "biến động"),
        ("momentum", "động lượng"),
        ("mean-reversion", "hồi quy về trung bình"),
        ("institutional", "tổ chức"),
        ("order flow", "dòng lệnh"),
        ("noise", "nhiễu"),
    ]
    for old, new in replacements:
        out = out.replace(old, new)

    if out.startswith("This factor"):
        out = out.replace("This factor", "Nhân tố này", 1)
    return out


_VI_VAR_LABELS: Dict[str, str] = {
    "$close": "giá đóng cửa",
    "$open": "giá mở cửa",
    "$high": "giá cao nhất trong ngày",
    "$low": "giá thấp nhất trong ngày",
    "$volume": "khối lượng giao dịch",
    "$return": "lợi suất ngày",
}


def _looks_vietnamese(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if "Nhân tố" in text:
        return True
    # Basic heuristic: presence of Vietnamese-specific characters.
    return any(ch in text for ch in "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def _build_factor_description_vi_from_expr(expr: str) -> str:
    if not isinstance(expr, str) or not expr.strip():
        return ""

    ops_raw = re.findall(r"[A-Z_]+(?=\()", expr)
    ops = list(dict.fromkeys(ops_raw))
    vars_used = list(dict.fromkeys(re.findall(r"\$\w+", expr)))
    windows = re.findall(r"(?:,\s*)(\d+)", expr)

    op_chain = " → ".join(ops[:8]) if ops else "các toán tử kỹ thuật"
    var_labels = [_VI_VAR_LABELS.get(v, v) for v in vars_used[:4]]
    var_part = ", ".join(var_labels) if var_labels else "dữ liệu giá/khối lượng"

    win_part = ""
    if windows:
        uniq = list(dict.fromkeys(windows))
        win_part = f" trên các cửa sổ {', '.join(uniq[:3])} ngày"

    theme_hint = ""
    if any(v in vars_used for v in ("$volume",)):
        theme_hint = "dòng tiền/khối lượng"
    elif any(op in ops for op in ("DELTA", "RET", "ROC")):
        theme_hint = "động lượng ngắn hạn"
    elif any(op in ops for op in ("ZSCORE", "MEAN", "EMA", "MA")):
        theme_hint = "xu hướng và hồi quy về trung bình"
    elif any(op in ops for op in ("STD", "VAR", "KURT", "SKEW")):
        theme_hint = "biến động và rủi ro"

    s1 = f"Nhân tố này áp dụng chuỗi toán tử {op_chain} lên {var_part}{win_part} để trích xuất tín hiệu thống kê từ cấu trúc dữ liệu.".strip()
    s2_parts: List[str] = []
    if any(op in ops for op in ("RANK", "CS_RANK")):
        s2_parts.append("Có chuẩn hoá theo lát cắt ngang (RANK) để giảm ảnh hưởng đồng biến thị trường")
    if "TS_RANK" in ops:
        s2_parts.append("Có chuẩn hoá theo chuỗi thời gian (TS_RANK) để làm nổi bật mức tương đối trong lịch sử")
    if theme_hint:
        s2_parts.append(f"Tín hiệu thiên về {theme_hint}")

    s2 = (". ".join(s2_parts) + ".") if s2_parts else ""
    return (s1 + (" " + s2 if s2 else "")).strip()


def _factor_description_for_display(factor_description: str, factor_expression: str) -> str:
    """Return Vietnamese description for UI when configured, without relying on partial phrase replacement."""
    lang = os.environ.get("FACTOR_DESCRIPTION_LANG", "vi").strip().lower()
    if not lang.startswith("vi"):
        return factor_description

    if _looks_vietnamese(factor_description):
        return factor_description

    vi = _build_factor_description_vi_from_expr(factor_expression)
    if vi:
        return vi

    # Last resort: best-effort translation of stored English.
    return _translate_factor_description_to_vi(factor_description)


def _market_suitability_line_for_display(market_suitability: Dict[str, Any]) -> str:
    """Render one concise Vietnamese line naming only the best-fit market."""
    if not isinstance(market_suitability, dict) or not market_suitability:
        return ""
    label = str(market_suitability.get("label_vi") or "").strip()
    if not label:
        return ""
    return f"Alpha chiến lược này đạt hiệu quả tốt nhất với thị trường: {label}."


def _first_metric(bt: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    """Get first available metric from aliases and cast to float."""
    for key in keys:
        if key in bt:
            return _safe_float(bt.get(key), default)
    return default


def _extract_bt_metrics(backtest_results: Dict[str, Any]) -> Dict[str, float]:
    """Normalize key metrics from heterogeneous backtest payloads."""
    bt = backtest_results or {}
    ic = _first_metric(bt, [
        "IC", "ic", "information_coefficient",
        "1day.excess_return_without_cost.information_coefficient",
        "1day.excess_return_with_cost.information_coefficient",
    ])
    icir = _first_metric(bt, [
        "ICIR", "icir", "information_coefficient_ir",
        "1day.excess_return_without_cost.information_coefficient_ir",
        "1day.excess_return_with_cost.information_coefficient_ir",
    ])
    rank_ic = _first_metric(bt, [
        "Rank IC", "rank_ic", "RankIC", "rankIC",
        "1day.excess_return_without_cost.rank_ic",
        "1day.excess_return_with_cost.rank_ic",
    ])
    rank_icir = _first_metric(bt, [
        "Rank ICIR", "rank_ic_ir", "RankICIR", "rankICIR",
        "1day.excess_return_without_cost.rank_ic_ir",
        "1day.excess_return_with_cost.rank_ic_ir",
    ])

    annual_return = _first_metric(bt, [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_without_cost.annualized_return",
        "annual_return", "annualized_return", "ARR", "arr",
    ])
    information_ratio = _first_metric(bt, [
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_without_cost.information_ratio",
        "information_ratio", "IR", "ir", "sharpe", "sharpe_ratio",
    ])
    max_drawdown = _first_metric(bt, [
        "1day.excess_return_with_cost.max_drawdown",
        "1day.excess_return_without_cost.max_drawdown",
        "max_drawdown", "MDD", "mdd",
    ])

    # Turnover / trading frequency proxy (higher => noisier + more cost sensitive).
    turnover = _first_metric(bt, [
        "1day.ffr",
        "ffr",
        "turnover",
    ])

    train_l2 = _first_metric(bt, ["l2.train", "train_l2"])
    valid_l2 = _first_metric(bt, ["l2.valid", "valid_l2"])

    return {
        "ic": ic,
        "icir": icir,
        "rank_ic": rank_ic,
        "rank_icir": rank_icir,
        "annual_return": annual_return,
        "information_ratio": information_ratio,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "train_l2": train_l2,
        "valid_l2": valid_l2,
    }


def _compute_factor_score(metrics: Dict[str, float]) -> float:
    """
    Composite ranking score:
    - prioritize cross-sectional predictive strength (Rank IC / IC)
    - reward stability (ICIR / Rank ICIR)
    - penalize clear overfit patterns (large train-valid gap + weak stability)
    """
    signal = max(abs(metrics["rank_ic"]), abs(metrics["ic"]))
    stability = max(metrics["rank_icir"], metrics["icir"], 0.0)
    # If no usable backtest metrics are present, force this factor to the bottom.
    # This prevents "IC=0 placeholder" factors from dominating the top list.
    if (
        signal == 0.0
        and stability == 0.0
        and metrics["annual_return"] == 0.0
        and metrics["information_ratio"] == 0.0
        and metrics["max_drawdown"] == 0.0
    ):
        return -1.0

    # Keep signed performance terms, but cap their influence so noisy
    # single-run realized metrics do not overwhelm predictive signal quality.
    perf_ir = metrics["information_ratio"]
    annual_ret = metrics["annual_return"]
    mdd_abs = abs(metrics["max_drawdown"]) if metrics["max_drawdown"] < 0 else metrics["max_drawdown"]
    mdd_capped = min(max(mdd_abs, 0.0), 1.0)
    turnover = metrics.get("turnover", 0.0) or 0.0

    # Generalization proxy: avoid overfitting to train while weak out-of-sample stability.
    l2_gap = abs(metrics["train_l2"] - metrics["valid_l2"])
    overfit_penalty = 0.0
    if signal >= 0.03 and stability < 0.2:
        overfit_penalty += 0.02
    if l2_gap > 0.01:
        overfit_penalty += min(0.03, l2_gap)

    # Explicit strategy-risk penalties.
    if annual_ret < 0:
        overfit_penalty += min(0.025, abs(annual_ret) * 0.10)
    if mdd_capped > 0.35:
        overfit_penalty += min(0.03, (mdd_capped - 0.35) * 0.04)

    # High turnover often indicates noisy/unstable signals and high cost sensitivity.
    # Penalize above a moderate threshold; keep penalty bounded.
    if turnover > 0.55:
        overfit_penalty += min(0.03, (turnover - 0.55) * 0.06)

    ir_component = max(-0.04, min(0.05, perf_ir * 0.06))
    return_component = max(-0.03, min(0.05, annual_ret * 0.20))

    return (
        0.45 * signal
        + 0.35 * stability
        + ir_component
        + return_component
        - overfit_penalty
    )


def _classify_quality(backtest_results: Dict[str, Any]) -> str:
    """Classify factor quality using predictive signal + stability (not IR alone)."""
    if not backtest_results:
        return "low"

    m = _extract_bt_metrics(backtest_results)
    score = _compute_factor_score(m)

    signal = max(abs(m["rank_ic"]), abs(m["ic"]))
    stability = max(m["rank_icir"], m["icir"], 0.0)
    turnover = m.get("turnover", 0.0) or 0.0
    mdd_abs = abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else m["max_drawdown"]

    # High quality (realistic): strong predictive signal + strong stability + acceptable
    # realized performance under costs, with controlled turnover and drawdown.
    if (
        score >= 0.08
        and signal >= 0.045
        and stability >= 0.30
        and m["annual_return"] >= 0
        and m["information_ratio"] >= 0
        and mdd_abs <= 0.55
        and turnover <= 0.70
    ):
        return "high"

    # Medium quality: usable signal and stability, but may still be weak on realized metrics.
    # Reject clearly noisy/high-churn candidates.
    if score >= 0.03 and signal >= 0.02 and stability >= 0.10 and turnover <= 0.85:
        return "medium"
    return "low"


def _enrich_direction(direction: str) -> str:
    """Attach strict quantitative guardrails to user direction guidance."""
    base = (direction or "").strip()
    constraints = (
        "Focus on momentum mining with short-term reversal and trading-volume regime signals; "
        "prefer logically simple, interpretable formulas; avoid over-nested operators and noisy redundancy; "
        "prioritize robust RankIC/IC stability over in-sample creativity; enforce out-of-sample consistency."
    )
    if not base:
        return constraints
    return f"{base}. {constraints}"


def _ensure_minimum_high_quality(factors: List[Dict[str, Any]]) -> bool:
    """
    Guarantee at least one high-quality candidate for a run by promoting
    the best-scoring factor when no high exists.
    Returns True if fallback promotion is applied.
    """
    if not factors:
        return False

    # Realism-first default: do NOT force a "high" label unless explicitly enabled.
    enable_promotion = os.environ.get("FACTOR_ENABLE_RELATIVE_HIGH_PROMOTION", "0").strip().lower() in ("1", "true", "yes", "on")
    if not enable_promotion:
        return False

    if any(f.get("quality") == "high" for f in factors):
        return False

    # Choose the best factor(s) by composite score under current scoring logic.
    # If there are ties (or near-ties), promote all tied best factors to avoid
    # confusing "same metrics but different quality" outcomes.
    scores: List[float] = []
    best_score = float("-inf")
    for factor in factors:
        bt = factor.get("backtestResults") or factor.get("backtest_results") or {}
        score = _compute_factor_score(_extract_bt_metrics(bt))
        scores.append(score)
        if score > best_score:
            best_score = score

    # Guardrails: don't promote when the entire batch is clearly untradeable under costs.
    # This avoids the confusing "Cao nhưng vẫn lỗ" outcome.
    eps = 1e-9
    promoted = False
    for factor, score in zip(factors, scores):
        if score >= best_score - eps:
            bt = factor.get("backtestResults") or factor.get("backtest_results") or {}
            m = _extract_bt_metrics(bt)
            mdd_abs = abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else m["max_drawdown"]
            if m["annual_return"] < -0.02 and m["information_ratio"] < -0.10:
                continue
            if mdd_abs > 0.80:
                continue
            factor["quality"] = "high"
            factor["qualitySource"] = "relative_top_pick"
            promoted = True
    return promoted


async def _broadcast(task_id: str, message: Dict[str, Any]):
    """Send a JSON message to all WebSocket clients for a task."""
    if task_id not in ws_connections:
        return
    dead: List[WebSocket] = []
    for ws in ws_connections[task_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[task_id].remove(ws)


# ========================== Mining Process ==========================

async def _run_mining(task_id: str, req: MiningStartRequest):
    """
    Launch the actual QuantaAlpha mining experiment as a subprocess
    and stream its output over WebSocket.
    """
    task = tasks[task_id]
    try:
        # Build the command
        env = os.environ.copy()
        # Load .env into env
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # Safety defaults for web-triggered runs: avoid hanging factor execution forever.
        env.setdefault("FACTOR_CoSTEER_FILE_BASED_EXECUTION_TIMEOUT", "180")

        # Prefer project venv python for factor.py execution when available.
        preferred_factor_python = str(Path.home() / "vinh" / ".venv" / "bin" / "python")
        if Path(preferred_factor_python).exists():
            env.setdefault("FACTOR_CoSTEER_python_bin", preferred_factor_python)

        # Use experiment_id as suffix to guarantee isolation
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env["EXPERIMENT_ID"] = experiment_id
        
        # Enforce unique library suffix if not provided
        if not req.librarySuffix:
            req.librarySuffix = experiment_id
            # Update task config so frontend knows the suffix
            task["config"]["librarySuffix"] = req.librarySuffix
            
        env["FACTOR_LIBRARY_SUFFIX"] = req.librarySuffix

        results_base = dotenv.get("DATA_RESULTS_DIR", str(PROJECT_ROOT / "data" / "results"))
        env["WORKSPACE_PATH"] = f"{results_base}/workspace_{experiment_id}"
        env["PICKLE_CACHE_FOLDER_PATH_STR"] = f"{results_base}/pickle_cache_{experiment_id}"

        os.makedirs(env["WORKSPACE_PATH"], exist_ok=True)
        os.makedirs(env["PICKLE_CACHE_FOLDER_PATH_STR"], exist_ok=True)

        # Qlib symlink
        qlib_data = dotenv.get("QLIB_DATA_DIR", "")
        if qlib_data:
            qlib_symlink_dir = Path.home() / ".qlib" / "qlib_data"
            qlib_symlink_dir.mkdir(parents=True, exist_ok=True)
            cn_data_link = qlib_symlink_dir / "cn_data"
            if not cn_data_link.exists() or os.readlink(str(cn_data_link)) != qlib_data:
                if cn_data_link.is_symlink():
                    cn_data_link.unlink()
                cn_data_link.symlink_to(qlib_data)

        # Build a temporary config with frontend parameter overrides
        base_config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
        config_path_to_use = str(base_config_path)

        try:
            with open(base_config_path, "r", encoding="utf-8") as _f:
                run_cfg = yaml.safe_load(_f) or {}

            # Apply frontend overrides
            if req.numDirections is not None:
                run_cfg.setdefault("planning", {})["num_directions"] = req.numDirections
            if req.maxRounds is not None:
                run_cfg.setdefault("evolution", {})["max_rounds"] = req.maxRounds
            if req.maxLoops is not None:
                run_cfg.setdefault("execution", {})["max_loops"] = req.maxLoops
            if req.factorsPerHypothesis is not None:
                run_cfg.setdefault("factor", {})["factors_per_hypothesis"] = req.factorsPerHypothesis

            # Apply parallel execution override from frontend
            if req.parallelEnabled is not None:
                run_cfg.setdefault("evolution", {})["parallel_enabled"] = req.parallelEnabled
                run_cfg.setdefault("execution", {})["parallel_execution"] = req.parallelEnabled

            # Apply quality gate override from frontend
            if req.qualityGateEnabled is not None:
                qg = run_cfg.setdefault("quality_gate", {})
                if req.qualityGateEnabled:
                    # Enable all guards when user enables quality gate to reduce overfit and weak factors.
                    qg["complexity_enabled"] = True
                    qg["redundancy_enabled"] = True
                    qg["consistency_enabled"] = True
                else:
                    # Disable quality gate: disable all
                    qg["consistency_enabled"] = False
                    qg["complexity_enabled"] = False
                    qg["redundancy_enabled"] = False

            # Write to a temporary file so the original is untouched
            tmp_dir = Path(env.get("WORKSPACE_PATH", "/tmp"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_cfg = tmp_dir / "experiment_override.yaml"
            with open(tmp_cfg, "w", encoding="utf-8") as _f:
                yaml.safe_dump(run_cfg, _f, allow_unicode=True, default_flow_style=False)
            config_path_to_use = str(tmp_cfg)
        except Exception as cfg_err:
            # Fall back to original config if anything fails
            import traceback
            traceback.print_exc()

        # Build CLI args
        enhanced_direction = _enrich_direction(req.direction)

        cmd = [
            sys.executable, "-u", "-m", "quantaalpha.cli", "mine",
            "--direction", enhanced_direction,
            "--config_path", config_path_to_use,
        ]

        task["status"] = "running"
        task["progress"]["phase"] = "planning"
        task["progress"]["message"] = "Starting experiment..."
        task["updatedAt"] = _now()

        await _broadcast(task_id, {
            "type": "progress",
            "taskId": task_id,
            "data": task["progress"],
            "timestamp": _now(),
        })

        # Launch subprocess
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Stream stdout line by line
        line_count = 0
        current_phase = "planning"

        # Noisy patterns to suppress (shared with backtest)
        _MINING_NOISE = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "FutureWarning",
            "UserWarning",
            "Training until validation scores",
            "Did not meet early stopping",
        )

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            
            # Mirror to backend console/log
            print(f"[{task_id}] {line}", flush=True)

            line_count += 1

            # Skip noisy warnings
            if any(p in line for p in _MINING_NOISE):
                continue

            # Detect phase from log messages
            new_phase = current_phase
            if "factor_propose" in line:
                new_phase = "evolving"
            elif "factor_backtest" in line or "backtest" in line.lower():
                new_phase = "backtesting"
            elif "feedback" in line:
                new_phase = "analyzing"
            elif "factor_calculate" in line:
                new_phase = "evolving"
            elif "planning" in line.lower():
                new_phase = "planning"
            elif "evolution complete" in line.lower() or "program complete" in line.lower() or "experiment done" in line.lower():
                new_phase = "completed"

            if new_phase != current_phase:
                current_phase = new_phase
                task["progress"]["phase"] = current_phase
                task["progress"]["message"] = line[:200]
                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

            # Send log every line (throttle to avoid flooding)
            if line_count % 3 == 0 or "INFO" in line or "ERROR" in line or "WARNING" in line:
                level = "info"
                if "ERROR" in line or "Error" in line:
                    level = "error"
                elif "WARNING" in line or "Warning" in line:
                    level = "warning"
                elif "complete" in line.lower() or "done" in line.lower() or "success" in line.lower():
                    level = "success"

                log_entry = {
                    "id": _gen_id(),
                    "timestamp": _now(),
                    "level": level,
                    "message": line[:500],
                }
                task["logs"].append(log_entry)
                # Keep only last 500 logs in memory
                if len(task["logs"]) > 500:
                    task["logs"] = task["logs"][-500:]

                await _broadcast(task_id, {
                    "type": "log",
                    "taskId": task_id,
                    "data": log_entry,
                    "timestamp": _now(),
                })

            # Extract metrics from log lines like "RankIC=0.0016"
            if "RankIC=" in line:
                try:
                    rank_ic_str = line.split("RankIC=")[1].split(",")[0].split(")")[0]
                    task["metrics"]["rankIc"] = float(rank_ic_str)
                    await _broadcast(task_id, {
                        "type": "metrics",
                        "taskId": task_id,
                        "data": task["metrics"],
                        "timestamp": _now(),
                    })
                except Exception:
                    pass
            
            # Check for factor saving to update top factors list
            if "saved" in line.lower() or "factor" in line.lower():
                _update_mining_metrics(task)
                if task.get("metrics"):
                     await _broadcast(task_id, {
                        "type": "result",
                        "taskId": task_id,
                        "data": {"status": task["status"], "metrics": task["metrics"]},
                        "timestamp": _now(),
                    })

        exit_code = await proc.wait()
        task["pid"] = None

        if exit_code == 0:
            task["status"] = "completed"
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "Experiment completed"
        else:
            task["status"] = "failed"
            task["progress"]["message"] = f"Experiment failed (exit code: {exit_code})"

        task["updatedAt"] = _now()

        # Load final factor count from the library JSON
        # Prefer the library file matching the librarySuffix for this experiment
        _update_mining_metrics(task)

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {"status": task["status"], "metrics": task["metrics"]},
            "timestamp": _now(),
        })

    except Exception as e:
        task["status"] = "failed"
        task["progress"]["message"] = f"Error: {str(e)}"
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


# ========================== API Endpoints ==========================

@app.get("/")
async def root():
    return {"message": "QuantaAlpha API", "version": "2.0.0"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": _now()}


# ========================== Sova Chat API ==========================

class SovaChatContext(BaseModel):
    asset: Optional[str] = None
    timeframe: Optional[str] = None
    regime: Optional[Dict[str, Any]] = None
    backtest_summary: Optional[Dict[str, Any]] = None
    candle_snapshot: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, Any]]] = None  # last 5-10 messages


class SovaChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    context: Optional[SovaChatContext] = None


# Lazy-load orchestrator (avoid import cost at startup)
_sova_orch = None
_sova_orch_lock = threading.Lock()


def _get_sova_orchestrator():
    global _sova_orch
    if _sova_orch is None:
        with _sova_orch_lock:
            if _sova_orch is None:
                try:
                    from sova_orchestrator import SovaAnalysisOrchestrator
                    _sova_orch = SovaAnalysisOrchestrator(project_root=PROJECT_ROOT)
                    logger.info("[SovaChat] Orchestrator loaded ✓")
                except Exception as exc:
                    logger.error(f"[SovaChat] Failed to load orchestrator: {exc}")
                    _sova_orch = None
    return _sova_orch


@app.post("/api/sova/chat")
async def sova_chat(req: SovaChatRequest):
    """
    Sova AI chat endpoint.
    Accepts a user message + trading context → returns SovaAnalysis JSON
    that includes natural-language summary and chart overlay data.
    """
    orch = _get_sova_orchestrator()
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail="Sova orchestrator not available. Check server logs."
        )

    ctx: Dict[str, Any] = {}
    if req.context:
        ctx = req.context.model_dump(exclude_none=True)

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: orch.analyze(req.message, ctx)
        )
        return {
            "success": True,
            "data": result.to_dict(),
        }
    except Exception as exc:
        logger.exception(f"[SovaChat] analyze() error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ---- Mining endpoints ----

@app.post("/api/v1/mining/start", response_model=ApiResponse)
async def start_mining(req: MiningStartRequest):
    """Start a new factor mining experiment."""
    task_id = _gen_id()
    task = {
        "taskId": task_id,
        "status": "running",
        "config": req.model_dump(),
        "progress": {
            "phase": "parsing",
            "currentRound": 0,
            "totalRounds": req.maxRounds or 3,
            "progress": 0,
            "message": "Initializing experiment...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {
            "ic": 0, "icir": 0, "rankIc": 0, "rankIcir": 0,
            "annualReturn": 0, "sharpeRatio": 0, "maxDrawdown": 0,
            "totalFactors": 0, "highQualityFactors": 0,
            "mediumQualityFactors": 0, "lowQualityFactors": 0,
        },
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch the mining process in background
    asyncio.create_task(_run_mining(task_id, req))

    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="Experiment started",
    )


@app.get("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def get_mining_status(task_id: str):
    """Get task status."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/mining/{task_id}", response_model=ApiResponse)
async def cancel_mining(task_id: str):
    """Cancel a running mining task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            pid = task["pid"]
            # Try graceful termination first
            os.kill(pid, signal.SIGTERM)
            
            # Wait briefly for cleanup (0.5s)
            for _ in range(5):
                try:
                    os.kill(pid, 0) # Check if alive
                    await asyncio.sleep(0.1)
                except ProcessLookupError:
                    break
            
            # Force kill if still running
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="Task cancelled")


@app.get("/api/v1/mining/tasks/list", response_model=ApiResponse)
async def list_tasks():
    """List all tasks."""
    task_list = sorted(tasks.values(), key=lambda t: t["createdAt"], reverse=True)
    return ApiResponse(success=True, data={"tasks": task_list})


# ---- Factor library endpoints ----

@app.get("/api/v1/factors", response_model=ApiResponse)
async def get_factors(
    quality: Optional[str] = Query(None, description="Filter by quality: high/medium/low"),
    search: Optional[str] = Query(None, description="Search by factor name"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    library: Optional[str] = Query(None, description="Specific library file name"),
):
    """Get factors from the factor library JSON."""
    # Find the most recent factor library
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        # Fallback: check if file exists at project root (legacy location)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(
                success=True,
                data={"factors": [], "total": 0, "limit": limit, "offset": offset,
                      "libraries": []},
            )
        lib_path = jsons[0]

    try:
        raw = _load_factor_library(lib_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read factor library: {e}")

    factors_dict = raw.get("factors", {})
    metadata = raw.get("metadata", {})

    # Convert dict to list with quality classification
    factors_list: List[Dict[str, Any]] = []
    for factor_id, factor_info in factors_dict.items():
        if not isinstance(factor_info, dict):
            continue
        bt = factor_info.get("backtest_results", {})
        q = _classify_quality(bt)
        m = _extract_bt_metrics(bt)

        desc = _factor_description_for_display(
            factor_info.get("factor_description", ""),
            factor_info.get("factor_expression", ""),
        )
        market_suitability = factor_info.get("market_suitability", {})
        market_line = _market_suitability_line_for_display(market_suitability)
        if market_line:
            desc = (str(desc).rstrip() + "\n" + market_line).strip()
        
        factor_entry = {
            "factorId": factor_info.get("factor_id", factor_id),
            "factorName": factor_info.get("factor_name", "Unknown"),
            "factorExpression": factor_info.get("factor_expression", ""),
            "factorDescription": desc,
            "factorFormulation": factor_info.get("factor_formulation", ""),
            "quality": q,
            "backtestResults": bt,
            "marketSuitability": market_suitability,
            # Extract key metrics
            "ic": m["ic"],
            "icir": m["icir"],
            "rankIc": m["rank_ic"],
            "rankIcir": m["rank_icir"],
            "annualReturn": m["annual_return"],
            "maxDrawdown": m["max_drawdown"],
            "sharpeRatio": m["information_ratio"],
            "round": factor_info.get("evolution_metadata", {}).get("round", 0)
            if isinstance(factor_info.get("evolution_metadata"), dict) else 0,
            "direction": factor_info.get("evolution_metadata", {}).get("direction_index", "")
            if isinstance(factor_info.get("evolution_metadata"), dict) else "",
            "createdAt": factor_info.get("added_at", ""),
        }
        factors_list.append(factor_entry)

    # If strict absolute thresholds produce no high, promote top-ranked candidate
    # so each run still yields one actionable high-priority strategy.
    _ensure_minimum_high_quality(factors_list)

    # Apply filters
    if quality:
        factors_list = [f for f in factors_list if f["quality"] == quality]
    if search:
        search_lower = search.lower()
        factors_list = [
            f for f in factors_list
            if search_lower in f["factorName"].lower()
            or search_lower in f.get("factorDescription", "").lower()
            or search_lower in f.get("factorExpression", "").lower()
        ]

    total = len(factors_list)
    paginated = factors_list[offset: offset + limit]

    # Available library files
    all_libs = [Path(p).name for p in _find_factor_jsons()]

    return ApiResponse(
        success=True,
        data={
            "factors": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "metadata": metadata,
            "libraries": all_libs,
        },
    )


# ---- Factor cache endpoints ----
# IMPORTANT: These must be registered BEFORE /api/v1/factors/{factor_id}
# otherwise FastAPI matches "cache-status" as a factor_id parameter.

@app.get("/api/v1/factors/cache-status", response_model=ApiResponse)
async def get_cache_status(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Check cache status of factors in the specified factor library."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=True, data={
                "total": 0, "h5_cached": 0, "md5_cached": 0,
                "need_compute": 0, "factors": [],
            })
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    # Import from core library
    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.check_cache_status(lib_path)
    return ApiResponse(success=True, data=result)


@app.post("/api/v1/factors/warm-cache", response_model=ApiResponse)
async def warm_cache(
    library: Optional[str] = Query(None, description="Factor library JSON filename"),
):
    """Batch sync from result.h5 to MD5 cache directory."""
    if library:
        lib_path = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if not Path(lib_path).exists():
            alt = str(PROJECT_ROOT / library)
            if Path(alt).exists():
                lib_path = alt
    else:
        jsons = _find_factor_jsons()
        if not jsons:
            return ApiResponse(success=False, error="Factor library file not found")
        lib_path = jsons[0]

    if not Path(lib_path).exists():
        raise HTTPException(status_code=404, detail=f"Factor library not found: {library}")

    from quantaalpha.factors.library import FactorLibraryManager
    result = FactorLibraryManager.warm_cache_from_json(lib_path)
    # Build a clear message
    parts = []
    if result['synced']:
        parts.append(f"Synced {result['synced']} new")
    if result.get('already_cached'):
        parts.append(f"{result['already_cached']} already cached")
    if result.get('no_source'):
        parts.append(f"{result['no_source']} no H5 source (will compute from expression during backtest)")
    if result['failed']:
        parts.append(f"{result['failed']} failed")
    msg = ", ".join(parts) if parts else "Nothing to do"
    return ApiResponse(
        success=True,
        data=result,
        message=msg,
    )


# ---- Factor library list endpoint (must be BEFORE {factor_id} route) ----

@app.get("/api/v1/factors/libraries", response_model=ApiResponse)
async def list_factor_libraries():
    """List all factor library JSON files in the project root."""
    libs = [Path(p).name for p in _find_factor_jsons()]
    return ApiResponse(success=True, data={"libraries": libs})


@app.get("/api/v1/factors/{factor_id}", response_model=ApiResponse)
async def get_factor_detail(
    factor_id: str,
    library: Optional[str] = Query(None, description="Specific library file name"),
    preferListQuality: bool = Query(
        True,
        description="If true, try to use the list-view (post quality-floor) quality when available.",
    ),
):
    """Get full detail of a single factor."""
    candidate_paths: List[str] = []

    if library:
        preferred = str(PROJECT_ROOT / "data" / "factorlib" / library)
        if Path(preferred).exists():
            candidate_paths.append(preferred)
        else:
            legacy = str(PROJECT_ROOT / library)
            if Path(legacy).exists():
                candidate_paths.append(legacy)

    for lib_path in _find_factor_jsons():
        if lib_path not in candidate_paths:
            candidate_paths.append(lib_path)

    for lib_path in candidate_paths:
        try:
            raw = _load_factor_library(lib_path)
            factors = raw.get("factors", {})
            if factor_id in factors:
                info = factors[factor_id] if isinstance(factors[factor_id], dict) else {}
                bt = info.get("backtest_results", {})
                metrics = _extract_bt_metrics(bt)

                # Default to absolute quality; optionally prefer list-view quality (which includes
                # quality-floor promotion) for consistency with the UI.
                quality_abs = _classify_quality(bt)
                quality_out = quality_abs
                quality_source: Optional[str] = None
                if preferListQuality:
                    try:
                        # Build a minimal list (ids + metrics) to apply the same promotion logic.
                        factors_list: List[Dict[str, Any]] = []
                        for fid, finfo in factors.items():
                            if not isinstance(finfo, dict):
                                continue
                            bt_i = finfo.get("backtest_results", {})
                            m_i = _extract_bt_metrics(bt_i)
                            factors_list.append({
                                "factorId": finfo.get("factor_id", fid),
                                "factorName": finfo.get("factor_name", "Unknown"),
                                "ic": m_i["ic"],
                                "icir": m_i["icir"],
                                "rankIc": m_i["rank_ic"],
                                "rankIcir": m_i["rank_icir"],
                                "annualReturn": m_i["annual_return"],
                                "maxDrawdown": m_i["max_drawdown"],
                                "sharpeRatio": m_i["information_ratio"],
                                "backtestResults": bt_i,
                                "quality": _classify_quality(bt_i),
                            })
                        _ensure_minimum_high_quality(factors_list)
                        for f in factors_list:
                            if f.get("factorId") == info.get("factor_id", factor_id):
                                quality_out = f.get("quality", quality_abs)
                                quality_source = f.get("qualitySource")
                                break
                    except Exception:
                        pass

                factor_entry = {
                    **info,
                    "factorId": info.get("factor_id", factor_id),
                    "factorName": info.get("factor_name", "Unknown"),
                    "factorExpression": info.get("factor_expression", ""),
                    "factorDescription": _factor_description_for_display(
                        info.get("factor_description", ""),
                        info.get("factor_expression", ""),
                    ),
                    "factorFormulation": info.get("factor_formulation", ""),
                    "backtestResults": bt,
                    "quality": quality_out,
                    "qualitySource": quality_source,
                    "ic": metrics["ic"],
                    "icir": metrics["icir"],
                    "rankIc": metrics["rank_ic"],
                    "rankIcir": metrics["rank_icir"],
                    "annualReturn": metrics["annual_return"],
                    "sharpeRatio": metrics["information_ratio"],
                    "maxDrawdown": metrics["max_drawdown"],
                    "createdAt": info.get("added_at", ""),
                    "sourceLibrary": Path(lib_path).name,
                }
                return ApiResponse(success=True, data={"factor": factor_entry})
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Factor not found")


# ---- Backtest endpoints ----

@app.post("/api/v1/backtest/start", response_model=ApiResponse)
async def start_backtest(req: BacktestStartRequest):
    """Start an independent backtest."""
    task_id = _gen_id()
    config_path = req.configPath or str(PROJECT_ROOT / "configs" / "backtest.yaml")

    task = {
        "taskId": task_id,
        "status": "running",
        "type": "backtest",
        "config": {**req.model_dump(), "configPath": config_path},
        "progress": {
            "phase": "backtesting",
            "currentRound": 0,
            "totalRounds": 1,
            "progress": 0,
            "message": "Starting backtest...",
            "timestamp": _now(),
        },
        "logs": [],
        "metrics": {},
        "result": None,
        "pid": None,
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    tasks[task_id] = task

    # Launch backtest in background
    asyncio.create_task(_run_backtest(task_id, req, config_path))
    return ApiResponse(
        success=True,
        data={"taskId": task_id, "task": task},
        message="Backtest started",
    )


@app.get("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def get_backtest_status(task_id: str):
    """Get backtest task status and results."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return ApiResponse(success=True, data={"task": tasks[task_id]})


@app.delete("/api/v1/backtest/{task_id}", response_model=ApiResponse)
async def cancel_backtest(task_id: str):
    """Cancel a running backtest task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.get("pid"):
        try:
            os.kill(task["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    task["status"] = "cancelled"
    task["updatedAt"] = _now()
    await _broadcast(task_id, {
        "type": "result",
        "taskId": task_id,
        "data": {"status": "cancelled"},
        "timestamp": _now(),
    })
    return ApiResponse(success=True, message="Backtest cancelled")


async def _run_backtest(task_id: str, req: BacktestStartRequest, config_path: str):
    """Run the independent backtest (V2) as a subprocess."""
    task = tasks[task_id]
    try:
        env = os.environ.copy()
        dotenv = _load_dotenv_dict()
        env.update(dotenv)

        # --- Resolve factor JSON path ---
        # Frontend sends just the filename (e.g. "all_factors_library_test3hjback.json")
        # We need to resolve it to the full path under data/factorlib/
        factor_json_input = req.factorJson
        factor_json_path = Path(factor_json_input)
        if not factor_json_path.is_absolute():
            # Check data/factorlib/ first
            candidate = PROJECT_ROOT / "data" / "factorlib" / factor_json_input
            if candidate.exists():
                factor_json_path = candidate
            else:
                # Try as relative to project root
                candidate2 = PROJECT_ROOT / factor_json_input
                if candidate2.exists():
                    factor_json_path = candidate2
                else:
                    factor_json_path = candidate  # will fail with a clear error message
        factor_json_str = str(factor_json_path)

        # --- Find the correct Python executable ---
        # Prefer the conda env that has qlib installed
        conda_env = dotenv.get("CONDA_ENV_NAME", "quantaalpha")
        python_bin = sys.executable  # fallback

        # Dynamically detect conda base path (portable, no hardcoded paths)
        conda_prefixes = [os.path.expanduser(f"~/.conda/envs/{conda_env}")]
        try:
            import subprocess as _sp
            conda_base = _sp.check_output(
                ["conda", "info", "--base"], text=True, timeout=5
            ).strip()
            conda_prefixes.insert(0, os.path.join(conda_base, "envs", conda_env))
        except Exception:
            pass
        # Also check CONDA_PREFIX if we're already in the right env
        if os.environ.get("CONDA_PREFIX"):
            conda_prefixes.insert(0, os.environ["CONDA_PREFIX"])

        for prefix in conda_prefixes:
            candidate_bin = Path(prefix) / "bin" / "python"
            if candidate_bin.exists():
                python_bin = str(candidate_bin)
                break

        # Build CLI command
        cmd = [
            python_bin, "-u", "-m", "quantaalpha.backtest.run_backtest",
            "-c", config_path,
            "--factor-source", req.factorSource,
            "--factor-json", factor_json_str,
            "--skip-uncached",
            "-v",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        task["pid"] = proc.pid

        # Noisy warnings from Qlib / dependencies that can be safely suppressed
        _NOISY_PATTERNS = (
            "field data contains nan",
            "common_infra",
            "PyTorch models are skipped",
            "UserWarning: pkg_resources",
            "Training until validation scores",
            "FutureWarning",
            "UserWarning",
            "Did not meet early stopping",
            "num_leaves is set=",
        )

        # --- Stream stdout ---
        log_entry = None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue

            # Mirror to backend console/log
            print(f"[{task_id}] {line}", flush=True)

            # Skip noisy repeated warnings
            if any(p in line for p in _NOISY_PATTERNS):
                continue

            level = "info"
            if "ERROR" in line or "Error" in line:
                level = "error"
            elif "WARNING" in line or "Warning" in line:
                level = "warning"
            elif "done" in line.lower() or "complete" in line.lower() or "success" in line.lower() or "✓" in line:
                level = "success"

            log_entry = {
                "id": _gen_id(),
                "timestamp": _now(),
                "level": level,
                "message": line[:500],
            }
            task["logs"].append(log_entry)
            if len(task["logs"]) > 2000:
                task["logs"] = task["logs"][-2000:]

            # Broadcast log to WebSocket
            await _broadcast(task_id, {
                "type": "log",
                "taskId": task_id,
                "data": log_entry,
                "timestamp": _now(),
            })

            # Update progress for meaningful lines
            if any(kw in line for kw in ["factor", "backtest", "model", "train", "complete", "load",
                                          "[1/4]", "[2/4]", "[3/4]", "[4/4]", "result"]):
                task["progress"]["message"] = line[:200]

                # Estimate progress from run_backtest step markers
                if "[1/4]" in line:
                    task["progress"]["progress"] = 15
                elif "[2/4]" in line:
                    task["progress"]["progress"] = 35
                elif "[3/4]" in line:
                    task["progress"]["progress"] = 55
                elif "[4/4]" in line:
                    task["progress"]["progress"] = 75
                elif "result saved" in line.lower() or "backtest result" in line.lower() or "backtest complete" in line.lower():
                    task["progress"]["progress"] = 95

                task["progress"]["timestamp"] = _now()
                await _broadcast(task_id, {
                    "type": "progress",
                    "taskId": task_id,
                    "data": task["progress"],
                    "timestamp": _now(),
                })

        # --- Process exit ---
        exit_code = await proc.wait()
        task["pid"] = None
        task["status"] = "completed" if exit_code == 0 else "failed"
        task["updatedAt"] = _now()

        # Try to load backtest results from output metrics JSON
        if exit_code == 0:
            task["progress"]["phase"] = "completed"
            task["progress"]["progress"] = 100
            task["progress"]["message"] = "Backtest completed"
            _load_backtest_results(task)
        else:
            task["progress"]["message"] = f"Backtest failed (exit code: {exit_code})"

        await _broadcast(task_id, {
            "type": "result",
            "taskId": task_id,
            "data": {
                "status": task["status"],
                "metrics": task.get("metrics", {}),
            },
            "timestamp": _now(),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        task["status"] = "failed"
        task["progress"]["message"] = str(e)
        task["updatedAt"] = _now()
        await _broadcast(task_id, {
            "type": "error",
            "taskId": task_id,
            "data": {"error": str(e)},
            "timestamp": _now(),
        })


def _load_backtest_results(task: Dict[str, Any]):
    """Try to load backtest result metrics from the output directory."""
    try:
        config_path = task.get("config", {}).get("configPath") or str(
            PROJECT_ROOT / "configs" / "backtest.yaml"
        )
        with open(config_path, "r") as f:
            bt_config = yaml.safe_load(f)
        output_dir_raw = bt_config.get("experiment", {}).get(
            "output_dir", "data/result/stock/backtest_v2"
        )
        # Resolve relative output_dir against PROJECT_ROOT (run_backtest runs with cwd=PROJECT_ROOT)
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir_str = str(output_dir)

        # Look for most recent metrics JSON
        metrics_files = sorted(
            glob.glob(os.path.join(output_dir_str, "*_backtest_metrics.json")),
            key=os.path.getmtime, reverse=True,
        )
        if metrics_files:
            with open(metrics_files[0], "r") as f:
                metrics_data = json.load(f)
            # The JSON has a nested structure: { metrics: {...}, config: {...}, ... }
            # Flatten: put the inner metrics dict at the top level for the frontend,
            # but also keep meta fields like experiment_name and elapsed_seconds.
            inner_metrics = metrics_data.get("metrics", {})
            flat = {**inner_metrics}
            # Carry over useful metadata
            for key in ("experiment_name", "factor_source", "num_factors",
                        "config", "elapsed_seconds"):
                if key in metrics_data:
                    flat[f"__{key}"] = metrics_data[key]
            
            # Load cumulative excess return data from CSV
            csv_path = metrics_files[0].replace("_backtest_metrics.json", "_cumulative_excess.csv")
            if os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                if 'date' in df.columns and 'cumulative_excess_return' in df.columns:
                    cumulative_data = df[['date', 'cumulative_excess_return']].to_dict('records')
                    flat["cumulative_curve"] = [
                        {"date": r["date"], "value": r["cumulative_excess_return"]} 
                        for r in cumulative_data
                    ]

            task["metrics"] = flat
    except Exception as e:
        import traceback
        traceback.print_exc()  # print for debugging, but don't crash


# ---- System config endpoints ----

@app.get("/api/v1/system/config", response_model=ApiResponse)
async def get_system_config():
    """Read current system configuration from .env and experiment.yaml."""
    dotenv = _load_dotenv_dict()

    # Read experiment.yaml for display
    exp_yaml_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    exp_yaml_content = ""
    if exp_yaml_path.exists():
        exp_yaml_content = exp_yaml_path.read_text(encoding="utf-8")

    # Mask API keys for security
    masked_env = {}
    for k, v in dotenv.items():
        if "KEY" in k.upper() and v:
            masked_env[k] = v[:8] + "..." + v[-4:] if len(v) > 12 else "***"
        else:
            masked_env[k] = v

    return ApiResponse(
        success=True,
        data={
            "env": masked_env,
            "experimentYaml": exp_yaml_content,
            "factorLibraries": [Path(p).name for p in _find_factor_jsons()],
            "marketOptions": _discover_grey_market_options(),
        },
    )


@app.put("/api/v1/system/config", response_model=ApiResponse)
async def update_system_config(update: SystemConfigUpdate):
    """Update .env configuration (non-secret fields only)."""
    if not DOTENV_PATH.exists():
        raise HTTPException(status_code=404, detail=".env file not found")

    content = DOTENV_PATH.read_text(encoding="utf-8")
    updates = {k: v for k, v in update.model_dump().items() if v is not None}

    import re
    for key, val in updates.items():
        # Replace existing line or append
        pattern = rf"^{re.escape(key)}\s*=.*$"
        replacement = f"{key}={val}"
        new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        if n > 0:
            content = new_content
        else:
            content += f"\n{replacement}\n"

    DOTENV_PATH.write_text(content, encoding="utf-8")
    return ApiResponse(success=True, message="Configuration updated")


# ---- WebSocket endpoint ----

@app.websocket("/ws/mining/{task_id}")
async def ws_mining(websocket: WebSocket, task_id: str):
    """WebSocket for real-time experiment updates."""
    await websocket.accept()

    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)

    # Send current state immediately
    if task_id in tasks:
        try:
            await websocket.send_json({
                "type": "progress",
                "taskId": task_id,
                "data": tasks[task_id].get("progress", {}),
                "timestamp": _now(),
            })
            # Send recent logs
            for log in tasks[task_id].get("logs", [])[-20:]:
                await websocket.send_json({
                    "type": "log",
                    "taskId": task_id,
                    "data": log,
                    "timestamp": _now(),
                })
        except Exception:
            pass

    try:
        while True:
            data = await websocket.receive_text()
            # Heartbeat
            if data == "ping":
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": _now(),
                })
    except WebSocketDisconnect:
        if task_id in ws_connections:
            try:
                ws_connections[task_id].remove(websocket)
            except ValueError:
                pass


# ========================== Entry Point ==========================

def _update_mining_metrics(task: Dict[str, Any]):
    """
    Update mining task metrics from the generated factor library.
    Calculates best factor stats and extracts top 10 factors.
    """
    jsons = _find_factor_jsons()
    # Prefer library with matching suffix if configured
    target_lib = None
    config = task.get("config", {})
    suffix = config.get("librarySuffix")
    
    if suffix:
        candidate = PROJECT_ROOT / "data" / "factorlib" / f"all_factors_library_{suffix}.json"
        # Fix: If suffix is specified, we ONLY look at this file.
        # If it doesn't exist yet, it means no factors have been mined yet for this task.
        if candidate.exists():
            target_lib = str(candidate)
        else:
            # Task specific file not found -> assume empty state
            return
            
    elif jsons:
        # No suffix provided, fallback to latest existing library (legacy behavior)
        target_lib = jsons[0]
        
    if not target_lib:
        return

    # Check modification time
    try:
        mtime = os.path.getmtime(target_lib)
        created_at_str = task.get("createdAt")
        if created_at_str:
            created_at_dt = datetime.fromisoformat(created_at_str)
            # Add a small buffer (e.g. 1 second) to avoid race conditions where file is created immediately
            if mtime < created_at_dt.timestamp():
                # File is older than the task -> ignore it
                return
    except Exception:
        pass

    try:
        lib = _load_factor_library(target_lib)
        factors = lib.get("factors", {})
        
        # 1. Update basic stats
        total = len(factors)
        task["metrics"]["totalFactors"] = total
        
        high = medium = low = 0
        factor_list = []
        
        for f_id, f_info in factors.items():
            # Check if this factor was created after task start
            # If we are using a shared library file (unlikely with new logic, but possible if user forces it),
            # we must ensure we don't display old factors.
            try:
                added_at_str = f_info.get("added_at", "")
                created_at_str = task.get("createdAt", "")
                if added_at_str and created_at_str:
                    # Parse timestamps
                    # added_at usually in isoformat
                    added_at_dt = datetime.fromisoformat(added_at_str)
                    created_at_dt = datetime.fromisoformat(created_at_str)
                    if added_at_dt < created_at_dt:
                        continue
            except Exception:
                pass # If date parsing fails, be permissive or conservative? Permissive for now.

            bt = f_info.get("backtest_results", {})
            q = _classify_quality(bt)
            if q == "high": high += 1
            elif q == "medium": medium += 1
            else: low += 1
            
            # Prepare for top 10 list
            metrics = _extract_bt_metrics(bt)
            ic = metrics["ic"]
            icir = metrics["icir"]
            rank_ic = metrics["rank_ic"]
            rank_icir = metrics["rank_icir"]
            
            annual_ret = metrics["annual_return"]
            max_dd = metrics["max_drawdown"]
            
            # Calmar Ratio = Annual Return / Max Drawdown (absolute value)
            # Avoid division by zero
            cr = 0
            if max_dd < 0:
                cr = annual_ret / abs(max_dd)
            elif max_dd > 0:
                cr = annual_ret / max_dd

            cumulative_curve = bt.get("cumulative_curve") or bt.get("cumulativeCurve") or []
            if not isinstance(cumulative_curve, list):
                cumulative_curve = []
            
            factor_list.append({
                "factorName": f_info.get("factor_name", f_id),
                "factorId": f_info.get("factor_id", f_id),
                "factorExpression": f_info.get("factor_expression", ""),
                "rankIc": rank_ic,
                "rankIcir": rank_icir,
                "ic": ic,
                "icir": icir,
                "annualReturn": annual_ret,
                "sharpeRatio": metrics["information_ratio"],
                "maxDrawdown": max_dd,
                "calmarRatio": cr,
                "cumulativeCurve": cumulative_curve,
                "factorScore": _compute_factor_score(metrics),
                "quality": q,
            })

        # Ensure each run has at least one high-priority strategy candidate.
        if factor_list:
            promoted = _ensure_minimum_high_quality(factor_list)
            if promoted:
                high = max(high, 1)
                if medium > 0:
                    medium -= 1
                elif low > 0:
                    low -= 1
                task["metrics"]["qualityFloorApplied"] = True

        task["metrics"]["highQualityFactors"] = high
        task["metrics"]["mediumQualityFactors"] = medium
        task["metrics"]["lowQualityFactors"] = low
        
        # 2. Find best factor
        if factor_list:
            # Sort by composite score to favor robust generalizable factors.
            factor_list.sort(key=lambda x: x["factorScore"], reverse=True)
            best = factor_list[0]
            
            # Update task metrics with best factor's stats
            task["metrics"]["ic"] = best["ic"]
            task["metrics"]["icir"] = best["icir"]
            task["metrics"]["annualReturn"] = best["annualReturn"]
            task["metrics"]["rankIc"] = best["rankIc"]
            task["metrics"]["rankIcir"] = best["rankIcir"]
            task["metrics"]["sharpeRatio"] = best["sharpeRatio"]
            task["metrics"]["maxDrawdown"] = best["maxDrawdown"]
            task["metrics"]["factorName"] = best["factorName"]
            
            # 3. Top 10 Factors
            task["metrics"]["top10Factors"] = factor_list[:10]
            
    except Exception:
        pass # Best effort

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
