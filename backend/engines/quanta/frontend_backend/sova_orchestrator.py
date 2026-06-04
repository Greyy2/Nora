"""
SovaAnalysisOrchestrator — 3-role AI engine for the Trading page.

Role 1 (Finder):    Scan alpha result folders → collect best factors
Role 2 (Evaluator): Score, rank, filter factors by quality gates
Role 3 (Narrator):  Compose natural-language explanation + chart overlay schema

Returns a SovaAnalysis JSON consumed by the frontend chart overlay.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("QuantaAlpha.SovaOrchestrator")

# ─────────────────────────────────────────────────────────────────
# Data classes for the output schema
# ─────────────────────────────────────────────────────────────────

@dataclass
class SovaAction:
    """A single recommended trading action (entry / exit / partial)."""
    type: str          # "buy" | "sell" | "close" | "watch"
    time: Optional[int] = None     # unix timestamp (seconds), None = now
    price: Optional[float] = None  # reference price
    label: str = ""

@dataclass
class SovaZone:
    """A price zone to render as a coloured rectangle on the chart."""
    start: Optional[int] = None    # unix timestamp start
    end:   Optional[int] = None    # unix timestamp end (None = open-ended)
    low:   float = 0.0
    high:  float = 0.0
    label: str = ""
    color: str = "rgba(59,130,246,0.15)"  # default blue

@dataclass
class SovaIndicator:
    key:         str
    value:       Any
    human_label: str

@dataclass
class SovaAnalysis:
    summary:       str
    confidence:    float          # 0.0 – 1.0
    actions:       List[SovaAction]   = field(default_factory=list)
    zones:         List[SovaZone]     = field(default_factory=list)
    indicators:    List[SovaIndicator] = field(default_factory=list)
    strategy_pick: Optional[Dict[str, Any]] = None
    risk_note:     str = ""
    factors_used:  List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary":       self.summary,
            "confidence":    round(self.confidence, 3),
            "actions":       [asdict(a) for a in self.actions],
            "zones":         [asdict(z) for z in self.zones],
            "indicators":    [asdict(i) for i in self.indicators],
            "strategy_pick": self.strategy_pick,
            "risk_note":     self.risk_note,
            "factors_used":  self.factors_used,
        }


# ─────────────────────────────────────────────────────────────────
# Helper: find latest alpha result files
# ─────────────────────────────────────────────────────────────────

def _find_alpha_result_files(project_root: Path, limit: int = 5) -> List[Path]:
    """
    Scan data/result, data/results, and data/factorlib for the most recent
    JSON files that contain factor/backtest information.
    """
    patterns = [
        project_root / "data" / "result" / "**" / "*.json",
        project_root / "data" / "results" / "**" / "*.json",
        project_root / "data" / "factorlib" / "*.json",
        project_root / "data" / "factor" / "**" / "*.json",
    ]
    found: List[Path] = []
    seen: set = set()
    for pat in patterns:
        for p in glob.glob(str(pat), recursive=True):
            if p not in seen:
                seen.add(p)
                found.append(Path(p))

    # Sort newest first
    found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return found[:limit * 4]  # over-fetch; we'll filter below


def _load_factor_data(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Role 1 — Alpha Finder
# ─────────────────────────────────────────────────────────────────

class _AlphaFinder:
    """Scans result folders and extracts usable alpha factor records."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def collect(self, max_factors: int = 20) -> List[Dict[str, Any]]:
        """Return a list of factor dicts with standardised keys."""
        files = _find_alpha_result_files(self.project_root, limit=max_factors)
        factors: List[Dict[str, Any]] = []

        for fp in files:
            data = _load_factor_data(fp)
            if not data:
                continue

            # Handle factor library format: dict of factor_name → info
            if isinstance(data, dict):
                for name, info in data.items():
                    if not isinstance(info, dict):
                        continue
                    if "expression" not in info and "formulation" not in info:
                        continue
                    rec = self._normalise(name, info, fp)
                    if rec:
                        factors.append(rec)
                    if len(factors) >= max_factors:
                        break

            # Handle list format
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("factor_name") or item.get("name") or "unknown"
                    rec = self._normalise(name, item, fp)
                    if rec:
                        factors.append(rec)
                    if len(factors) >= max_factors:
                        break

            if len(factors) >= max_factors:
                break

        logger.info(f"[AlphaFinder] Collected {len(factors)} factor records from {len(files)} files")
        return factors

    @staticmethod
    def _normalise(name: str, info: Dict[str, Any], source: Path) -> Optional[Dict[str, Any]]:
        expr = info.get("expression") or info.get("formulation") or ""
        if not expr:
            return None

        bt = info.get("backtestResults") or info.get("backtest_results") or {}
        if isinstance(bt, str):
            try:
                bt = json.loads(bt)
            except Exception:
                bt = {}

        def _f(keys):
            for k in keys:
                v = bt.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
            return 0.0

        ic      = _f(["IC", "ic", "information_coefficient"])
        icir    = _f(["ICIR", "icir"])
        rank_ic = _f(["Rank IC", "rank_ic", "RankIC"])
        arr     = _f(["annual_return", "annualized_return", "ARR",
                       "1day.excess_return_with_cost.annualized_return"])
        mdd     = _f(["max_drawdown", "MDD", "mdd",
                       "1day.excess_return_with_cost.max_drawdown"])
        ir      = _f(["information_ratio", "IR", "sharpe",
                       "1day.excess_return_with_cost.information_ratio"])

        quality = info.get("quality") or "unknown"

        return {
            "name":        name,
            "expression":  expr,
            "description": info.get("description") or "",
            "ic":          ic,
            "icir":        icir,
            "rank_ic":     rank_ic,
            "annual_return": arr,
            "max_drawdown":  mdd,
            "information_ratio": ir,
            "quality":     quality,
            "source_file": str(source),
        }


# ─────────────────────────────────────────────────────────────────
# Role 2 — Alpha Evaluator
# ─────────────────────────────────────────────────────────────────

class _AlphaEvaluator:
    """Scores, ranks, and filters alpha factors. Applies risk guardrails."""

    # Weights for composite score
    W_SIGNAL    = 0.40
    W_STABILITY = 0.35
    W_RETURN    = 0.15
    W_IR        = 0.10

    def score(self, f: Dict[str, Any]) -> float:
        signal    = max(abs(f["rank_ic"]), abs(f["ic"]))
        stability = max(f["icir"], 0.0)
        ret_norm  = max(-0.5, min(1.0, f["annual_return"]))
        ir_norm   = max(-0.5, min(1.0, f["information_ratio"] * 0.1))
        mdd_pen   = max(0.0, abs(f["max_drawdown"]) - 0.35) * 0.1

        if signal == 0 and stability == 0 and f["annual_return"] == 0:
            return -1.0  # placeholder / empty factor

        return (
            self.W_SIGNAL    * signal
            + self.W_STABILITY * stability
            + self.W_RETURN    * ret_norm
            + self.W_IR        * ir_norm
            - mdd_pen
        )

    def evaluate(self, factors: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        if not factors:
            return []

        # Compute score
        scored = [(self.score(f), f) for f in factors]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Risk guardrail: reject factors with extreme drawdown
        passed = [
            f for s, f in scored
            if s > -0.5 and abs(f["max_drawdown"]) < 0.85
        ]

        result = []
        for f in passed[:top_n]:
            f = f.copy()
            f["score"] = round(self.score(f), 4)
            result.append(f)

        logger.info(f"[AlphaEvaluator] {len(passed)} factors passed guardrails → top {len(result)} selected")
        return result

    def classify_regime_from_factors(self, factors: List[Dict[str, Any]]) -> str:
        """Infer dominant regime from factor expressions."""
        exprs = " ".join(f.get("expression", "") for f in factors).lower()
        if "delta" in exprs or "momentum" in exprs:
            return "TRENDING"
        if "zscore" in exprs or "mean" in exprs or "revers" in exprs:
            return "MEAN_REVERSION"
        if "std" in exprs or "var" in exprs:
            return "HIGH_VOLATILITY"
        return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────
# Role 3 — Sova Narrator (Natural Language + Overlay Builder)
# ─────────────────────────────────────────────────────────────────

_REGIME_VI = {
    "TRENDING":       "xu hướng rõ ràng",
    "MEAN_REVERSION": "hồi quy về trung bình",
    "HIGH_VOLATILITY": "biến động cao",
    "NEUTRAL":        "trung tính / sideways",
    "TRENDING_BULL":  "xu hướng tăng",
    "TRENDING_DOWN":  "xu hướng giảm",
    "TRENDING_UP":    "xu hướng tăng",
}

_QUALITY_VI = {
    "high":    "chất lượng cao ✦",
    "medium":  "chất lượng trung bình",
    "low":     "chất lượng thấp",
    "unknown": "chưa xác định",
}

_STRATEGY_TEMPLATES = {
    "TRENDING": {
        "title": "Momentum Trend Follow",
        "description": "Theo xu hướng chính. Mua breakout, đặt stoploss dưới swing low gần nhất.",
        "risk_pct": 1.5,
    },
    "MEAN_REVERSION": {
        "title": "Range Bounce",
        "description": "Thị trường đang dao động trong vùng. Mua vùng hỗ trợ, bán vùng kháng cự.",
        "risk_pct": 1.0,
    },
    "HIGH_VOLATILITY": {
        "title": "Volatility Breakout",
        "description": "Biến động cao — chờ breakout xác nhận trước khi vào lệnh. Giảm size 50%.",
        "risk_pct": 0.75,
    },
    "NEUTRAL": {
        "title": "Wait & Watch",
        "description": "Chưa có tín hiệu rõ ràng. Quan sát thêm, không vào lệnh mới.",
        "risk_pct": 0.0,
    },
}


class _SovaNarrator:
    """
    Composes the final Sova response:
    - Natural-language summary (tiếng Việt, dài, logic, professional)
    - Overlay schema (actions + zones + indicators)
    """

    def narrate(
        self,
        message: str,
        best_factors: List[Dict[str, Any]],
        regime: str,
        context: Dict[str, Any],
    ) -> SovaAnalysis:
        asset      = context.get("asset", "tài sản")
        timeframe  = context.get("timeframe", "không xác định")
        regime_vi  = _REGIME_VI.get(regime, regime)
        strategy   = _STRATEGY_TEMPLATES.get(regime, _STRATEGY_TEMPLATES["NEUTRAL"])

        # Confidence: average factor score capped 0–1
        if best_factors:
            avg_score = sum(f.get("score", 0) for f in best_factors) / len(best_factors)
            confidence = round(min(0.95, max(0.10, avg_score * 3.5)), 2)
        else:
            confidence = 0.15

        # ── Build summary text ──────────────────────────────────────
        factor_lines = []
        for f in best_factors[:3]:
            name  = f["name"]
            ic    = f["ic"]
            arr   = f["annual_return"]
            qual  = _QUALITY_VI.get(f["quality"], f["quality"])
            expr_short = f["expression"][:60] + ("..." if len(f["expression"]) > 60 else "")
            factor_lines.append(
                f"  • **{name}** ({qual}) — IC={ic:.3f}, Lợi nhuận năm={arr*100:.1f}%\n"
                f"    Công thức: `{expr_short}`"
            )

        factors_block = "\n".join(factor_lines) if factor_lines else "  • Chưa có dữ liệu alpha đủ chất lượng."

        ic_best   = best_factors[0]["ic"]      if best_factors else 0
        arr_best  = best_factors[0]["annual_return"] if best_factors else 0
        mdd_best  = best_factors[0]["max_drawdown"]  if best_factors else 0
        ir_best   = best_factors[0]["information_ratio"] if best_factors else 0

        # Assess factor quality narrative
        if ic_best >= 0.05:
            ic_comment = f"IC={ic_best:.3f} rất tốt — tín hiệu dự báo mạnh và đáng tin cậy"
        elif ic_best >= 0.03:
            ic_comment = f"IC={ic_best:.3f} ở mức chấp nhận được — tín hiệu có giá trị nhất định"
        elif ic_best > 0:
            ic_comment = f"IC={ic_best:.3f} còn yếu — cần thêm xác nhận từ các chỉ báo khác"
        else:
            ic_comment = "Chưa có IC đáng kể — không nên dựa vào alpha này để giao dịch"

        arr_comment = (
            f"lợi nhuận năm {arr_best*100:.1f}% (tốt)" if arr_best > 0.10
            else f"lợi nhuận năm {arr_best*100:.1f}% (thấp/âm — cần cẩn thận)" if arr_best < 0
            else f"lợi nhuận năm {arr_best*100:.1f}%"
        )

        risk_note = ""
        if abs(mdd_best) > 0.35:
            risk_note = (
                f"⚠️ Drawdown tối đa {abs(mdd_best)*100:.0f}% — rủi ro cao. "
                "Hãy giảm kích thước vị thế xuống còn 50% so với bình thường."
            )
        elif abs(mdd_best) > 0.20:
            risk_note = f"Drawdown {abs(mdd_best)*100:.0f}% — quản lý rủi ro chặt."

        # Assess regime recommendation
        if regime in ("TRENDING", "TRENDING_BULL", "TRENDING_UP"):
            regime_action = (
                "Xu hướng đang rõ ràng → **chiến lược momentum** phù hợp nhất lúc này. "
                "Tìm điểm vào lệnh sau các nhịp pullback nhỏ, đặt stoploss dưới swing low gần nhất."
            )
        elif regime == "MEAN_REVERSION":
            regime_action = (
                "Thị trường đang dao động trong vùng → **chiến lược mua hỗ trợ/bán kháng cự** hiệu quả hơn. "
                "Không nên đuổi theo breakout vì xác suất revert về giữa vùng rất cao."
            )
        elif regime == "HIGH_VOLATILITY":
            regime_action = (
                "Biến động đang tăng cao → **giảm kích thước vị thế** và chỉ vào lệnh khi có xác nhận rõ ràng. "
                "Trailing stop hoặc time-based exit thường hiệu quả hơn fixed target trong giai đoạn này."
            )
        else:
            regime_action = (
                "Thị trường đang **sideways / không có xu hướng rõ ràng** → "
                "Kiên nhẫn chờ breakout xác nhận, tránh vào lệnh khi chưa có tín hiệu."
            )

        summary = f"""📊 **Phân tích thị trường {asset} — Khung {timeframe}**

Sova đã quét và đánh giá các alpha factor từ kết quả chạy gần nhất. Đây là những gì tìm thấy:

**1. Trạng thái thị trường**
Regime hiện tại: **{regime_vi}**. {regime_action}

**2. Alpha factors tốt nhất (đã lọc qua Quality Gate)**
{factors_block}

**3. Đánh giá chất lượng tín hiệu**
Factor hàng đầu có {ic_comment}, {arr_comment}. Sharpe ratio (IR) = {ir_best:.2f}.

**4. Chiến lược đề xuất: {strategy["title"]}**
{strategy["description"]}
→ Risk mỗi lệnh: **{strategy["risk_pct"]}% vốn**

**5. Kết luận**
{'Bạn có thể cân nhắc vào lệnh theo chiến lược trên với kích thước vừa phải.' if confidence > 0.5 else 'Nên chờ thêm tín hiệu xác nhận — confidence hiện tại còn thấp ({:.0f}%).'.format(confidence * 100)}
Mức độ tin cậy tổng hợp: **{confidence*100:.0f}%**"""

        # ── Build actions ────────────────────────────────────────────
        actions = self._build_actions(regime, context)

        # ── Build zones ──────────────────────────────────────────────
        zones = self._build_zones(regime, context)

        # ── Build indicators ─────────────────────────────────────────
        indicators = []
        if best_factors:
            f0 = best_factors[0]
            indicators = [
                SovaIndicator("ic",             round(f0["ic"], 4),            "IC tốt nhất"),
                SovaIndicator("icir",           round(f0["icir"], 4),          "ICIR"),
                SovaIndicator("annual_return",  round(f0["annual_return"], 4), "Lợi nhuận năm"),
                SovaIndicator("max_drawdown",   round(f0["max_drawdown"], 4),  "Drawdown tối đa"),
                SovaIndicator("confidence",     confidence,                    "Mức tin cậy Sova"),
            ]

        return SovaAnalysis(
            summary       = summary,
            confidence    = confidence,
            actions       = actions,
            zones         = zones,
            indicators    = indicators,
            strategy_pick = strategy,
            risk_note     = risk_note,
            factors_used  = [f["name"] for f in best_factors],
        )

    # ── Action builder ───────────────────────────────────────────────

    def _build_actions(self, regime: str, context: Dict[str, Any]) -> List[SovaAction]:
        now = int(time.time())
        candles = context.get("candle_snapshot") or []
        last_price = None
        if candles:
            try:
                last_price = float(candles[-1].get("close") or candles[-1].get("c", 0))
            except Exception:
                last_price = None

        actions: List[SovaAction] = []

        if regime in ("TRENDING", "TRENDING_BULL", "TRENDING_UP"):
            if last_price:
                entry = round(last_price * 0.997, 4)  # slight pullback
                tp    = round(last_price * 1.025, 4)
                sl    = round(last_price * 0.985, 4)
                actions = [
                    SovaAction("watch", now,           last_price, "Giá hiện tại"),
                    SovaAction("buy",   now + 300,     entry,      "Điểm vào đề xuất (pullback)"),
                    SovaAction("sell",  now + 3600*8,  tp,         "Take Profit (+2.5%)"),
                    SovaAction("sell",  now + 300,     sl,         "Stop Loss (-1.5%)"),
                ]
        elif regime == "MEAN_REVERSION":
            if last_price:
                sup  = round(last_price * 0.985, 4)
                res  = round(last_price * 1.015, 4)
                actions = [
                    SovaAction("buy",  now + 600, sup, "Vùng hỗ trợ — mua khi giá về"),
                    SovaAction("sell", now + 600, res, "Vùng kháng cự — bán khi giá chạm"),
                ]
        else:
            actions = [SovaAction("watch", now, last_price, "Chờ xác nhận — không vào lệnh mới")]

        return actions

    # ── Zone builder ─────────────────────────────────────────────────

    def _build_zones(self, regime: str, context: Dict[str, Any]) -> List[SovaZone]:
        candles = context.get("candle_snapshot") or []
        if not candles:
            return []

        try:
            prices = [float(c.get("close") or c.get("c", 0)) for c in candles if c]
            prices = [p for p in prices if p > 0]
            if not prices:
                return []
        except Exception:
            return []

        now     = int(time.time())
        last_p  = prices[-1]
        max_p   = max(prices)
        min_p   = min(prices)

        zones: List[SovaZone] = []

        if regime in ("TRENDING", "TRENDING_BULL", "TRENDING_UP"):
            # Support zone (recent low band)
            support_low  = round(min_p * 0.995, 4)
            support_high = round(min_p * 1.010, 4)
            zones.append(SovaZone(
                start=now - 3600*24, end=None,
                low=support_low, high=support_high,
                label="Support Zone", color="rgba(34,197,94,0.15)",
            ))
        elif regime == "MEAN_REVERSION":
            # Both support and resistance zones
            mid = (max_p + min_p) / 2
            zones.append(SovaZone(
                start=now - 3600*48, end=None,
                low=round(min_p, 4), high=round(min_p * 1.012, 4),
                label="Hỗ trợ", color="rgba(34,197,94,0.15)",
            ))
            zones.append(SovaZone(
                start=now - 3600*48, end=None,
                low=round(max_p * 0.988, 4), high=round(max_p, 4),
                label="Kháng cự", color="rgba(239,68,68,0.15)",
            ))
            zones.append(SovaZone(
                start=now - 3600*48, end=None,
                low=round(mid * 0.995, 4), high=round(mid * 1.005, 4),
                label="Vùng trung tính", color="rgba(148,163,184,0.10)",
            ))
        elif regime == "HIGH_VOLATILITY":
            # Danger zone: ±2% from last price
            zones.append(SovaZone(
                start=now - 3600*12, end=None,
                low=round(last_p * 0.98, 4), high=round(last_p * 1.02, 4),
                label="Vùng biến động cao ⚠️", color="rgba(251,191,36,0.12)",
            ))

        return zones


# ─────────────────────────────────────────────────────────────────
# Public Orchestrator
# ─────────────────────────────────────────────────────────────────

class SovaAnalysisOrchestrator:
    """
    Public entry-point. Combines all 3 roles.

    Usage:
        orch = SovaAnalysisOrchestrator(project_root=Path("/path/to/QuantaAlpha"))
        result = orch.analyze(
            message="Thị trường đang thế nào?",
            context={
                "asset": "BTCUSDT",
                "timeframe": "1h",
                "regime": {"current": "TRENDING_BULL", "distribution": {...}},
                "backtest_summary": {...},
                "candle_snapshot": [...],
                "history": [...]
            }
        )
        return result.to_dict()
    """

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            # Default: 3 levels up from this file
            project_root = Path(__file__).resolve().parents[3]
        self.project_root = Path(project_root)
        self.finder    = _AlphaFinder(self.project_root)
        self.evaluator = _AlphaEvaluator()
        self.narrator  = _SovaNarrator()

    def analyze(self, message: str, context: Dict[str, Any]) -> SovaAnalysis:
        """
        Full 3-role pipeline:
          1. Finder  → collect raw alpha factors
          2. Evaluator → score + filter
          3. Narrator → build natural-language summary + overlay
        """
        logger.info(f"[SovaOrchestrator] Analyzing: '{message[:80]}'")

        # Role 1: Find
        raw_factors = self.finder.collect(max_factors=30)

        # Role 2: Evaluate
        best_factors = self.evaluator.evaluate(raw_factors, top_n=5)

        # Regime from context or inferred from factors
        regime_ctx = context.get("regime") or {}
        regime = (
            regime_ctx.get("current")
            or self.evaluator.classify_regime_from_factors(best_factors)
        )

        # Role 3: Narrate
        analysis = self.narrator.narrate(
            message      = message,
            best_factors = best_factors,
            regime       = regime,
            context      = context,
        )

        logger.info(
            f"[SovaOrchestrator] Done — confidence={analysis.confidence}, "
            f"factors={len(best_factors)}, regime={regime}"
        )
        return analysis
