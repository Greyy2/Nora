"""
Sova Cloud Brain  —  Real LLM Reasoning Layer
==============================================
NOT a template system. This module sends REAL prompts to Qwen/DeepSeek and
expects REAL mathematical reasoning in return.

Two modes of operation:
1. refine_hypothesis()    — LLM reads MarketDNA → proposes factor direction
2. refine_expression()    — LLM reads GA candidates → picks/improves best one
3. rerank_alphas()        — LLM compares candidates → selects the strongest
"""

import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("SOVA.CloudBrain")


def _resolve_explanation_mode() -> str:
    """
    Resolve explanation policy from environment.

    Modes:
    - auto: detailed when cloud LLM is available, concise on fallback
    - detailed: always request deep explanation from cloud
    - concise: always return key-point style output
    """
    mode = os.environ.get("SOVA_EXPLANATION_MODE", "auto").strip().lower()
    if mode in {"auto", "detailed", "concise"}:
        return mode
    return "auto"


def _first_n_sentences(text: str, n: int = 2) -> str:
    """Trim text to at most n sentences for concise fallback output."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()


def _merge_reasoning(local_text: str, cloud_text: str, mode: str) -> str:
    """Merge local deterministic reasoning with cloud insights without duplicating verbosity."""
    local_clean = (local_text or "").strip()
    cloud_clean = (cloud_text or "").strip()

    if not local_clean:
        return cloud_clean
    if not cloud_clean:
        return local_clean

    if mode == "concise":
        # Keep local as authoritative baseline, append one cloud cross-check sentence.
        return f"{_first_n_sentences(local_clean, 1)} {_first_n_sentences(cloud_clean, 1)}".strip()

    # Detailed modes: local 4-line structure + concise cloud validation line.
    cloud_line = _first_n_sentences(cloud_clean, 1)
    if cloud_line:
        return f"{local_clean}\n5) Cloud Cross-Check: {cloud_line}"
    return local_clean


def _mode_wants_detailed(mode: str) -> bool:
    """Return True when requests should ask for structured deep reasoning."""
    return mode in {"auto", "detailed"}


def _resolve_local_diversity_mode() -> str:
    """Control local expression diversity pressure: off | balanced | high."""
    mode = os.environ.get("SOVA_LOCAL_DIVERSITY", "balanced").strip().lower()
    if mode in {"off", "balanced", "high"}:
        return mode
    return "balanced"


def _safe_env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _safe_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _safe_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}

# ─── Qlib expression grammar (Synced with ALPHA_GENE_REGISTRY) ──────────────
_BASE_VALID_VARS = {"$open", "$close", "$high", "$low", "$volume", "$return"}


def _load_valid_vars() -> set:
    extras_raw = str(os.environ.get("SOVA_EXTRA_ALLOWED_VARIABLES", "") or "").strip()
    extra_vars = set()
    if extras_raw:
        for token in re.split(r"[\s,]+", extras_raw):
            t = token.strip()
            if not t:
                continue
            if not t.startswith("$"):
                t = f"${t}"
            if re.match(r"^\$[A-Za-z_]\w*$", t):
                extra_vars.add(t)
    return set(_BASE_VALID_VARS) | extra_vars


_VALID_VARS = _load_valid_vars()
_VALID_OPS   = {
    "RANK", "TS_RANK", "TS_MEAN", "TS_STD", "TS_CORR", "TS_MAX", "TS_MIN",
    "TS_SKEW", "TS_KURT", "DELTA", "DELAY", "SIGN", "LOG", "ABS", "POWER",
    "MIN", "MAX", "EMA", "SMA", "WMA", "TS_SUM", "TS_MEDIAN", "TS_ARGMAX",
    "TS_ARGMIN", "TS_COVARIANCE"
}


class QwenFreeClient:
    """
    Client for Qwen/DeepSeek via SiliconFlow or any OpenAI-compatible endpoint.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY", "")
        )
        # Prefer explicit Qwen/SOVA endpoint first, then generic OpenAI-compatible URL.
        env_base_url = (
            os.environ.get("SOVA_CLOUD_BASE_URL")
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.base_url = (env_base_url or base_url).rstrip("/")
        self.no_auth = _safe_env_bool("SOVA_CLOUD_NO_AUTH", False)
        self.model = (
            os.environ.get("REASONING_MODEL")
            or os.environ.get("CLOUD_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        )
        self.default_temperature = _safe_env_float("SOVA_CLOUD_TEMPERATURE", 0.55)
        self.default_max_tokens = _safe_env_int("SOVA_CLOUD_MAX_TOKENS", 1200)
        self.default_top_p = _safe_env_float("SOVA_CLOUD_TOP_P", 0.90)

    def _chat_url(self) -> str:
        # Allow users to pass full chat endpoint directly as base_url.
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not self.no_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(
        self,
        prompt: str,
        system_prompt: str = "You are SOVA, a Senior Quant AI.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if not self.api_key and not self.no_auth:
            return ""

        url = self._chat_url()
        headers = self._headers()
        t = self.default_temperature if temperature is None else float(temperature)
        mx = self.default_max_tokens if max_tokens is None else int(max_tokens)
        tp = self.default_top_p if top_p is None else float(top_p)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": t,
            "max_tokens": mx,
            "top_p": tp,
        }
        try:
            logger.info(f"[CloudBrain] → {self.model} | prompt={len(prompt)} chars")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            logger.info(f"[CloudBrain] ← response={len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"[CloudBrain] Request failed: {e}")
            return ""


class RerankClient:
    """SiliconFlow Rerank API — used to score candidate alphas."""
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.siliconflow.cn/v1"):
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
        env_base_url = (
            os.environ.get("SOVA_CLOUD_BASE_URL")
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.base_url = (env_base_url or base_url).rstrip("/")
        self.no_auth = _safe_env_bool("SOVA_CLOUD_NO_AUTH", False)
        self.model = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

    def _rerank_url(self) -> str:
        if self.base_url.endswith("/rerank"):
            return self.base_url
        return f"{self.base_url}/rerank"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if not self.no_auth and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def rerank(self, query: str, documents: List[str], top_n: int = 1) -> List[Dict[str, Any]]:
        if not self.api_key and not self.no_auth:
            return []

        url = self._rerank_url()
        headers = self._headers()
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception as e:
            logger.error(f"[Reranker] Failed: {e}")
            return []


def _extract_expressions(text: str) -> List[str]:
    """
    Extract all Qlib-style factor expressions from LLM response text.
    Handles code blocks, bullet lists, inline formulas, and assignment-style lines.
    """
    def _clean_line(line: str) -> str:
        s = str(line or "").strip().strip("`")
        # Remove markdown bullets / ordered list prefixes but preserve unary negatives.
        s = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", s)
        # Strip common prose labels from LLM outputs, e.g. "Best:", "Expression:".
        s = re.sub(r"^(?:best|expression|expr|candidate)\s*:\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*", "", s)
        # Remove sentence-ending punctuation accidentally attached to expressions.
        s = re.sub(r"([)\]0-9])\s*[.;:]$", r"\1", s)
        return s.strip().rstrip(",")

    def _looks_like_expr(s: str) -> bool:
        if not s:
            return False
        if len(s) < 8 or len(s) > 1200:
            return False
        if s.count("(") != s.count(")"):
            return False
        if "$" not in s:
            return False
        if not re.search(r"[A-Z_]+\(", s):
            return False
        if any(tok in s for tok in ("{", "}", "[", "]", '"')):
            return False
        return True

    candidates: List[str] = []

    # 1) Code blocks are highest-priority because they are usually clean expressions.
    for block in re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL):
        for raw_line in block.splitlines():
            line = _clean_line(raw_line)
            if _looks_like_expr(line):
                candidates.append(line)

    # 2) Line-level extraction from free text / bullets.
    for raw_line in (text or "").splitlines():
        line = _clean_line(raw_line)
        if _looks_like_expr(line):
            candidates.append(line)

    # 3) Span extraction for inline sentences that contain formulas.
    src = text or ""
    for m in re.finditer(r"\b[A-Z_]+\s*\(", src):
        start = m.start()
        depth = 0
        end = None
        for i in range(start, len(src)):
            ch = src[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    j = i + 1
                    # Continue through arithmetic tails at top-level, e.g. A(...) / (B(...) + 1e-8)
                    while j < len(src):
                        ch2 = src[j]
                        if ch2 in "\n;`":
                            break
                        if ch2 in "." and j + 1 < len(src) and src[j + 1].isspace():
                            break
                        j += 1
                    end = j
                    break
        if end is None:
            continue
        seg = _clean_line(src[start:end])
        if _looks_like_expr(seg):
            candidates.append(seg)

    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for c in candidates:
        k = re.sub(r"\s+", "", c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def _extract_float_metric(text: str, patterns: List[str], default: float = 0.0) -> float:
    """Extract a float metric from free-form summary text using ordered regex patterns."""
    src = text or ""
    for pat in patterns:
        m = re.search(pat, src, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            return float(m.group(1))
        except Exception:
            continue
    return default


def _is_parentheses_balanced(expr: str) -> bool:
    return (expr or "").count("(") == (expr or "").count(")")


def _resolve_normalization_policy() -> str:
    policy = str(os.environ.get("SOVA_NORMALIZATION_POLICY", "adaptive") or "adaptive").strip().lower()
    if policy in {"adaptive", "forced", "raw"}:
        return policy
    return "adaptive"


def _looks_scale_stable(expr: str) -> bool:
    e = str(expr or "").upper()
    if not e:
        return False
    if "TS_STD(" in e and "/" in e:
        return True
    if "TS_MIN(" in e and "TS_MAX(" in e and "/" in e:
        return True
    if "TS_CORR(" in e or "TS_SKEW(" in e:
        return True
    if "TS_MEAN(SIGN(" in e:
        return True
    return False


def _ensure_normalization(expr: str, is_forex: bool) -> str:
    """Ensure expression normalization follows adaptive policy, not blind wrapping."""
    e = (expr or "").strip().strip("`")
    if not e:
        return ""
    if re.match(r"^(RANK|TS_RANK)\(", e, flags=re.IGNORECASE):
        return e

    policy = _resolve_normalization_policy()
    if policy == "raw":
        return e
    if policy == "forced":
        return f"TS_RANK({e}, 60)" if is_forex else f"RANK({e})"

    # Adaptive: keep raw if already scale-stable, otherwise normalize.
    if _looks_scale_stable(e):
        return e
    return f"TS_RANK({e}, 60)" if is_forex else f"RANK({e})"


def _build_local_rationale(regime: str, theme: str, dna_summary: str, is_forex: bool = False) -> str:
    """Construct a deterministic, metric-grounded rationale when cloud LLM is unavailable."""
    hurst = _extract_float_metric(
        dna_summary,
        [r"Hurst:\s*([\d\.]+)", r"Hurst.*?([\d\.]+)"],
        default=0.50,
    )
    entropy = _extract_float_metric(
        dna_summary,
        [r"Entropy:\s*([\d\.]+)", r"Spectral\s*Entropy:\s*([\d\.]+)"],
        default=1.20,
    )
    volatility = _extract_float_metric(
        dna_summary,
        [r"Volatility:\s*([\d\.]+)", r"volatility\s*[:=]\s*([\d\.]+)"],
        default=0.015,
    )
    trend = _extract_float_metric(
        dna_summary,
        [r"Trend:\s*([\-\d\.]+)", r"trend\s*[:=]\s*([\-\d\.]+)"],
        default=0.0,
    )

    if hurst > 0.55:
        thesis = "persistent trend regime"
        mechanism = "return continuation dominates short-horizon reversal"
        ops = ["DELTA", "EMA", "TS_CORR"]
    elif hurst < 0.45:
        thesis = "mean-reverting regime"
        mechanism = "temporary dislocations revert toward local equilibrium"
        ops = ["TS_MEAN", "TS_STD", "RANK"]
    else:
        thesis = "mixed regime"
        mechanism = "signal quality depends on volatility-normalized cross-effects"
        ops = ["TS_CORR", "TS_STD", "DELTA"]

    if entropy > 1.5 and "TS_RANK" not in ops:
        ops.append("TS_RANK")
    if volatility > 0.03 and "TS_STD" not in ops:
        ops.append("TS_STD")
    if trend > 0.01 and "EMA" not in ops:
        ops.append("EMA")

    windows = [5, 20, 60] if abs(trend) > 0.01 else [10, 20, 40]
    norm = (
        "adaptive (RAW when scale-stable, else TS_RANK(..., 60))"
        if is_forex
        else "adaptive (RAW when scale-stable, else RANK(...))"
    )

    alt_a_ops = ["TS_CORR", "EMA", "TS_STD"]
    alt_b_ops = ["TS_MEAN", "TS_STD", "SIGN"]
    if "MOMENTUM" in (theme or "").upper():
        alt_a_ops = ["DELTA", "EMA", "TS_CORR"]
    elif "REVERSION" in (theme or "").upper():
        alt_b_ops = ["TS_MEAN", "TS_STD", "RANK"]

    return (
        f"1) Theory: Regime={regime} indicates {thesis}; Theme={theme} aligns with this market structure.\n"
        f"2) Mechanism: Hurst={hurst:.3f}, Entropy={entropy:.3f}, Vol={volatility:.3f} imply {mechanism}.\n"
        f"3) Primary Prescription: Use operators {', '.join(ops[:4])} with windows {windows}, normalize by {norm}.\n"
        f"4) Alternative Branch A: Use {', '.join(alt_a_ops)} for stronger trend-continuation sensitivity under volatility changes.\n"
        f"5) Alternative Branch B: Use {', '.join(alt_b_ops)} for anti-noise, mean-reversion resilience during entropy spikes.\n"
        "6) Risk Control & Validation: penalize nested RANK/depth, enforce OOS checks by regime, and reject unstable IC/MDD trade-offs."
    )


def _expression_diversity_score(expr: str) -> float:
    """Estimate structural diversity to avoid collapsing to narrow formula families."""
    e = (expr or "").upper()
    if not e:
        return 0.0
    ops = set(re.findall(r"[A-Z_]+(?=\()", e))
    vars_ = set(re.findall(r"\$[A-Z_]+", e))
    depth = e.count("(")
    diversity = len(ops) * 0.22 + len(vars_) * 0.18
    if 4 <= depth <= 10:
        diversity += 0.25
    return diversity


def _local_expression_score(expr: str, regime: str, theme: str, is_forex: bool = False) -> float:
    """Heuristic quality score for offline candidate selection."""
    e = (expr or "").strip().upper()
    if not e or not _is_parentheses_balanced(e):
        return -1e9

    score = 0.0
    score += 0.6 * e.count("TS_CORR")
    score += 0.35 * e.count("TS_STD")
    score += 0.30 * e.count("EMA")
    score += 0.25 * e.count("DELTA")
    score += 0.20 * e.count("TS_MEAN")

    if e.startswith("RANK(") or e.startswith("TS_RANK("):
        score += 0.25
    if "RANK(RANK(" in e:
        score -= 1.2

    # Complexity penalties
    score -= max(0, len(e) - 180) * 0.003
    score -= max(0, e.count("(") - 10) * 0.08

    # Theme/regime nudges
    t = (theme or "").upper()
    r = (regime or "").upper()
    if "MOMENTUM" in t or "TREND" in r:
        score += 0.35 * e.count("DELTA")
        score += 0.20 * e.count("EMA")
    if "REVERSION" in t:
        score += 0.30 * e.count("TS_MEAN")
        score += 0.20 * e.count("TS_STD")

    # Forex often lacks stable volume field quality.
    if is_forex and "$VOLUME" in e:
        score -= 0.6

    return score


def _pick_best_expression_local(candidates: List[str], regime: str, theme: str, is_forex: bool) -> Optional[str]:
    """Choose best candidate with precision-first scoring and controlled diversity pressure."""
    if not candidates:
        return None

    scored: List[Tuple[str, float, float]] = []
    for c in candidates:
        q = _local_expression_score(c, regime, theme, is_forex)
        d = _expression_diversity_score(c)
        scored.append((c, q, d))

    scored = [x for x in scored if x[1] > -1e8]
    if not scored:
        return _ensure_normalization(candidates[0], is_forex)

    scored.sort(key=lambda x: x[1], reverse=True)
    mode = _resolve_local_diversity_mode()

    if mode == "off":
        return _ensure_normalization(scored[0][0], is_forex)

    best_quality = scored[0][1]
    quality_band = 0.18 if mode == "balanced" else 0.32
    pool = [x for x in scored[:5] if x[1] >= best_quality - quality_band]
    if not pool:
        pool = scored[:1]

    diversity_weight = 0.20 if mode == "balanced" else 0.45
    pick = max(pool, key=lambda x: x[1] + diversity_weight * x[2])
    return _ensure_normalization(pick[0], is_forex)


def _parse_council_context_hints(council_context: Optional[Any]) -> Dict[str, Any]:
    """Extract light-weight optimization hints from optional council context."""
    hints = {
        "prefer_rank": False,
        "prefer_low_risk": False,
        "operator_bias": [],
        "avoid_ops": [],
        "theme_override": None,
    }
    if not council_context:
        return hints

    try:
        directive = str(getattr(council_context, "improvement_directive", "") or "")
        txt = directive.lower()

        if any(k in txt for k in ["rank", "cross-sectional", "cross sectional"]):
            hints["prefer_rank"] = True
        if any(k in txt for k in ["stability", "drawdown", "risk", "robust"]):
            hints["prefer_low_risk"] = True

        op_nudges = getattr(council_context, "operator_nudges", []) or []
        if isinstance(op_nudges, str):
            op_nudges = [op_nudges]
        hints["operator_bias"] = [str(x).upper() for x in op_nudges if x]

        theme_override = getattr(council_context, "theme_override", None)
        if theme_override:
            hints["theme_override"] = str(theme_override)

        for op in ["DELTA", "TS_CORR", "TS_STD", "TS_MEAN", "EMA", "RANK", "TS_RANK"]:
            if f"avoid {op.lower()}" in txt:
                hints["avoid_ops"].append(op)
    except Exception:
        pass

    return hints


def _pick_best_expression_local_with_context(
    candidates: List[str],
    regime: str,
    theme: str,
    is_forex: bool,
    council_context: Optional[Any] = None,
) -> Optional[str]:
    """Context-aware local selector that blends base score, diversity, and council hints."""
    if not candidates:
        return None

    hints = _parse_council_context_hints(council_context)
    theme_used = hints.get("theme_override") or theme
    diversity_mode = _resolve_local_diversity_mode()
    diversity_weight = 0.0 if diversity_mode == "off" else (0.16 if diversity_mode == "balanced" else 0.30)

    scored: List[Tuple[str, float]] = []
    for c in candidates:
        q = _local_expression_score(c, regime, theme_used, is_forex)
        if q <= -1e8:
            continue

        e = (c or "").upper()
        bonus = 0.0

        for op in hints.get("operator_bias", []):
            bonus += 0.10 * e.count(op)
        for op in hints.get("avoid_ops", []):
            bonus -= 0.25 * e.count(op)

        if hints.get("prefer_rank") and (e.startswith("RANK(") or e.startswith("TS_RANK(")):
            bonus += 0.18

        if hints.get("prefer_low_risk"):
            for op in ("TS_STD", "TS_MEAN", "EMA", "RANK", "TS_RANK"):
                bonus += 0.05 * e.count(op)
            bonus -= 0.07 * e.count("POWER")

        d = _expression_diversity_score(c)
        scored.append((c, q + bonus + diversity_weight * d))

    if not scored:
        return _pick_best_expression_local(candidates, regime, theme, is_forex)

    scored.sort(key=lambda x: x[1], reverse=True)
    return _ensure_normalization(scored[0][0], is_forex)


def _local_refine_from_feedback(expression: str, diagnosis: str, regime: str, is_forex: bool) -> str:
    """Offline surgical refiner used when cloud synthesis is unavailable."""
    expr = (expression or "").strip()
    if not expr:
        return ""

    d = (diagnosis or "").lower()
    base = expr
    variants: List[str] = [base]

    # Remove common redundancy pattern.
    base = re.sub(r"RANK\(\s*RANK\(", "RANK(", base, flags=re.IGNORECASE)
    variants.append(base)

    if "overfit" in d or "complex" in d or "fragile" in d:
        # Compress very large lookback windows to improve robustness.
        def _shrink_window(m: re.Match) -> str:
            try:
                w = int(m.group(1))
                return ", 20)" if w > 60 else m.group(0)
            except Exception:
                return m.group(0)
        variants.append(re.sub(r",\s*(\d{2,3})\)", _shrink_window, base))

    elif "weak" in d or "low" in d or "noisy" in d:
        enhancers = (
            [
                "TS_CORR($close, $volume, 20)",
                "EMA(DELTA($close, 3), 10)",
                "TS_STD($return, 20)",
            ]
            if not is_forex
            else [
                "TS_CORR($close, DELTA($close, 1), 20)",
                "EMA(DELTA($close, 3), 10)",
                "TS_STD($return, 20)",
            ]
        )
        variants.extend([f"({base}) + {enh}" for enh in enhancers])

    elif "reversion" in d and "mismatch" in d:
        variants.append(f"({base}) + TS_MEAN($return, 10)")
        variants.append(f"({base}) + TS_STD($return, 20)")

    elif "trend" in d and "mismatch" in d:
        variants.append(f"({base}) + DELTA($close, 5)")
        variants.append(f"({base}) + EMA(DELTA($close, 3), 10)")

    # Remove broken variants and choose best via local scorer.
    valid = [v for v in variants if _is_parentheses_balanced(v)]
    if not valid:
        valid = [expr]

    theme_hint = "MOMENTUM_FLOW" if "trend" in d else "MEAN_REVERSION" if "reversion" in d else "LIQUIDITY_DYNAMICS"
    chosen = _pick_best_expression_local(valid, regime, theme_hint, is_forex)
    if chosen:
        return chosen

    return _ensure_normalization(expr, is_forex)


class HybridReasoningMatrix:
    """
    Orchestrates between Genetic Algorithm output and real LLM reasoning.

    The GA generates CANDIDATE expressions quickly.
    The LLM REASONS about them — picking the most mathematically sound one
    OR proposing a superior variant.
    """

    def __init__(self):
        self.cloud_brain = QwenFreeClient()
        self.reranker = RerankClient()
        self._explanation_mode = _resolve_explanation_mode()
        self._last_refine_source = "unknown"
        self._last_feedback_refine_source = "unknown"

    # ── 1. HYPOTHESIS: LLM reads raw market data and proposes strategy ────────

    def refine_hypothesis(
        self,
        local_hypothesis: Dict[str, Any],
        dna_summary: str,
        recall_summary: str = "",
    ) -> str:
        """
        Ask the real LLM to THINK about the market regime and propose a
        mathematically grounded factor hypothesis.

        Returns the LLM's raw reasoning text (used for logging/context).
        """
        regime = local_hypothesis.get("regime", "UNKNOWN")
        theme  = local_hypothesis.get("primary_theme", local_hypothesis.get("theme", "unknown"))
        is_forex = bool(local_hypothesis.get("is_forex", False))

        if _mode_wants_detailed(self._explanation_mode):
            hyp_temp = _safe_env_float("SOVA_CLOUD_HYPOTHESIS_TEMP", 0.30)
            hyp_max_tokens = _safe_env_int("SOVA_CLOUD_HYPOTHESIS_MAX_TOKENS", 1400)
            hyp_top_p = _safe_env_float("SOVA_CLOUD_HYPOTHESIS_TOP_P", 0.92)
        else:
            hyp_temp = _safe_env_float("SOVA_CLOUD_HYPOTHESIS_TEMP_CONCISE", 0.20)
            hyp_max_tokens = _safe_env_int("SOVA_CLOUD_HYPOTHESIS_MAX_TOKENS_CONCISE", 720)
            hyp_top_p = _safe_env_float("SOVA_CLOUD_HYPOTHESIS_TOP_P_CONCISE", 0.85)

        # Local brain always runs first; cloud acts as an enhancement layer.
        local_reasoning = self.generate_mathematical_rationale(
            dna_summary=dna_summary,
            regime=regime,
            theme=theme,
            is_forex=is_forex,
        )

        wants_detailed = _mode_wants_detailed(self._explanation_mode)
        if wants_detailed:
            system = (
                "You are a GPT-5.1-grade senior quantitative researcher at an institutional hedge fund.\n"
                "### Elite Quant Reasoning Protocol ###\n"
                "Before proposing any factor, complete a 3-step reasoning chain:\n"
                "  Step 1 (Theory): market inefficiency + academic basis.\n"
                "  Step 2 (Mechanism): causal alpha transmission mechanism.\n"
                "  Step 3 (Prescription): operators, windows, normalization.\n"
                "### Risk Discipline ###\n"
                "- Target IC 0.04-0.08, ICIR > 0.5, MDD < 8%.\n"
                "- Prefer: RANK, EMA, TS_STD, TS_CORR.\n"
                "Respond in ENGLISH only."
            )
            response_format = (
                "Return exactly 3 numbered lines:\n"
                "1) Theory (<=18 words)\n"
                "2) Mechanism (<=18 words)\n"
                "3) Prescription with concrete operators/windows (<=22 words)"
            )
        else:
            system = (
                "You are a senior quant AI. Output only key conclusions with high precision. "
                "No filler, no narrative. Respond in ENGLISH only."
            )
            response_format = (
                "Return 1-2 short sentences only: core mechanism and operator prescription."
            )

        prompt = f"""Given the following market state, complete the 3-step Elite Reasoning Protocol.

## Market Regime DNA
{dna_summary}

## Regime & Theme
Regime: {regime}
Strategy Theme: {theme}

## Memory Context (successful past patterns)
{recall_summary if recall_summary else "No prior successful patterns in this regime."}

## Your Task — Complete ALL 3 Steps:
**Step 1 (Theory)**: Why does this regime favor the stated theme? (market inefficiency + academic basis, 2 sentences)
**Step 2 (Mechanism)**: What mathematical mechanism generates predictive alpha in this regime? (specific, testable)
**Step 3 (Prescription)**: Which Qlib operators, windows, and normalization? (e.g., RANK(TS_CORR($close, $volume, 10)))

Do NOT output the full expression yet — only reasoning.

## Response Format
{response_format}
"""
        response = self.cloud_brain.chat(
            prompt,
            system,
            temperature=hyp_temp,
            max_tokens=hyp_max_tokens,
            top_p=hyp_top_p,
        )
        if response:
            logger.info(f"[LLM-Hypothesis-GPT5.1] Regime={regime} | theme={theme}\n{response[:400]}")
            cleaned = response.strip()
            merged = _merge_reasoning(local_reasoning, cleaned, self._explanation_mode)
            if self._explanation_mode == "concise":
                return _first_n_sentences(merged, n=2)
            return merged

        fallback = local_reasoning
        return _first_n_sentences(fallback, n=2) if fallback else f"{theme}: use stable normalized operators for this regime."

    def generate_mathematical_rationale(self, dna_summary: str, regime: str, theme: str, is_forex: bool = False) -> str:
        """
        MATH-FIRST Fallback: Derive rationale purely from MarketDNA metrics.
        No generic templates, just direct logic.
        """
        return _build_local_rationale(regime, theme, dna_summary, is_forex=is_forex)

    def refine_expression(
        self,
        candidates: List[str],
        regime: str,
        theme: str,
        dna_summary: str,
        is_forex: bool = False,
        council_context: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Ask the LLM to evaluate GA-generated expressions and either:
        - Select the best one with mathematical justification, OR
        - Propose an improved expression based on its reasoning.

        Always returns a valid Qlib expression or None if LLM fails.
        """
        self._last_refine_source = "unknown"
        if not candidates:
            return None

        norm = (
            "adaptive (prefer RAW when scale-stable, otherwise TS_RANK(..., 60))"
            if is_forex
            else "adaptive (prefer RAW when scale-stable, otherwise RANK(...))"
        )
        market_type = "single-asset Forex/XAU" if is_forex else "cross-sectional Stocks"
        cands_text = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(candidates[:8]))

        wants_detailed = _mode_wants_detailed(self._explanation_mode)
        hints = _parse_council_context_hints(council_context)
        council_directive = ""
        if council_context:
            directive = str(getattr(council_context, "improvement_directive", "") or "")
            if directive:
                council_directive = f"\n## Council Directive\n{directive}\n"

        if wants_detailed:
            refine_temp = _safe_env_float("SOVA_CLOUD_REFINE_TEMP", 0.18)
            refine_max_tokens = _safe_env_int("SOVA_CLOUD_REFINE_MAX_TOKENS", 900)
            refine_top_p = _safe_env_float("SOVA_CLOUD_REFINE_TOP_P", 0.82)
        else:
            refine_temp = _safe_env_float("SOVA_CLOUD_REFINE_TEMP_CONCISE", 0.12)
            refine_max_tokens = _safe_env_int("SOVA_CLOUD_REFINE_MAX_TOKENS_CONCISE", 420)
            refine_top_p = _safe_env_float("SOVA_CLOUD_REFINE_TOP_P_CONCISE", 0.70)

        system = (
            "You are a GPT-5.1-grade senior quant researcher. Evaluate factor expressions for an institutional alpha model.\n"
            "### Elite Self-Critique Protocol ###\n"
            "For each candidate: (a) compute likely IC range, (b) identify weaknesses, (c) propose if a fix is possible.\n"
            "After evaluating all candidates, select the BEST or propose an improved alternative.\n"
            "### Risk Discipline ###\n"
            "- Target: IC 0.04-0.08, MDD < 8%, ICIR > 0.5.\n"
            "- Prefer stable operators: RANK, EMA, TS_STD, TS_CORR.\n"
            f"Output ONLY valid Qlib expressions using: {', '.join(sorted(_VALID_VARS))} and operators: {', '.join(sorted(_VALID_OPS))}.\n"
            "Respond in ENGLISH only."
        )

        response_format = (
            "Return EXACTLY ONE expression on its own line, then one short justification sentence."
            if wants_detailed
            else "Return EXACTLY ONE valid expression only. No explanation."
        )

        prompt = f"""## Context
Market: {market_type}
Regime: {regime}
Strategy Theme: {theme}
        Normalization Policy: {norm}

## Market DNA Summary
{dna_summary}

## Candidate Expressions (generated by Genetic Algorithm)
{cands_text}
{council_directive}

## Allowed Variables
{', '.join(sorted(_VALID_VARS))}  (NO other variables allowed)

## Allowed Operators
{', '.join(sorted(_VALID_OPS))}  (NO other operators allowed)

## Your Task
Analyze the candidates above. Select the ONE best expression OR propose a superior improved expression that:
1. Captures the **{theme.replace('_', ' ').lower()}** signal in the **{regime}** regime
2. Uses non-redundant operators (e.g., NOT RANK(RANK(...)))
3. Has balanced parentheses and only allowed variables/operators
4. Uses adaptive normalization: keep RAW only when the expression is already scale-stable; otherwise normalize appropriately

## Response Format
{response_format}

Example:
RANK(TS_CORR($close, $volume, 20))
Justification: Captures volume-price co-movement which leads price in TRENDING regimes.

Your answer:"""

        response = self.cloud_brain.chat(
            prompt,
            system,
            temperature=refine_temp,
            max_tokens=refine_max_tokens,
            top_p=refine_top_p,
        )
        if not response:
            local = _pick_best_expression_local_with_context(
                candidates,
                regime,
                hints.get("theme_override") or theme,
                is_forex,
                council_context=council_context,
            )
            if local:
                self._last_refine_source = "local"
                logger.info(f"[LOCAL-Refine] ✓ Regime={regime} | chose: {local[:100]}")
                return local
            self._last_refine_source = "local"
            return _ensure_normalization(candidates[0], is_forex)

        # Parse response
        exprs = _extract_expressions(response)
        if exprs:
            chosen = exprs[0]
            self._last_refine_source = "cloud"
            logger.info(f"[LLM-Refine] ✓ Regime={regime} | chose: {chosen[:100]}")
            return chosen

        # Fallback: look for expression inline in first non-empty line
        for line in response.splitlines():
            line = line.strip()
            if (
                "$" in line
                and line.count("(") == line.count(")")
                and (re.match(r"^(?:[A-Z_]+\(|\()", line) is not None)
            ):
                self._last_refine_source = "cloud"
                logger.info(f"[LLM-Refine] ✓ Inline parse: {line[:100]}")
                return line

        logger.warning(f"[LLM-Refine] Could not parse expression from response:\n{response[:200]}")
        local = _pick_best_expression_local_with_context(
            candidates,
            regime,
            hints.get("theme_override") or theme,
            is_forex,
            council_context=council_context,
        )
        if local:
            self._last_refine_source = "local"
            logger.info(f"[LOCAL-Refine] ✓ fallback pick for {regime}: {local[:100]}")
            return local
        self._last_refine_source = "local"
        return _ensure_normalization(candidates[0], is_forex)

    def rerank_alphas(self, query: str, alphas: List[str]) -> Optional[str]:
        """Use Cloud Reranker to pick the best-fitting Alpha for the regime."""
        results = self.reranker.rerank(query, alphas, top_n=1)
        if results:
            best = results[0]["document"]["text"]
            score = results[0]["relevance_score"]
            logger.info(f"[Reranker] Best: {best[:80]} (score={score:.4f})")
            return best
        return None

    def refine_expression_from_feedback(
        self,
        expression: str,
        ic: float,
        icir: float,
        regime: str,
        diagnosis: str,
        dna_summary: str,
        is_forex: bool = False,
        council_context: Optional[Any] = None,
        is_expansion: bool = False,
        prescription: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        The "Closed-Loop Evolution" core:
        LLM analyzes a previous formula and its performance, then
        evolves it into a descendant branch (Surgical Fix or Strategic Expansion).
        """
        self._last_feedback_refine_source = "unknown"
        norm = (
            "adaptive (prefer RAW when scale-stable, otherwise TS_RANK(..., 60))"
            if is_forex
            else "adaptive (prefer RAW when scale-stable, otherwise RANK(...))"
        )
        market_type = "single-asset Forex/XAU" if is_forex else "cross-sectional Stocks"
        wants_detailed = _mode_wants_detailed(self._explanation_mode)
        hints = _parse_council_context_hints(council_context)

        if is_expansion:
            fb_temp = _safe_env_float("SOVA_CLOUD_FEEDBACK_EXPAND_TEMP", 0.24)
            fb_max_tokens = _safe_env_int("SOVA_CLOUD_FEEDBACK_EXPAND_MAX_TOKENS", 1000)
            fb_top_p = _safe_env_float("SOVA_CLOUD_FEEDBACK_EXPAND_TOP_P", 0.88)
        else:
            fb_temp = _safe_env_float("SOVA_CLOUD_FEEDBACK_FIX_TEMP", 0.14)
            fb_max_tokens = _safe_env_int("SOVA_CLOUD_FEEDBACK_FIX_MAX_TOKENS", 820)
            fb_top_p = _safe_env_float("SOVA_CLOUD_FEEDBACK_FIX_TOP_P", 0.78)

        if not wants_detailed:
            fb_temp = _safe_env_float("SOVA_CLOUD_FEEDBACK_TEMP_CONCISE", min(fb_temp, 0.15))
            fb_max_tokens = _safe_env_int("SOVA_CLOUD_FEEDBACK_MAX_TOKENS_CONCISE", min(fb_max_tokens, 520))
            fb_top_p = _safe_env_float("SOVA_CLOUD_FEEDBACK_TOP_P_CONCISE", min(fb_top_p, 0.80))

        system = (
            "You are Sova, an elite quantitative researcher. You specialize in alpha perfection. "
            "Your goal is to reach IC > 0.06 within 5 rounds. You do NOT guess; you perform surgical math. "
            "Respond in ENGLISH only. Output ONLY valid Qlib expressions."
        )

        # ── Council & Prescription Engagement ────────────────────────────
        council_directive = ""
        if council_context:
            council_directive = f"\n## AI Council Master Directive\n{getattr(council_context, 'improvement_directive', 'Prioritize stability and low-risk operators.')}"

        prescriptive_guidance = ""
        if prescription:
            s_ops = ", ".join(prescription.get("suggested_ops", []))
            instr = prescription.get("instruction", "")
            prescriptive_guidance = f"\n## Surgical Prescription\n- Recommended Operators: {s_ops}\n- Expert Advice: {instr}"
            if prescription.get("needs_rank"):
                prescriptive_guidance += "\n- CRITICAL: Ensure the final expression is wrapped in RANK() or TS_RANK()."

        # ── Mode Selection ───────────────────────────────────────────────
        mode_instruction = ""
        if is_expansion:
             mode_instruction = (
                 "### MODE: STRATEGIC EXPANSION ###\n"
                 "Objective: Increase alpha magnitude without losing the original logic.\n"
                 "Technique: Interaction terms (variable A * variable B) or multi-horizon scaling (TS_MEAN short / TS_MEAN long).\n"
                 "Requirement: IC must improve significantly (>0.01 IC jump)."
             )
        else:
             mode_instruction = (
                 f"### MODE: SURGICAL FIX ###\n"
                 f"Objective: Correct the identified '{diagnosis}' failure with minimal structural change.\n"
                 "Technique: Replace one operator, adjust one window, or change the normalization layer.\n"
                 "Requirement: Precise correction. No 'hallucinated' variables."
             )

        prompt = f"""## Evolution Context
Original Expression: {expression}
Performance: IC={ic:.4f}, ICIR={icir:.4f}
Regime: {regime}
Diagnosis: {diagnosis}
Market: {market_type}
Normalization Policy: {norm}
{council_directive}
{prescriptive_guidance}

## Evolution Strategy
{mode_instruction}

## Market DNA Summary
{dna_summary}

## Allowed Variables
{', '.join(sorted(_VALID_VARS))}

## Allowed Operators
{', '.join(sorted(_VALID_OPS))}

## Response Format
{('Return EXACTLY ONE improved expression on its own line, then exactly ONE sentence of mathematical justification.' if wants_detailed else 'Return EXACTLY ONE improved expression only. No explanation.')}

Your answer:"""

        response = self.cloud_brain.chat(
            prompt,
            system,
            temperature=fb_temp,
            max_tokens=fb_max_tokens,
            top_p=fb_top_p,
        )
        if not response:
            local = _local_refine_from_feedback(expression, diagnosis, regime, is_forex)
            if local:
                logger.info(f"[LOCAL-Synthesis] ✓ fallback for {diagnosis}: {expression[:70]} → {local[:100]}")
                self._last_feedback_refine_source = "local"
                return _pick_best_expression_local_with_context(
                    [expression, local],
                    regime,
                    hints.get("theme_override") or ("MOMENTUM_FLOW" if "trend" in diagnosis else "MEAN_REVERSION"),
                    is_forex,
                    council_context=council_context,
                ) or local
            self._last_feedback_refine_source = "local"
            return _ensure_normalization(expression, is_forex)

        exprs = _extract_expressions(response)
        if exprs:
            chosen = exprs[0]
            self._last_feedback_refine_source = "cloud"
            logger.info(f"[LLM-Synthesis] ✓ Fix for {diagnosis}: {expression} → {chosen}")
            return chosen

        # Fallback inline parse
        for line in response.splitlines():
            line = line.strip()
            if (
                "$" in line
                and line.count("(") == line.count(")")
                and (re.match(r"^(?:[A-Z_]+\(|\()", line) is not None)
            ):
                self._last_feedback_refine_source = "cloud"
                logger.info(f"[LLM-Synthesis] ✓ Inline parse fix: {line[:100]}")
                return line

        local = _local_refine_from_feedback(expression, diagnosis, regime, is_forex)
        if local:
            logger.info(f"[LOCAL-Synthesis] ✓ parse-fallback for {diagnosis}: {local[:100]}")
            self._last_feedback_refine_source = "local"
            return _pick_best_expression_local_with_context(
                [expression, local],
                regime,
                hints.get("theme_override") or ("MOMENTUM_FLOW" if "trend" in diagnosis else "MEAN_REVERSION"),
                is_forex,
                council_context=council_context,
            ) or local

        self._last_feedback_refine_source = "local"
        return _ensure_normalization(expression, is_forex)


class SovaNarrativeEngine:
    """
    Sova's Language Voice Layer.

    Transforms mathematical reasoning (Hurst, IC, Regime, Operators) into
    fluent, intelligent natural language — like an elite quant analyst
    briefing their portfolio team.

    This runs in PARALLEL with mathematical synthesis. Two outputs, one engine:
     - [MATH] Tight, symbolic, operator-precise.
     - [NARRATIVE] Fluent, confident, explainable like GPT.
    """

    _SYSTEM_PROMPT_DETAILED = (
        "You are Sova, an elite AI quantitative analyst at a top-tier hedge fund. "
        "Explain with expert depth, clear structure, and causal precision. "
        "Use plain professional language, avoid filler, and tie every claim to metrics/operators. "
        "Respond in English only."
    )

    _SYSTEM_PROMPT_CONCISE = (
        "You are Sova, an elite AI quantitative analyst. "
        "Respond with only key points, high precision, and no unnecessary words. "
        "Respond in English only."
    )

    # Smart offline fallback narratives keyed by regime + theme
    _FALLBACK_NARRATIVES = {
        ("EXPONENTIAL_BULL", "MOMENTUM_FLOW"): (
            "The market is in a persistent upward regime with strong trend conviction. "
            "I'm positioning with momentum-following factors that capture velocity signals, "
            "prioritizing DELTA and correlation-based operators to ride the continuation."
        ),
        ("MEAN_REVERSION_ZONE", "MEAN_REVERSION"): (
            "Price dispersion has stretched beyond statistical norms in this regime — "
            "a classic setup for reversion. I'm deploying Z-score structures to capture "
            "the predictable snapback toward equilibrium."
        ),
        ("CHAOTIC_NOISE", "LIQUIDITY_DYNAMICS"): (
            "This is a noise-dominated environment with elevated entropy. The playbook here "
            "is to focus on structural liquidity signals — volume-price divergence and "
            "microstructure patterns — rather than directional momentum."
        ),
        ("STABLE_ACCUMULATION", "MOMENTUM_FLOW"): (
            "The market is in quiet accumulation mode — low volatility, persistent drift. "
            "I'm using trend-following and relative strength factors to capture "
            "the gradual institutional buildup before the next expansion phase."
        ),
        ("CAPITULATION_CRASH", "MEAN_REVERSION"): (
            "Panic selling has created extreme price dislocations. "
            "In these conditions, contrarian mean-reversion is the highest-conviction trade — "
            "deploying range-position and Z-score structures to fade the overreaction."
        ),
    }

    def __init__(self):
        self._client = QwenFreeClient()
        self._explanation_mode = _resolve_explanation_mode()

    def narrate(
        self,
        regime: str,
        primary_theme: str,
        dna_summary: str,
        operators_chosen: List[str],
        recall_expressions: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a fluent natural language narrative for Sova's current reasoning.

        Returns a 2-3 sentence explanation that sounds like an intelligent
        senior quant — not a template, not dry code output.
        """
        ops_str = ", ".join(operators_chosen[:4]) if operators_chosen else "composite operators"
        recall_str = ""
        if recall_expressions:
            recall_str = f"\nHistorical edge:\n" + "\n".join(
                f"  • {e[:60]}..." for e in recall_expressions[:2]
            )

        wants_detailed = self._explanation_mode in {"auto", "detailed"}
        system_prompt = self._SYSTEM_PROMPT_DETAILED if wants_detailed else self._SYSTEM_PROMPT_CONCISE

        if wants_detailed:
            narrative_temp = _safe_env_float("SOVA_CLOUD_NARRATIVE_TEMP", 0.34)
            narrative_max_tokens = _safe_env_int("SOVA_CLOUD_NARRATIVE_MAX_TOKENS", 720)
            narrative_top_p = _safe_env_float("SOVA_CLOUD_NARRATIVE_TOP_P", 0.95)
        else:
            narrative_temp = _safe_env_float("SOVA_CLOUD_NARRATIVE_TEMP_CONCISE", 0.18)
            narrative_max_tokens = _safe_env_int("SOVA_CLOUD_NARRATIVE_MAX_TOKENS_CONCISE", 280)
            narrative_top_p = _safe_env_float("SOVA_CLOUD_NARRATIVE_TOP_P_CONCISE", 0.80)

        style_instruction = (
            "Return a long, highly detailed, and natural-sounding explanation (at least 3-4 paragraphs).\n"
            "Include:\n"
            "1) A deep analysis of the current regime and market context.\n"
            "2) The economic and financial rationale behind the strategy.\n"
            "3) A thorough explanation of how the complex mathematical formula (operators) translates this rationale into actionable signals.\n"
            "4) Detailed risk-control mechanisms embedded in the formula.\n"
            "Write in a fluent, professional, yet natural and accessible tone."
            if wants_detailed
            else "Return 1-2 short sentences with only the main actionable conclusion."
        )

        prompt = f"""## Current Market State
{dna_summary}

## My Analysis
- Regime: {regime.replace("_", " ")}
- Strategy: {primary_theme.replace("_", " ")}
- Operators selected: {ops_str}
{recall_str}

{style_instruction}
Make the explanation unambiguous and directly tied to the provided metrics."""

        try:
            response = self._client.chat(
                prompt,
                system_prompt,
                temperature=narrative_temp,
                max_tokens=narrative_max_tokens,
                top_p=narrative_top_p,
            )
            if response and len(response.strip()) > 20:
                logger.info(f"[Narrative] Generated {len(response)} chars for {regime}/{primary_theme}")
                cleaned = response.strip()
                if self._explanation_mode == "concise":
                    return _first_n_sentences(cleaned, n=2)
                return cleaned
        except Exception as e:
            logger.debug(f"[Narrative] LLM unavailable: {e}")

        # Fallback: smart context-aware offline narrative
        fallback = self._FALLBACK_NARRATIVES.get(
            (regime, primary_theme),
            self._FALLBACK_NARRATIVES.get((regime, "MOMENTUM_FLOW"))
        )
        if fallback:
            # If cloud is unavailable, always return concise and precise key points.
            return _first_n_sentences(fallback, n=2)

        # Final generic fallback: data-derived, still readable
        h_text = ""
        if "Hurst" in dna_summary:
            try:
                h = float(re.search(r"Hurst.*?([0-9.]+)", dna_summary).group(1))
                if h > 0.55:
                    h_text = "The market displays trend persistence (Hurst > 0.5), reinforcing momentum-type signals. "
                elif h < 0.45:
                    h_text = "The market shows mean-reverting behavior (Hurst < 0.5), favoring reversion strategies. "
            except: pass

        final_text = (
            f"{h_text}In this {regime.replace('_', ' ').lower()} environment, "
            f"I'm deploying {ops_str} to capture the dominant structural signal. "
            f"My selection is grounded in the current regime DNA and prior performance data."
        )
        return _first_n_sentences(final_text, n=2)
