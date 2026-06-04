"""
SOVA COUNCIL BRIDGE — 5-Layer Intelligence Interface
=====================================================

Translates QuantaAlpha's CouncilVerdict + PrismData into Sova's internal
cognitive structures (feedback, hypothesis direction, operator hints).

Enables Sova to act as:
  1. Mathematical Scientist — Prism-aware hypothesis generation
  2. Rigorous Tester       — Dual-market failure diagnosis
  3. Council Participant   — CouncilVerdict-driven self-correction
  4. Adaptive Learner      — Cross-season lesson accumulation

Architecture:
  QuantaAlpha AlphaCouncil → CouncilVerdict
    ↓ (via this bridge)
  CouncilContext → HypothesisGenerator.generate(council_context=...)
    ↓
  Enriched operator/theme/window selection + reasoning chain injection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("SOVA.CouncilBridge")


# ── Verdict Classification (mirror of QuantaAlpha's enum) ────────────────────
# Kept as string constants to avoid hard import from QuantaAlpha.
VERDICT_PARADIGM_SHIFT        = "PARADIGM_SHIFT"
VERDICT_THEORY_PRACTICAL      = "THEORY_PRACTICAL_MISMATCH"
VERDICT_REGIME_MISMATCH       = "REGIME_MISMATCH"
VERDICT_OVERFITTING           = "OVERFITTING"
VERDICT_WEAK_UNIVERSALITY     = "WEAK_UNIVERSALITY"
VERDICT_STRONG_PERFORMER      = "STRONG_PERFORMER"
VERDICT_SPECIALIST            = "SPECIALIST"
VERDICT_GENERALIST            = "GENERALIST"

# ── Prism signal names ────────────────────────────────────────────────────────
PRISM_TRENDING  = "trending"
PRISM_REVERTING = "reverting"
PRISM_VOLATILE  = "volatile"
PRISM_QUIET     = "quiet"
PRISM_MIXED     = "mixed"


@dataclass
class CouncilContext:
    """
    Structured cognitive context passed from QuantaAlpha Council → Sova.

    Sova's HypothesisGenerator uses this instead of (or alongside) the
    plain text `feedback` to steer:
      - operator family selection (theory vs reality correction)
      - hypothesis axiom priority (shift if paradigm-level failure)
      - window cluster (regime mismatch → shorten/lengthen)
      - normalization mode (overfit → prefer raw / simpler)
      - theme override (actual vs theoretical divergence)
    """
    # ── Council verdict fields ─────────────────────────────────────────────
    verdict_classification: str = ""          # e.g. PARADIGM_SHIFT, OVERFITTING
    market_classification:  str = ""          # GENERALIST / SPECIALIST / NEITHER
    stock_fit_score:        float = 0.5
    forex_fit_score:        float = 0.5
    improvement_directive:  str = ""           # Free-text from Council
    consensus_action:       str = ""           # ITERATE / REJECT / ACCEPT / REFINE
    season:                 int = 1

    # ── Prism denoised context ─────────────────────────────────────────────
    prism_signal:           str = ""           # trending / reverting / volatile / quiet
    prism_clean_trend:      float = 0.0        # denoised trend strength (-1 to +1)
    prism_clean_vol:        float = 0.0        # denoised volatility level (0 to 1)
    prism_clean_momentum:   float = 0.0        # denoised momentum score (-1 to +1)
    prism_dominant_period:  int = 20           # dominant cycle period (bars)
    prism_summary_text:     str = ""           # full natural-language Prism summary

    # ── Dual-market performance (raw from backtest) ────────────────────────
    stock_ic:       float = 0.0
    stock_icir:     float = 0.0
    stock_sharpe:   float = 0.0
    stock_mdd:      float = 0.0
    forex_sharpe:   float = 0.0
    forex_cagr:     float = 0.0
    forex_mdd:      float = 0.0
    forex_quality:  str = ""                   # cao / trung_binh / thap

    # ── Derived guidance (computed by bridge, consumed by Sova) ───────────
    operator_nudges:   List[str] = field(default_factory=list)  # extra ops to prefer
    operator_penalties: List[str] = field(default_factory=list) # ops to avoid
    theme_override:    str = ""                # override primary theme if nonempty
    window_bias:       str = ""                # "shorter" / "longer" / ""
    normalization_override: str = ""           # "RAW" / "RANK" / "TS_RANK" / ""
    axiom_shift:       bool = False            # True = full paradigm shift, try new family
    confidence_floor:  float = 0.5            # minimum hypothesis confidence
    council_brief:     str = ""               # multi-line text injected into reasoning_chain

    # ── GPT-5.1 Elite Council Fields ──────────────────────────────────────────
    quality_tier:       str = ""      # "trash" / "weak" / "good" / "excellent"
    evolution_mandate:  str = ""      # FUNDAMENTAL_REDESIGN / SURGICAL_IC_BOOST / REFINE_MDD_FIX / PROTECT_DIVERSIFY
    target_ic:          float = 0.06  # Precise IC target for the next round

    def is_populated(self) -> bool:
        return bool(self.verdict_classification or self.prism_signal)

    def log_summary(self) -> str:
        return (
            f"[CouncilContext] Season={self.season} | "
            f"verdict={self.verdict_classification} | mkt={self.market_classification} | "
            f"stock_fit={self.stock_fit_score:.0%} forex_fit={self.forex_fit_score:.0%} | "
            f"prism={self.prism_signal} trend={self.prism_clean_trend:+.2f} | "
            f"action={self.consensus_action}"
        )


# ── Classification → operator/theme/window mapping ───────────────────────────
_VERDICT_OPERATOR_NUDGES: Dict[str, List[str]] = {
    VERDICT_PARADIGM_SHIFT:   ["TS_CORR", "TS_SKEW", "TS_KURT"],   # need structural insight
    VERDICT_THEORY_PRACTICAL: ["RANK", "TS_MEAN", "DELTA"],          # simplify, lift reality
    VERDICT_REGIME_MISMATCH:  ["TS_STD", "TS_MEAN", "SIGN"],         # cross-regime stability
    VERDICT_OVERFITTING:      ["RANK", "SIGN", "ABS"],                # simpler, de-noised
    VERDICT_WEAK_UNIVERSALITY:["TS_RANK", "TS_CORR", "DELTA"],       # time-series generality
    VERDICT_STRONG_PERFORMER: ["DELTA", "TS_MEAN", "TS_STD"],        # exploit momentum
    VERDICT_SPECIALIST:       ["TS_RANK", "TS_MEAN", "EMA"],         # time-series focused
    VERDICT_GENERALIST:       ["RANK", "TS_CORR", "TS_SKEW"],       # cross-sectional + corr
}

_VERDICT_OPERATOR_PENALTIES: Dict[str, List[str]] = {
    VERDICT_OVERFITTING:      ["TS_SKEW", "TS_KURT", "TS_ARGMAX"],  # complex ops overfit
    VERDICT_REGIME_MISMATCH:  ["TS_SKEW", "TS_KURT"],
    VERDICT_PARADIGM_SHIFT:   ["DELTA", "TS_MEAN"],                  # too basic, paradigm broken
}

_VERDICT_THEME_OVERRIDE: Dict[str, str] = {
    VERDICT_PARADIGM_SHIFT:   "REGIME_TRANSITION",
    VERDICT_REGIME_MISMATCH:  "REGIME_TRANSITION",
    VERDICT_OVERFITTING:      "MEAN_REVERSION",      # reversion = simpler, more stable
    VERDICT_WEAK_UNIVERSALITY:"LIQUIDITY_DYNAMICS",
}

_VERDICT_WINDOW_BIAS: Dict[str, str] = {
    VERDICT_OVERFITTING:      "shorter",   # reduce in-sample fitting
    VERDICT_PARADIGM_SHIFT:   "shorter",   # regime shift → shorter lookback
    VERDICT_STRONG_PERFORMER: "longer",    # amplify signal
    VERDICT_WEAK_UNIVERSALITY:"shorter",
}

_VERDICT_NORM_OVERRIDE: Dict[str, str] = {
    VERDICT_OVERFITTING:  "RAW",           # stop cross-sectional rank exploitation
    VERDICT_SPECIALIST:   "TS_RANK",       # single-asset friendly
}

_PRISM_OPERATOR_NUDGES: Dict[str, List[str]] = {
    PRISM_TRENDING:  ["DELTA", "EMA", "TS_MEAN"],
    PRISM_REVERTING: ["TS_MEAN", "TS_STD", "RANK"],
    PRISM_VOLATILE:  ["TS_STD", "TS_SKEW", "ABS"],
    PRISM_QUIET:     ["TS_CORR", "SIGN", "DELTA"],
    PRISM_MIXED:     ["TS_CORR", "RANK", "TS_STD"],
}

_PRISM_THEME_NUDGE: Dict[str, str] = {
    PRISM_TRENDING:  "MOMENTUM_FLOW",
    PRISM_REVERTING: "MEAN_REVERSION",
    PRISM_VOLATILE:  "RISK_MANAGEMENT",
    PRISM_QUIET:     "LIQUIDITY_DYNAMICS",
}


# ── Public API ────────────────────────────────────────────────────────────────

def build_council_context_from_verdict(
    verdict_dict: Dict[str, Any],
    prism_data: Optional[Dict[str, Any]] = None,
    backtest_metrics: Optional[Dict[str, Any]] = None,
    season: int = 1,
) -> CouncilContext:
    """
    Convert raw dicts from QuantaAlpha's CouncilVerdict + PrismData into
    a fully derived CouncilContext ready for Sova.

    Parameters
    ----------
    verdict_dict : dict
        Output of `CouncilVerdict.to_dict()` (from QuantaAlpha council).
    prism_data : dict, optional
        Output of `PrismContextInjector.build_context_dict()`.
    backtest_metrics : dict, optional
        Flat dict with keys like 'stock_ic', 'forex_sharpe', etc.
    season : int
        Current council season number.
    """
    v = verdict_dict or {}
    p = prism_data or {}
    bm = backtest_metrics or {}

    # ── Pull verdict fields ───────────────────────────────────────────────
    classification = str(v.get("classification", "") or "")
    market_cls = str(v.get("market_classification", "") or "")
    stock_fit = float(v.get("stock_fit_score", 0.5) or 0.5)
    forex_fit = float(v.get("forex_fit_score", 0.5) or 0.5)
    directive = str(v.get("improvement_directive", "") or "")
    action = str(v.get("consensus_action", "") or "")

    # ── Pull Prism fields ─────────────────────────────────────────────────
    prism_signal = str(p.get("signal", "") or "")
    prism_trend  = float(p.get("clean_trend", 0.0) or 0.0)
    prism_vol    = float(p.get("clean_volatility", 0.0) or 0.0)
    prism_mom    = float(p.get("clean_momentum", 0.0) or 0.0)
    prism_period = int(p.get("dominant_period", 20) or 20)
    prism_text   = str(p.get("summary_text", "") or "")

    # ── Pull backtest metrics ─────────────────────────────────────────────
    stock_ic    = float(bm.get("stock_ic", 0.0) or 0.0)
    stock_icir  = float(bm.get("stock_icir", 0.0) or 0.0)
    stock_sharpe= float(bm.get("stock_sharpe", 0.0) or 0.0)
    stock_mdd   = float(bm.get("stock_mdd", 0.0) or 0.0)
    forex_sharpe= float(bm.get("forex_sharpe", 0.0) or 0.0)
    forex_cagr  = float(bm.get("forex_cagr", 0.0) or 0.0)
    forex_mdd   = float(bm.get("forex_mdd", 0.0) or 0.0)
    forex_quality = str(bm.get("forex_quality", "") or "")

    # ── Derive Sova guidance ──────────────────────────────────────────────
    op_nudges   = list(_VERDICT_OPERATOR_NUDGES.get(classification, []))
    op_penalties= list(_VERDICT_OPERATOR_PENALTIES.get(classification, []))
    theme_ov    = _VERDICT_THEME_OVERRIDE.get(classification, "")
    win_bias    = _VERDICT_WINDOW_BIAS.get(classification, "")
    norm_ov     = _VERDICT_NORM_OVERRIDE.get(classification, "")
    axiom_shift = (classification == VERDICT_PARADIGM_SHIFT)

    # Prism enriches operator nudges
    prism_ops  = _PRISM_OPERATOR_NUDGES.get(prism_signal, [])
    for op in prism_ops:
        if op not in op_nudges:
            op_nudges.append(op)

    # Prism theme nudge (only if no verdct-level override)
    if not theme_ov and prism_signal:
        theme_ov = _PRISM_THEME_NUDGE.get(prism_signal, "")

    # Confidence floor: high fit = confident, low fit = explore more
    avg_fit = (stock_fit + forex_fit) / 2.0
    conf_floor = max(0.3, min(0.9, avg_fit))

    # ── Build council brief for reasoning chain ────────────────────────────
    brief_parts: List[str] = []
    if classification:
        brief_parts.append(f"[Council S{season}] Verdict: {classification}")
    if market_cls:
        brief_parts.append(f"[Council] Market class: {market_cls} | stock={stock_fit:.0%} forex={forex_fit:.0%}")
    if directive:
        brief_parts.append(f"[Council] Directive: {directive[:300]}")
    if prism_signal:
        brief_parts.append(
            f"[Prism] Denoised signal: {prism_signal} | "
            f"trend={prism_trend:+.2f} vol={prism_vol:.2f} mom={prism_mom:+.2f}"
        )
    if stock_ic:
        brief_parts.append(f"[Market] Stock IC={stock_ic:.4f} ICIR={stock_icir:.2f} Sharpe={stock_sharpe:.2f}")
    if forex_sharpe:
        brief_parts.append(f"[Market] Forex Sharpe={forex_sharpe:.2f} CAGR={forex_cagr:.2%} MDD={forex_mdd:.2%}")

    # ── Compute GPT-5.1 Elite Council mandate ────────────────────────────────
    # 4-tier quality classification based on IC level.
    # Each tier activates a different evolution mandate for the next round.
    if stock_ic < 0.02:
        quality_tier = "trash"
        evolution_mandate = "FUNDAMENTAL_REDESIGN"  # Start over from new mathematical foundation
        target_ic = 0.04
    elif stock_ic < 0.04:
        quality_tier = "weak"
        evolution_mandate = "SURGICAL_IC_BOOST"      # Focus surgery on improving IC alone
        target_ic = 0.05
    elif stock_ic < 0.06:
        quality_tier = "good"
        evolution_mandate = "REFINE_MDD_FIX"         # IC is acceptable, now tighten risk
        target_ic = 0.07
    else:
        quality_tier = "excellent"
        evolution_mandate = "PROTECT_DIVERSIFY"      # Protect the alpha, diversify variants
        target_ic = stock_ic + 0.01  # Marginal IC improvement, focus on diversification

    # ── Enrich council brief with Elite mandate ───────────────────────────────
    brief_parts.append(
        f"[Council-GPT5.1] Quality: {quality_tier.upper()} | Mandate: {evolution_mandate} | "
        f"Target IC: {target_ic:.3f} | Current IC: {stock_ic:.4f}"
    )

    ctx = CouncilContext(
        verdict_classification=classification,
        market_classification=market_cls,
        stock_fit_score=stock_fit,
        forex_fit_score=forex_fit,
        improvement_directive=directive,
        consensus_action=action,
        season=season,
        prism_signal=prism_signal,
        prism_clean_trend=prism_trend,
        prism_clean_vol=prism_vol,
        prism_clean_momentum=prism_mom,
        prism_dominant_period=prism_period,
        prism_summary_text=prism_text,
        stock_ic=stock_ic,
        stock_icir=stock_icir,
        stock_sharpe=stock_sharpe,
        stock_mdd=stock_mdd,
        forex_sharpe=forex_sharpe,
        forex_cagr=forex_cagr,
        forex_mdd=forex_mdd,
        forex_quality=forex_quality,
        operator_nudges=op_nudges,
        operator_penalties=op_penalties,
        theme_override=theme_ov,
        window_bias=win_bias,
        normalization_override=norm_ov,
        axiom_shift=axiom_shift,
        confidence_floor=conf_floor,
        council_brief="\n".join(brief_parts),
        quality_tier=quality_tier,
        evolution_mandate=evolution_mandate,
        target_ic=target_ic,
    )

    logger.info(ctx.log_summary())
    return ctx


def apply_council_context_to_prompt_spec(
    base_prompt_spec: Dict[str, Any],
    ctx: Optional[CouncilContext],
) -> Dict[str, Any]:
    """
    Merge CouncilContext guidance into an existing prompt_spec dict.

    Called by sova_adapter before passing prompt_spec to HypothesisGenerator.
    Enriches (does NOT overwrite) families, operators, normalization, windows.
    """
    spec = dict(base_prompt_spec or {})
    if ctx is None or not ctx.is_populated():
        return spec

    # ── Operator nudges (prepend, preserve existing) ─────────────────────
    existing_ops = list(spec.get("operator_hints", []))
    nudged_ops = []
    for op in ctx.operator_nudges:
        if op not in existing_ops and op not in ctx.operator_penalties:
            nudged_ops.append(op)
    spec["operator_hints"] = nudged_ops + existing_ops

    # Remove penalised operators
    spec["operator_hints"] = [
        op for op in spec["operator_hints"]
        if op not in ctx.operator_penalties
    ]

    # ── Theme override ────────────────────────────────────────────────────
    if ctx.theme_override:
        existing_themes = list(spec.get("theme_hints", []))
        if ctx.theme_override not in existing_themes:
            spec["theme_hints"] = [ctx.theme_override] + existing_themes

    # ── Normalization override ────────────────────────────────────────────
    if ctx.normalization_override:
        existing_norms = list(spec.get("normalization_modes", []))
        if ctx.normalization_override not in existing_norms:
            spec["normalization_modes"] = [ctx.normalization_override] + existing_norms

    # ── Window bias ───────────────────────────────────────────────────────
    if ctx.window_bias:
        existing_windows = list(spec.get("windows", [5, 10, 20]))
        if ctx.window_bias == "shorter":
            spec["windows"] = sorted({max(2, w // 2) for w in existing_windows} | {3, 5, 10})
        elif ctx.window_bias == "longer":
            spec["windows"] = sorted({min(120, w * 2) for w in existing_windows} | {20, 40, 60})

    # ── Axiom shift: reset families to explore new territory ─────────────
    if ctx.axiom_shift:
        current_families = set(spec.get("families", []))
        all_families = {
            "momentum", "mean_reversion", "flow", "correlation",
            "volatility", "range", "microstructure", "regime", "skewness",
        }
        unexplored = list(all_families - current_families)
        import random
        if unexplored:
            spec["families"] = random.sample(unexplored, min(3, len(unexplored)))
            logger.info(f"[Council] Axiom shift → exploring new families: {spec['families']}")

    # ── Always propagate Prism dominant period as a hint window ──────────
    if ctx.prism_dominant_period and ctx.prism_dominant_period > 0:
        wins = set(spec.get("windows", [5, 10, 20]))
        p = ctx.prism_dominant_period
        wins.add(p)
        wins.add(max(2, p // 2))
        wins.add(min(120, p * 2))
        spec["windows"] = sorted(wins)

    return spec


def apply_council_context_to_feedback(
    base_feedback: Dict[str, Any],
    ctx: Optional[CouncilContext],
) -> Dict[str, Any]:
    """
    Enrich the standard feedback dict with Council/Prism signals so that
    HypothesisGenerator.generate(feedback=...) steers correctly.
    """
    fb = dict(base_feedback or {})
    if ctx is None or not ctx.is_populated():
        return fb

    # Map verdict → suggested_intent
    _VERDICT_TO_INTENT = {
        VERDICT_OVERFITTING:      "SIMPLIFY",
        VERDICT_PARADIGM_SHIFT:   "DIVERSIFY",
        VERDICT_REGIME_MISMATCH:  "DIVERSIFY",
        VERDICT_THEORY_PRACTICAL: "EXPLOIT",
        VERDICT_WEAK_UNIVERSALITY:"EXPLOIT",
        VERDICT_STRONG_PERFORMER: "EXPLOIT",
    }
    if ctx.verdict_classification in _VERDICT_TO_INTENT:
        fb["suggested_intent"] = _VERDICT_TO_INTENT[ctx.verdict_classification]

    # Propagate dominant errors from directive text
    dominant_errors = []
    directive_lower = ctx.improvement_directive.lower()
    _DIRECTIVE_ERROR_MAP = {
        "wrong direction":   "WRONG_DIRECTION",
        "treating noise":    "TREATING_NOISE_AS_SIGNAL",
        "held too long":     "HELD_TOO_LONG",
        "chasing trend":     "CHASING_TREND_LATE",
        "premature exit":    "PREMATURE_EXIT",
        "stop too tight":    "STOP_TOO_TIGHT",
        "oversized":         "OVERSIZED_POSITION",
        "regime mismatch":   "REGIME_MISMATCH",
    }
    for keyword, error_code in _DIRECTIVE_ERROR_MAP.items():
        if keyword in directive_lower:
            dominant_errors.append(error_code)
    if dominant_errors:
        fb["dominant_errors"] = dominant_errors

    # Carry council brief as supplemental context
    if ctx.council_brief:
        fb["council_brief"] = ctx.council_brief

    # Carry Prism signal for direct conditioning
    fb["prism_signal"] = ctx.prism_signal
    fb["prism_clean_trend"] = ctx.prism_clean_trend

    # Carry last IC so HypothesisGenerator can compute reward
    if ctx.stock_ic:
        fb["last_ic"] = ctx.stock_ic

    return fb


def extract_dual_market_metrics(backtest_result: Any) -> Dict[str, float]:
    """
    Extract per-market summary metrics from a QuantaAlpha experiment object.

    Returns flat dict compatible with build_council_context_from_verdict's
    backtest_metrics parameter.
    """
    out: Dict[str, float] = {}

    # ── Stock metrics ─────────────────────────────────────────────────────
    try:
        result_table = getattr(backtest_result, "result", None) or {}
        if isinstance(result_table, dict):
            # IC
            for key in ("IC", "ic", "1day.ic", "5day.ic"):
                v = result_table.get(key)
                if isinstance(v, dict):
                    v = list(v.values())[0] if v else None
                if v is not None:
                    try:
                        out["stock_ic"] = float(v)
                        break
                    except Exception:
                        pass
            # ICIR
            for key in ("ICIR", "icir"):
                v = result_table.get(key)
                if isinstance(v, dict):
                    v = list(v.values())[0] if v else None
                if v is not None:
                    try:
                        out["stock_icir"] = float(v)
                        break
                    except Exception:
                        pass
            # Sharpe
            for key in ("1day.excess_return_with_cost.information_ratio",
                        "information_ratio", "sharpe"):
                v = result_table.get(key)
                if isinstance(v, dict):
                    v = list(v.values())[0] if v else None
                if v is not None:
                    try:
                        out["stock_sharpe"] = float(v)
                        break
                    except Exception:
                        pass
            # MDD
            for key in ("1day.excess_return_with_cost.max_drawdown", "max_drawdown"):
                v = result_table.get(key)
                if isinstance(v, dict):
                    v = list(v.values())[0] if v else None
                if v is not None:
                    try:
                        out["stock_mdd"] = float(v)
                        break
                    except Exception:
                        pass
    except Exception:
        pass

    # ── Forex metrics ─────────────────────────────────────────────────────
    try:
        forex_summary = getattr(backtest_result, "forex_backtest_results", None) or {}
        if isinstance(forex_summary, dict):
            by_factor = forex_summary.get("by_factor", {}) or {}
            sharpes, cagrs, mdds = [], [], []
            quality_counts = {"cao": 0, "trung_binh": 0, "thap": 0}
            for fv in by_factor.values():
                if not isinstance(fv, dict) or not fv.get("success"):
                    continue
                m = fv.get("metrics") or {}
                try:
                    if m.get("sharpe") is not None:
                        sharpes.append(float(m["sharpe"]))
                except Exception:
                    pass
                try:
                    if m.get("cagr") is not None:
                        cagrs.append(float(m["cagr"]))
                except Exception:
                    pass
                try:
                    if m.get("max_drawdown_pct") is not None:
                        mdds.append(float(m["max_drawdown_pct"]))
                except Exception:
                    pass
                q = str(fv.get("quality", "thap")).strip().lower()
                if q in quality_counts:
                    quality_counts[q] += 1

            def _avg(lst: list) -> float:
                return sum(lst) / len(lst) if lst else 0.0

            out["forex_sharpe"] = _avg(sharpes)
            out["forex_cagr"]   = _avg(cagrs)
            out["forex_mdd"]    = _avg(mdds)

            # Quality bucket
            if quality_counts["cao"] > 0:
                out["forex_quality"] = "cao"
            elif quality_counts["trung_binh"] > 0:
                out["forex_quality"] = "trung_binh"
            else:
                out["forex_quality"] = "thap"
    except Exception:
        pass

    return out


def build_prism_context_dict(prism_injector_output: Any) -> Dict[str, Any]:
    """
    Convert PrismContextInjector output into a simple dict for the bridge.
    Accepts either a dict (from build_context_dict) or any object with attrs.
    """
    if prism_injector_output is None:
        return {}
    if isinstance(prism_injector_output, dict):
        return prism_injector_output
    # Object with attributes
    try:
        return {
            "signal":         getattr(prism_injector_output, "signal", ""),
            "clean_trend":    getattr(prism_injector_output, "clean_trend", 0.0),
            "clean_volatility": getattr(prism_injector_output, "clean_volatility", 0.0),
            "clean_momentum": getattr(prism_injector_output, "clean_momentum", 0.0),
            "dominant_period": getattr(prism_injector_output, "dominant_period", 20),
            "summary_text":   getattr(prism_injector_output, "summary_text", ""),
        }
    except Exception:
        return {}
