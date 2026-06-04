"""
SOVA RIGHT BRAIN: The Reasoning Matrix (LLM Core)
Role: Recursive Thinking, Alpha Evolution, Strategic Advisory, Trade Analysis
Architecture: Chain-of-Thought Engine + Genetic Evolver (RL-guided) + Chess-Style Trade Analyst
Intelligence Level: Senior AI Agent - Creative Mind with Continuous Self-Learning
"""

import os
import json
import re
import math
import logging
import random
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, deque, Counter
from pathlib import Path

import numpy as np

from sova_paths import memory_path, get_vortex_root

# ─── Risk Discipline Operators ─────────────────────────────────────────────
# These operators produce more stable, risk-adjusted signals.
# Applying a SMALL prior bias (1.2x) encourages stability without blocking creativity.
# Reduced from the original 2.5x which suppressed mathematical diversity.
LOW_RISK_OPERATORS = {
    "RANK", "TS_RANK", "EMA", "SMA", "TS_MEAN", "TS_STD", "TS_CORR",
    "TS_MEDIAN", "TS_MIN", "TS_MAX"
}

# ─── Creative Exploration Operators ──────────────────────────────────────────
# Operators that enable novel mathematical structures beyond cross-sectional RANK.
# Given a slight boost to ensure they compete on equal footing with LOW_RISK_OPERATORS.
CREATIVE_OPERATORS = {
    "DELTA", "TS_SKEW", "TS_KURT", "TS_COVARIANCE", "TS_ARGMAX", "TS_ARGMIN",
    "LOG", "EXP", "SIGN", "ABS", "SQRT", "INV"
}


class ThompsonSampler:
    """
    Bayesian Multi-Armed Bandit using Thompson Sampling.
    Each arm (operator / template / strategy) has Beta(alpha, beta) prior.
    After each trial: success -> alpha += reward, failure -> beta += 1.
    Sampling draws a probability from the posterior and picks argmax.
    This replaces all random.choice calls with learned probability weights.
    """

    def __init__(self, arms: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.arms = list(arms)
        self.alpha = {}
        self.beta = {}
        
        for a in arms:
            # Apply calibrated prior bias:
            # - LOW_RISK_OPERATORS: 1.2x boost for stability stability (was 2.5x — too aggressive, blocked creativity)
            # - CREATIVE_OPERATORS: 1.15x boost to ensure they compete effectively
            # - Other operators: neutral prior
            if a.upper() in LOW_RISK_OPERATORS:
                self.alpha[a] = prior_alpha * 1.2  # Mild stability bias
                self.beta[a] = prior_beta
            elif a.upper() in CREATIVE_OPERATORS:
                self.alpha[a] = prior_alpha * 1.15  # Creativity encouragement
                self.beta[a] = prior_beta
            else:
                self.alpha[a] = prior_alpha
                self.beta[a] = prior_beta

    def sample(self, deterministic: bool = True) -> str:
        """Draw one sample from each arm's Beta posterior, return arm with highest draw.
        If deterministic=True, returns the arm with the highest expected value (alpha / (alpha + beta)).
        """
        if deterministic:
            expected_values = {a: self.alpha[a] / (self.alpha[a] + self.beta[a]) for a in self.arms}
            return max(expected_values, key=expected_values.__getitem__)
        
        draws = {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return max(draws, key=draws.__getitem__)

    def sample_top_k(self, k: int, deterministic: bool = True) -> List[str]:
        """Sample without replacement: draw from posteriors, return top-k arms.
        If deterministic=True, uses expected values instead of random beta draws.
        """
        if deterministic:
            expected_values = {a: self.alpha[a] / (self.alpha[a] + self.beta[a]) for a in self.arms}
            return sorted(expected_values, key=expected_values.__getitem__, reverse=True)[:k]
            
        draws = {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return sorted(draws, key=draws.__getitem__, reverse=True)[:k]

    def update(self, arm: str, reward: float):
        """
        Update posterior.
        reward > 0 (typically 0..1 scale of IC) -> strengthen arm.
        Graduated rewards:
        - IC >= 0.08: Institutional-grade super-reward (5x) — peak alpha
        - IC >= 0.05: Strong reward (3x) — good alpha
        - IC >= 0.02: Standard reward (1x) — usable alpha
        - reward < 0: Penalize arm
        """
        if arm not in self.alpha:
            self.alpha[arm] = 1.0
            self.beta[arm] = 1.0
        
        if reward >= 0.08:
            # Super-reward for institutional-grade alpha (IC 0.08+)
            self.alpha[arm] += reward * 5.0
        elif reward >= 0.05:
            # Strong reward for good alpha (IC 0.05+)
            self.alpha[arm] += reward * 3.0
        elif reward >= 0.02:
            # Standard reward for usable alpha (IC 0.02+)
            self.alpha[arm] += reward
        elif reward > 0:
            # Weak reward — just slightly reinforce
            self.alpha[arm] += reward * 0.5
        else:
            self.beta[arm] += abs(reward) + 0.1

    def add_arm(self, arm: str):
        if arm not in self.alpha:
            self.alpha[arm] = 1.0
            self.beta[arm] = 1.0
            self.arms.append(arm)

    def get_probabilities(self) -> Dict[str, float]:
        """Return mean of each arm's Beta posterior = alpha/(alpha+beta)."""
        return {a: self.alpha[a] / (self.alpha[a] + self.beta[a]) for a in self.arms}

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": self.arms, "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThompsonSampler":
        obj = cls(d["arms"])
        obj.alpha = d["alpha"]
        obj.beta = d["beta"]
        return obj


class ReasoningMemory:
    """
    Persistent memory for learned reasoning weights.
    Saves/loads ThompsonSampler state per (regime, context) key.
    Enables cross-session learning.
    """
    SAVE_PATH = memory_path("reasoning_weights.json")

    def __init__(self):
        self._samplers: Dict[str, ThompsonSampler] = {}
        self._load()

    def get_sampler(self, key: str, arms: List[str]) -> ThompsonSampler:
        if key not in self._samplers:
            self._samplers[key] = ThompsonSampler(arms)
        else:
            # Add any new arms not seen before
            for arm in arms:
                self._samplers[key].add_arm(arm)
        return self._samplers[key]

    def update(self, key: str, arm: str, reward: float):
        """Update (and create if needed) the sampler for a key.

        This is critical for continuous learning: we must be able to learn from
        new arms/keys that appear during evolution/refinement.
        """
        try:
            if key not in self._samplers:
                self._samplers[key] = ThompsonSampler([arm])
            else:
                self._samplers[key].add_arm(arm)
            self._samplers[key].update(arm, float(reward))
        except Exception:
            # Never break the main loop due to memory persistence.
            pass
        self._save()

    def _load(self):
        try:
            self.SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if self.SAVE_PATH.exists():
                with open(self.SAVE_PATH) as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    self._samplers[k] = ThompsonSampler.from_dict(v)
        except Exception:
            pass

    def _save(self):
        try:
            with open(self.SAVE_PATH, "w") as f:
                json.dump({k: v.to_dict() for k, v in self._samplers.items()}, f, indent=2)
        except Exception:
            pass


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

from sova_matching_learning import (
    NeuralPerceptionLayer, SynapticVortexMemory, GovernanceOracle,
    ICCalculator, MemoryImpression, BacktestMetrics, SignalDenoiser,
    EnsembleJudge, MarketDNA, AdvancedSignalProcessor
)

from summarizer import StrategicSummarizer                                                                                                                                                                                                                                                                                                                                                                                                                                               # HybridReasoningMatrix no longer used — all reasoning is internal

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SOVA.RightBrain")

# ─── SOVA ALPHA GENE REGISTRY (Operators & Constants) ──────────────────────
STOCK_ALPHA_REGISTRY = {
    "UNARY": ["LOG", "ABS", "SIGN", "SQRT", "INV", "EXP"],
    "BINARY": ["DELAY", "DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SKEW", "TS_KURT",
               "TS_SUM", "TS_MEDIAN", "TS_ARGMAX", "TS_ARGMIN", "EMA", "SMA", "WMA", "TS_RANK"],
    "TRINARY": ["TS_CORR", "TS_COVARIANCE"],
    "VAR": ["$close", "$open", "$high", "$low", "$volume", "$return"],
    "ARITHMETIC": ["+", "-", "*", "/"],
    "DERIVED": [
        "$close - $open", "$high - $low", "($close - $low) / ($high - $low + 1e-8)",
        "$close / $open - 1", "($high + $low) / 2", "$volume * $close",
        "DELTA($close, 1)", "($close - TS_MEAN($close, 20)) / (TS_STD($close, 20) + 1e-8)",
        "LOG($close / DELAY($close, 1) + 1e-8)", "RANK($close) - RANK($open)"
    ],
    "SPECTRAL": ["TS_SKEW", "TS_KURT"],
    "CROSS_ASSET": [
        "TS_CORR($close, $volume, {w})",
        "TS_CORR($close, $open, {w})",
        "TS_CORR(DELTA($close, 1), $volume, {w})",
    ],
    "REGIME_ADAPTIVE": [
        "RANK(EMA(DELTA($close, {w1}), {w2})) - 0.5",
        "TS_RANK(TS_SKEW($close, {w}), {w2})",
    ]
}

FOREX_ALPHA_REGISTRY = {
    "UNARY": ["LOG", "ABS", "SIGN", "SQRT", "EMA", "SMA"],
    "BINARY": ["DELAY", "DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "EMA", "SMA", "WMA"],
    "TRINARY": ["TS_CORR", "TS_COVARIANCE"],
    "VAR": ["$close", "$open", "$high", "$low", "$volume"],
    "ARITHMETIC": ["+", "-", "*", "/"],
    "DERIVED": [
        "($close / DELAY($close, 1) - 1)", 
        "($high - $low) / ($close + 1e-8)",
        "DELTA($close, 1) / (TS_STD($close, 20) + 1e-8)",
        "EMA(DELTA($close, 1), 5)"
    ],
    "SPECTRAL": ["TS_SKEW", "TS_KURT"],
    "REGIME_ADAPTIVE": [
        "SIGN(TS_MEAN(DELTA($close, 1), {w})) * ABS(TS_STD($close, {w}))",
        "($high - $low) / (EMA($close, {w}) + 1e-8)",
    ]
}

FOREX_XAU_REGISTRY = {
    "UNARY": ["LOG", "ABS", "SIGN", "SQRT", "INV", "EXP"],
    "BINARY": ["DELAY", "DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SKEW", "TS_KURT",
               "TS_SUM", "TS_MEDIAN", "TS_ARGMAX", "TS_ARGMIN", "EMA", "SMA", "WMA"],
    "TRINARY": ["TS_CORR", "TS_COVARIANCE"],
    "VAR": ["$close", "$open", "$high", "$low", "$volume"],
    "ARITHMETIC": ["+", "-", "*", "/"],
    # GPT-5.1: Time-series specialized axioms for single-asset forex/XAU
    "SPECTRAL": ["TS_SKEW", "TS_KURT"],
    "REGIME_ADAPTIVE": [
        "SIGN(TS_MEAN(DELTA($close, 1), {w})) * ABS(TS_STD($close, {w}))",
        "($high - $low) / (EMA($close, {w}) + 1e-8)",
    ]
}

# GPT-5.1: Deep Formula Archetypes
# Stock archetypes use RANK for cross-sectional edge.
STOCK_ARCHETYPES = {
    "MEAN_REVERSION": "RANK(($close - TS_MEAN($close, {w})) / (TS_STD($close, {w}) + 1e-8))",
    "MOMENTUM": "RANK(DELTA(EMA($close, {w1}), {w2}))",
    "VOLATILITY_SKEW": "-1 * RANK(TS_SKEW($close, {w}))",
    "ADAPTIVE_AUTOCORRELATION": "RANK(TS_CORR($close, DELAY($close, {w1}), {w2}))",
    "HIGH_LOW_DYNAMIC": "RANK(EMA(($high - $low) / (TS_MEAN($close, {w}) + 1e-8), {w2}))",
}

# Forex archetypes use Time-Series persistence (no RANK).
FOREX_ARCHETYPES = {
    "TREND_FOLLOWING": "EMA(DELTA($close, 1), {w}) / (TS_STD($close, {w}) + 1e-8)",
    "MEAN_REVERSION": "($close - TS_MEAN($close, {w})) / (TS_STD($close, {w}) + 1e-8)",
    "RANGE_SCALPING": "($close - TS_MIN($low, {w})) / (TS_MAX($high, {w}) - TS_MIN($low, {w}) + 1e-8) - 0.5",
    "VOLATILITY_BREAKOUT": "EMA(($high - $low) / (TS_MEAN($close, {w}) + 1e-8), {w2})",
}

# GPT-5.1 Axis 7: Fractal Archetypes (Multi-Horizon Comparison)
FRACTAL_ARCHETYPES = {
    "FRACTAL_MOMENTUM": "RANK(DELTA($close, {w1}) / (TS_STD($close, {w2}) + 1e-8))",
    "FRACTAL_REVERSAL": "RANK(($close - TS_MEAN($close, {w2})) / (TS_STD($close, {w2}) + 1e-8)) * SIGN(DELTA($close, {w1}))",
    "FRACTAL_VOLUME_FLOW": "RANK(TS_CORR(DELTA($close, {w1}), DELTA($volume, {w1}), {w2}))",
    "FRACTAL_SKEW_REVERSION": "RANK(TS_SKEW($close, {w2})) * -1 * SIGN(DELTA($close, {w1}))",
}

# GPT-5.1 Axis 10: Elite Institutional Archetypes (Target 0.08 IC)
ELITE_ARCHETYPES = {
    "ELITE_INTRADAY_GAP": "RANK(($close - $open) / ($open + 1e-8))",
    "ELITE_LIQUIDITY_SYNERGY": "RANK(TS_CORR(RANK(DELTA($close, 1)), RANK($volume), {w}))",
    "ELITE_VOL_SCALED_BULL": "RANK(DELTA($close, {w}) / (TS_STD($close, {w2}) + 1e-8)) * RANK($volume)",
    "ELITE_MEAN_REVERSION_PRO": "RANK(($close - TS_MEAN($close, {w})) / (TS_STD($close, {w}) + 1e-8)) * -SIGN(TS_SKEW($close, {w}))",
}

# GPT-5.1 Axis 8: Institutional Golden Seeds (Proven Mathematical Foundations)
INSTITUTIONAL_GOLDEN_SEEDS = [
    "RANK(TS_CORR(RANK(DELTA($close, 1)), RANK($volume), 10))",  # VPIN-inspired synergy
    "RANK(DELTA($close, 5) / (TS_MEAN($volume * ABS(DELTA($close, 1)), 20) + 1e-8))", # Amihud Illiquidity
    "RANK(DELTA($close, 20)) / (TS_STD(DELTA($close, 1), 60) + 1e-8)", # Vol-Adjusted Momentum
    "RANK(EMA(DELTA($close, 1), 5)) - RANK(EMA(DELTA($close, 1), 20))", # Dual-Horizon Flow
]

# ── Compatibility aliases expected by sova_adapter.py ────────────────────────
# STRATEGIC_AXIOMS / FOREX_XAU_AXIOMS are synonym exports for the registries
STRATEGIC_AXIOMS = STOCK_ALPHA_REGISTRY
FOREX_XAU_AXIOMS = FOREX_ALPHA_REGISTRY
INSTITUTIONAL_GOLDEN_SEEDS_DICT = {f"GOLDEN_SEED_{i}": f for i, f in enumerate(INSTITUTIONAL_GOLDEN_SEEDS)}
DEEP_FORMULA_ARCHETYPES = {**STOCK_ARCHETYPES, **FOREX_ARCHETYPES, **FRACTAL_ARCHETYPES, **ELITE_ARCHETYPES, **INSTITUTIONAL_GOLDEN_SEEDS_DICT}

REGIME_STRATEGY_MAP = {
    "CAPITULATION_CRASH": {
        "primary": "LIQUIDITY_DYNAMICS",
        "secondary": "RISK_MANAGEMENT",
        "operators": ["TS_STD", "TS_SKEW", "RANK", "TS_MAX"],
        "windows": [5, 10, 20],
        "aggression": 0.2,
        "description": "Crisis mode: protect capital, seek liquidity, asymmetric risk"
    },
    "EXPONENTIAL_BULL": {
        "primary": "MOMENTUM_FLOW",
        "secondary": "REGIME_TRANSITION",
        "operators": ["RANK", "DELTA", "TS_CORR", "TS_MEAN", "EMA"],
        "windows": [5, 10, 20, 60],
        "aggression": 0.8,
        "description": "Trend amplification: ride momentum, scale positions, watch reversals"
    },
    "CHAOTIC_NOISE": {
        "primary": "MEAN_REVERSION",
        "secondary": "RISK_MANAGEMENT",
        "operators": ["RANK", "TS_STD", "TS_MEAN", "Ref"],
        "windows": [5, 10, 20],
        "aggression": 0.3,
        "description": "Noise filtering: tight bands, quick reversals, small positions"
    },
    "MEAN_REVERSION_ZONE": {
        "primary": "MEAN_REVERSION",
        "secondary": "LIQUIDITY_DYNAMICS",
        "operators": ["RANK", "TS_MEAN", "DELTA", "TS_STD"],
        "windows": [5, 10, 20],
        "aggression": 0.6,
        "description": "Statistical edge: fade extremes, scale into deviations"
    },
    "DISTRIBUTION_ANOMALY": {
        "primary": "LIQUIDITY_DYNAMICS",
        "secondary": "REGIME_TRANSITION",
        "operators": ["TS_CORR", "TS_SKEW", "RANK", "DELTA", "TS_STD"],
        "windows": [5, 10, 20],
        "aggression": 0.4,
        "description": "Volume divergence: detect distribution, prepare for transition"
    },
    "STABLE_ACCUMULATION": {
        "primary": "MOMENTUM_FLOW",
        "secondary": "MEAN_REVERSION",
        "operators": ["RANK", "DELTA", "TS_MEAN", "TS_CORR", "EMA"],
        "windows": [5, 10, 20, 60],
        "aggression": 0.7,
        "description": "Steady growth: trend following with reversion filters"
    },
    "STABLE_EROSION": {
        "primary": "RISK_MANAGEMENT",
        "secondary": "MEAN_REVERSION",
        "operators": ["RANK", "TS_STD", "TS_MIN", "DELTA"],
        "windows": [5, 10, 20],
        "aggression": 0.3,
        "description": "Defensive: short bias, tight stops, selective entries"
    },
    "NEUTRAL": {
        "primary": "MEAN_REVERSION",
        "secondary": "MOMENTUM_FLOW",
        "operators": ["RANK", "DELTA", "TS_MEAN", "TS_STD"],
        "windows": [5, 10, 20],
        "aggression": 0.5,
        "description": "Balanced: equal weight to momentum and reversion signals"
    }
}

SINGLE_ASSET_STRATEGY_MAP = {
    "XAU_SPECIFICS": {
        "primary": "MEAN_REVERSION",
        "secondary": "LIQUIDITY_DYNAMICS",
        "operators": ["TS_RANK", "TS_MEAN", "DELTA", "TS_STD", "TS_CORR"],
        "windows": [2, 5, 13, 21, 55],
        "aggression": 0.5,
        "description": "Gold Specific: focus on mean reversion from extremes and volume-driven shifts"
    },
    "BTC_SPECIFICS": {
        "primary": "MOMENTUM_FLOW",
        "secondary": "REGIME_TRANSITION",
        "operators": ["TS_RANK", "DELTA", "TS_MEAN", "TS_STD", "LOG", "SIGN"],
        "windows": [3, 8, 20, 50],
        "aggression": 0.7,
        "description": "Crypto Specific: focus on momentum clusters and heavy-tail volatility"
    },
    "GENERIC_SINGLE": {
        "primary": "MEAN_REVERSION",
        "secondary": "MOMENTUM_FLOW",
        "operators": ["TS_RANK", "DELTA", "TS_MEAN", "TS_STD"],
        "windows": [5, 10, 20],
        "aggression": 0.5,
        "description": "Default single asset strategy: balanced transition from levels"
    }
}

TRADE_ERROR_TAXONOMY = {
    "CHASING_TREND_LATE": {
        "pattern": "Entry after >70% of move completed, high adverse excursion",
        "correction": "Use momentum persistence filters, wait for pullback confirmation",
        "severity": "HIGH"
    },
    "TREATING_NOISE_AS_SIGNAL": {
        "pattern": "Entry during high entropy/fractal regime, quick reversal",
        "correction": "Apply spectral entropy gate, require volume confirmation",
        "severity": "HIGH"
    },
    "PREMATURE_EXIT": {
        "pattern": "Exit before reaching calculated target, move continued in favor",
        "correction": "Widen trailing stops in trending regimes, use ATR-based targets",
        "severity": "MEDIUM"
    },
    "HELD_TOO_LONG": {
        "pattern": "Failed to exit when regime transitioned, gave back profits",
        "correction": "Implement regime-change stop, monitor transition probability",
        "severity": "HIGH"
    },
    "WRONG_DIRECTION": {
        "pattern": "Trade against dominant regime direction",
        "correction": "Align trade direction with regime classification, use regime confidence gate",
        "severity": "CRITICAL"
    },
    "OVERSIZED_POSITION": {
        "pattern": "Position too large for current volatility regime",
        "correction": "Scale position inversely with realized volatility, cap at 2% risk",
        "severity": "HIGH"
    },
    "MISSED_OPPORTUNITY": {
        "pattern": "Clear signal aligned with regime but no entry taken",
        "correction": "Lower confidence threshold in high-conviction regimes",
        "severity": "LOW"
    },
    "STOP_TOO_TIGHT": {
        "pattern": "Stopped out within normal noise band, price resumed direction",
        "correction": "Set stops beyond ATR multiple for regime volatility level",
        "severity": "MEDIUM"
    },
    "STOP_TOO_WIDE": {
        "pattern": "Stop allowed excessive drawdown beyond statistical expectation",
        "correction": "Tighten stops in compressing volatility, cap at 1.5x ATR",
        "severity": "MEDIUM"
    },
    "CORRELATION_BLINDNESS": {
        "pattern": "Multiple correlated positions amplified drawdown",
        "correction": "Monitor portfolio correlation matrix, limit correlated exposure",
        "severity": "HIGH"
    },
    "REGIME_MISMATCH": {
        "pattern": "Strategy designed for different regime applied incorrectly",
        "correction": "Validate strategy-regime alignment before execution",
        "severity": "CRITICAL"
    },
    "LIQUIDITY_TRAP": {
        "pattern": "Entry in thin liquidity, excessive slippage on exit",
        "correction": "Check volume surge ratio before entry, avoid thin liquidity periods",
        "severity": "HIGH"
    }
}


class StrategicTokenizer:
    def __init__(self):
        self.gene_registry = STOCK_ALPHA_REGISTRY

    def tokenize(self, expression: str) -> List[str]:
        return re.findall(r'[A-Z_]+\(|\$\w+|[\d.]+|[()+\-*/,]', expression)

    def classify_token(self, token: str) -> str:
        for category, genes in self.gene_registry.items():
            if token.rstrip('(') in genes or token in genes:
                return category
        if token.startswith("$"):
            return "VAR"
        if token in '()+-*/,':
            return "SYNTAX"
        if re.match(r'[\d.]+', token):
            return "CONSTANT"
        return "UNKNOWN"

    def extract_structure(self, expression: str) -> Dict[str, Any]:
        tokens = self.tokenize(expression)
        categories = [self.classify_token(t) for t in tokens]
        
        return {
            "tokens": tokens,
            "categories": categories,
            "depth": expression.count("("),
            "num_operators": sum(1 for c in categories if c in ("UNARY", "BINARY", "TRINARY")),
            "num_variables": sum(1 for c in categories if c == "VAR"),
            "unique_variables": len(set(t for t, c in zip(tokens, categories) if c == "VAR")),
            "complexity_score": len(tokens) + expression.count("(") * 2,
            "operator_set": set(t.rstrip('(') for t, c in zip(tokens, categories) if c in ("UNARY", "BINARY", "TRINARY"))
        }


class MemoryPatternMiner:
    """
    ELITE AI CAPABILITY: Search and extract successful patterns from memory vortex.
    Analyzes high-IC formulas to identify building blocks, operator combinations,
    and structural motifs that consistently generate alpha.
    
    This enables AI to "learn from experience" instead of random generation.
    """
    def __init__(self):
        self.tokenizer = StrategicTokenizer()
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        
    def mine_successful_patterns(self, memories: List['MemoryImpression'], 
                                 min_ic: float = 0.02) -> Dict[str, Any]:
        """
        Extract building blocks from high-quality memories.
        Returns patterns organized by: operators, operator_chains, variable_usage, windows.
        """
        if not memories:
            return {"operators": Counter(), "chains": Counter(), "variables": Counter(), 
                    "windows": Counter(), "structures": []}
        
        # Filter to successful memories only
        successful = [m for m in memories if m.Metrics.IC >= min_ic]
        if not successful:
            successful = sorted(memories, key=lambda x: x.Metrics.IC, reverse=True)[:max(1, len(memories)//3)]
        
        operator_freq = Counter()
        chain_freq = Counter()  # 2-operator sequences
        variable_freq = Counter()
        window_freq = Counter()
        structural_patterns = []
        
        for mem in successful:
            expr = mem.Expression
            structure = self.tokenizer.extract_structure(expr)
            
            # Extract operators
            for op in structure["operator_set"]:
                operator_freq[op] += mem.Metrics.IC  # Weight by IC, not just count
            
            # Extract operator chains (consecutive operator usage)
            tokens = structure["tokens"]
            for i in range(len(tokens) - 1):
                if tokens[i].rstrip('(') in operator_freq and tokens[i+1].rstrip('(') in operator_freq:
                    chain = f"{tokens[i].rstrip('(')}→{tokens[i+1].rstrip('(')}"
                    chain_freq[chain] += mem.Metrics.IC * 0.5
            
            # Extract variables
            vars_in_expr = re.findall(r'\$\w+', expr)
            for v in vars_in_expr:
                variable_freq[v] += mem.Metrics.IC
            
            # Extract windows
            windows = re.findall(r',\s*(\d+)\s*\)', expr)
            for w in windows:
                window_freq[int(w)] += mem.Metrics.IC * 0.3
            
            # Store full structural pattern
            structural_patterns.append({
                "expression": expr,
                "ic": mem.Metrics.IC,
                "fitness": mem.Metrics.Fitness,
                "operators": list(structure["operator_set"]),
                "variables": list(set(vars_in_expr)),
                "depth": structure["depth"],
                "complexity": structure["complexity_score"]
            })
        
        return {
            "operators": operator_freq,
            "chains": chain_freq,
            "variables": variable_freq,
            "windows": window_freq,
            "structures": sorted(structural_patterns, key=lambda x: x["fitness"], reverse=True)
        }
    
    def search_similar_contexts(self, target_regime: str, target_dna: 'MarketDNA',
                                all_memories: Dict[str, List['MemoryImpression']]) -> List['MemoryImpression']:
        """
        Search across ALL regime memories for formulas that worked in similar market conditions.
        Uses DNA similarity (volatility, hurst, trend) to find analogous contexts.
        """
        similar = []
        
        for regime, memories in all_memories.items():
            for mem in memories:
                # Simple similarity score based on regime affinity
                if regime == target_regime:
                    similarity = 1.0
                elif regime in ["EXPONENTIAL_BULL", "STABLE_ACCUMULATION"] and target_regime in ["EXPONENTIAL_BULL", "STABLE_ACCUMULATION"]:
                    similarity = 0.7
                elif regime in ["CAPITULATION_CRASH", "STABLE_EROSION"] and target_regime in ["CAPITULATION_CRASH", "STABLE_EROSION"]:
                    similarity = 0.7
                elif regime in ["CHAOTIC_NOISE", "MEAN_REVERSION_ZONE"] and target_regime in ["CHAOTIC_NOISE", "MEAN_REVERSION_ZONE"]:
                    similarity = 0.6
                else:
                    similarity = 0.3
                
                # Boost by fitness
                score = similarity * mem.Metrics.Fitness * mem.Reinforcement
                similar.append((mem, score))
        
        # Return top candidates
        similar.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in similar[:20]]


class MathematicalLogicValidator:
    """
    🧠 ELITE MATHEMATICAL VALIDATOR: Ensures Alpha formulas are 'Ideal Rules'.

    Checks for:
    - Scale Invariance (Axiom 1: Independence from absolute price levels)
    - Signal Consistency (Axiom 2: Economic logic aligns with regime)
    - SNR Protection (Axiom 3: No recursive noise amplification)
    - Stability Guards (Axiom 4: Division-by-zero prevention)
    """

    @staticmethod
    def validate_logic(expression: str, regime: str, operators_used: List[str]) -> Tuple[bool, str, float]:
        """
        Returns: (is_valid, reasoning, quality_score)
        Axiomatic check for formula perfection.
        """
        quality_score = 0.6  # Start higher for elite tier
        issues = []
        insights = []
        expression_upper = expression.upper()

        # 1. Axiom 1: Scale Invariance (Mandatory)
        # Good alpha should be independent of absolute price levels.
        has_normalization = any(op in expression_upper for op in ["RANK", "ZSCORE", "TS_RANK", "EMA", "SMA", "/"])
        if not has_normalization and "$" in expression:
            issues.append("Violation: Scale Invariance (Raw price usage without normalization)")
            quality_score -= 0.4
        else:
            insights.append("Axiom Met: Scale Invariant signal structure")
            quality_score += 0.05

        # 2. Axiom 2: Signal-to-Noise Ratio (SNR) Protection
        # Avoid recursive nesting of high-moment operators that amplify noise.
        if expression_upper.count("TS_SKEW") > 1 or expression_upper.count("TS_KURT") > 1:
            issues.append("Violation: Noise Amplification (Recursive high-moment operators)")
            quality_score -= 0.3

        # 3. Axiom 3: Mathematical Stability Guards
        if "/" in expression and "1E-" not in expression_upper and "+ 0." not in expression_upper:
            issues.append("Violation: Mathematical Fragility (Division without epsilon guard)")
            quality_score -= 0.2
        else:
            if "/" in expression:
                insights.append("Axiom Met: Stability guards present in division")

        # 4. Regime-Specific Mathematical Alignment
        if regime in ["EXPONENTIAL_BULL", "STABLE_ACCUMULATION"]:
            if "DELTA" in operators_used or "TS_CORR" in operators_used:
                insights.append(f"Strategic Fit: Momentum-aligned for {regime}")
                quality_score += 0.12
        elif regime in ["MEAN_REVERSION_ZONE", "CHAOTIC_NOISE"]:
            if "TS_MEAN" in operators_used or "TS_STD" in operators_used:
                insights.append(f"Strategic Fit: Reversion-aligned for {regime}")
                quality_score += 0.12

        # 5. Financial Interpretability & Information Depth
        depth = expression.count("(")
        if depth > 8:
            issues.append("Concern: Excessive nesting - risk of overfitting on noise")
            quality_score -= 0.2
        elif 3 <= depth <= 6:
            insights.append("Structural Merit: Balanced complexity vs clarity")
            quality_score += 0.10

        if "$volume" in expression_upper:
            insights.append("Structural Merit: Multi-factor (Price + Volume) integration")
            quality_score += 0.10

        # Final Evaluation
        quality_score = max(0.0, min(1.0, quality_score))
        is_valid = quality_score >= 0.45  # Stricter threshold for 'Elite' Alpha

        reasoning = "Axiomatic Logic Analysis:\n"
        if insights:
            reasoning += "✓ Successes:\n  - " + "\n  - ".join(insights) + "\n"
        if issues:
            reasoning += "✗ Failures:\n  - " + "\n  - ".join(issues) + "\n"
        reasoning += f"Mathematical Integrity Score: {quality_score:.2f}/1.00"

        return is_valid, reasoning, quality_score

# ═══════════════════════════════════════════════════════════════════════════════
#  ALPHA EXPRESSION AST: Tree-based representation for formula manipulation
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaASTNode:
    """Recursive AST node representing an alpha expression.
    
    Types:
      - VAR: leaf node ($close, $volume, etc.)
      - CONST: numeric constant (1e-8, 0.5, etc.)
      - UNARY: single-child operator (RANK, ABS, SIGN, LOG, etc.)
      - BINARY_OP: two children with arithmetic (+, -, *, /)
      - FUNC: operator with arguments — func(child, window) or func(child1, child2, window)
    """
    __slots__ = ('kind', 'value', 'children', 'window')

    def __init__(self, kind: str, value: str, children: list = None, window: int = None):
        self.kind = kind       # VAR, CONST, UNARY, BINARY_OP, FUNC
        self.value = value     # operator name, variable name, or constant
        self.children = children or []
        self.window = window   # lookback window for time-series operators

    def to_expr(self) -> str:
        """Serialize AST back to Qlib expression string."""
        if self.kind == 'VAR':
            return self.value
        if self.kind == 'CONST':
            return self.value
        if self.kind == 'UNARY':
            return f"{self.value}({self.children[0].to_expr()})"
        if self.kind == 'BINARY_OP':
            left = self.children[0].to_expr()
            right = self.children[1].to_expr()
            return f"({left} {self.value} {right})"
        if self.kind == 'FUNC':
            args = [c.to_expr() for c in self.children]
            if self.window is not None:
                args.append(str(self.window))
            return f"{self.value}({', '.join(args)})"
        return self.value

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def collect_nodes(self) -> list:
        """Flatten all nodes in pre-order for tree surgery."""
        result = [self]
        for c in self.children:
            result.extend(c.collect_nodes())
        return result

    def clone(self) -> 'AlphaASTNode':
        return AlphaASTNode(
            self.kind, self.value,
            [c.clone() for c in self.children],
            self.window
        )

    def structural_signature(self) -> str:
        """Shape-only signature (no variable names or windows) for diversity comparison."""
        if self.kind in ('VAR', 'CONST'):
            return self.kind[0]
        child_sigs = ','.join(c.structural_signature() for c in self.children)
        return f"{self.value}({child_sigs})"


def _parse_alpha_expr(expr: str) -> Optional[AlphaASTNode]:
    """Parse Qlib expression string into AlphaASTNode tree.
    
    Handles: FUNC(args), $variable, numeric, (expr op expr)
    This is a recursive descent parser — no external dependencies.
    """
    expr = expr.strip()
    if not expr:
        return None

    # Variable
    if expr.startswith('$') and re.match(r'^\$\w+$', expr):
        return AlphaASTNode('VAR', expr)

    # Numeric constant
    if re.match(r'^-?[\d.]+(?:e[+-]?\d+)?$', expr):
        return AlphaASTNode('CONST', expr)

    # Function call: NAME(args) — verify the closing paren matches the opening one
    m = re.match(r'^([A-Z_]+)\(', expr)
    if m:
        func_name = m.group(1)
        # Find the matching closing paren for the opening one
        depth = 1
        start = m.end()
        match_end = -1
        for i in range(start, len(expr)):
            if expr[i] == '(':
                depth += 1
            elif expr[i] == ')':
                depth -= 1
                if depth == 0:
                    match_end = i
                    break
        # Only treat as function call if matching ) is the LAST character
        if match_end == len(expr) - 1:
            inner = expr[start:match_end]
            # Split arguments at top-level commas
            args = _split_top_level(inner, ',')
            children = []
            window = None
            for i, arg in enumerate(args):
                arg = arg.strip()
                # Last arg might be a window (integer)
                if i == len(args) - 1 and re.match(r'^\d+$', arg):
                    window = int(arg)
                else:
                    child = _parse_alpha_expr(arg)
                    if child:
                        children.append(child)

            if func_name in ('RANK', 'ABS', 'SIGN', 'LOG', 'SQRT', 'INV', 'EXP') and not window:
                return AlphaASTNode('UNARY', func_name, children[:1])
            return AlphaASTNode('FUNC', func_name, children, window)

    # Parenthesized binary operation: (left OP right)
    if expr.startswith('(') and expr.endswith(')'):
        inner = expr[1:-1]
        # Find top-level arithmetic operator
        for op in ['+', '-', '*', '/']:
            parts = _split_top_level(inner, op)
            if len(parts) == 2:
                left = _parse_alpha_expr(parts[0].strip())
                right = _parse_alpha_expr(parts[1].strip())
                if left and right:
                    return AlphaASTNode('BINARY_OP', op, [left, right])
    
    # Fallback: try without outer parens for bare binary ops like "A / (B + 1e-8)"
    for op in ['+', '-']:  # Lower precedence first
        parts = _split_top_level(expr, op)
        if len(parts) == 2:
            left = _parse_alpha_expr(parts[0].strip())
            right = _parse_alpha_expr(parts[1].strip())
            if left and right:
                return AlphaASTNode('BINARY_OP', op, [left, right])
    for op in ['*', '/']:
        parts = _split_top_level(expr, op)
        if len(parts) == 2:
            left = _parse_alpha_expr(parts[0].strip())
            right = _parse_alpha_expr(parts[1].strip())
            if left and right:
                return AlphaASTNode('BINARY_OP', op, [left, right])

    # If we can't parse, return as a raw leaf
    return AlphaASTNode('VAR', expr) if '$' in expr else AlphaASTNode('CONST', expr)


def _split_top_level(s: str, sep: str) -> list:
    """Split string by separator only at the top level (not inside parentheses)."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0 and len(sep) == 1:
            parts.append(''.join(current))
            current = []
            continue
        current.append(ch)
    parts.append(''.join(current))
    return parts if len(parts) > 1 else [s]


# ═══════════════════════════════════════════════════════════════════════════════
#  SMART EPSILON: Context-adaptive numerical stability constants
# ═══════════════════════════════════════════════════════════════════════════════

def _smart_eps(denom_type: str = 'default') -> str:
    """Return context-appropriate epsilon for division safety.

    Different denominators need different epsilon magnitudes:
    - TS_STD of returns (order ~0.01): 1e-8 is fine
    - $volume (order ~1e6-1e9): 1e-8 is negligible, need 1e-4
    - ($high - $low) price range (order ~0.01-10): 1e-6
    - TS_MEAN of prices: 1e-8 is fine
    - Correlation denominators (order ~1): 1e-6
    """
    _EPS_MAP = {
        'std':         '1e-8',    # TS_STD generally small — 1e-8 OK
        'volume':      '1e-4',    # $volume is large — bigger eps avoids false precision
        'price_range': '1e-6',    # ($high - $low) in price units
        'correlation': '1e-6',    # Correlation-derived denominators
        'mean':        '1e-8',    # TS_MEAN denominators
        'price':       '1e-6',    # Raw price denominators ($close, $high, etc.)
        'default':     '1e-8',
    }
    return _EPS_MAP.get(denom_type, '1e-8')


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _resolve_normalization_policy() -> str:
    """Normalization policy: adaptive | forced | raw."""
    policy = str(os.environ.get("SOVA_NORMALIZATION_POLICY", "adaptive") or "adaptive").strip().lower()
    if policy in {"adaptive", "forced", "raw"}:
        return policy
    return "adaptive"


def _outer_norm_operator(expr: str) -> str:
    m = re.match(r'^\s*([A-Z_]+)\s*\(', str(expr or ""))
    return m.group(1).upper() if m else ""


def _strip_outer_norm(expr: str) -> str:
    """Remove one outer RANK/TS_RANK wrapper when present."""
    e = str(expr or "").strip()
    m_rank = re.match(r'^RANK\((.*)\)$', e, flags=re.IGNORECASE)
    if m_rank:
        return m_rank.group(1).strip()
    m_tsr = re.match(r'^TS_RANK\((.*),\s*\d+\)$', e, flags=re.IGNORECASE)
    if m_tsr:
        return m_tsr.group(1).strip()
    return e


def _expression_scale_stable(expr: str) -> bool:
    """Heuristic for signals that are already numerically comparable without rank wrapping."""
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


def _adaptive_wrap_expression(
    expr: str,
    single_asset_mode: bool = False,
    preferred_mode: Optional[str] = None,
    window: int = 60,
) -> str:
    """Adaptive normalization wrapper to avoid rigid rank-only outputs."""
    e = str(expr or "").strip()
    if not e:
        return e

    policy = _resolve_normalization_policy()
    target_mode = (preferred_mode or ("TS_RANK" if single_asset_mode else "RANK")).upper()
    if target_mode not in {"RAW", "RANK", "TS_RANK"}:
        target_mode = "TS_RANK" if single_asset_mode else "RANK"

    outer = _outer_norm_operator(e)
    inner = _strip_outer_norm(e)

    def _wrap(mode: str, body: str) -> str:
        if mode == "RAW":
            return body
        if mode == "TS_RANK":
            return f"TS_RANK({body}, {max(20, int(window))})"
        return f"RANK({body})"

    # Raw policy deliberately strips hard rank wrappers when present.
    if policy == "raw":
        return inner

    # Forced policy guarantees normalized output by chosen mode.
    if policy == "forced":
        mode = target_mode if target_mode != "RAW" else ("TS_RANK" if single_asset_mode else "RANK")
        base = inner if outer in {"RANK", "TS_RANK"} else e
        return _wrap(mode, base)

    # Adaptive policy: harmonize market-specific wrapper if already wrapped.
    if outer in {"RANK", "TS_RANK"}:
        if single_asset_mode and outer == "RANK":
            return _wrap("TS_RANK", inner)
        if (not single_asset_mode) and outer == "TS_RANK":
            return _wrap("RANK", inner)
        if _env_flag("SOVA_ALLOW_RAW_FROM_WRAPPED", True) and _expression_scale_stable(inner):
            return inner
        return e

    # Not wrapped yet: keep raw if structurally normalized, else wrap.
    if _expression_scale_stable(e):
        return e
    mode = target_mode if target_mode != "RAW" else ("TS_RANK" if single_asset_mode else "RANK")
    return _wrap(mode, e)


# ═══════════════════════════════════════════════════════════════════════════════
#  SKELETON LEARNER: Learn which expression SHAPES produce high IC
# ═══════════════════════════════════════════════════════════════════════════════

class SkeletonLearner:
    """Learns which expression structures consistently produce high IC.
    
    When a formula like `RANK(DELTA($close, 5) / (TS_STD($close, 20) + 1e-8))`
    achieves IC=0.06, this class extracts the abstract skeleton:
        `RANK(DELTA(V, Ws) / (TS_STD(V, Wl) + EPS))`
    and stores it ranked by IC. Future generations can instantiate top skeletons
    with new variables/windows for consistent high-IC output.
    """
    
    def __init__(self, max_skeletons: int = 30):
        self._max_skeletons = max_skeletons
        # skeleton_key → {ic_sum, count, best_ic, best_expr, skeleton}
        self._registry: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def _extract_skeleton(expr: str) -> str:
        """Extract abstract skeleton by replacing variables and windows with placeholders."""
        skeleton = expr
        # Replace all $variables with V placeholder
        skeleton = re.sub(r'\$\w+', 'V', skeleton)
        # Replace window parameters (numbers after comma in function calls) with W
        skeleton = re.sub(r',\s*\d+\)', ', W)', skeleton)
        # Replace epsilon values
        skeleton = re.sub(r'\d+e-\d+', 'EPS', skeleton)
        skeleton = re.sub(r'0\.0+\d*', 'EPS', skeleton)
        return skeleton
    
    def record(self, expression: str, ic: float, icir: float = 0.0):
        """Record a successful expression and its skeleton."""
        if ic < 0.015:  # Don't learn from noise
            return
        
        skeleton = self._extract_skeleton(expression)
        key = skeleton
        
        if key not in self._registry:
            self._registry[key] = {
                'ic_sum': 0.0, 'count': 0, 'best_ic': 0.0,
                'best_expr': '', 'skeleton': skeleton,
                'avg_icir': 0.0, 'icir_sum': 0.0,
            }
        
        entry = self._registry[key]
        entry['ic_sum'] += ic
        entry['icir_sum'] += icir
        entry['count'] += 1
        entry['avg_icir'] = entry['icir_sum'] / entry['count']
        if ic > entry['best_ic']:
            entry['best_ic'] = ic
            entry['best_expr'] = expression
        
        # Prune if too many — remove lowest avgIC skeletons
        if len(self._registry) > self._max_skeletons:
            sorted_keys = sorted(
                self._registry.keys(),
                key=lambda k: self._registry[k]['ic_sum'] / max(1, self._registry[k]['count'])
            )
            for k in sorted_keys[:len(self._registry) - self._max_skeletons]:
                del self._registry[k]
    
    def get_top_skeletons(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top-N skeletons ranked by average IC × count (favors both quality and consistency)."""
        if not self._registry:
            return []
        ranked = sorted(
            self._registry.values(),
            key=lambda e: (e['ic_sum'] / max(1, e['count'])) * min(3, e['count']),
            reverse=True
        )
        return ranked[:n]
    
    def generate_from_skeleton(self, skeleton_entry: Dict[str, Any],
                                variables: List[str], windows: List[int],
                                generation: int = 0) -> Optional[str]:
        """Instantiate a skeleton with specific variables and windows.
        
        Takes the best_expr from a skeleton entry and creates a variant
        with different variables and window sizes.
        """
        best_expr = skeleton_entry.get('best_expr', '')
        if not best_expr:
            return None
        
        result = best_expr
        
        # Substitute variables: replace each $var found with a rotated variable
        used_vars = re.findall(r'\$\w+', result)
        if used_vars and variables:
            for i, old_var in enumerate(set(used_vars)):
                new_var = variables[(generation + i) % len(variables)]
                if new_var != old_var:
                    result = result.replace(old_var, new_var, 1)
        
        # Substitute windows: find all numeric windows and replace with rotated ones  
        if windows:
            def _replace_window(match):
                idx = _replace_window.counter
                _replace_window.counter += 1
                return f', {windows[(generation + idx) % len(windows)]})'
            _replace_window.counter = 0
            result = re.sub(r',\s*\d+\)', _replace_window, result)
        
        return result if result != best_expr else None


# Singleton skeleton learner
_skeleton_learner = SkeletonLearner()


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL COMPOSITION ENGINE: The real AI brain — no external API calls
# ═══════════════════════════════════════════════════════════════════════════════

class InternalCompositionEngine:
    """Self-contained mathematical reasoning engine for alpha formula generation.
    
    This IS the SOVA brain. It replaces all external LLM calls with:
    1. Operator Algebra: mathematical rules about which operators combine well
    2. Financial Theory: hypotheses from academic literature mapped to formulas
    3. AST Genetic Programming: tree-based crossover/mutation for evolution
    4. Quality Scoring: internal mathematical validation without external calls
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Operator Algebra Layer                                     │
    │  → Compatibility matrix: which ops combine meaningfully     │
    │  → Argument type system: each op knows its input/output     │
    │  → Composition rules: how to chain ops for financial logic  │
    ├─────────────────────────────────────────────────────────────┤
    │  Financial Hypothesis Layer                                 │
    │  → 10 canonical alpha families (Jegadeesh, Fama, etc.)      │
    │  → Each family → operator chain + variable roles            │
    │  → Regime → family affinity scoring                         │
    ├─────────────────────────────────────────────────────────────┤
    │  Genetic Programming Layer                                  │
    │  → AST-based crossover: swap meaningful subtrees            │
    │  → AST-based mutation: change nodes with type checking      │
    │  → Tournament selection with diversity pressure             │
    ├─────────────────────────────────────────────────────────────┤
    │  Quality Validation Layer                                   │
    │  → Structural checks (depth, complexity, balance)           │
    │  → Semantic checks (operator logic, financial meaning)      │
    │  → Diversity checks (structural signature comparison)       │
    └─────────────────────────────────────────────────────────────┘
    """

    # Operator compatibility: which operators can wrap which other operators
    # Key: outer op → Set of inner ops that combine well with it
    OPERATOR_AFFINITY = {
        'RANK':    {'DELTA', 'TS_CORR', 'TS_STD', 'TS_SKEW', 'TS_KURT', 'TS_MEAN', 'TS_MIN', 'TS_MAX'},
        'TS_RANK': {'DELTA', 'TS_CORR', 'TS_STD', 'TS_SKEW', 'TS_MEAN'},
        'DELTA':   {'$var', 'TS_MEAN', 'TS_STD', 'EMA', 'SMA'},
        'TS_CORR': {'$var', 'DELTA'},
        'TS_STD':  {'$var', 'DELTA', 'TS_MEAN'},
        'TS_MEAN': {'$var', 'DELTA', 'SIGN'},
        'TS_SKEW': {'$var', 'DELTA'},
        'TS_KURT': {'$var', 'DELTA'},
        'SIGN':    {'DELTA', 'TS_MEAN'},
        'ABS':     {'DELTA', 'TS_CORR'},
        'LOG':     {'$var'},
        'EMA':     {'$var', 'DELTA'},
        'SMA':     {'$var'},
        'WMA':     {'$var'},
    }

    # Financial hypothesis templates: academic theory → formula structure
    HYPOTHESIS_LIBRARY = [
        # ── ELITE HIGH-IC ALPHA HYPOTHESES (PROVEN CSI300) ──────────────────────
        {
            'name': 'alpha_wqa_oscillator',
            'theory': 'Cross-sectional normalized volume-weighted price stochastic (Elite 0.08+ IC potential)',
            'structure': lambda v, ws, wl: f"RANK(TS_MEAN((({v} - TS_MIN($low, {wl})) / (TS_MAX($high, {wl}) - TS_MIN($low, {wl}) + 1e-8)) * $volume, {ws})) * -1.0",
            'regimes': {'EXPONENTIAL_BULL': 1.0, 'MEAN_REVERSION_ZONE': 1.0, 'STABLE_ACCUMULATION': 0.9, 'NEUTRAL': 0.8},
            'variables': ['$close', '$vwap'],
        },
        {
            'name': 'alpha_pressure_wqa101',
            'theory': 'Overnight gap vs intraday volatility rank scaling (Elite 0.06+ IC potential)',
            'structure': lambda v, ws, wl: f"RANK(-1.0 * TS_CORR(DELTA({v}, 1), DELTA($volume, 1), {wl})) * RANK(TS_STD({v}, {ws}))",
            'regimes': {'CHAOTIC_NOISE': 1.0, 'DISTRIBUTION_ANOMALY': 0.9, 'CAPITULATION_CRASH': 1.0},
            'variables': ['$close', '$open'],
        },
        {
            'name': 'alpha_liquidity_shock',
            'theory': 'Extreme liquidity shock premium via rank difference (Elite 0.05+ IC potential)',
            'structure': lambda v, ws, wl: f"(RANK(TS_MEAN($volume, {ws})) - RANK(TS_MEAN($volume, {wl}))) * -1.0 * RANK(TS_CORR({v}, $volume, {ws}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 1.0, 'EXPONENTIAL_BULL': 0.8, 'NEUTRAL': 0.9},
            'variables': ['$vwap', '$close'],
        },
        {
            'name': 'alpha_asymmetric_reversal',
            'theory': 'Asymmetric volatility-scaled reversal (Elite 0.07+ IC potential)',
            'structure': lambda v, ws, wl: f"RANK(-1.0 * ({v} - TS_MEAN({v}, {wl})) / (TS_STD({v}, {wl}) + 1e-8)) * RANK(TS_MEAN($volume, {ws}))",
            'regimes': {'MEAN_REVERSION_ZONE': 1.0, 'CHAOTIC_NOISE': 0.9, 'STABLE_ACCUMULATION': 0.8},
            'variables': ['$close', '$vwap'],
        },
        # ────────────────────────────────────────────────────────────────────────
        # Jegadeesh & Titman (1993): Cross-sectional momentum
        {
            'name': 'momentum_jt93',
            'theory': 'Past winners outperform losers over 3-12 months',
            'structure': lambda v, ws, wl: f"RANK(DELTA({v}, {ws}) / (TS_STD({v}, {wl}) + {_smart_eps('std')}))",
            'regimes': {'EXPONENTIAL_BULL': 1.0, 'STABLE_ACCUMULATION': 0.8, 'NEUTRAL': 0.5},
            'variables': ['$close', '$return'],
        },
        # DeBondt & Thaler (1985): Long-term reversal
        {
            'name': 'reversal_dt85',
            'theory': 'Extreme losers revert over long horizons',
            'structure': lambda v, ws, wl: f"RANK(({v} - TS_MEAN({v}, {wl})) / (TS_STD({v}, {wl}) + {_smart_eps('std')}))",
            'regimes': {'MEAN_REVERSION_ZONE': 1.0, 'CHAOTIC_NOISE': 0.7, 'NEUTRAL': 0.6},
            'variables': ['$close', '$return'],
        },
        # Kyle (1985): Informed trading via price-volume correlation
        {
            'name': 'informed_trading_kyle85',
            'theory': 'Price-volume co-movement reveals informed order flow',
            'structure': lambda v, ws, wl: f"RANK(TS_CORR(DELTA({v}, 1), $volume, {wl}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 1.0, 'STABLE_ACCUMULATION': 0.7, 'EXPONENTIAL_BULL': 0.6},
            'variables': ['$close'],
        },
        # Baker, Bradley, Wurgler (2011): Low-volatility anomaly
        {
            'name': 'low_vol_bbw11',
            'theory': 'Low-volatility stocks earn higher risk-adjusted returns',
            'structure': lambda v, ws, wl: f"RANK(TS_STD({v}, {ws}) / (TS_STD({v}, {wl}) + {_smart_eps('std')}))",
            'regimes': {'STABLE_ACCUMULATION': 0.9, 'MEAN_REVERSION_ZONE': 0.7, 'NEUTRAL': 0.6},
            'variables': ['$close', '$return'],
        },
        # Barberis & Huang (2008): Skewness preference
        {
            'name': 'skew_preference_bh08',
            'theory': 'Negative skew premium — investors overpay for lottery stocks',
            'structure': lambda v, ws, wl: f"RANK(TS_SKEW({v}, {wl}))",
            'regimes': {'CHAOTIC_NOISE': 0.9, 'DISTRIBUTION_ANOMALY': 0.7, 'NEUTRAL': 0.5},
            'variables': ['$return', '$close'],
        },
        # Amihud (2002): Illiquidity premium
        {
            'name': 'illiquidity_amihud02',
            'theory': 'Price impact per unit volume predicts returns',
            'structure': lambda v, ws, wl: f"RANK(TS_MEAN(ABS(DELTA({v}, 1)) / ($volume + {_smart_eps('volume')}), {wl}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 0.9, 'CAPITULATION_CRASH': 0.7, 'STABLE_EROSION': 0.6},
            'variables': ['$close'],
        },
        # Faber (2007): Moving average trend following
        {
            'name': 'trend_following_faber07',
            'theory': 'Price above/below moving average predicts trend continuation',
            'structure': lambda v, ws, wl: f"RANK(({v} - TS_MEAN({v}, {wl})) * TS_MEAN(SIGN(DELTA({v}, 1)), {ws}))",
            'regimes': {'EXPONENTIAL_BULL': 0.9, 'STABLE_ACCUMULATION': 0.8, 'NEUTRAL': 0.5},
            'variables': ['$close'],
        },
        # Grinblatt & Han (2005): Disposition effect / anchoring
        {
            'name': 'anchoring_gh05',
            'theory': 'Distance from reference point (high/low) drives behavior',
            'structure': lambda v, ws, wl: f"RANK(({v} - TS_MIN({v}, {wl})) / (TS_MAX({v}, {wl}) - TS_MIN({v}, {wl}) + {_smart_eps('price_range')}))",
            'regimes': {'MEAN_REVERSION_ZONE': 0.9, 'STABLE_ACCUMULATION': 0.6, 'NEUTRAL': 0.5},
            'variables': ['$close', '$high', '$low'],
        },
        # Hasbrouck (1991): Microstructure efficiency
        {
            'name': 'microstructure_hasbrouck91',
            'theory': 'Intraday price patterns reveal market efficiency breakdown',
            'structure': lambda v, ws, wl: f"RANK(($close - $open) / ($high - $low + {_smart_eps('price_range')}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 0.8, 'CHAOTIC_NOISE': 0.7, 'CAPITULATION_CRASH': 0.6},
            'variables': ['$close'],
        },
        # Moskowitz, Ooi, Pedersen (2012): Time-series momentum
        {
            'name': 'tsmom_mop12',
            'theory': 'Own past returns predict future returns (time-series, not cross-section)',
            'structure': lambda v, ws, wl: f"TS_RANK(DELTA({v}, {ws}) / (TS_STD({v}, {wl}) + {_smart_eps('std')}), {max(20, wl)})",
            'regimes': {'EXPONENTIAL_BULL': 0.9, 'STABLE_ACCUMULATION': 0.7, 'NEUTRAL': 0.5},
            'variables': ['$close', '$return'],
        },
        # ── NEW TEMPLATES (8 additional alpha families) ──────────────────────
        # Chordia & Shivakumar (2002): Price-volume divergence
        {
            'name': 'pv_divergence_cs02',
            'theory': 'Divergence between price momentum and volume change reveals informed activity',
            'structure': lambda v, ws, wl: f"RANK(DELTA({v}, {ws}) - TS_MEAN(DELTA($volume, {ws}), {wl}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 0.9, 'STABLE_ACCUMULATION': 0.8, 'NEUTRAL': 0.6, 'EXPONENTIAL_BULL': 0.5},
            'variables': ['$close', '$return'],
        },
        # Bali, Cakici, Whitelaw (2011): MAX effect (lottery demand)
        {
            'name': 'max_effect_bcw11',
            'theory': 'Stocks with extreme recent highs are overpriced due to lottery demand',
            'structure': lambda v, ws, wl: f"RANK(TS_MAX({v}, {ws}) / (TS_MEAN({v}, {wl}) + {_smart_eps('price')}))",
            'regimes': {'CHAOTIC_NOISE': 0.9, 'EXPONENTIAL_BULL': 0.7, 'DISTRIBUTION_ANOMALY': 0.6, 'NEUTRAL': 0.5},
            'variables': ['$close', '$return'],
        },
        # Choi & Sias (2009): Institutional herding via volume pattern
        {
            'name': 'herding_cs09',
            'theory': 'Abnormal volume clustering signals institutional herding behavior',
            'structure': lambda v, ws, wl: f"RANK(TS_CORR($volume, ABS(DELTA({v}, 1)), {wl}) * SIGN(DELTA({v}, {ws})))",
            'regimes': {'DISTRIBUTION_ANOMALY': 0.9, 'EXPONENTIAL_BULL': 0.7, 'STABLE_ACCUMULATION': 0.6, 'NEUTRAL': 0.5},
            'variables': ['$close'],
        },
        # Connors RSI variant: Multi-timeframe mean reversion
        {
            'name': 'multi_tf_reversion',
            'theory': 'Multi-timeframe deviation from mean captures reversion across horizons',
            'structure': lambda v, ws, wl: f"RANK(({v} - TS_MEAN({v}, {ws})) / (TS_STD({v}, {ws}) + {_smart_eps('std')}) - ({v} - TS_MEAN({v}, {wl})) / (TS_STD({v}, {wl}) + {_smart_eps('std')}))",
            'regimes': {'MEAN_REVERSION_ZONE': 0.95, 'STABLE_EROSION': 0.7, 'NEUTRAL': 0.6, 'CHAOTIC_NOISE': 0.5},
            'variables': ['$close', '$return'],
        },
        # Garman-Klass (1980): Intraday range volatility
        {
            'name': 'range_vol_gk80',
            'theory': 'High-low range is a more efficient volatility estimator than close-to-close',
            'structure': lambda v, ws, wl: f"RANK(TS_MEAN(($high - $low) / ($close + {_smart_eps('price')}), {ws}) / (TS_MEAN(($high - $low) / ($close + {_smart_eps('price')}), {wl}) + {_smart_eps('std')}))",
            'regimes': {'CHAOTIC_NOISE': 0.85, 'CAPITULATION_CRASH': 0.8, 'STABLE_ACCUMULATION': 0.7, 'NEUTRAL': 0.5},
            'variables': ['$close'],
        },
        # Parkinson (1980): High-low range estimator for breakout
        {
            'name': 'parkinson_hl80',
            'theory': 'Narrowing high-low range precedes breakouts — volatility compression signal',
            'structure': lambda v, ws, wl: f"RANK(TS_MIN(($high - $low) / ($close + {_smart_eps('price')}), {wl}) / (TS_MAX(($high - $low) / ($close + {_smart_eps('price')}), {wl}) + {_smart_eps('std')}))",
            'regimes': {'STABLE_ACCUMULATION': 0.9, 'MEAN_REVERSION_ZONE': 0.7, 'NEUTRAL': 0.6},
            'variables': ['$close'],
        },
        # Roll (1984): Effective spread from serial covariance
        {
            'name': 'roll_spread_84',
            'theory': 'Serial covariance of returns estimates effective bid-ask spread (liquidity proxy)',
            'structure': lambda v, ws, wl: f"RANK(TS_MEAN(DELTA({v}, 1) * DELAY(DELTA({v}, 1), 1), {wl}))",
            'regimes': {'DISTRIBUTION_ANOMALY': 0.85, 'CAPITULATION_CRASH': 0.8, 'STABLE_EROSION': 0.7, 'NEUTRAL': 0.5},
            'variables': ['$close', '$return'],
        },
        # Markowitz (1952) variant: Mean-variance efficiency score
        {
            'name': 'mean_var_efficiency',
            'theory': 'Return-to-risk ratio across windows captures mean-variance efficiency',
            'structure': lambda v, ws, wl: f"RANK(TS_MEAN(DELTA({v}, {ws}), {wl}) / (TS_STD(DELTA({v}, {ws}), {wl}) + {_smart_eps('std')}))",
            'regimes': {'STABLE_ACCUMULATION': 0.9, 'EXPONENTIAL_BULL': 0.7, 'MEAN_REVERSION_ZONE': 0.6, 'NEUTRAL': 0.65},
            'variables': ['$close', '$return'],
        },
    ]

    # Operator argument types for type-safe AST manipulation
    OPERATOR_SIGNATURES = {
        # op_name: (n_expr_args, needs_window)
        'RANK': (1, False), 'ABS': (1, False), 'SIGN': (1, False),
        'LOG': (1, False), 'SQRT': (1, False), 'INV': (1, False), 'EXP': (1, False),
        'DELTA': (1, True), 'DELAY': (1, True),
        'TS_MEAN': (1, True), 'TS_STD': (1, True), 'TS_MAX': (1, True), 'TS_MIN': (1, True),
        'TS_SKEW': (1, True), 'TS_KURT': (1, True), 'TS_SUM': (1, True),
        'TS_MEDIAN': (1, True), 'TS_ARGMAX': (1, True), 'TS_ARGMIN': (1, True),
        'TS_RANK': (1, True), 'EMA': (1, True), 'SMA': (1, True), 'WMA': (1, True),
        'TS_CORR': (2, True), 'TS_COVARIANCE': (2, True),
    }

    # Regime → list of hypothesis names that work well, ordered by affinity
    REGIME_HYPOTHESIS_AFFINITY = {
        'EXPONENTIAL_BULL':    ['alpha_wqa_oscillator', 'alpha_liquidity_shock', 'momentum_jt93', 'tsmom_mop12', 'trend_following_faber07', 'pv_divergence_cs02', 'mean_var_efficiency'],
        'STABLE_ACCUMULATION': ['alpha_wqa_oscillator', 'alpha_asymmetric_reversal', 'low_vol_bbw11', 'trend_following_faber07', 'momentum_jt93', 'informed_trading_kyle85', 'parkinson_hl80', 'range_vol_gk80', 'mean_var_efficiency'],
        'MEAN_REVERSION_ZONE': ['alpha_wqa_oscillator', 'alpha_asymmetric_reversal', 'reversal_dt85', 'anchoring_gh05', 'low_vol_bbw11', 'multi_tf_reversion', 'parkinson_hl80'],
        'CHAOTIC_NOISE':       ['alpha_pressure_wqa101', 'alpha_asymmetric_reversal', 'skew_preference_bh08', 'reversal_dt85', 'microstructure_hasbrouck91', 'max_effect_bcw11', 'range_vol_gk80'],
        'DISTRIBUTION_ANOMALY':['alpha_pressure_wqa101', 'alpha_liquidity_shock', 'informed_trading_kyle85', 'illiquidity_amihud02', 'microstructure_hasbrouck91', 'pv_divergence_cs02', 'herding_cs09', 'roll_spread_84'],
        'CAPITULATION_CRASH':  ['alpha_pressure_wqa101', 'illiquidity_amihud02', 'skew_preference_bh08', 'reversal_dt85', 'range_vol_gk80', 'roll_spread_84'],
        'STABLE_EROSION':      ['alpha_wqa_oscillator', 'low_vol_bbw11', 'reversal_dt85', 'illiquidity_amihud02', 'multi_tf_reversion', 'roll_spread_84'],
        'NEUTRAL':             ['alpha_wqa_oscillator', 'alpha_pressure_wqa101', 'alpha_liquidity_shock', 'alpha_asymmetric_reversal', 'momentum_jt93', 'reversal_dt85', 'low_vol_bbw11', 'informed_trading_kyle85', 'multi_tf_reversion', 'mean_var_efficiency'],
    }

    def __init__(self):
        self._reasoning_mem = ReasoningMemory()
        self.logic_validator = MathematicalLogicValidator()
        self._hypothesis_map = {h['name']: h for h in self.HYPOTHESIS_LIBRARY}

    # ── Core composition: build formula from financial theory ──────────────

    def compose_from_hypothesis(self, regime: str, variables: List[str], windows: List[int],
                                generation: int = 0, single_asset_mode: bool = False) -> str:
        """Generate alpha formula by selecting and instantiating a financial hypothesis.
        
        Selection via Thompson Sampling: hypotheses that produced good IC in the past
        are sampled more frequently. New hypotheses start with uniform prior.
        """
        # Get regime-appropriate hypothesis candidates
        candidates = self.REGIME_HYPOTHESIS_AFFINITY.get(regime,
                     self.REGIME_HYPOTHESIS_AFFINITY['NEUTRAL'])
        
        # Thompson Sampling over hypothesis space
        sampler = self._reasoning_mem.get_sampler(f"{regime}:hypothesis", candidates)
        chosen_name = sampler.sample()
        hyp = self._hypothesis_map.get(chosen_name, self.HYPOTHESIS_LIBRARY[0])

        # Select variable: prefer hypothesis-recommended, rotate per generation
        hyp_vars = [v for v in hyp['variables'] if v in variables or v in ('$close', '$return')]
        all_vars = list(dict.fromkeys(hyp_vars + variables))[:6]
        var = all_vars[generation % len(all_vars)] if all_vars else '$close'

        # Select windows: rotate across generations for timeframe diversity
        window_pairs = []
        for ws in windows:
            for wl in windows:
                if wl > ws:
                    window_pairs.append((ws, wl))
        if not window_pairs:
            window_pairs = [(5, 20)]
        ws, wl = window_pairs[generation % len(window_pairs)]

        expr = hyp['structure'](var, ws, wl)
        expr = _adaptive_wrap_expression(
            expr,
            single_asset_mode=single_asset_mode,
            preferred_mode="TS_RANK" if single_asset_mode else "RANK",
            window=max(20, wl),
        )
        return expr

    def compose_interaction(self, regime: str, variables: List[str], windows: List[int],
                           generation: int = 0, single_asset_mode: bool = False) -> str:
        """Generate multi-component interaction formula.
        
        Combines TWO different hypothesis families with an arithmetic operator.
        This creates formulas that capture compound market dynamics.
        """
        candidates = self.REGIME_HYPOTHESIS_AFFINITY.get(regime,
                     self.REGIME_HYPOTHESIS_AFFINITY['NEUTRAL'])
        if len(candidates) < 2:
            candidates = list(self._hypothesis_map.keys())[:4]

        # Select two DIFFERENT hypotheses
        hyp1_name = candidates[generation % len(candidates)]
        hyp2_name = candidates[(generation + 1) % len(candidates)]
        if hyp2_name == hyp1_name and len(candidates) > 1:
            hyp2_name = candidates[(generation + 2) % len(candidates)]

        hyp1 = self._hypothesis_map[hyp1_name]
        hyp2 = self._hypothesis_map[hyp2_name]

        # Different variables for each component
        all_vars = list(dict.fromkeys(variables + ['$close', '$volume', '$return', '$high', '$low']))[:6]
        v1 = all_vars[generation % len(all_vars)]
        v2 = all_vars[(generation + 1) % len(all_vars)]
        if v2 == v1 and len(all_vars) > 1:
            v2 = all_vars[(generation + 2) % len(all_vars)]

        # Different timeframes
        ws_list = sorted(set(windows))
        if len(ws_list) >= 2:
            ws1, wl1 = ws_list[0], ws_list[-1]
            ws2 = ws_list[min(1, len(ws_list) - 1)]
            wl2 = ws_list[max(0, len(ws_list) - 2)]
        else:
            ws1, wl1 = (5, 20)
            ws2, wl2 = (10, 40)

        # Build components — strip outer RANK for interaction
        comp1 = hyp1['structure'](v1, ws1, wl1)
        comp2 = hyp2['structure'](v2, ws2, max(wl2, ws2 + 1))
        # Remove RANK/TS_RANK wrapper from components before combining
        comp1_inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', comp1)
        comp2_inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', comp2)

        # Choose arithmetic combiner via Thompson Sampling
        arith_ops = ['*', '/', '+', '-']
        combiner = self._reasoning_mem.get_sampler(f"{regime}:interaction_op", arith_ops).sample()

        combined = f"({comp1_inner}) {combiner} ({comp2_inner})"
        return _adaptive_wrap_expression(
            combined,
            single_asset_mode=single_asset_mode,
            preferred_mode="TS_RANK" if single_asset_mode else "RANK",
            window=max(20, wl1),
        )

    # ── AST-based genetic operations ──────────────────────────────────────

    def ast_crossover(self, parent_a: str, parent_b: str, generation: int = 0) -> Optional[str]:
        """Tree-level crossover: swap compatible subtrees between two formulas.
        
        Unlike string-level crossover, this produces syntactically valid children
        by matching subtree types (signal nodes swap with signal nodes, etc.).
        """
        tree_a = _parse_alpha_expr(parent_a)
        tree_b = _parse_alpha_expr(parent_b)
        if not tree_a or not tree_b:
            return None

        # Clone to avoid mutating originals
        child = tree_a.clone()
        donor = tree_b.clone()

        # Collect nodes that have children AND are not the root (prefer mid-tree)
        child_nodes = child.collect_nodes()
        donor_nodes = donor.collect_nodes()

        # Find parent→child edges in the child tree for safe replacement
        edges = []  # (parent_node, child_index)
        for node in child_nodes:
            for idx, c in enumerate(node.children):
                if c.depth() >= 1:  # prefer deeper subtrees
                    edges.append((node, idx))

        # Find donor subtrees of reasonable size
        donor_subtrees = [n for n in donor_nodes if 1 <= n.depth() <= 4]
        if not donor_subtrees:
            donor_subtrees = [n for n in donor_nodes if n.children]
        if not edges or not donor_subtrees:
            return None

        edge = edges[generation % len(edges)]
        donor_sub = donor_subtrees[(generation + 1) % len(donor_subtrees)]
        edge[0].children[edge[1]] = donor_sub.clone()

        result = child.to_expr()
        if len(result) > 240:
            return None
        return result

    def ast_mutation(self, expression: str, operators: List[str], variables: List[str],
                     windows: List[int], regime: str = 'NEUTRAL',
                     generation: int = 0) -> Optional[str]:
        """Type-safe AST mutation: change a node while respecting operator signatures.
        
        Mutation types (selected by Thompson Sampling):
        - variable_swap: change $close → $volume (preserves structure)
        - window_shift: change window parameter (preserves semantics, shifts timeframe)
        - operator_replace: change DELTA → TS_MEAN (preserves arg count)
        - subtree_regrow: replace a subtree with a fresh hypothesis component
        """
        tree = _parse_alpha_expr(expression)
        if not tree:
            return None

        child = tree.clone()
        all_nodes = child.collect_nodes()

        # Select mutation type via Thompson Sampling
        # Select mutation type via Thompson Sampling
        mut_types = ['variable_swap', 'window_shift', 'operator_replace', 'subtree_regrow', 'composite']
        mut_type = self._reasoning_mem.get_sampler(f"{regime}:ast_mutation", mut_types).sample()

        if mut_type == 'variable_swap':
            var_nodes = [n for n in all_nodes if n.kind == 'VAR' and n.value.startswith('$')]
            if var_nodes:
                target = var_nodes[generation % len(var_nodes)]
                new_var = variables[(generation + 1) % len(variables)] if variables else '$close'
                target.value = new_var

        elif mut_type == 'window_shift':
            windowed = [n for n in all_nodes if n.window is not None]
            if windowed:
                target = windowed[generation % len(windowed)]
                target.window = windows[generation % len(windows)] if windows else 10

        elif mut_type == 'operator_replace':
            func_nodes = [n for n in all_nodes if n.kind == 'FUNC' and n.value in self.OPERATOR_SIGNATURES]
            if func_nodes:
                target = func_nodes[generation % len(func_nodes)]
                n_args, needs_window = self.OPERATOR_SIGNATURES[target.value]
                # Find operators with same signature
                compatible = [op for op, sig in self.OPERATOR_SIGNATURES.items()
                             if sig == (n_args, needs_window) and op != target.value and op in operators]
                if compatible:
                    new_op = compatible[generation % len(compatible)]
                    target.value = new_op

        elif mut_type == 'subtree_regrow':
            # Replace a subtree with a hypothesis-based component (much richer than old DELTA-only)
            func_nodes = [n for n in all_nodes if n.children and n.depth() >= 1]
            if func_nodes:
                target = func_nodes[generation % len(func_nodes)]
                v = variables[generation % len(variables)] if variables else '$close'
                ws = windows[generation % len(windows)] if windows else 5
                wl = windows[(generation + 1) % len(windows)] if windows and len(windows) > 1 else 20
                
                # Use regime-aware hypothesis template for richer subtree regrowth
                fresh_expr = None
                regime_hypotheses = self.REGIME_HYPOTHESIS_AFFINITY.get(regime, [])
                if regime_hypotheses:
                    hyp_name = regime_hypotheses[generation % len(regime_hypotheses)]
                    hyp = self._hypothesis_map.get(hyp_name)
                    if hyp:
                        try:
                            hyp_v = v if v in hyp.get('variables', [v]) else hyp['variables'][0]
                            full_expr = hyp['structure'](hyp_v, ws, wl)
                            # Extract inner part (remove outer RANK wrapper to use as subtree)
                            import re as _re
                            inner = _re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', full_expr)
                            fresh_tree = _parse_alpha_expr(inner)
                            if fresh_tree and fresh_tree.depth() <= 3:
                                fresh_expr = inner
                        except Exception:
                            pass
                
                if fresh_expr:
                    fresh_tree = _parse_alpha_expr(fresh_expr)
                    if fresh_tree:
                        target.children[0] = fresh_tree
                    else:
                        # Fallback to diverse simple templates
                        fresh = AlphaASTNode('FUNC', 'DELTA', [AlphaASTNode('VAR', v)], ws)
                        target.children[0] = fresh
                else:
                    # Diverse fallback templates instead of always DELTA
                    _fallback_templates = [
                        lambda: AlphaASTNode('FUNC', 'DELTA', [AlphaASTNode('VAR', v)], ws),
                        lambda: AlphaASTNode('FUNC', 'TS_STD', [AlphaASTNode('VAR', v)], wl),
                        lambda: AlphaASTNode('FUNC', 'EMA', [AlphaASTNode('VAR', v)], ws),
                        lambda: AlphaASTNode('FUNC', 'TS_MEAN', [
                            AlphaASTNode('FUNC', 'ABS', [
                                AlphaASTNode('FUNC', 'DELTA', [AlphaASTNode('VAR', v)], 1)
                            ])
                        ], wl),
                    ]
                    fresh = _fallback_templates[generation % len(_fallback_templates)]()
                    target.children[0] = fresh

        elif mut_type == 'composite':
            # Chain 2 mutations for faster exploration (e.g., variable_swap + window_shift)
            sub_types = ['variable_swap', 'window_shift', 'operator_replace']
            import random as _rng
            picked = _rng.sample(sub_types, min(2, len(sub_types)))
            for sub_mut in picked:
                if sub_mut == 'variable_swap':
                    var_nodes = [n for n in child.collect_nodes() if n.kind == 'VAR' and n.value.startswith('$')]
                    if var_nodes:
                        t = var_nodes[generation % len(var_nodes)]
                        t.value = variables[(generation + 1) % len(variables)] if variables else '$close'
                elif sub_mut == 'window_shift':
                    windowed = [n for n in child.collect_nodes() if n.window is not None]
                    if windowed:
                        t = windowed[generation % len(windowed)]
                        t.window = windows[(generation + 1) % len(windows)] if windows else 15
                elif sub_mut == 'operator_replace':
                    func_nodes = [n for n in child.collect_nodes() if n.kind == 'FUNC' and n.value in self.OPERATOR_SIGNATURES]
                    if func_nodes:
                        t = func_nodes[generation % len(func_nodes)]
                        n_args, needs_window = self.OPERATOR_SIGNATURES[t.value]
                        compatible = [op for op, sig in self.OPERATOR_SIGNATURES.items()
                                     if sig == (n_args, needs_window) and op != t.value]
                        if compatible:
                            t.value = compatible[generation % len(compatible)]

        result = child.to_expr()
        if len(result) > 240:
            return None
        return result

    # ── Internal refinement: replaces _cloud_refine ───────────────────────

    def refine(self, candidate: str, patterns: Dict[str, Any], regime: str,
               operators: List[str], variables: List[str], windows: List[int],
               generation: int = 0,
               single_asset_mode: bool = False) -> str:
        """Refine a formula using internal algebraic rules — NO external API calls.
        
        Refinement strategies (applied based on quality analysis):
        1. Add volatility normalization if missing
        2. Add diversity operator if too simple
        3. Replace redundant operators
        4. Inject cross-variable interaction
        """
        tree = _parse_alpha_expr(candidate)
        if not tree:
            return candidate

        ops_used = set(re.findall(r'[A-Z_]+(?=\()', candidate))
        vars_used = set(re.findall(r'\$\w+', candidate))

        # Strategy 1: Add volatility normalization if raw signal
        if 'TS_STD' not in ops_used and tree.depth() < 3:
            v = list(vars_used)[0] if vars_used else '$close'
            w = windows[-1] if windows else 20
            return f"({candidate}) / (TS_STD({v}, {w}) + {_smart_eps('std')})"

        # Strategy 2: Add cross-variable interaction if single-variable
        if len(vars_used) <= 1 and len(variables) > 1:
            other_var = [v for v in variables if v not in vars_used]
            if other_var:
                v2 = other_var[generation % len(other_var)]
                w = windows[generation % len(windows)] if windows else 10
                inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', candidate)
                return _adaptive_wrap_expression(
                    f"({inner}) * ({v2} / (TS_MEAN({v2}, {w}) + {_smart_eps('mean')}))",
                    single_asset_mode=single_asset_mode,
                    preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                    window=max(20, w),
                )

        # Strategy 3: Enhance with trend quality filter
        if 'SIGN' not in ops_used and 'TS_MEAN' in ops_used:
            v = list(vars_used)[0] if vars_used else '$close'
            w = windows[0] if windows else 5
            inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', candidate)
            return _adaptive_wrap_expression(
                f"({inner}) * TS_MEAN(SIGN(DELTA({v}, 1)), {w})",
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, w),
            )

        return candidate

    # ── Full generation pipeline ──────────────────────────────────────────

    def generate_candidates(self, regime: str, operators: List[str],
                           variables: List[str], windows: List[int],
                           generation: int = 0,
                           single_asset_mode: bool = False,
                           recall: Optional[List] = None,
                           num_candidates: int = 3) -> List[str]:
        """Generate diverse alpha candidates using pure internal reasoning.
        
        Pipeline (intelligence-priority order):
        0. Skeleton-based: reuse proven expression shapes with new vars/windows
        1. Hypothesis-based composition (from financial theory)
        2. Interaction composition (multi-hypothesis compound)
        3. AST crossover from memory (if available)
        4. AST mutation from best memory (if available)
        5. Quality validation + refinement
        """
        candidates = []

        # 0. SKELETON-FIRST: reuse proven winning expression shapes
        #    This is the AI's strongest signal — shapes that already produced IC > 0.05
        top_skeletons = _skeleton_learner.get_top_skeletons(n=3)
        for sk_idx, sk in enumerate(top_skeletons):
            if len(candidates) >= num_candidates:
                break
            sk_expr = _skeleton_learner.generate_from_skeleton(
                sk, variables, windows, generation=generation + sk_idx
            )
            if sk_expr and sk_expr not in candidates:
                candidates.append(sk_expr)
                logger.info(f"[SKELETON] Reusing shape (avgIC={sk['ic_sum']/max(1,sk['count']):.4f}): {sk_expr[:80]}")

        # 1. Pure hypothesis composition
        hyp_formula = self.compose_from_hypothesis(
            regime, variables, windows, generation, single_asset_mode
        )
        if hyp_formula not in candidates:
            candidates.append(hyp_formula)

        # 2. Multi-hypothesis interaction
        if len(candidates) < num_candidates + 1:
            interaction = self.compose_interaction(
                regime, variables, windows, generation, single_asset_mode
            )
            if interaction not in candidates:
                candidates.append(interaction)

        # 3. AST crossover from memory patterns
        if recall and len(recall) >= 2 and len(candidates) < num_candidates + 2:
            parent_a = recall[0].Expression
            parent_b = recall[min(1, len(recall) - 1)].Expression
            child = self.ast_crossover(parent_a, parent_b, generation)
            if child and child not in candidates:
                candidates.append(child)

        # 4. AST mutation from best memory
        if recall and len(candidates) < num_candidates + 3:
            best_parent = max(recall, key=lambda x: x.Metrics.Fitness * x.Reinforcement)
            mutated = self.ast_mutation(
                best_parent.Expression, operators, variables, windows, regime, generation
            )
            if mutated and mutated != best_parent.Expression and mutated not in candidates:
                candidates.append(mutated)

        # 5. Validate and refine each candidate
        refined = []
        for cand in candidates:
            is_valid, reasoning, score = self.logic_validator.validate_logic(cand, regime, operators)
            if score < 0.45:
                cand = self.refine(
                    cand,
                    {},
                    regime,
                    operators,
                    variables,
                    windows,
                    generation,
                    single_asset_mode=single_asset_mode,
                )
            if len(cand) <= 240:
                refined.append(cand)
            else:
                refined.append(cand[:240])

        return refined


# ═══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK REFINEMENT ENGINE: Learn from backtest failures, refine formulas
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackRefinementEngine:
    """Learns from backtest IC/ICIR feedback to intelligently refine failing formulas.

    Unlike simple regeneration, this engine DIAGNOSES why a formula failed
    and applies targeted surgical fixes:
    - Overfitting → simplify (reduce depth, strip redundant operators)
    - Weak signal → amplify (add cross-variable interaction, change window)
    - Wrong regime → adapt (switch momentum↔reversion operators)
    - Numerical instability → stabilize (add normalization, fix denominators)
    """

    # Failure diagnosis rules: (condition_fn, diagnosis, fix_strategy)
    DIAGNOSIS_RULES = [
        ("overfit",   "IC decent but ICIR < 0.3 → overfitting to specific market phase"),
        ("weak",      "IC < 0.015 → signal too weak, need different operator family"),
        ("noise",     "IC ≈ 0 → no signal detected, formula captures only noise"),
        ("unstable",  "IC volatile across periods → numerical instability or regime mismatch"),
        ("toolong",   "Expression > 2000 chars → over-engineered, signal diluted by complexity"),
        ("qlib_physics", "Formula violates Qlib data physics (e.g. using $open without lag, missing 1e-8 safety)"),
    ]

    # Axis 9: Quanta-Native Physics (Deep Reasoning Foundation)
    QUANTA_PHYSICS_KNOWLEDGE = {
        "NAN_RISK": "Qlib $open often contains NaNs at the start of sessions; always use TS_MEAN($open, 1) or TS_FILLNA to stabilize.",
        "CAUSALITY": "Causing lookahead bias: using $close[t] to predict return[t]. Always target DELTA($close, 1) or DELTA($open, -1) correctly.",
        "VOL_SKEW": "Volume is unscaled; raw $volume adds noise. Always normalize via RANK($volume) or $volume / TS_MEAN($volume, 20).",
        "INTRA_DAY": "Alpha often lives in ($close - $open). Standard returns are noisy. Try capturing intra-day gaps.",
        "STABILITY": "Numerical instability in LOG; always use LOG(ABS(x) + 1e-8) to prevent -inf/nan crashes.",
    }

    # Surgical fix strategies mapped to each diagnosis
    FIX_STRATEGIES = {
        "overfit": [
            "strip_outer_layer",      # Remove outermost operator wrapper
            "shorten_windows",        # Cut lookback windows by 30-50%
            "simplify_to_core",       # Extract core 2-operator sub-expression
        ],
        "weak": [
            "add_cross_variable",     # Inject price-volume interaction
            "switch_operator_family", # Change DELTA→TS_CORR or TS_MEAN→TS_STD
            "change_primary_variable",# Switch $close→$volume or $return
            "add_volatility_norm",    # Add risk normalization
        ],
        "noise": [
            "rebuild_from_hypothesis",# Full reset using InternalCompositionEngine
            "flip_direction",         # Try inverse signal (multiply by -1)
            "switch_operator_family", # Completely change approach
        ],
        "unstable": [
            "add_volatility_norm",    # Wrap with / (TS_STD + 1e-8)
            "add_rank_wrapper",       # Add RANK() for cross-sectional stability
            "widen_windows",          # Increase lookback for smoother signal
        ],
        "toolong": [
            "ast_prune_deepest",      # Remove the deepest subtree branch
            "collapse_redundant",     # Merge nested same-type operators
            "extract_strongest_branch",# Keep only the highest-quality sub-expression
        ],
        "direction_wrong": [
            "flip_direction",         # Signal is inverted — multiply by -1
            "rebuild_from_hypothesis",# Start over with different theory
            "change_primary_variable",# Different variable may fix direction
        ],
        "overcrowded": [
            "simplify_to_core",       # Too many ops — extract the strongest 2-op core
            "ast_prune_deepest",      # Remove deepest branch to reduce noise
            "strip_outer_layer",      # Remove outermost wrapper
            "collapse_redundant",     # Merge consecutive same-type operators
        ],
        "window_mismatch": [
            "shorten_windows",        # Windows too long for current regime
            "widen_windows",          # Windows too short for current regime
            "add_volatility_norm",    # Normalize to make signal window-robust
        ],
        # ── Dual-market diagnoses (NEW — from Council) ─────────────────────
        "forex_only_weak": [
            "add_ts_rank_wrapper",    # TS_RANK works better for forex single-asset
            "shorten_windows",        # Forex signals typically shorter horizon
            "add_volatility_norm",    # Normalize to handle FX vol regime
            "switch_operator_family", # Try different operator family
        ],
        "stock_only_weak": [
            "add_rank_wrapper",       # RANK() makes signal cross-sectional for stock
            "change_primary_variable",# $volume / $turnover helps stock cross-section
            "add_cross_variable",     # Price-volume interaction is stock-specific
            "widen_windows",          # Stock signals can use longer lookbacks
        ],
        "paradigm_shift": [
            "rebuild_from_hypothesis",# Total rebuild with new theory
            "switch_operator_family", # Change entire operator family tree
            "flip_direction",         # If theory is wrong, try opposite signal
        ],
    }

    def __init__(self):
        self._reasoning_mem = ReasoningMemory()
        self._composition_engine = InternalCompositionEngine()
        self.logic_validator = MathematicalLogicValidator()
        self._refinement_history: deque = deque(maxlen=200)
        self._failure_patterns: Dict[str, List[Dict]] = defaultdict(list)  # diagnosis → [{expr, ic, fix_applied}]
        # Axis 12: Self-Aware Reflection Buffer (Limb-specific memory)
        self._self_correction_buffer: Dict[str, List[str]] = defaultdict(list) # expression_root → [failed_diagnoses]

    def diagnose(self, expression: str, ic: float, icir: float = 0.0,
                 regime: str = "NEUTRAL",
                 council_context: Optional[Any] = None) -> str:
        """Diagnose WHY a formula failed — GPT-5.1 Root Cause Analysis.

        Returns a precise diagnosis key for the refinement engine.
        """
        depth = expression.count('(')
        n_ops = len(re.findall(r'[A-Z_]+(?=\()', expression))

        # ── Axis 1: Council Strategic Mandate (Highest Priority) ──────────
        if council_context is not None:
            mandate = getattr(council_context, "evolution_mandate", "")
            if mandate == "FUNDAMENTAL_REDESIGN":
                return "paradigm_shift"
            if mandate == "SURGICAL_IC_BOOST":
                return "weak_signal"
            if mandate == "REFINE_MDD_FIX":
                return "overfit_risk"

        # ── Axis 2: Market Type Dual-Pass Check ───────────────────────────
        if council_context is not None:
            stock_fit = getattr(council_context, "stock_fit_score", 0.5)
            forex_fit = getattr(council_context, "forex_fit_score", 0.5)
            if stock_fit > 0.6 and forex_fit < 0.35: return "forex_specific_failure"
            if forex_fit > 0.6 and stock_fit < 0.35: return "stock_specific_failure"

        # ── Axis 3: Statistical Root Cause Analysis ──────────────────────
        # Category A: Directional Failure
        if ic < -0.005: return "inverse_correlation"
        
        # Category B: Complexity/Overfitting Root Cause
        if n_ops >= 15 and ic < 0.02: return "diluted_by_complexity"
        if len(expression) > 2000 and ic < 0.02: return "toolong"
        if ic >= 0.03 and icir < 0.35: return "unstable_overfit"
        
        # Category C: Window/Regime Root Cause
        if 0.01 <= ic < 0.03 and icir < 0.2: return "regime_window_mismatch"
        
        # Category D: Noise Root Cause
        if abs(ic) <= 0.005: return "pure_noise"

        # Fallback
        return "weak_signal"
        # Has some IC but not stable enough
        if icir < 0.2 and ic >= 0.015:
            return "unstable"
        return "weak"  # default

    def get_surgical_prescription(self, diagnosis: str) -> Dict[str, Any]:
        """Provides specific mathematical operators to fix a diagnosed failure."""
        presets = {
            "paradigm_shift": ["TS_CORR", "CROSS_ASSET_STD", "DELTA_LOG"],
            "weak_signal": ["RANK", "TS_RANK", "SIGN", "BINS"],
            "overfit_risk": ["EMA", "TS_MEAN", "LOW_PASS_FILTER"],
            "forex_specific_failure": ["DELTA", "ABS", "TS_MIN", "TS_MAX"],
            "stock_specific_failure": ["RANK", "TS_RANK", "SIGN"],
            "inverse_correlation": ["NEGATE", "SIGN"],
            "diluted_by_complexity": ["TS_MEAN", "SUM"],
            "unstable_overfit": ["EMA", "TS_STD_NORM", "WMA"],
            "regime_window_mismatch": ["TS_MEAN_20", "TS_RANK_60", "DELAY_40"],
            "pure_noise": ["RANK(DELTA)", "VOLATILITY_ADAPTIVE_STRETCH"],
        }
        ops = presets.get(diagnosis, ["RANK", "DELTA", "TS_MEAN"])
        
        # Axis 9: Quanta-Native Reasoning (Expert Diagnostics)
        # We inject specific Qlib "physics" knowledge to guide the LLM
        guidance = f"Prioritize using these operators: {', '.join(ops)}."
        
        # Inject Quanta Physics into the instruction
        if diagnosis == "weak_signal":
            guidance += f" {self.QUANTA_PHYSICS_KNOWLEDGE['INTRA_DAY']} {self.QUANTA_PHYSICS_KNOWLEDGE['VOL_SKEW']}"
        elif diagnosis == "unstable_overfit":
            guidance += f" {self.QUANTA_PHYSICS_KNOWLEDGE['STABILITY']}"
        elif diagnosis == "pure_noise":
            guidance += f" {self.QUANTA_PHYSICS_KNOWLEDGE['NAN_RISK']} Use TS_RANK for stationarity."
        elif diagnosis == "qlib_physics":
            guidance = f"CRITICAL FIX: {self.QUANTA_PHYSICS_KNOWLEDGE['CAUSALITY']} {self.QUANTA_PHYSICS_KNOWLEDGE['STABILITY']}"
            
        return {
            "suggested_ops": ops,
            "instruction": guidance,
            "needs_rank": diagnosis in ["weak_signal", "pure_noise", "stock_specific_failure", "qlib_physics"]
        }

    def refine_from_feedback(self, expression: str, ic: float, icir: float = 0.0,
                             regime: str = "NEUTRAL",
                             operators: Optional[List[str]] = None,
                             variables: Optional[List[str]] = None,
                             windows: Optional[List[int]] = None,
                             generation: int = 0,
                             preferred_strategies: Optional[List[str]] = None,
                             dna: Optional[Any] = None,
                             council_context: Optional[Any] = None) -> Tuple[str, str, str]:
        """Refine a failing formula — exhaustive multi-strategy approach.

        Tries REAL LLM SYNTHESIS first (if available), then falls back to
        5 heuristic strategies with quality scoring.
        Returns: (refined_expression, diagnosis, fix_applied)
        """
        operators = operators or ["DELTA", "TS_MEAN", "TS_STD", "TS_CORR", "RANK", "TS_RANK"]
        variables = variables or ["$close", "$volume", "$return", "$high", "$low"]
        windows = windows or [5, 10, 20, 40, 60]
        single_asset_hint = expression.strip().startswith("TS_RANK(")

        diagnosis = self.diagnose(expression, ic, icir, regime, council_context=council_context)
        prescription = self.get_surgical_prescription(diagnosis)

        # ── 🧠 REAL LLM SYNTHESIS: Let the LLM fix the formula directly ──
        try:
            from sova_cloud_brain import HybridReasoningMatrix
            cloud = HybridReasoningMatrix()
            
            # Convert DNA to summary if provided
            dna_summary = ""
            if dna:
                # Try to use a helper if it exists, otherwise generic str
                if hasattr(dna, "to_dict"):
                    dna_summary = json.dumps(dna.to_dict(), indent=2)
                else:
                    dna_summary = str(dna)
            
            # Axis 12: Inject Self-Aware Reflection into the prompt
            reflection_note = ""
            root_expr = expression.split('(')[0] if '(' in expression else expression
            if root_expr in self._self_correction_buffer:
                past_failures = ", ".join(set(self._self_correction_buffer[root_expr]))
                reflection_note = f" SELF-AWARENESS: Your previous attempts for this logic failed due to: {past_failures}. DO NOT repeat these same patterns."

            refined_llm = cloud.refine_expression_from_feedback(
                expression=expression,
                ic=ic,
                icir=icir,
                regime=regime,
                diagnosis=diagnosis + reflection_note,
                dna_summary=dna_summary,
                is_forex=single_asset_hint,
                council_context=council_context,
                is_expansion=("expand" in (preferred_strategies or [])) or (generation > 0 and abs(ic) < 0.01),
                prescription=prescription
            )
            
            # Record failure in self-correction buffer if IC is low
            if ic < 0.01:
                self._self_correction_buffer[root_expr].append(diagnosis)
            
            if refined_llm:
                source = getattr(cloud, "_last_feedback_refine_source", "unknown")
                if source == "cloud":
                    fix_tag = "cloud_synthesis"
                elif source == "local":
                    fix_tag = "local_synthesis"
                else:
                    fix_tag = "llm_synthesis"
                logger.info(f"[FeedbackRefiner] ✅ Synthesis succeeded for {diagnosis} via {source}")
                return refined_llm, diagnosis, fix_tag
        except Exception as e:
            logger.debug(f"[FeedbackRefiner] LLM Synthesis skipped: {e}")

        # Fallback to heuristic strategies
        strategies = list(self.FIX_STRATEGIES.get(diagnosis, ["add_volatility_norm"]))
        if preferred_strategies:
            preferred = [s for s in preferred_strategies if s in strategies]
            trailing = [s for s in strategies if s not in preferred]
            strategies = preferred + trailing

        # Exhaustive approach: try up to 5 strategies, keep the best result
        MAX_ATTEMPTS = 5
        best_refined = None
        best_score = -1.0
        best_strategy = strategies[0] if strategies else "rebuild_from_hypothesis"

        # Order strategies by Thompson Sampling posteriors for this diagnosis
        fix_key = f"feedback_fix:{regime}:{diagnosis}"
        sampler = self._reasoning_mem.get_sampler(fix_key, strategies)
        
        tried_strategies = []
        for attempt in range(min(MAX_ATTEMPTS, len(strategies))):
            # Sample strategy (Thompson Sampling naturally explores vs exploits)
            strategy = sampler.sample()
            if strategy in tried_strategies:
                # Force untried strategy
                untried = [s for s in strategies if s not in tried_strategies]
                if not untried:
                    break
                strategy = untried[0]
            tried_strategies.append(strategy)

            refined = self._apply_fix(expression, strategy, operators, variables, windows, regime, generation + attempt)

            if refined and refined != expression:
                is_valid, _, score = self.logic_validator.validate_logic(refined, regime, operators)
                if is_valid and score > best_score:
                    best_refined = refined
                    best_score = score
                    best_strategy = strategy
                    logger.info(f"[REFINE] Strategy '{strategy}' produced score={score:.3f}: {refined[:70]}")

        if not best_refined or best_refined == expression:
            # Nuclear option: rebuild from scratch using hypothesis engine
            candidates = self._composition_engine.generate_candidates(
                regime,
                operators,
                variables,
                windows,
                generation=generation,
                single_asset_mode=single_asset_hint,
                num_candidates=2,
            )
            # Pick the candidate with highest quality score
            for cand in candidates:
                _, _, score = self.logic_validator.validate_logic(cand, regime, operators)
                if score > best_score:
                    best_refined = cand
                    best_score = score
                    best_strategy = "rebuild_from_hypothesis"
            if not best_refined:
                best_refined = candidates[0] if candidates else expression
                best_strategy = "rebuild_from_hypothesis"

        # Record for learning
        self._refinement_history.append({
            "original": expression, "refined": best_refined,
            "diagnosis": diagnosis, "strategy": best_strategy,
            "ic": ic, "regime": regime,
            "attempts_tried": len(tried_strategies),
            "best_score": best_score,
        })

        return best_refined, diagnosis, best_strategy

    def record_refinement_outcome(self, original: str, refined: str, diagnosis: str,
                                   strategy: str, new_ic: float, regime: str = "NEUTRAL"):
        """Feed back the IC result of a refinement to improve future fix selection."""
        reward = new_ic - 0.01  # baseline: any IC > 0.01 is positive
        fix_key = f"feedback_fix:{regime}:{diagnosis}"
        self._reasoning_mem.update(fix_key, strategy, reward)

        self._failure_patterns[diagnosis].append({
            "expression": original[:80], "refined": refined[:80],
            "strategy": strategy, "ic_improvement": new_ic,
        })
        # Keep only recent patterns
        if len(self._failure_patterns[diagnosis]) > 50:
            self._failure_patterns[diagnosis] = self._failure_patterns[diagnosis][-40:]

    def _apply_fix(self, expression: str, strategy: str,
                   operators: List[str], variables: List[str], windows: List[int],
                   regime: str, generation: int) -> Optional[str]:
        """Apply a specific fix strategy to a formula."""
        single_asset_hint = expression.strip().startswith("TS_RANK(")
        preferred_wrap = "TS_RANK" if single_asset_hint else "RANK"

        if strategy == "strip_outer_layer":
            # Remove outermost RANK/TS_RANK wrapper
            m = re.match(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', expression)
            if m:
                inner = m.group(1)
                return inner
            return None

        elif strategy == "shorten_windows":
            # Reduce all window parameters by ~40%
            def _shorten(match):
                w = int(match.group(1))
                new_w = max(2, int(w * 0.6))
                return f", {new_w})"
            return re.sub(r',\s*(\d+)\)', _shorten, expression)

        elif strategy == "simplify_to_core":
            # Parse AST and extract the deepest meaningful 2-op sub-expression
            tree = _parse_alpha_expr(expression)
            if not tree:
                return None
            # Find the deepest FUNC node with at least one FUNC child
            all_nodes = tree.collect_nodes()
            func_nodes = [n for n in all_nodes if n.kind == 'FUNC' and n.children]
            if len(func_nodes) >= 2:
                # Take the second-deepest function as core
                core = func_nodes[min(1, len(func_nodes) - 1)]
                core_expr = core.to_expr()
                if len(core_expr) > 10:
                    return _adaptive_wrap_expression(
                        core_expr,
                        single_asset_mode=single_asset_hint,
                        preferred_mode=preferred_wrap,
                        window=60,
                    )
            return None

        elif strategy == "add_cross_variable":
            other_vars = [v for v in variables if v not in expression]
            if not other_vars:
                return None
            v2 = other_vars[generation % len(other_vars)]
            w = windows[generation % len(windows)]
            inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', expression)
            return _adaptive_wrap_expression(
                f"({inner}) * ({v2} / (TS_MEAN({v2}, {w}) + 1e-8))",
                single_asset_mode=single_asset_hint,
                preferred_mode=preferred_wrap,
                window=max(20, int(w)),
            )

        elif strategy == "switch_operator_family":
            # Swap primary operator to a different family
            swaps = {"DELTA": "TS_MEAN", "TS_MEAN": "DELTA", "TS_STD": "TS_CORR",
                     "TS_CORR": "TS_STD", "TS_SKEW": "TS_STD"}
            result = expression
            for old_op, new_op in swaps.items():
                if old_op + "(" in result:
                    result = result.replace(old_op + "(", new_op + "(", 1)
                    break
            return result if result != expression else None

        elif strategy == "change_primary_variable":
            vars_in_expr = re.findall(r'\$\w+', expression)
            if not vars_in_expr:
                return None
            primary = vars_in_expr[0]
            alternatives = [v for v in variables if v != primary]
            if not alternatives:
                return None
            new_var = alternatives[generation % len(alternatives)]
            return expression.replace(primary, new_var, 1)

        elif strategy == "rebuild_from_hypothesis":
            candidates = self._composition_engine.generate_candidates(
                regime,
                operators,
                variables,
                windows,
                generation=generation,
                single_asset_mode=single_asset_hint,
                num_candidates=1,
            )
            return candidates[0] if candidates else None

        elif strategy == "flip_direction":
            inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', expression)
            return _adaptive_wrap_expression(
                f"-1 * ({inner})",
                single_asset_mode=single_asset_hint,
                preferred_mode=preferred_wrap,
                window=60,
            )

        elif strategy == "add_volatility_norm":
            v = re.findall(r'\$\w+', expression)
            v = v[0] if v else "$close"
            w = windows[-1] if windows else 20
            inner = re.sub(r'^(?:RANK|TS_RANK)\((.+?)(?:,\s*\d+)?\)$', r'\1', expression)
            return _adaptive_wrap_expression(
                f"({inner}) / (TS_STD({v}, {w}) + 1e-8)",
                single_asset_mode=single_asset_hint,
                preferred_mode=preferred_wrap,
                window=max(20, int(w)),
            )

        elif strategy == "add_rank_wrapper":
            if not expression.startswith("RANK(") and not expression.startswith("TS_RANK("):
                return _adaptive_wrap_expression(
                    expression,
                    single_asset_mode=single_asset_hint,
                    preferred_mode=preferred_wrap,
                    window=60,
                )
            return None

        elif strategy == "widen_windows":
            def _widen(match):
                w = int(match.group(1))
                new_w = min(120, int(w * 1.5))
                return f", {new_w})"
            return re.sub(r',\s*(\d+)\)', _widen, expression)

        elif strategy == "ast_prune_deepest":
            tree = _parse_alpha_expr(expression)
            if not tree:
                return None
            all_nodes = tree.collect_nodes()
            # Find deepest node and replace it with a simple variable
            deepest = max(all_nodes, key=lambda n: n.depth())
            if deepest.kind in ('VAR', 'CONST'):
                # Already a leaf — prune its parent's deepest child instead
                for node in all_nodes:
                    for i, child in enumerate(node.children):
                        if child.depth() == tree.depth() - 1 and child.children:
                            v = variables[generation % len(variables)]
                            node.children[i] = AlphaASTNode('VAR', v)
                            result = tree.to_expr()
                            return result if len(result) <= 240 else None
            return None

        elif strategy == "collapse_redundant":
            # Remove nested same-function calls: RANK(RANK(x)) → RANK(x)
            result = expression
            for op in ['RANK', 'ABS', 'SIGN']:
                result = re.sub(rf'{op}\({op}\(', f'{op}(', result)
                # Fix paren balance
                if result.count('(') > result.count(')'):
                    pass  # already handled
                elif result.count(')') > result.count('('):
                    result = result[:result.rfind(')')] + result[result.rfind(')') + 1:]
            return result if result != expression else None

        elif strategy == "extract_strongest_branch":
            tree = _parse_alpha_expr(expression)
            if not tree:
                return None
            # Take the child with the most operators (richest signal)
            if tree.children:
                best_child = max(tree.children, key=lambda c: c.size())
                child_expr = best_child.to_expr()
                if len(child_expr) > 10 and '$' in child_expr:
                    return _adaptive_wrap_expression(
                        child_expr,
                        single_asset_mode=single_asset_hint,
                        preferred_mode=preferred_wrap,
                        window=60,
                    )
            return None

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ALPHA THEORY DESCRIBER: Generate mathematical theory explanation for formulas
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaTheoryDescriber:
    """Generates human-readable mathematical theory descriptions for alpha formulas.

    Acts like a quantitative mathematician explaining the financial intuition and
    mathematical foundation behind each formula — in precise academic language.
    """

    # Operator → mathematical meaning + financial interpretation
    OPERATOR_SEMANTICS = {
        "DELTA":    ("finite difference operator Δ_w", "captures price momentum (rate of change over w bars)"),
        "TS_MEAN":  ("exponential/simple moving average μ_w", "smooths noise to reveal underlying trend"),
        "TS_STD":   ("rolling standard deviation σ_w", "measures local volatility / risk"),
        "TS_CORR":  ("rolling Pearson correlation ρ_w", "quantifies co-movement between two time series"),
        "TS_SKEW":  ("rolling third central moment γ_w", "detects return distribution asymmetry (crash risk)"),
        "TS_KURT":  ("rolling excess kurtosis κ_w", "measures tail heaviness — extreme event frequency"),
        "TS_RANK":  ("rolling temporal percentile rank", "normalizes signal within its own history (time-series z-score)"),
        "TS_MAX":   ("rolling maximum", "identifies local price ceiling / resistance level"),
        "TS_MIN":   ("rolling minimum", "identifies local price floor / support level"),
        "RANK":     ("cross-sectional percentile rank", "normalizes signal across all instruments at each time step"),
        "SIGN":     ("signum function sgn(x)", "extracts pure direction (+1/0/-1), removing magnitude noise"),
        "ABS":      ("absolute value |x|", "captures magnitude regardless of direction"),
        "DELAY":    ("lag operator L_w", "shifts time series by w bars for autoregressive comparison"),
        "LOG":      ("natural logarithm ln(x)", "compresses range, converts multiplicative to additive dynamics"),
    }

    # Composition pattern → academic theory reference
    COMPOSITION_THEORIES = {
        ("DELTA", "TS_STD"):   "Sharpe ratio decomposition — momentum normalized by volatility (risk-adjusted return)",
        ("DELTA", "TS_MEAN"):  "Smoothed momentum — trend acceleration after noise filtration (Moskowitz et al. 2012)",
        ("DELTA", "SIGN"):     "Binary momentum — direction-only signal stripping magnitude noise",
        ("TS_CORR", "DELTA"):  "Kyle (1985) informed trading — price-volume correlation reveals institutional flow",
        ("TS_STD", "TS_STD"):  "Volatility ratio — short/long vol compression detects regime transitions",
        ("TS_MEAN", "TS_MEAN"): "Double smoothing — Hodrick-Prescott-style trend extraction",
        ("TS_MAX", "TS_MIN"):  "Channel oscillator — bounded [0,1] mean-reversion signal (support/resistance)",
        ("TS_SKEW",):         "Barberis & Huang (2008) — lottery premium from skewness mispricing",
        ("SIGN", "TS_MEAN"):   "Trend consistency — directional batting average (what fraction of bars moved up/down)",
    }

    # Variable → financial meaning
    VARIABLE_MEANINGS = {
        "$close":  "closing price — primary price signal, most liquid and widely tracked",
        "$open":   "opening price — encodes overnight information gap and opening auction dynamics",
        "$high":   "intraday high — measures buying pressure / bullish conviction",
        "$low":    "intraday low — measures selling pressure / bearish conviction",
        "$volume": "trading volume — liquidity proxy, institutional participation indicator",
        "$return": "period return — standardized price change, stationary by construction",
    }

    def describe(self, expression: str, regime: str = "NEUTRAL",
                 ic: float = 0.0, icir: float = 0.0) -> str:
        """Generate an institutional-grade mathematical theory description.

        Produces a deep-dive academic narrative mirroring a professional quant 
        research paper, covering structural encoding, financial grounding, 
        and market regime resilience.
        """
        ops = list(dict.fromkeys(re.findall(r'[A-Z_]+(?=\()', expression)))
        vars_used = list(dict.fromkeys(re.findall(r'\$\w+', expression)))
        windows = [int(w) for w in re.findall(r',\s*(\d+)\)', expression)]
        depth = expression.count('(')

        # 1. Structural Mathematical Encoding
        math_parts = []
        for op in ops:
            sem = self.OPERATOR_SEMANTICS.get(op)
            if sem:
                math_parts.append(f"  • {op} ({sem[0]}): {sem[1]}")
        math_section = "**Structural Encoding:**\n" + "\n".join(math_parts) if math_parts else ""

        # 2. Financial Theoretical Grounding
        theory = None
        ops_tuple = tuple(ops[:2])
        for pattern, desc in self.COMPOSITION_THEORIES.items():
            if all(p in ops for p in pattern):
                theory = desc
                break
        
        if not theory:
            if "TS_CORR" in ops:
                theory = "Co-movement analysis — identifies lead-lag relationships or institutional order flow footprints."
            elif "TS_SKEW" in ops or "TS_KURT" in ops:
                theory = "Higher-moment extraction — captures distributional tail risk mispricing (lottery effect theory)."
            elif "DELTA" in ops and "RANK" in ops:
                theory = "Cross-sectional momentum — exploits relative strength dispersion under behavioral anchoring."
            else:
                theory = "Dynamic factor encoding — identifies latent price-volume efficiencies within specific market microstructure."

        # 3. Time-Scale Interaction Analysis
        if windows:
            short_w = min(windows)
            long_w = max(windows)
            window_analysis = (
                f"**Multi-Scale Dynamics & Logic:** The strategy operates across a {short_w}-bar to {long_w}-bar horizon. "
                f"These layers interact systematically to isolate robust market behaviors. "
                f"{'Short-term windows focus on immediate liquidity-driven micro-patterns, rapidly adapting to new information. ' if short_w <= 10 else ''}"
                f"{'Medium-to-long term windows filter out microstructure noise to establish the structural trend baseline and risk boundaries. ' if long_w >= 30 else ''}"
                f"{'The interplay between these varied time-scales implies a cross-frequency mechanism designed to reject short-lived noise bursts while maintaining stable core directional conviction. ' if long_w > 2 * short_w else ''}"
            )
        else:
            window_analysis = "**Time-Scale Dynamics & Logic:** A robust cross-sectional normalization approach operating without immediate temporal lookback dependency, isolating instantaneous relative strength."

        # 4. Regime Adaptability & Resilience
        regime_narratives = {
            "NEUTRAL": "Highly effective in stationary market environments, where historical mean and variance provide a reliable predictive anchor.",
            "EXPONENTIAL_BULL": "Specifically designed to surf upward convexity — the formula captures the rapid acceleration phase of strong institutional accumulation while filtering out minor retracements.",
            "CAPITULATION_CRASH": "Acts as a defensive contrarian signal — precisely identifying extreme oversold conditions or liquidity black holes to execute strategic mean-reversion trades.",
            "CHAOTIC_NOISE": "Operates with a high-frequency noise rejection mechanism — aggressively using statistical smoothing and variance bounds to find true stability amidst chaotic market entropy.",
            "DISTRIBUTION_ANOMALY": "Exploits non-normal return distributions — strategically capturing skewness-driven alpha and tail risks often completely missed by standard linear pricing models.",
            "MEAN_REVERSION_ZONE": "Targeted at classic mean-reverting regimes — the logic directly quantifies the elastic strain of price drifting too far from its moving equilibrium, anticipating the inevitable snapback.",
        }
        regime_context = regime_narratives.get(regime, f"Specifically calibrated for {regime} market conditions.")

        # 5. Conviction & Risk Assessment
        complexity_label = "Institutional-Standard" if 3 <= depth <= 5 else ("Over-Simplified" if depth < 3 else "High-Complexity/Research-Grade")
        complexity_note = f"**Research Profile:** {complexity_label} architecture (depth={depth}). "
        if depth > 6:
            complexity_note += "Caution: Structural complexity exceeds typical 'parsimony' thresholds; monitor for overfitting."
        
        # 6. Empirical Synthesis
        perf_synthesis = ""
        if ic > 0:
            quality = "Alpha Prime (High Conviction)" if ic >= 0.05 else ("Institutional Grade" if ic >= 0.035 else "Exploratory Signal")
            perf_synthesis = (
                f"\n\n**Empirical Performance Synthesis:**\n"
                f"Current validation yields an IC of {ic:.4f} ({quality}). "
                f"This signal demonstrates {'strong standalone predictive accuracy' if ic >= 0.05 else 'dependable correlation to forward returns'}. "
                f"Statistical significance is enhanced by the {'multi-scale' if len(set(windows)) > 1 else 'structural'} design."
            )

        narrative = (
            f"### Professional Strategy Narrative: {theory}\n\n"
            f"**Hypothesis Overview:** This alpha signal treats `{expression}` as a proxy for specific market drivers. "
            f"It formalizes the intuition that {theory.lower().strip('.')}.\n\n"
            f"{math_section}\n\n"
            f"{window_analysis}\n\n"
            f"**Regime Adaptability:** {regime_context}\n\n"
            f"{complexity_note}"
            f"{perf_synthesis}"
        )
        return narrative

    def describe_compact(self, expression: str, regime: str = "NEUTRAL") -> str:
        """Generate a one-paragraph compact theory description (for JSON responses)."""
        ops = list(dict.fromkeys(re.findall(r'[A-Z_]+(?=\()', expression)))
        vars_used = list(dict.fromkeys(re.findall(r'\$\w+', expression)))
        windows = [int(w) for w in re.findall(r',\s*(\d+)\)', expression)]

        # Find matching theory
        theory = None
        for pattern, desc in self.COMPOSITION_THEORIES.items():
            if all(p in ops for p in pattern):
                theory = desc
                break
        if not theory:
            if "DELTA" in ops:
                theory = "momentum persistence (Jegadeesh & Titman 1993)"
            elif "TS_CORR" in ops:
                theory = "informed trading microstructure (Kyle 1985)"
            elif "TS_STD" in ops:
                theory = "volatility-adjusted risk premium"
            else:
                theory = "cross-sectional factor pricing"

        op_chain = " → ".join(ops[:4])
        var_str = ", ".join(vars_used[:3])
        w_str = f"lookback {min(windows)}-{max(windows)} bars" if windows else "cross-sectional"

        return (
            f"Based on {theory}. "
            f"The [{op_chain}] operator chain applied to [{var_str}] ({w_str}) "
            f"decomposes the signal into: "
            + "; ".join(
                f"{op} = {self.OPERATOR_SEMANTICS[op][1]}" 
                for op in ops[:3] if op in self.OPERATOR_SEMANTICS
            )
            + f". Under {regime} regime, "
            f"{'momentum signals dominate' if 'DELTA' in ops else 'mean-reversion signals dominate' if 'TS_MEAN' in ops else 'microstructure signals dominate'}."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  SMART COMPLEXITY CONTROL: Intelligent formula simplification
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexityController:
    """Controls formula complexity to prevent over-engineering.

    Rather than blindly truncating long formulas, this controller:
    1. Measures structural complexity (depth, operator count, redundancy)
    2. Identifies which parts of the formula carry signal vs noise
    3. Prunes intelligently — keeping the strongest signal branches
    """

    MAX_DEPTH = 6
    MAX_LENGTH = 220
    MAX_OPERATORS = 8

    def __init__(self):
        self.logic_validator = MathematicalLogicValidator()

    def should_simplify(self, expression: str) -> Tuple[bool, str]:
        """Check if a formula needs simplification and why."""
        depth = expression.count('(')
        ops = re.findall(r'[A-Z_]+(?=\()', expression)
        length = len(expression)

        if length > self.MAX_LENGTH:
            return True, f"too_long ({length} > {self.MAX_LENGTH})"
        if depth > self.MAX_DEPTH:
            return True, f"too_deep ({depth} > {self.MAX_DEPTH})"
        if len(ops) > self.MAX_OPERATORS:
            return True, f"too_many_ops ({len(ops)} > {self.MAX_OPERATORS})"

        # Check for redundant patterns
        for op in ['RANK', 'ABS', 'SIGN']:
            if f'{op}({op}(' in expression:
                return True, f"redundant_{op}"
        # Check for excessive smoothing (4+ nested TS_MEAN)
        if expression.count('TS_MEAN(') >= 4:
            return True, "excessive_smoothing"

        return False, "ok"

    def simplify(self, expression: str, variables: Optional[List[str]] = None,
                 windows: Optional[List[int]] = None) -> str:
        """Intelligently simplify a formula while preserving signal quality."""
        variables = variables or ["$close", "$volume", "$return"]
        windows = windows or [5, 10, 20, 40]

        needs, reason = self.should_simplify(expression)
        if not needs:
            return expression

        # Step 1: Remove redundant wrappers
        result = expression
        for op in ['RANK', 'ABS', 'SIGN']:
            while f'{op}({op}(' in result:
                result = result.replace(f'{op}({op}(', f'{op}(', 1)
                # Fix balanced parens
                if result.count('(') > result.count(')'):
                    pass
                else:
                    result += ')' * (result.count('(') - result.count(')'))

        # Step 2: If still too long, use AST pruning
        if len(result) > self.MAX_LENGTH:
            tree = _parse_alpha_expr(result)
            if tree and tree.depth() > 3:
                result = self._prune_to_depth(tree, max_depth=4, variables=variables, windows=windows)

        # Step 3: If STILL too long, extract core expression
        if len(result) > self.MAX_LENGTH:
            tree = _parse_alpha_expr(result)
            if tree and tree.children:
                # Keep richest child branch
                best = max(tree.children, key=lambda c: c.size())
                core = best.to_expr()
                if len(core) > 10 and '$' in core:
                    result = f"RANK({core})" if not core.startswith('RANK(') else core

        # Step 4: Final truncation safety (should rarely trigger)
        if len(result) > 240:
            result = result[:235] + ")"
            # Balance parens
            diff = result.count('(') - result.count(')')
            if diff > 0:
                result += ')' * diff
            elif diff < 0:
                result = '(' * abs(diff) + result

        return result

    def _prune_to_depth(self, tree: AlphaASTNode, max_depth: int,
                        variables: List[str], windows: List[int]) -> str:
        """Prune AST tree to maximum depth, replacing deep branches with simple vars."""
        if tree.depth() <= max_depth:
            return tree.to_expr()

        pruned = tree.clone()
        self._prune_recursive(pruned, 0, max_depth, variables, windows)
        return pruned.to_expr()

    def _prune_recursive(self, node: AlphaASTNode, current_depth: int,
                         max_depth: int, variables: List[str], windows: List[int]):
        """Recursively prune nodes deeper than max_depth."""
        for i, child in enumerate(node.children):
            if current_depth + 1 >= max_depth:
                # Replace this child with a simple variable
                if child.children:  # only prune non-leaf nodes
                    # Preserve the variable from the subtree if possible
                    child_vars = re.findall(r'\$\w+', child.to_expr())
                    v = child_vars[0] if child_vars else variables[i % len(variables)]
                    node.children[i] = AlphaASTNode('VAR', v)
            else:
                self._prune_recursive(child, current_depth + 1, max_depth, variables, windows)


class CreativeRecombinator:
    """
    ELITE AI CAPABILITY: Creatively synthesize new alpha formulas by recombining
    successful patterns from memory in novel ways.

    Uses InternalCompositionEngine for ALL reasoning — fully self-contained,
    NO external API calls.
    """
    def __init__(self, cloud_brain=None):
        self.tokenizer = StrategicTokenizer()
        self.logic_validator = MathematicalLogicValidator()
        self._engine = InternalCompositionEngine()
        
    def recombine_patterns(self, patterns: Dict[str, Any], 
                          operators: List[str], variables: List[str], windows: List[int],
                          prompt_spec: Optional[Dict[str, Any]] = None,
                          diversity_mode: str = "balanced",
                          regime: str = "NEUTRAL",
                          use_reasoning: bool = True,
                          single_asset_mode: bool = False) -> str:
        """
        Create new formula by mixing top patterns from memory.
        
        diversity_mode:
          - "balanced": mix operators and variables evenly
          - "aggressive": deeper nesting, more operators
          - "simple": shallow formulas with clear structure
        """
        if not patterns or not patterns["operators"]:
            # Fallback if no patterns
            return self._fallback_composition(operators, variables, windows, single_asset_mode=single_asset_mode)
        
        # Extract top patterns
        top_ops = [op for op, _ in patterns["operators"].most_common(5)]
        top_vars = [v for v, _ in patterns["variables"].most_common(3)]
        top_windows = [w for w, _ in patterns["windows"].most_common(5)]
        
        # Merge with current gene pool
        final_ops = list(dict.fromkeys(top_ops + operators))[:6]
        final_vars = list(dict.fromkeys(top_vars + variables))[:4]
        final_windows = sorted(set(top_windows + windows))[:5]
        
        if not final_windows:
            final_windows = [5, 10, 20]
        
        # Apply prompt hints if available
        if prompt_spec:
            prompt_ops = prompt_spec.get("operator_hints", [])
            prompt_vars = prompt_spec.get("variable_hints", [])
            prompt_wins = [int(w) for w in prompt_spec.get("windows", []) if isinstance(w, (int, float))]
            
            for po in prompt_ops[:3]:
                if po not in final_ops and len(final_ops) < 6:
                    final_ops.insert(0, po)
            for pv in prompt_vars[:2]:
                if pv not in final_vars and len(final_vars) < 4:
                    final_vars.insert(0, pv)
            if prompt_wins:
                final_windows = sorted(set(prompt_wins + final_windows))[:6]
        
        # Build formula based on diversity mode
        if diversity_mode == "simple":
            candidate = self._build_simple_formula(final_ops, final_vars, final_windows, patterns, single_asset_mode=single_asset_mode)
        elif diversity_mode == "aggressive":
            candidate = self._build_complex_formula(final_ops, final_vars, final_windows, patterns, single_asset_mode=single_asset_mode)
        else:
            candidate = self._build_balanced_formula(final_ops, final_vars, final_windows, patterns, single_asset_mode=single_asset_mode)
        
        # 🧠 DEEPSEEK-GRADE VALIDATION: Check mathematical logic
        if use_reasoning:
            is_valid, reasoning, quality_score = self.logic_validator.validate_logic(
                candidate, regime, final_ops
            )
            
            # If quality is low, use internal composition engine for refinement
            if quality_score < 0.60:
                logger.info(f"[REASONING] Quality {quality_score:.2f} < 0.60, invoking Internal Engine for refinement...")
                refined = self._internal_refine(
                    candidate,
                    patterns,
                    regime,
                    final_ops,
                    final_vars,
                    final_windows,
                    single_asset_mode=single_asset_mode,
                )
                if refined and refined != candidate:
                    # Re-validate refined formula
                    is_valid_refined, reasoning_refined, quality_refined = self.logic_validator.validate_logic(
                        refined, regime, final_ops
                    )
                    if quality_refined > quality_score:
                        logger.info(f"[REASONING] Internal refinement improved quality: {quality_score:.2f} → {quality_refined:.2f}")
                        return refined
            
            if not is_valid:
                logger.warning(f"[LOGIC-FAIL] Formula rejected: {reasoning}")
                # Fall back to simpler, safer formula
                return self._build_simple_formula(final_ops, final_vars, final_windows, patterns, single_asset_mode=single_asset_mode)
        
        return candidate
    
    def _internal_refine(self, candidate: str, patterns: Dict[str, Any], regime: str,
                        operators: List[str], variables: List[str], windows: List[int],
                        single_asset_mode: bool = False) -> Optional[str]:
        """Use InternalCompositionEngine to refine low-quality formula — fully self-contained, NO external API."""
        try:
            refined = self._engine.refine(
                candidate,
                patterns,
                regime,
                operators,
                variables,
                windows,
                generation=0,
                single_asset_mode=single_asset_mode,
            )
            if refined and refined != candidate and '(' in refined and '$' in refined:
                return refined
        except Exception as e:
            logger.error(f"[INTERNAL-REFINE] Failed: {e}")
        return None
    
    def _build_simple_formula(self, ops: List[str], vars: List[str], windows: List[int], 
                             patterns: Dict[str, Any],
                             single_asset_mode: bool = False) -> str:
        """Build shallow, interpretable formula."""
        v = vars[0] if vars else "$close"
        w = windows[0] if windows else 10
        
        # Look for successful simple structures
        simple_structures = [s for s in patterns.get("structures", []) if s["depth"] <= 3]
        if simple_structures:
            # Mimic structure but with new operators/variables
            template = simple_structures[0]["expression"]
            # Simple substitution
            for old_v in ["$close", "$open", "$high", "$low", "$volume"]:
                if old_v in template and old_v != v:
                    template = template.replace(old_v, v, 1)
                    break
            return _adaptive_wrap_expression(
                template,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, int(w)),
            )
        
        # Default simple formula
        if "DELTA" in ops:
            return f"DELTA({v}, {w})"
        elif "TS_MEAN" in ops:
            return f"{v} / (TS_MEAN({v}, {w}) + 1e-8) - 1"
        else:
            return _adaptive_wrap_expression(
                v,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, int(w)),
            )
    
    def _build_balanced_formula(self, ops: List[str], vars: List[str], windows: List[int],
                               patterns: Dict[str, Any],
                               single_asset_mode: bool = False) -> str:
        """Build medium complexity formula using operator chains from memory."""
        chains = patterns.get("chains", Counter())
        top_chain = chains.most_common(1)
        
        v1 = vars[0] if vars else "$close"
        v2 = vars[1] if len(vars) > 1 else "$volume"
        w1 = windows[0] if windows else 10
        w2 = windows[1] if len(windows) > 1 else 20
        
        # Use discovered operator chain if available
        if top_chain:
            chain_str = top_chain[0][0]
            if "→" in chain_str:
                op1, op2 = chain_str.split("→")
                # Build formula using this chain
                if op1 in ["DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN"]:
                    inner = f"{op1}({v1}, {w1})"
                    if op2 in ["RANK", "SIGN", "ABS"]:
                        return f"{op2}({inner})"
                    elif op2 in ["TS_MEAN", "TS_STD"]:
                        return f"{op2}({inner}, {w2})"
        
        # Default balanced formula
        if "TS_CORR" in ops and len(vars) >= 2:
            return f"TS_CORR({v1}, {v2}, {w1})"
        elif "TS_STD" in ops and "DELTA" in ops:
            return f"DELTA({v1}, {w1}) / (TS_STD({v1}, {w2}) + 1e-8)"
        elif "TS_MEAN" in ops:
            return f"({v1} - TS_MEAN({v1}, {w1})) / (TS_STD({v1}, {w1}) + 1e-8)"
        else:
            return _adaptive_wrap_expression(
                f"DELTA({v1}, {w1})",
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, int(w2)),
            )
    
    def _build_complex_formula(self, ops: List[str], vars: List[str], windows: List[int],
                              patterns: Dict[str, Any],
                              single_asset_mode: bool = False) -> str:
        """Build deep, sophisticated formula by combining multiple patterns."""
        complex_structures = [s for s in patterns.get("structures", []) 
                             if s["depth"] >= 4 and s["fitness"] > 0.5]
        
        if complex_structures:
            # Take inspiration from most complex successful formula
            base = complex_structures[0]["expression"]
            # Could do sophisticated AST-based recombination here
            # For now, return the successful complex pattern
            return _adaptive_wrap_expression(
                base,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, int((windows + [20])[0])),
            )
        
        # Build complex composition
        v1 = vars[0] if vars else "$close"
        v2 = vars[1] if len(vars) > 1 else "$volume"
        v3 = vars[2] if len(vars) > 2 else "$high"
        w1, w2, w3 = (windows + [5, 10, 20])[:3]
        
        # Multi-layer formula
        if "TS_CORR" in ops and "TS_STD" in ops and "DELTA" in ops:
            term1 = f"TS_CORR(DELTA({v1}, {w1}), {v2}, {w2})"
            term2 = f"(({v1} - TS_MEAN({v1}, {w3})) / (TS_STD({v1}, {w3}) + 1e-8))"
            return f"({term1} * {term2})"
        elif "TS_SKEW" in ops and "TS_STD" in ops:
            return f"(TS_SKEW({v1}, {w1}) / (TS_STD({v1}, {w2}) + 1e-8)) * (DELTA({v1}, {w1}) / ({v1} + 1e-8))"
        else:
            # Nested transformation
            return _adaptive_wrap_expression(
                f"(DELTA({v1}, {w1}) / (TS_STD({v1}, {w2}) + 1e-8)) * ({v2} / (TS_MEAN({v2}, {w3}) + 1e-8))",
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, int(w3)),
            )
    
    def _fallback_composition(self, operators: List[str], variables: List[str], 
                             windows: List[int],
                             single_asset_mode: bool = False) -> str:
        """Fallback if no patterns available."""
        v = variables[0] if variables else "$close"
        w = windows[0] if windows else 10
        op = operators[0] if operators else "DELTA"
        expr = f"{op}({v}, {w})"
        return _adaptive_wrap_expression(
            expr,
            single_asset_mode=single_asset_mode,
            preferred_mode="RAW",
            window=max(20, int(w)),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  GPT-5.1 AXIS 1+4: CREATIVE IDEATION ENGINE
#  Generates formula ideas from deep mathematical root principles.
#  Unlike template selection, this synthesizes *novel* ideas from first principles.
# ═══════════════════════════════════════════════════════════════════════════════

class CreativeIdeationEngine:
    """
    GPT-5.1 Ideation Layer: Generates alpha ideas from mathematical foundations.

    Instead of picking from pre-defined templates, this engine reasons about
    market structure and derives formulas from quantitative theory:
      - Mean-Reversion Theory (Ornstein-Uhlenbeck, Hurst)
      - Momentum Factor (Jegadeesh-Titman, cross-sectional)
      - Entropy-Based Signals (information theory, predictive uncertainty)
      - Spectral Analysis (frequency decomposition, cycle detection)
    """

    # Mathematical foundation → formula generation strategy map
    MATH_FOUNDATIONS = {
        "MEAN_REVERSION": {
            "theory": "Prices revert to equilibrium; deviation is predictive",
            "operators": ["TS_MEAN", "TS_STD", "RANK", "DELTA"],
            "patterns": ["MEAN_REVERSION_ZSCORE", "INTRADAY_REVERSAL", "ADAPTIVE_AUTOCORRELATION"],
        },
        "MOMENTUM": {
            "theory": "Past winners continue winning; cross-sectional rank predicts",
            "operators": ["DELTA", "EMA", "RANK", "TS_RANK"],
            "patterns": ["MOMENTUM_ACCELERATION", "CROSS_SECTIONAL_MOMENTUM", "MULTI_TIMEFRAME_ALIGNMENT"],
        },
        "ENTROPY": {
            "theory": "High statistical uncertainty → reversals; low uncertainty → trend",
            "operators": ["TS_KURT", "TS_SKEW", "SIGN", "ABS"],
            "patterns": ["SKEWNESS_REVERSAL", "ENTROPY_SIGNAL", "SPECTRAL_MOMENTUM"],
        },
        "SPECTRAL": {
            "theory": "Price has cyclical components at multiple frequencies",
            "operators": ["TS_CORR", "EMA", "TS_SKEW", "RANK"],
            "patterns": ["ADAPTIVE_AUTOCORRELATION", "SPECTRAL_MOMENTUM", "REGIME_MOMENTUM_FILTER"],
        },
        "LIQUIDITY": {
            "theory": "Volume and price interaction reveals institutional activity",
            "operators": ["TS_CORR", "LOG", "RANK", "DELTA"],
            "patterns": ["VOLUME_WEIGHTED_MOMENTUM", "LIQUIDITY_IMPACT", "RANGE_BREAKOUT"],
        },
        "VOLATILITY": {
            "theory": "Volatility clustering predicts future risk-adjusted returns",
            "operators": ["TS_STD", "ABS", "RANK", "EMA"],
            "patterns": ["VOLATILITY_REGIME", "HIGH_LOW_DYNAMIC", "RANGE_BREAKOUT"],
        },
    }

    def __init__(self):
        self._idea_history: List[str] = []

    def _fill_archetype(self, pattern: str, windows: List[int]) -> str:
        """Fill a DEEP_FORMULA_ARCHETYPES template with concrete window values."""
        tmpl = DEEP_FORMULA_ARCHETYPES.get(pattern, "")
        if not tmpl:
            return ""
        w_sorted = sorted(windows)
        w1 = w_sorted[0] if len(w_sorted) > 0 else 5
        w2 = w_sorted[1] if len(w_sorted) > 1 else 20
        w3 = w_sorted[2] if len(w_sorted) > 2 else 40
        # also pick a 'w' (mid-range)
        w = w_sorted[len(w_sorted) // 2]
        result = (
            tmpl.replace("{w1}", str(w1))
                .replace("{w2}", str(w2))
                .replace("{w3}", str(w3))
                .replace("{w}", str(w))
        )
        return result

    def _calculate_structural_risk(self, expr: str) -> float:
        """Heuristic risk score (0.0=safe, 1.0=dangerous)."""
        risk = 0.0
        # Check for division without safety
        if '/' in expr and '+ 1e-' not in expr:
            risk += 0.4
        # Check for deep nesting of volatile ops
        volatile_ops = ["EXP", "LOG", "TS_SKEW", "TS_KURT"]
        for op in volatile_ops:
            if expr.count(op) > 1:
                risk += 0.2
        # Check for too many operators (overfit risk)
        n_ops = len(re.findall(r'[A-Z_]+\(', expr))
        if n_ops > 4:
            risk += 0.1 * (n_ops - 4)
        return min(1.0, risk)

    def ideate(
        self,
        dna: "MarketDNA",
        regime: str,
        round_idx: int = 1,
        single_asset_mode: bool = False,
        windows: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        ELITE IDEATION: Generates a high-precision, risk-embedded alpha formula.
        Structure: (Core Signal / Volatility Guard) * Regime Filter (Axiomatic Rule)
        """
        _windows = windows or [5, 10, 20, 40]

        # Phase 1: Select math foundation based on DNA signals
        foundation = self._select_foundation(dna, regime, single_asset_mode)
        foundation_info = self.MATH_FOUNDATIONS[foundation]

        # Phase 2: Select archetype and build Core Signal
        arch_map = FOREX_ARCHETYPES if single_asset_mode else STOCK_ARCHETYPES
        if round_idx >= 2:
            arch_map = {**arch_map, **FRACTAL_ARCHETYPES, **ELITE_ARCHETYPES, **INSTITUTIONAL_GOLDEN_SEEDS_DICT}
        
        archetype_keys = sorted(list(arch_map.keys()))
        # Deterministic selection based on regime and round to eliminate randomness
        import hashlib
        seed_str = f"{regime}_{round_idx}_{'forex' if single_asset_mode else 'stock'}"
        idx = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % len(archetype_keys)
        key = archetype_keys[idx]
        pattern = arch_map[key]

        core_formula = self._fill_archetype(pattern, _windows)
        if not core_formula:
            ops = foundation_info["operators"]
            core_formula = f"({ops[0]}($close, {_windows[0]}) - {ops[1]}($close, {_windows[-1]}))"

        # Phase 3: Embed ELITE RISK GUARD (Mandatory for Institutional Grade)
        # We normalize the core signal by trailing volatility to ensure scale invariance and risk control.
        risk_guard = "/ (TS_STD($close, 20) + 1e-8)"
        if dna.volatility > 0.05:
            # Aggressive guard for high volatility regimes
            risk_guard = "/ (TS_STD($close, 10) * 1.5 + 1e-8)"
            
        # Phase 4: Inject REGIME FILTER (The 'Ideal Rule' Gate)
        # Prevents execution during chaotic noise or regime-mismatch periods.
        regime_gate = "1.0"
        if regime == "CHAOTIC_NOISE" or dna.hurst < 0.45:
            # Mean-reversion gate or noise filter (using Qlib native syntax: A > B, or just rely on sign)
            regime_gate = "((SIGN(TS_STD(ABS(DELTA($close, 1)), 10) - TS_STD(ABS(DELTA($close, 1)), 60)) + 1.0) / 2.0)"
        elif dna.hurst > 0.55:
            # Momentum persistence gate
            regime_gate = "((SIGN(DELTA($close, 5)) + 1.0) / 2.0 + 0.1)"

        # Final Assembly: (Core / Guard) * Gate
        final_formula = f"(({core_formula}) {risk_guard}) * {regime_gate}"

        # Step 5: Adaptive normalization (not blindly rank-only).
        final_formula = _adaptive_wrap_expression(
            final_formula,
            single_asset_mode=single_asset_mode,
            preferred_mode="TS_RANK" if single_asset_mode else "RANK",
            window=max(20, int(np.median(_windows)) if _windows else 20),
        )

        # Record for session uniqueness
        self._idea_history.append(final_formula)
        if len(self._idea_history) > 50:
            self._idea_history.pop(0)

        # Phase 5: Strategic Reasoning
        theory = foundation_info["theory"]
        reasoning = (
            f"**Precision Strategy ({foundation})**: {theory}.\n\n"
            f"**Scenario Mapping**: In '{regime}' regime, Hurst={dna.hurst:.3f}. "
            f"Embedded Risk Shielding ({risk_guard}) compensates for {dna.volatility_regime}. "
            f"Regime Gate ensures Axiomatic Integrity by filtering out low-conviction noise.\n\n"
            f"**Math Rationale**: Uses {key} archetype with multi-window lookback {_windows}. "
            f"Target IC 0.06-0.08 with high stability via Monte Carlo simulation."
        )

        return {
            "foundation": foundation,
            "pattern": key,
            "formula": final_formula,
            "risk_score": self._calculate_structural_risk(final_formula),
            "reasoning": reasoning,
            "math_basis": f"{foundation}: {theory[:100]}",
        }

    def _select_foundation(self, dna: "MarketDNA", regime: str, single_asset_mode: bool) -> str:
        """Select mathematical foundation based on market DNA signals."""
        # Entropy-based signals when uncertainty is high
        if getattr(dna, "spectral_entropy", 0.0) > 0.6:
            return "ENTROPY"
        # Momentum when Hurst > 0.6 (strong persistence)
        if getattr(dna, "hurst", 0.5) > 0.60:
            return "MOMENTUM"
        # Mean-Reversion when Hurst < 0.4
        if getattr(dna, "hurst", 0.5) < 0.40:
            return "MEAN_REVERSION"
        # Volatility signals for volatile regimes
        if getattr(dna, "volatility", 0.0) > 0.02:
            return "VOLATILITY"
        # Liquidity analysis (default for cross-sectional stock)
        if not single_asset_mode:
            return "LIQUIDITY"
        # Spectral for single-asset forex
        return "SPECTRAL"


# ═══════════════════════════════════════════════════════════════════════════════
#  GPT-5.1 AXIS 11: QUANTA BRANCHER
#  Splits a single hypothesis into multiple distinct mathematical "limbs".
# ═══════════════════════════════════════════════════════════════════════════════

class QuantaBrancher:
    """
    Implements the 'Quanta' part of the fusion: 1 Root -> N Limbs.
    Each limb represents a unique mathematical interpretation of the core theory.
    """
    def __init__(self):
        self._composer = DeepFormulaComposer()

    def branch(self, theory: str, dna: "MarketDNA", regime: str, 
               operators: List[str], variables: List[str], windows: List[int],
               n_limbs: int = 4, single_asset_mode: bool = False) -> List[Dict[str, Any]]:
        """Splits an idea root into multiple branches for the 'Miner' stage."""
        limbs = []
        
        # Branch 1: Momentum Limb (High conviction/persistence)
        limbs.append({
            "limb_name": "Momentum_Limb",
            "theory": f"{theory} | Interpreted via trend persistence.",
            "formula": self._composer.compose(2, dna, regime, operators, variables, windows, single_asset_mode)
        })
        
        # Branch 2: Mean-Reversion Limb (Counter-trend/Mean-correction)
        limbs.append({
            "limb_name": "MeanReversion_Limb",
            "theory": f"{theory} | Interpreted via statistical mean-reversion.",
            "formula": self._composer.compose(3, dna, regime, operators, variables, windows, single_asset_mode)
        })
        
        # Branch 3: Multi-Horizon Limb (Fractal/Noise-resistant)
        limbs.append({
            "limb_name": "Fractal_Limb",
            "theory": f"{theory} | Interpreted via multi-timeframe fractal logic.",
            "formula": self._composer.compose(4, dna, regime, operators, variables, windows, single_asset_mode)
        })
        
        # Branch 4: Volatility/Liquidity Limb (Microstructure-aware)
        limbs.append({
            "limb_name": "Microstructure_Limb",
            "theory": f"{theory} | Interpreted via price-volume interaction physics.",
            "formula": self._composer.compose(5, dna, regime, operators, variables, windows, single_asset_mode)
        })
        
        return limbs[:n_limbs]


# ═══════════════════════════════════════════════════════════════════════════════
#  GPT-5.1 AXIS 1: DEEP FORMULA COMPOSER
#  Round-adaptive tree-structured formula synthesis.
#  Complexity scales with the evolution round (Round 1 = simple, Round 5 = deep).
# ═══════════════════════════════════════════════════════════════════════════════

class DeepFormulaComposer:
    """
    Composes formulas with tree-structured nesting, not just linear concatenation.
    Complexity is calibrated to the round index:
      Round 1: 1-2 operators (seeds, clean, interpretable)
      Round 2-3: 2-3 operators (multi-layer)
      Round 4-5: 3-4 operators (deep, spectral, cross-variable)
    """

    COMPLEXITY_MAP = {
        1: {"depth": 1, "n_operators": 2, "use_archetype": True},  # Raised from False → True to enable elite ideation in Round 1
        2: {"depth": 2, "n_operators": 3, "use_archetype": True},
        3: {"depth": 2, "n_operators": 3, "use_archetype": True},
        4: {"depth": 3, "n_operators": 4, "use_archetype": True},
        5: {"depth": 3, "n_operators": 5, "use_archetype": True},
    }

    def __init__(self):
        self._ideation = CreativeIdeationEngine()

    def compose(
        self,
        round_idx: int,
        dna: "MarketDNA",
        regime: str,
        operators: List[str],
        variables: List[str],
        windows: List[int],
        single_asset_mode: bool = False,
    ) -> str:
        """Compose a round-appropriate formula using tree structure."""
        config = self.COMPLEXITY_MAP.get(round_idx, self.COMPLEXITY_MAP[5])

        # For rounds 2+, use creative ideation engine to generate deep archetypes
        if config["use_archetype"]:
            idea = self._ideation.ideate(
                dna=dna, regime=regime, round_idx=round_idx,
                single_asset_mode=single_asset_mode, windows=windows,
            )
            if idea.get("formula"):
                return idea["formula"]

        # Fallback: manual tree composition
        return self._build_tree_formula(
            depth=config["depth"],
            ops=operators,
            vars=variables,
            windows=windows,
            single_asset_mode=single_asset_mode,
        )

    def _build_tree_formula(
        self, depth: int, ops: List[str], vars: List[str], windows: List[int],
        single_asset_mode: bool = False
    ) -> str:
        """Build a nested formula tree of given depth."""
        vars_ = vars if vars else ["$close", "$volume", "$open"]
        ops_ = ops if ops else (["DELTA", "TS_MEAN", "TS_STD"] if single_asset_mode else ["DELTA", "TS_MEAN", "TS_RANK"])
        wins_ = sorted(windows) if windows else [5, 10, 20]

        v1, v2 = vars_[0], vars_[min(1, len(vars_)-1)]
        w1 = wins_[0]
        w2 = wins_[min(1, len(wins_)-1)]
        norm = lambda x: _adaptive_wrap_expression(
            x,
            single_asset_mode=single_asset_mode,
            preferred_mode="TS_RANK" if single_asset_mode else "RANK",
            window=max(20, w2),
        )

        if depth == 1:
            return norm(f"DELTA({v1}, {w1})")

        if depth == 2:
            if "TS_CORR" in ops_:
                return norm(f"TS_CORR({v1}, {v2}, {w1})")
            elif "EMA" in ops_ and "DELTA" in ops_:
                return norm(f"EMA(DELTA({v1}, {w1}), {w2})")
            else:
                return norm(f"({v1} - TS_MEAN({v1}, {w1})) / (TS_STD({v1}, {w1}) + 1e-8)")

        # depth == 3
        w3 = wins_[min(2, len(wins_)-1)]
        if "TS_CORR" in ops_ and len(vars_) >= 2:
            inner1 = f"EMA(DELTA({v1}, {w1}), {w2})"
            inner2 = _adaptive_wrap_expression(
                "$volume",
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, w2),
            )
            return norm(f"TS_CORR({inner1}, {inner2}, {w3})")
        
        if depth == 4:
            v3 = vars_[min(2, len(vars_)-1)]
            w4 = wins_[min(3, len(wins_)-1)]
            core = f"TS_RANK(TS_CORR({v1}, {v2}, {w1}), {w2}) * SIGN(DELTA({v3}, {w3}))"
            return norm(f"TS_MEAN({core}, {w4})")

        if depth >= 5:
            v3 = vars_[min(2, len(vars_)-1)]
            w4 = wins_[min(3, len(wins_)-1)]
            w5 = wins_[min(4, len(wins_)-1)] if len(wins_) > 4 else w4 + 10
            # Example of long, complex institutional formula
            inner_momentum = f"(TS_RANK(DELTA({v1}, {w1}), {w3}) - TS_RANK(DELTA({v1}, {w2}), {w4}))"
            vol_guard = f"(TS_STD({v1}, {w1}) / (TS_STD({v1}, {w4}) + 1e-8))"
            flow_filter = f"SIGN(TS_CORR({v2}, $volume, {w2}))"
            return norm(f"TS_MEAN(({inner_momentum} / ({vol_guard} + 1e-8)) * {flow_filter}, {w5})")

        return norm(f"EMA(DELTA({v1}, {w1}), {w2}) * SIGN(TS_MEAN(DELTA({v2}, {w1}), {w3}))")


class HypothesisGenerator:
    def __init__(self):
        self.strategy_map = REGIME_STRATEGY_MAP
        self.hypothesis_history: deque = deque(maxlen=200)
        self.successful_hypotheses: List[Dict[str, Any]] = []
        self.failed_hypotheses: List[Dict[str, Any]] = []
        self._reasoning_mem = ReasoningMemory()
        # GPT-5.1: Deep composition engines
        self._ideation = CreativeIdeationEngine()
        self._deep_composer = DeepFormulaComposer()

    def generate(self, regime: str, dna: MarketDNA,
                 recall: List[MemoryImpression],
                 feedback: Optional[Dict[str, Any]] = None,
                 single_asset_mode: bool = False,
                 engine=None,
                 prompt_spec: Optional[Dict[str, Any]] = None,
                 council_context: Optional[Any] = None) -> Dict[str, Any]:
        """
        Synthesize market DNA and memory into a testable hypothesis.
        Axiom and gene selection are driven by learned Thompson Sampling weights,
        not random.choice — every hypothesis builds on past success/failure.

        council_context : CouncilContext (from sova_council_bridge)
            When provided, the Council's verdict and Prism denoised signals are
            fused into hypothesis direction — operator selection, theme override,
            window bias, normalization mode, and axiom shift are all steered by
            the Council's assessment of the PRIOR alpha's failure/success.
        """
        # ── Layer 4 → Layer 2: Apply CouncilContext if provided ───────────
        # The bridge translates CouncilVerdict + PrismData into concrete
        # prompt_spec enrichments and feedback signals for gene selection.
        if council_context is not None:
            try:
                from sova_council_bridge import (
                    apply_council_context_to_prompt_spec,
                    apply_council_context_to_feedback,
                )
                prompt_spec = apply_council_context_to_prompt_spec(
                    prompt_spec or {}, council_context
                )
                feedback = apply_council_context_to_feedback(
                    feedback or {}, council_context
                )
                logger.info(
                    f"[Council→Sova] Context applied: verdict={council_context.verdict_classification} "
                    f"prism={council_context.prism_signal} "
                    f"ops+={council_context.operator_nudges[:3]} "
                    f"axiom_shift={council_context.axiom_shift}"
                )
            except Exception as _ce:
                logger.debug(f"[Council→Sova] Context apply skipped: {_ce}")
        if engine:
            strategy_map = engine.get_strategy_map()
            strategy_config = strategy_map.get(regime, strategy_map.get("NEUTRAL", strategy_map.get("CHAOTIC_NOISE")))
        else:
            if single_asset_mode:
                strategy_config = SINGLE_ASSET_STRATEGY_MAP.get("GENERIC_SINGLE")
                if regime in ["NEWS_SHOCK_VOLATILITY", "LIQUIDITY_GAP_RISK"]:
                    strategy_config = SINGLE_ASSET_STRATEGY_MAP.get("XAU_SPECIFICS")
            else:
                strategy_config = self.strategy_map.get(regime, self.strategy_map["NEUTRAL"])

        primary_theme = strategy_config["primary"]
        secondary_theme = strategy_config["secondary"]

        theme_hints = (prompt_spec or {}).get("theme_hints", [])
        if theme_hints:
            primary_theme = theme_hints[0]
            if len(theme_hints) > 1 and theme_hints[1] != primary_theme:
                secondary_theme = theme_hints[1]

        # Steer secondary_theme based on dominant trade errors this session
        # Each error type maps to the corrective analytical lens needed
        _ERROR_THEME_MAP = {
            "WRONG_DIRECTION":          "REGIME_TRANSITION",
            "TREATING_NOISE_AS_SIGNAL": "MEAN_REVERSION",
            "HELD_TOO_LONG":            "RISK_MANAGEMENT",
            "CHASING_TREND_LATE":       "RISK_MANAGEMENT",
            "PREMATURE_EXIT":           "MOMENTUM_FLOW",
            "STOP_TOO_TIGHT":           "RISK_MANAGEMENT",
            "OVERSIZED_POSITION":       "RISK_MANAGEMENT",
            "REGIME_MISMATCH":          "REGIME_TRANSITION",
        }
        if feedback:
            for err in feedback.get("dominant_errors", []):
                override = _ERROR_THEME_MAP.get(err)
                if override and override != primary_theme:
                    secondary_theme = override
                    break  # Apply the most critical corrective lens


        reasoning_chain = [
            f"[SENSE] Market DNA: {regime} | Vol={dna.volatility:.3f} Hurst={dna.hurst:.3f} FD={dna.fractal_dimension:.3f}",
            f"[CONTEXT] Regime confidence={dna.regime_confidence:.2f}, Transition prob={dna.regime_transition_probability:.2f}",
            f"[STRATEGY] Primary: {primary_theme} | Secondary: {secondary_theme}",
            f"[MEMORY] Retrieved {len(recall)} historical imprints for cross-reference",
        ]

        if prompt_spec:
            reasoning_chain.append(
                f"[PROMPT-MATH] families={prompt_spec.get('families', [])[:4]} | "
                f"operators={prompt_spec.get('operator_hints', [])[:5]} | "
                f"variables={prompt_spec.get('variable_hints', [])[:4]} | "
                f"norm={prompt_spec.get('normalization_modes', [])[:3]}"
            )

        if feedback:
            feedback_intent = feedback.get("suggested_intent", "")
            if "SIMPLIFY" in feedback_intent:
                reasoning_chain.append("[FEEDBACK] Previous attempt over-complex. Reducing structural depth.")
            elif "DIVERSIFY" in feedback_intent:
                reasoning_chain.append("[FEEDBACK] Previous attempt redundant. Shifting search space.")
            elif "EXPLOIT" in feedback_intent:
                reasoning_chain.append("[FEEDBACK] Previous attempt promising. Refining parameters.")

            # ── Council/Prism brief injected into chain ───────────────────
            if feedback.get("council_brief"):
                reasoning_chain.append(
                    "[COUNCIL] " + str(feedback["council_brief"])[:600]
                )
            if feedback.get("dominant_errors"):
                reasoning_chain.append(
                    "[COUNCIL-ERRORS] " + ", ".join(feedback["dominant_errors"][:5])
                )
            if feedback.get("prism_signal"):
                reasoning_chain.append(
                    f"[PRISM] Signal={feedback['prism_signal']} "
                    f"trend={feedback.get('prism_clean_trend', 0.0):+.2f}"
                )

        # ── Council theme override (wins over regime default) ─────────────
        if council_context is not None and getattr(council_context, "theme_override", ""):
            primary_theme = council_context.theme_override
            reasoning_chain.append(
                f"[COUNCIL-THEME] Theme overridden to: {primary_theme} "
                f"(verdict={council_context.verdict_classification})"
            )
        if council_context is not None and getattr(council_context, "axiom_shift", False):
            reasoning_chain.append(
                "[COUNCIL-AXIOM] Paradigm shift detected — exploring fundamentally new formula families"
            )

        # ── IC Target Driven Reasoning ────────────────────────────────────
        # Drive structural depth based on maturity of the search.
        # Early rounds (generation < 2) target IC 0.04 (parsmonious).
        # Late rounds target IC 0.08+ (complex interaction).
        generation_idx = (prompt_spec or {}).get("generation", 0)
        target_ic = 0.04 + (min(generation_idx, 5) * 0.016)  # Caps at ~0.12
        reasoning_chain.append(f"[GOAL] Structural target: IC ≥ {target_ic:.3f}")

        complexity_target = self._estimate_complexity(dna, strategy_config)
        if target_ic > 0.08:
            complexity_target = int(complexity_target * 1.25)
            reasoning_chain.append("[ADJUST] Target IC high — increasing structural interaction depth.")

        # ── GPT-5.1: Contextual Bandit Market Split ───────────────────────
        market_prefix = "FOREX:" if single_asset_mode else "STOCK:"
        primary_operators = strategy_config["operators"]
        windows = strategy_config["windows"]

        gene_selection = self._select_genes(dna, recall, primary_operators, windows,
                            single_asset_mode=single_asset_mode, engine=engine,
                            prompt_spec=prompt_spec)

        reasoning_chain.append(f"[SYNTHESIS] Target complexity={complexity_target}, Genes={gene_selection['operators']}")

        # Log learned weights for transparency
        probs = self._reasoning_mem.get_sampler(
            f"{market_prefix}{regime}:ops", gene_selection["operators"]
        ).get_probabilities()
        reasoning_chain.append(f"[LEARNED WEIGHTS] Op probs={{{', '.join(f'{k}:{v:.2f}' for k,v in list(probs.items())[:4])}}}")

        # ── 🧠 REAL LLM REASONING: Replace template axioms with actual AI thought ──
        # Instead of printing hard-coded "Entropy peaks mark the boundary..." strings,
        # the LLM READS the MarketDNA metrics and REASONS about why this regime exists
        # and which mathematical operators are most appropriate.
        _llm_hypothesis_text = ""
        try:
            from sova_cloud_brain import _first_n_sentences, _resolve_explanation_mode
            _local_dna_summary = "\n".join([
                f"- Regime: {regime} | Confidence: {dna.regime_confidence:.2f}",
                f"- Volatility: {dna.volatility:.3f} ({dna.volatility_regime})",
                f"- Hurst: {dna.hurst:.3f}  (>0.5=persistent, <0.5=mean-reverting)",
                f"- FractalDim: {dna.fractal_dimension:.3f} | Entropy: {dna.spectral_entropy:.3f}",
                f"- Trend: {dna.trend:.4f} ({dna.trend_regime}) | VolSurge: {dna.volume_surge:.2f}x",
                f"- MomentumPersistence: {dna.momentum_persistence:.3f} | MeanRevStrength: {dna.mean_reversion_strength:.3f}",
            ])
            _recall_summary = (
                "Past successes in this regime:\n"
                + "\n".join(f"  - {r.Expression[:70]}" for r in recall[:3])
            ) if recall else ""

            _local_hypothesis = {
                "regime": regime,
                "primary_theme": primary_theme,
                "gene_selection": gene_selection,
                "reasoning_chain": reasoning_chain,
                "is_forex": bool(single_asset_mode),
            }
            _explanation_mode = _resolve_explanation_mode()
            _is_concise = _explanation_mode == "concise"

            # Local-first unified reasoning path (cloud is optional enhancer).
            from sova_cloud_brain import HybridReasoningMatrix
            _local_brain = HybridReasoningMatrix()
            _llm_hypothesis_text = _local_brain.refine_hypothesis(
                local_hypothesis=_local_hypothesis,
                dna_summary=_local_dna_summary,
                recall_summary=_recall_summary,
            )
            if _llm_hypothesis_text:
                _llm_reason_for_chain = _llm_hypothesis_text.strip()
                if _is_concise:
                    _llm_reason_for_chain = _first_n_sentences(_llm_reason_for_chain, n=2)
                reasoning_chain.append(f"[LLM-REASONING] {_llm_reason_for_chain[:500]}")
                logger.info(f"[HypothesisGen] unified reasoning injected for {regime}")
            else:
                _fallback_reason = _local_brain.generate_mathematical_rationale(
                    dna_summary=_local_dna_summary,
                    regime=regime,
                    theme=primary_theme,
                    is_forex=bool(single_asset_mode),
                )
                if _is_concise:
                    _fallback_reason = _first_n_sentences(_fallback_reason, n=2)
                reasoning_chain.append(f"[MATH-RATIONALE] {_fallback_reason[:500]}")
        except Exception as _e:
            # Fall back to data-derived mathematical rationale
            try:
                from sova_cloud_brain import HybridReasoningMatrix
                _fallback_rationale = HybridReasoningMatrix().generate_mathematical_rationale(_local_dna_summary, regime, primary_theme)
                try:
                    from sova_cloud_brain import _first_n_sentences, _resolve_explanation_mode
                    if _resolve_explanation_mode() == "concise":
                        _fallback_rationale = _first_n_sentences(_fallback_rationale, n=2)
                except Exception:
                    pass
                reasoning_chain.append(f"[MATH-RATIONALE] {_fallback_rationale}")
            except:
                reasoning_chain.append("[MATH-RATIONALE] Metrics-based synthesis fallback activated.")
            logger.debug(f"[HypothesisGen] LLM reasoning skipped: {_e}")

        # ── NARRATIVE LAYER: Sova speaks fluently about its decision ────────────
        # This runs after the math layer. Same data, different language register.
        _sova_narrative = ""
        try:
            from sova_cloud_brain import SovaNarrativeEngine
            _recall_exprs = [imp.Expression for imp in recall[:2]] if recall else []
            _ops_chosen = gene_selection.get("operators", [])[:5]
            _sova_narrative = SovaNarrativeEngine().narrate(
                regime=regime,
                primary_theme=primary_theme,
                dna_summary=_local_dna_summary,
                operators_chosen=_ops_chosen,
                recall_expressions=_recall_exprs,
            )
            if _sova_narrative:
                reasoning_chain.append(f"[NARRATIVE] {_sova_narrative}")
        except Exception as _ne:
            logger.debug(f"[Narrative] Skipped: {_ne}")

        hypothesis = {
            "regime": regime,
            "primary_theme": primary_theme,
            "secondary_theme": secondary_theme,
            "reasoning_chain": reasoning_chain,
            "narrative": _sova_narrative,
            "complexity_target": complexity_target,
            "gene_selection": gene_selection,
            "aggression": strategy_config["aggression"],
            "description": strategy_config["description"],
            "seed_tokens": self._extract_seed_tokens(recall),
            "timestamp": datetime.now().isoformat(),
            "target_ic": target_ic
        }

        # ── Phase 2: Self-Refinement Loop (Axiom-Level) ──────────────────
        # In a real Sova session, we would call Evolver.mutate_surgical here.
        # For now, we tag the hypothesis with 'needs_refinement' if critique fails.
        critique = self._self_critique(hypothesis)
        if critique["failed"]:
             hypothesis["reasoning_chain"].append(f"[CRITIQUE] Refinement note: {critique['reason']}")
             hypothesis["logic_nudge"] = critique["reason"]

        self.hypothesis_history.append(hypothesis)
        return hypothesis

    def _self_critique(self, hyp: Dict[str, Any]) -> Dict[str, Any]:
        """Internal Sova 'Double Check' logic for hypothesis sanity."""
        regime = hyp["regime"]
        ops = hyp["gene_selection"]["operators"]
        windows = hyp["gene_selection"]["windows"]

        # 1. Check for 'window starvation' in trending regimes
        if regime in ("EXPONENTIAL_BULL", "STABLE_ACCUMULATION"):
            if windows and max(windows) < 15:
                return {"failed": True, "reason": "Lookback starvation in trend regime. Needs longer temporal context."}
        
        # 2. Check for 'noise over-sensitivity' in chaotic regimes
        if regime == "CHAOTIC_NOISE" and "RANK" in ops and "TS_MEAN" not in ops and "SMA" not in ops:
             return {"failed": True, "reason": "High noise sensitivity. Needs temporal smoothing (TS_MEAN/SMA) before ranking."}

        return {"failed": False, "reason": ""}


    def _estimate_complexity(self, dna: MarketDNA, strategy_config: Dict) -> int:
        base = 20
        if dna.volatility > 0.40:
            base -= 5
        elif dna.volatility < 0.20:
            base += 5
        if dna.regime_confidence > 0.7:
            base += 3
        else:
            base -= 3
        if dna.spectral_entropy > 2.5:
            base -= 5
        return max(10, min(45, base))

    def _select_genes(self, dna: MarketDNA, recall: List[MemoryImpression],
                     preferred_operators: List[str], windows: List[int],
                     single_asset_mode: bool = False,
                     engine=None,
                     prompt_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if engine:
            registry = engine.get_gene_registry()
        else:
            registry = FOREX_ALPHA_REGISTRY if single_asset_mode else STOCK_ALPHA_REGISTRY

        # Count which operators appeared in successful recalled memories
        recall_operators = Counter()
        for imp in recall:
            ops = re.findall(r'[A-Z_]+', imp.Expression)
            # Weight by fitness: better memories contribute more signal
            weight = max(0.1, imp.Metrics.Fitness * imp.Reinforcement)
            for op in ops:
                recall_operators[op] += weight

        prompt_spec = prompt_spec or {}
        prompt_ops = prompt_spec.get("operator_hints", [])

        # Build operator pool: preferred from strategy + prompt + additional from memory
        all_ops = list(set(preferred_operators + prompt_ops + list(recall_operators.keys())))
        all_ops = [op for op in all_ops if
                   op in registry.get("BINARY", []) or
                   op in registry.get("UNARY", []) or
                   op in registry.get("TRINARY", [])]

        # Use Thompson Sampling per (regime proxy = dna.regime) x operator
        regime_key = f"{dna.regime}:ops"
        sampler = self._reasoning_mem.get_sampler(regime_key, all_ops or preferred_operators)
        # Warm-start sampler with recall signal: operators in successful memories get a boost
        for op, weight in recall_operators.items():
            sampler.update(op, weight * 0.1)  # gentle nudge, not overwrite

        selected_operators = sampler.sample_top_k(min(5, len(sampler.arms)))
        for op in prompt_ops:
            if op in all_ops and op not in selected_operators:
                if len(selected_operators) < 5:
                    selected_operators.append(op)
                else:
                    selected_operators[-1] = op

        # Variable selection driven by DNA state — deterministic, not random
        prompt_vars = [v for v in prompt_spec.get("variable_hints", []) if v in registry.get("VAR", []) or v == "$return"]
        selected_variables = prompt_vars[:]
        if not selected_variables:
            selected_variables = ["$close"]
        if dna.volume_surge > 1.3:
            selected_variables.append("$volume")
        if dna.microstructure_efficiency < 0.8:
            selected_variables.extend(["$high", "$low"])
        if abs(dna.trend) > 0.01:
            selected_variables.append("$open")

        if single_asset_mode and "$return" not in selected_variables:
            selected_variables.append("$return")

        # Window selection: cluster around the adaptive window from DNA
        adaptive = dna.adaptive_window_size
        prompt_windows = [int(w) for w in prompt_spec.get("windows", []) if isinstance(w, int)]
        derived_prompt_windows = []
        for pw in prompt_windows[:3]:
            derived_prompt_windows.extend([max(2, pw // 2), pw, min(120, pw * 2)])
        enriched_windows = sorted(set(windows + prompt_windows + derived_prompt_windows + [max(2, adaptive // 2), adaptive, min(60, adaptive * 2)]))

        return {
            "operators": selected_operators,
            "variables": list(dict.fromkeys(selected_variables)),
            "windows": enriched_windows,
            "recall_bias": dict(recall_operators.most_common(5))
        }

    def _extract_seed_tokens(self, recall: List[MemoryImpression]) -> List[str]:
        if not recall:
            return StrategicTokenizer().tokenize("RANK(DELTA($close, 5))")
        # Use highest-fitness memory as seed (not random)
        best = max(recall, key=lambda x: x.Metrics.Fitness * x.Reinforcement)
        return StrategicTokenizer().tokenize(best.Expression)

    def record_result(self, hypothesis: Dict[str, Any], success: bool, metrics: BacktestMetrics):
        """
        Feed IC outcome back into the Thompson Sampler for each operator/axiom used
        so future hypothesis generation learns from this experience.
        """
        regime = hypothesis.get("regime", "NEUTRAL")
        gene_sel = hypothesis.get("gene_selection", {})
        reward = metrics.IC if success else -abs(metrics.IC)

        # Reinforce operators
        for op in gene_sel.get("operators", []):
            self._reasoning_mem.update(f"{regime}:ops", op, reward)

        entry = {**hypothesis, "success": success, "metrics": asdict(metrics)}
        if success:
            self.successful_hypotheses.append(entry)
            if len(self.successful_hypotheses) > 100:
                self.successful_hypotheses = self.successful_hypotheses[-80:]
        else:
            self.failed_hypotheses.append(entry)
            if len(self.failed_hypotheses) > 100:
                self.failed_hypotheses = self.failed_hypotheses[-80:]


class RecursiveEvolver:
    """
    RL-guided Alpha Evolution engine.
    All synthesis and mutation decisions are driven by Thompson Sampling:
    - Template selection: which structural template to use per (regime, theme)
    - Mutation type: which mutation operator to apply based on past IC outcomes
    - Crossover strategy: which recombination approach has historically worked
    - Window selection: which lookback periods produce strongest signal per regime
    - Internal Composition Engine: algebraic + hypothesis-driven refinement (no external API)
    """

    _MUTATION_TYPES = ["variable_swap", "window_shift", "operator_wrap", "arithmetic_inject",
                       "ast_variable_swap", "ast_window_shift", "ast_operator_replace", "ast_subtree_regrow"]
    _CROSSOVER_TYPES = ["segment_recombination", "operator_substitution", "structural_fusion", "ast_subtree_swap"]
    _THEME_TEMPLATES = {
        "MEAN_REVERSION": ["zscore", "range_position", "ma_ratio", "delta_zscore"],
        "MOMENTUM_FLOW": ["raw_delta", "vol_corr", "signed_delta", "roc"],
        "LIQUIDITY_DYNAMICS": ["price_vol_corr", "vol_cv", "signed_vol_delta", "skew"],
        "REGIME_TRANSITION": ["vol_ratio", "vol_delta", "autocorr", "kurtosis"],
        "default": ["raw_delta", "ma_diff", "std", "zscore"],
    }

    def __init__(self, cloud_brain=None):
        self.tokenizer = StrategicTokenizer()
        self.evolution_history: deque = deque(maxlen=200)
        self.successful_patterns: List[str] = []
        self.failed_patterns: List[str] = []
        self._reasoning_mem = ReasoningMemory()
        self._composition_engine = InternalCompositionEngine()
        self._last_candidate_meta: Dict[str, List[Dict[str, Any]]] = {}
        self._meta_fifo: deque = deque(maxlen=2000)
        self._recent_expr_norm: deque = deque(maxlen=200)
        # Anti-collapse memory: repeatedly failed expressions are temporarily cooled down.
        self._expr_failure_streak: Dict[str, int] = defaultdict(int)
        self._expr_cooldown_until: Dict[str, int] = {}
        self._failure_event_idx: int = 0
        # cloud_brain kept for backward compat but NEVER used for generation
        self.cloud_brain = cloud_brain

    @staticmethod
    def _normalize_expr(expr: str) -> str:
        return re.sub(r"\s+", "", str(expr or ""))

    @staticmethod
    def _parens_balanced(expr: str) -> bool:
        bal = 0
        for ch in expr:
            if ch == "(":
                bal += 1
            elif ch == ")":
                bal -= 1
                if bal < 0:
                    return False
        return bal == 0

    def _validate_candidate(
        self,
        expr: str,
        operators: List[str],
        variables: List[str],
    ) -> bool:
        expr = str(expr or "").strip()
        if not expr:
            return False

        max_len = int(os.environ.get("SOVA_MAX_EXPR_LEN", "240") or "240")
        if len(expr) > max_len:
            return False
        if not self._parens_balanced(expr):
            return False

        structure = self.tokenizer.extract_structure(expr)
        max_depth = int(os.environ.get("SOVA_MAX_EXPR_DEPTH", "18") or "18")
        if int(structure.get("depth", 0) or 0) > max_depth:
            return False
        max_complexity = int(os.environ.get("SOVA_MAX_COMPLEXITY_SCORE", "260") or "260")
        if int(structure.get("complexity_score", 0) or 0) > max_complexity:
            return False

        # Ensure operators and variables are from allowed pools.
        allowed_ops = set(operators or [])
        for cat, genes in (self.tokenizer.gene_registry or {}).items():
            if cat != "VAR":
                allowed_ops.update([g for g in genes if isinstance(g, str)])

        func_ops = [t.rstrip('(') for t in structure.get("tokens", []) if isinstance(t, str) and t.endswith('(')]
        for op in func_ops:
            if op not in allowed_ops:
                return False

        allowed_vars = set(variables or []) | {"$close", "$open", "$high", "$low", "$volume", "$return"}
        vars_in_expr = re.findall(r"\$\w+", expr)
        if not vars_in_expr:
            return False
        for v in vars_in_expr:
            if v not in allowed_vars:
                return False
        return True

    def _postprocess_candidates(
        self,
        candidates: List[Optional[str]],
        operators: List[str],
        variables: List[str],
    ) -> List[str]:
        dedupe_recent = os.environ.get("SOVA_DEDUPE_RECENT", "1").strip().lower() not in {"0", "false", "no"}
        cooldown_enabled = os.environ.get("SOVA_FAIL_COOLDOWN", "1").strip().lower() not in {"0", "false", "no"}

        seen: set = set()
        out: List[str] = []
        for cand in candidates or []:
            if not cand:
                continue
            expr = str(cand).strip()
            if not expr:
                continue
            if not self._validate_candidate(expr, operators=operators, variables=variables):
                continue
            norm = self._normalize_expr(expr)
            if norm in seen:
                continue
            if cooldown_enabled:
                until = int(self._expr_cooldown_until.get(norm, -1))
                if until >= self._failure_event_idx:
                    continue
            if dedupe_recent and norm in set(self._recent_expr_norm):
                continue
            seen.add(norm)
            out.append(expr)
            if dedupe_recent:
                self._recent_expr_norm.append(norm)
        return out

    @staticmethod
    def _op_signature(expr: str) -> set:
        return set(re.findall(r"[A-Z_]+(?=\()", str(expr or "")))

    def _generate_counterfactual_candidates(
        self,
        recall: List[MemoryImpression],
        regime: str,
        variables: List[str],
        windows: List[int],
        single_asset_mode: bool = False,
    ) -> List[str]:
        """Generate adversarial/counterfactual candidates to escape local optima."""
        if not variables:
            variables = ["$close"]
        if not windows:
            windows = [5, 20, 60]

        var = self._sample_variable(regime, variables)
        alt_vars = [v for v in variables if v != var] or variables
        var2 = self._sample_variable(regime, alt_vars)
        vol_var = "$volume" if "$volume" in variables else var2
        w_short = self._sample_window(regime, windows)
        w_long = self._sample_window(regime, [w for w in windows if w != w_short] or windows)

        preferred_wrap = "TS_RANK" if single_asset_mode else "RANK"
        family_map = {
            "mean_reversion": _adaptive_wrap_expression(
                f"({var} - TS_MEAN({var}, {w_long})) / (TS_STD({var}, {w_long}) + 1e-8)",
                single_asset_mode=single_asset_mode,
                preferred_mode=preferred_wrap,
                window=max(20, w_long),
            ),
            "momentum": _adaptive_wrap_expression(
                f"DELTA({var}, {w_short}) / (TS_STD({var}, {w_long}) + 1e-8)",
                single_asset_mode=single_asset_mode,
                preferred_mode=preferred_wrap,
                window=max(20, w_long),
            ),
            "flow": _adaptive_wrap_expression(
                f"TS_CORR(DELTA({var}, 1), {vol_var}, {w_long})",
                single_asset_mode=single_asset_mode,
                preferred_mode=preferred_wrap,
                window=max(20, w_long),
            ),
            "range": _adaptive_wrap_expression(
                f"({var} - TS_MIN({var}, {w_long})) / (TS_MAX({var}, {w_long}) - TS_MIN({var}, {w_long}) + 1e-8)",
                single_asset_mode=single_asset_mode,
                preferred_mode=preferred_wrap,
                window=max(20, w_long),
            ),
        }

        mode = self._reasoning_mem.get_sampler(
            f"{regime}:counterfactual",
            list(family_map.keys()),
        ).sample()
        primary = family_map.get(mode, family_map["mean_reversion"])

        # Add one structurally different backup candidate.
        backup = family_map["flow"] if mode != "flow" else family_map["momentum"]

        # If recall exists, prefer candidates with lower operator overlap vs top memory.
        if recall:
            top_expr = str(recall[0].Expression or "")
            top_sig = self._op_signature(top_expr)
            ranked = sorted(
                [primary, backup],
                key=lambda e: len(self._op_signature(e).intersection(top_sig)),
            )
            return ranked
        return [primary, backup]

    def _register_candidate(self, expr: Optional[str], meta: Dict[str, Any]) -> None:
        if not expr:
            return
        expr = str(expr)
        if not expr.strip():
            return
        is_new = expr not in self._last_candidate_meta
        payload = dict(meta or {})
        payload.setdefault("ts_utc", datetime.utcnow().isoformat() + "Z")
        self._last_candidate_meta.setdefault(expr, []).append(payload)
        if is_new:
            self._meta_fifo.append(expr)
            # Bound the mapping size to avoid unbounded growth.
            while len(self._last_candidate_meta) > int(self._meta_fifo.maxlen or 2000):
                oldest = self._meta_fifo.popleft()
                self._last_candidate_meta.pop(oldest, None)

    def get_candidate_meta(self, expr: str) -> List[Dict[str, Any]]:
        return list(self._last_candidate_meta.get(expr, []))

    def _reward_from_metrics(
        self,
        ic: float,
        icir: Optional[float] = None,
        ir: Optional[float] = None,
        arr: Optional[float] = None,
        mdd: Optional[float] = None,
        rank_ic: Optional[float] = None,
        success: bool = True,
    ) -> float:
        """Map metrics into a bounded reward.

        Keep it simple and monotonic:
        - Favor IC/RankIC/ICIR/ARR
        - Penalize drawdown magnitude
        - Use tanh to keep updates stable
        """
        try:
            ic_v = float(ic or 0.0)
            icir_v = float(icir or 0.0)
            ir_v = float(ir or 0.0)
            arr_v = float(arr or 0.0)
            mdd_v = float(mdd or 0.0)
            rank_ic_v = float(rank_ic or 0.0)
        except Exception:
            ic_v, icir_v, ir_v, arr_v, mdd_v, rank_ic_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Include strategy IR (information ratio) as an additional stability/quality signal.
        raw = 2.0 * ic_v + 1.2 * rank_ic_v + 0.6 * icir_v + 0.7 * ir_v + 0.4 * arr_v - 0.8 * abs(mdd_v)
        shaped = float(np.tanh(raw * 5.0))
        if success:
            return max(0.0, shaped)
        return -max(0.05, abs(shaped))

    def evolve(self, hypothesis: Dict[str, Any],
               recall: List[MemoryImpression],
               generation: int = 0,
               single_asset_mode: bool = False,
               engine=None,
               prompt_spec: Optional[Dict[str, Any]] = None) -> List[str]:
        candidates = []
        gene_selection = hypothesis["gene_selection"]
        operators = gene_selection["operators"]
        variables = gene_selection["variables"]
        windows = gene_selection["windows"]
        seed_tokens = hypothesis["seed_tokens"]
        regime = hypothesis["regime"]
        theme = hypothesis["primary_theme"]

        candidates = []
        gene_selection = hypothesis["gene_selection"]
        operators = gene_selection["operators"]
        variables = gene_selection["variables"]
        windows = gene_selection["windows"]
        seed_tokens = hypothesis["seed_tokens"]
        regime = hypothesis["regime"]
        theme = hypothesis["primary_theme"]

        # Axis 8: Institutional Golden Seeds Protocol
        # Force-inject proven mathematical foundations into the first generation
        if generation == 0 and not single_asset_mode:
            for seed in INSTITUTIONAL_GOLDEN_SEEDS:
                if seed not in candidates:
                    candidates.append(seed)
                    self._register_candidate(seed, {
                        "origin": "golden_seed",
                        "regime": regime,
                        "theme": "INSTITUTIONAL_FOUNDATION",
                    })

        # 🧠 INTERNAL AI: Memory-Guided + Hypothesis-Driven Synthesis (no external API)
        if recall and len(recall) >= 3:
            pattern_miner = MemoryPatternMiner()
            patterns = pattern_miner.mine_successful_patterns(recall, min_ic=0.02)
            
            if patterns and patterns["operators"]:
                recombinator = CreativeRecombinator()
                
                # Generate 3 candidates with different creativity levels
                diversity_modes = ["simple", "balanced", "aggressive"]
                for mode in diversity_modes:
                    memory_guided_alpha = recombinator.recombine_patterns(
                        patterns, operators, variables, windows,
                        prompt_spec=prompt_spec,
                        diversity_mode=mode,
                        regime=regime,
                        use_reasoning=True,
                        single_asset_mode=single_asset_mode,
                    )
                    if memory_guided_alpha:
                        candidates.append(memory_guided_alpha)
                        self._register_candidate(memory_guided_alpha, {
                            "origin": "memory_guided",
                            "mode": mode,
                            "regime": regime,
                            "theme": theme,
                        })
                        logger.info(f"[MEMORY-GUIDED] {mode} synthesis: {memory_guided_alpha[:80]}...")

        # 🧬 HYPOTHESIS-DRIVEN: Generate from academic financial theory + operator algebra
        try:
            hypothesis_candidates = self._composition_engine.generate_candidates(
                regime,
                operators,
                variables,
                windows,
                generation=generation,
                single_asset_mode=single_asset_mode,
                num_candidates=3,
            )
            for hc in hypothesis_candidates:
                if hc and hc not in candidates:
                    candidates.append(hc)
                    self._register_candidate(hc, {
                        "origin": "hypothesis_engine",
                        "regime": regime,
                        "theme": theme,
                    })
                    logger.info(f"[HYPOTHESIS-ENGINE] {hc[:80]}...")
        except Exception as e:
            logger.warning(f"[HYPOTHESIS-ENGINE] Failed: {e}")

        if prompt_spec:
            prompt_result = self._synthesize_from_prompt_spec(
                regime, theme, operators, variables, windows,
                prompt_spec=prompt_spec,
                single_asset_mode=single_asset_mode,
                engine=engine,
                generation=generation,
            )
            if prompt_result:
                candidates.append(prompt_result)

        # 1. Template synthesis — Thompson-sampled template per theme
        template_result = self._synthesize_from_template(
            regime, theme, operators, variables, windows, single_asset_mode, engine
        )
        candidates.append(template_result)

        # 2. Seed-based evolution
        seed_result = self._synthesize_from_seed(
            seed_tokens, theme, operators, variables, windows, single_asset_mode, engine
        )
        candidates.append(seed_result)

        # 3. Composition synthesis with learned operator weights
        composition_result = self._synthesize_from_composition(
            operators, variables, windows, regime, single_asset_mode, engine
        )
        candidates.append(composition_result)
        self._register_candidate(composition_result, {
            "origin": "composition",
            "regime": regime,
            "theme": theme,
        })

        # 4. Crossover from best memories — strategy chosen by Thompson Sampling
        if recall and len(recall) >= 2:
            parent_a = recall[0].Expression
            parent_b = recall[min(1, len(recall) - 1)].Expression
            # Select crossover strategy via sampling
            crossover_strategy = self._reasoning_mem.get_sampler(
                f"{regime}:crossover", self._CROSSOVER_TYPES
            ).sample()
            if crossover_strategy == "ast_subtree_swap":
                crossover_result = self._composition_engine.ast_crossover(parent_a, parent_b)
            else:
                crossover_result = self._crossover(parent_a, parent_b, strategy=crossover_strategy)
            if crossover_result:
                candidates.append(crossover_result)
                self._register_candidate(crossover_result, {
                    "origin": "crossover",
                    "regime": regime,
                    "theme": theme,
                    "crossover_strategy": crossover_strategy,
                    "parents": [parent_a, parent_b],
                })

        # 5. Mutation — type chosen via Thompson Sampling per regime
        if recall:
            # Prefer highest-fitness parent for mutation seed
            best_parent = max(recall, key=lambda x: x.Metrics.Fitness * x.Reinforcement)
            mutation_type = self._reasoning_mem.get_sampler(
                f"{regime}:mutation", self._MUTATION_TYPES
            ).sample()
            if mutation_type.startswith("ast_"):
                mutated = self._composition_engine.ast_mutation(
                    best_parent.Expression, operators, variables, windows,
                    regime=regime, generation=generation
                )
            else:
                mutated = self._mutate(
                    best_parent.Expression, operators, variables, windows,
                    mutation_type=mutation_type,
                    single_asset_mode=single_asset_mode, engine=engine
                )
            if mutated and mutated != best_parent.Expression:
                candidates.append(mutated)
                self._register_candidate(mutated, {
                    "origin": "mutation",
                    "regime": regime,
                    "theme": theme,
                    "mutation_type": mutation_type,
                    "parent": best_parent.Expression,
                })

        # 6. Counterfactual synthesis: intentionally explore orthogonal structures.
        if recall:
            for cf in self._generate_counterfactual_candidates(
                recall=recall,
                regime=regime,
                variables=variables,
                windows=windows,
                single_asset_mode=single_asset_mode,
            ):
                if cf:
                    candidates.append(cf)
                    self._register_candidate(cf, {
                        "origin": "counterfactual",
                        "regime": regime,
                        "theme": theme,
                    })

        self.evolution_history.append({
            "generation": generation,
            "regime": regime,
            "candidates": candidates,
            "timestamp": datetime.now().isoformat()
        })

        # Final filter: keep only valid, non-duplicate, bounded-complexity expressions.
        candidates = self._postprocess_candidates(candidates, operators=operators, variables=variables)
        self.evolution_history[-1]["candidates"] = candidates
        return candidates

    def _synthesize_from_prompt_spec(self, regime: str, theme: str,
                                     operators: List[str], variables: List[str],
                                     windows: List[int],
                                     prompt_spec: Optional[Dict[str, Any]] = None,
                                     single_asset_mode: bool = False,
                                     engine=None,
                                     generation: int = 0) -> Optional[str]:
        prompt_spec = prompt_spec or {}
        families = prompt_spec.get("families", [])
        if not families:
            return None

        # DIVERSITY: Rotate variables based on generation index
        hinted_vars = [v for v in prompt_spec.get("variable_hints", []) if v in variables or v == "$return"]
        var_pool = list(dict.fromkeys(hinted_vars + variables + ["$close", "$volume", "$return", "$high", "$low"]))
        var_pool = [v for v in var_pool if v in variables or v in ("$close", "$volume", "$return", "$high", "$low", "$open")]
        var1 = var_pool[generation % len(var_pool)]
        var2 = var_pool[(generation + 1) % len(var_pool)]
        if var2 == var1:
            var2 = var_pool[(generation + 2) % len(var_pool)] if len(var_pool) > 2 else "$volume"

        # DIVERSITY: Rotate window sets per generation for different timeframes
        prompt_windows = [int(w) for w in prompt_spec.get("windows", []) if isinstance(w, int)]
        window_pool = prompt_windows or windows or [5, 10, 20]
        window_sets = [
            (max(2, min(window_pool)), max(window_pool), window_pool[min(len(window_pool) - 1, 1)] if len(window_pool) > 1 else max(min(window_pool) + 3, min(20, max(window_pool)))),
            (3, 20, 10), (5, 40, 15), (10, 60, 25), (2, 10, 5),
            (7, 30, 15), (15, 60, 30), (3, 15, 7),
        ]
        ws = window_sets[generation % len(window_sets)]
        short_w, long_w, mid_w = ws[0], ws[1], ws[2]

        normalization_modes = prompt_spec.get("normalization_modes", [])
        if generation % 3 < len(normalization_modes):
            norm_mode = normalization_modes[generation % 3]
        else:
            norm_mode = normalization_modes[0] if normalization_modes else ("TS_RANK" if single_asset_mode else "RANK")

        arithmetic_ops = prompt_spec.get("arithmetic_hints", []) or ["+", "-", "*", "/"]
        combiner = arithmetic_ops[generation % len(arithmetic_ops)]
        family = families[generation % len(families)]

        def _wrap(expr: str) -> str:
            preferred = norm_mode if norm_mode in {"RAW", "RANK", "TS_RANK"} else ("TS_RANK" if single_asset_mode else "RANK")
            if engine and _resolve_normalization_policy() == "forced" and preferred in {"RANK", "TS_RANK"}:
                return engine.wrap_expression(expr)
            return _adaptive_wrap_expression(
                expr,
                single_asset_mode=single_asset_mode,
                preferred_mode=preferred,
                window=max(20, long_w),
            )

        # ENHANCED BUILDING BLOCKS: More diverse, creative, and sophisticated
        building_blocks = {
            # === MOMENTUM FAMILY (Price Trend Signals) ===
            "momentum": f"DELTA({var1}, {short_w}) / (TS_STD({var1}, {long_w}) + 1e-8)",
            "acceleration": f"(DELTA({var1}, {short_w}) - DELTA({var1}, {mid_w})) / (TS_STD({var1}, {long_w}) + 1e-8)",
            "trend_consistency": f"TS_MEAN(SIGN(DELTA({var1}, 1)), {mid_w}) * ABS(DELTA({var1}, {short_w})) / ({var1} + 1e-8)",
            "momentum_strength": f"DELTA({var1}, {short_w}) * TS_MEAN(SIGN(DELTA({var1}, 1)), {mid_w}) / (TS_STD({var1}, {long_w}) + 1e-8)",
            
            # === MEAN-REVERSION FAMILY (Contrarian Signals) ===
            "mean_reversion": f"({var1} - TS_MEAN({var1}, {mid_w})) / (TS_STD({var1}, {long_w}) + 1e-8)",
            "oscillator": f"({var1} - TS_MEAN({var1}, {short_w})) / (TS_STD({var1}, {short_w}) + 1e-8)",
            "bollinger_position": f"({var1} - TS_MEAN({var1}, {mid_w})) / (2 * TS_STD({var1}, {mid_w}) + 1e-8)",
            "multi_scale_reversion": f"(({var1} - TS_MEAN({var1}, {short_w})) / (TS_STD({var1}, {short_w}) + 1e-8) - ({var1} - TS_MEAN({var1}, {long_w})) / (TS_STD({var1}, {long_w}) + 1e-8))",
            
            # === PRICE-VOLUME INTERACTION (Microstructure) ===
            "flow": f"TS_CORR(DELTA({var1}, 1), {var2 if var2 != var1 else '$volume'}, {mid_w})",
            "volume_divergence": f"TS_CORR(DELTA({var1}, {short_w}), DELTA({var2 if var2 != var1 else '$volume'}, {short_w}), {mid_w})",
            "informed_trading": f"TS_CORR(DELTA({var1}, 1), {var2 if var2 != var1 else '$volume'}, {short_w}) * DELTA({var1}, {short_w}) / (TS_STD({var1}, {mid_w}) + 1e-8)",
            "volume_momentum": f"(DELTA({var1}, {short_w}) / ({var1} + 1e-8)) * ({var2 if var2 != var1 else '$volume'} / (TS_MEAN({var2 if var2 != var1 else '$volume'}, {mid_w}) + 1e-8))",
            
            # === VOLATILITY & REGIME (Risk & State Detection) ===
            "volatility": f"TS_STD({var1}, {short_w}) / (TS_STD({var1}, {long_w}) + 1e-8)",
            "regime": f"DELTA(TS_STD({var1}, {short_w}), {mid_w})",
            "volatility_adjusted_momentum": f"DELTA({var1}, {short_w}) / (TS_STD({var1}, {short_w}) + 1e-8) * (TS_STD({var1}, {long_w}) / (TS_STD({var1}, {short_w}) + 1e-8))",
            "regime_adaptive": f"({var1} - TS_MEAN({var1}, {mid_w})) / (TS_STD({var1}, {mid_w}) + 1e-8) * (1 - TS_STD({var1}, {short_w}) / (TS_STD({var1}, {long_w}) + 1e-8))",
            
            # === RANGE & BREAKOUT (Support/Resistance) ===
            "range": f"({var1} - TS_MIN({var1}, {long_w})) / (TS_MAX({var1}, {long_w}) - TS_MIN({var1}, {long_w}) + 1e-8)",
            "breakout": f"({var1} / (TS_MAX({var1}, {long_w}) + 1e-8)) - 1",
            "channel_position": f"({var1} - TS_MIN({var1}, {mid_w})) / (TS_MAX({var1}, {mid_w}) - TS_MIN({var1}, {mid_w}) + 1e-8) - 0.5",
            "breakout_strength": f"({var1} - TS_MAX({var1}, {long_w})) / (TS_STD({var1}, {long_w}) + 1e-8)",
            
            # === MICROSTRUCTURE PATTERNS (Intraday Dynamics) ===
            "microstructure": "($close - $open) / ($high - $low + 1e-8)",
            "gap": "($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)",
            "intraday_trend": "($close - $open) / ($open + 1e-8)",
            "range_efficiency": "ABS($close - $open) / ($high - $low + 1e-8)",
            
            # === CORRELATION & CROSS-VARIABLE (Multi-Asset) ===
            "correlation": f"TS_CORR({var1}, {var2}, {mid_w})",
            "correlation_stability": f"TS_CORR({var1}, {var2}, {short_w}) - TS_CORR({var1}, {var2}, {long_w})",
            "cross_momentum": f"TS_CORR(DELTA({var1}, {short_w}), DELTA({var2}, {short_w}), {mid_w})",
            
            # === ADVANCED STATISTICS (Higher Moments) ===
            "skewness": f"TS_SKEW({var1 if var1 != '$close' else '$return'}, {mid_w})",
            "kurtosis_proxy": f"(TS_MAX({var1}, {mid_w}) - TS_MIN({var1}, {mid_w})) / (TS_STD({var1}, {mid_w}) + 1e-8)",
            "tail_risk": f"({var1} - TS_MEAN({var1}, {mid_w})) / (TS_MAX(ABS({var1} - TS_MEAN({var1}, {mid_w})), {long_w}) + 1e-8)",
            
            # === TREND QUALITY (Reliability Metrics) ===
            "trend_quality": f"TS_MEAN(SIGN(DELTA({var1}, 1)), {mid_w})",
            "trend_reliability": f"TS_MEAN(SIGN(DELTA({var1}, 1)), {mid_w}) * DELTA({var1}, {short_w}) / ({var1} + 1e-8)",
            "directional_conviction": f"ABS(TS_MEAN(SIGN(DELTA({var1}, 1)), {mid_w})) * ABS(DELTA({var1}, {short_w})) / (TS_STD({var1}, {long_w}) + 1e-8)",
        }

        primary = building_blocks.get(family, building_blocks["momentum"])
        if generation % 3 == 0:
            expr = primary
        elif generation % 3 == 1:
            secondary_family = families[(generation + 1) % len(families)]
            secondary = building_blocks.get(secondary_family, building_blocks["volatility"])
            expr = f"({primary}) {combiner} ({secondary})"
        else:
            secondary_family = families[(generation + 1) % len(families)]
            tertiary_family = families[(generation + 2) % len(families)] if len(families) > 2 else "trend_quality"
            secondary = building_blocks.get(secondary_family, building_blocks["volatility"])
            tertiary = building_blocks.get(tertiary_family, building_blocks["trend_quality"])
            expr = f"(({primary}) {combiner} ({secondary})) * ({tertiary})"

        out = _wrap(expr)
        self._register_candidate(out, {
            "origin": "prompt_spec",
            "regime": regime,
            "theme": theme,
            "family": family,
            "norm": norm_mode,
            "windows": [short_w, mid_w, long_w],
            "vars": [var1, var2],
        })
        return out

    def _sample_window(self, regime: str, windows: List[int]) -> int:
        """Select a lookback window via Thompson Sampling per regime."""
        str_windows = [str(w) for w in windows]
        sampler = self._reasoning_mem.get_sampler(f"{regime}:windows", str_windows)
        return int(sampler.sample())

    def _sample_variable(self, regime: str, variables: List[str]) -> str:
        """Select a variable via Thompson Sampling per regime."""
        sampler = self._reasoning_mem.get_sampler(f"{regime}:vars", variables)
        return sampler.sample()

    def _synthesize_from_template(self, regime: str, theme: str,
                                  operators: List[str], variables: List[str],
                                  windows: List[int],
                                  single_asset_mode: bool = False,
                                  engine=None) -> str:
        var = self._sample_variable(regime, variables)
        w1 = self._sample_window(regime, windows)
        remaining = [w for w in windows if w != w1] or windows
        w2 = self._sample_window(regime, remaining)

        def _wrap(expr):
            if engine and _resolve_normalization_policy() == "forced":
                return engine.wrap_expression(expr)
            return _adaptive_wrap_expression(
                expr,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, w2),
            )

        def _zscore(x, w):
            if engine:
                return engine.zscore_expression(x, w)
            return f"(({x} - TS_MEAN({x}, {w})) / (TS_STD({x}, {w}) + 1e-8))"

        # Template choice is learned per regime+theme pair
        available_templates = self._THEME_TEMPLATES.get(theme, self._THEME_TEMPLATES["default"])
        template_key = f"{regime}:{theme}:template"
        chosen_template = self._reasoning_mem.get_sampler(template_key, available_templates).sample()
        if theme == "MEAN_REVERSION":
            template_map = {
                "zscore": _wrap(f"({var} - TS_MEAN({var}, {w1})) / (TS_STD({var}, {w1}) + 1e-8)"),
                "range_position": _wrap(f"({var} - TS_MIN({var}, {w1})) / (TS_MAX({var}, {w1}) - TS_MIN({var}, {w1}) + 1e-8)"),
                "ma_ratio": _wrap(f"TS_MEAN({var}, {w1}) / TS_MEAN({var}, {w2}) - 1"),
                "delta_zscore": _wrap(_zscore(f"DELTA({var}, {w1})", w2)),
            }
        elif theme == "MOMENTUM_FLOW":
            vol_var = "$volume" if "$volume" in variables else var
            template_map = {
                "raw_delta": _wrap(f"DELTA({var}, {w1})"),
                "vol_corr": _wrap(f"TS_CORR(DELTA({var}, 1), {vol_var}, {w1})"),
                "signed_delta": _wrap(f"DELTA({var}, {w1}) * SIGN(TS_MEAN(DELTA({var}, 1), {w2}))"),
                "roc": _wrap(f"{var} / DELAY({var}, {w1}) - 1"),
            }
        elif theme == "LIQUIDITY_DYNAMICS":
            vol_var = "$volume" if "$volume" in variables else var
            template_map = {
                "price_vol_corr": _wrap(f"TS_CORR({var}, {vol_var}, {w1})"),
                "vol_cv": _wrap(f"TS_STD({vol_var}, {w1}) / (TS_MEAN({vol_var}, {w1}) + 1e-8)"),
                "signed_vol_delta": _wrap(f"DELTA({vol_var}, {w1}) * SIGN(DELTA({var}, {w2}))"),
                "skew": _wrap(f"TS_SKEW({var}, {w1})"),
            }
        elif theme == "REGIME_TRANSITION":
            w_min = min(windows)
            w_max = max(windows)
            template_map = {
                "vol_ratio": _wrap(f"TS_STD({var}, {w_min}) / (TS_STD({var}, {w_max}) + 1e-8)"),
                "vol_delta": _wrap(f"DELTA(TS_STD({var}, {w1}), {w2})"),
                "autocorr": _wrap(f"TS_CORR({var}, DELAY({var}, 1), {w1})"),
                "kurtosis": _wrap(f"TS_KURT({var}, {w1})"),
                "vol_skew": _wrap(f"TS_SKEW(DELTA({var}, 1), {w1})"),  # New: captures tail shifts
            }
        else:
            template_map = {
                "raw_delta": _wrap(f"DELTA({var}, {w1})"),
                "ma_diff": _wrap(f"{var} - TS_MEAN({var}, {w1})"),
                "std": _wrap(f"TS_STD({var}, {w1})"),
                "zscore": _wrap(f"({var} - TS_MEAN({var}, {w1})) / (TS_STD({var}, {w1}) + 1e-8)"),
                "spectral_divergence": _wrap(f"({var}/TS_MEAN({var}, {w1})) - (DELAY({var}, 1)/TS_MEAN({var}, {w2}))"), # New: divergence
            }

        out = template_map.get(chosen_template, list(template_map.values())[0])
        self._register_candidate(out, {
            "origin": "template",
            "regime": regime,
            "theme": theme,
            "template": chosen_template,
            "var": var,
            "windows": [w1, w2],
        })
        return out

    def _synthesize_from_seed(self, seed_tokens: List[str], theme: str,
                              operators: List[str], variables: List[str],
                              windows: List[int],
                              single_asset_mode: bool = False,
                              engine=None) -> str:
        seed_expr = "".join(seed_tokens)
        regime_proxy = theme  # use theme as regime proxy for seed-based mutations

        # Mutation via Thompson-sampled type
        mutation_type = self._reasoning_mem.get_sampler(
            f"seed:{theme}:mutation", self._MUTATION_TYPES
        ).sample()
        if engine:
            registry = engine.get_gene_registry()
        else:
            registry = FOREX_ALPHA_REGISTRY if single_asset_mode else STOCK_ALPHA_REGISTRY
        seed_expr = self._mutate(
            seed_expr, registry.get("BINARY", []) + registry.get("UNARY", []),
            variables, windows, mutation_type=mutation_type,
            single_asset_mode=single_asset_mode, engine=engine
        )

        def _wrap(e):
            if engine and _resolve_normalization_policy() == "forced":
                return engine.wrap_expression(e)
            return _adaptive_wrap_expression(
                e,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, w),
            )

        w = self._sample_window(theme, windows)
        vol_var = "$volume" if "$volume" in variables else variables[0] if variables else "$close"

        # Wrap seed according to theme
        enrichments = {
            "MEAN_REVERSION": _wrap(f"({seed_expr}) - TS_MEAN({seed_expr}, {w})"),
            "MOMENTUM_FLOW": _wrap(f"TS_CORR({seed_expr}, {vol_var}, {w})"),
            "LIQUIDITY_DYNAMICS": _wrap(f"TS_SKEW({seed_expr}, {w})"),
            "REGIME_TRANSITION": _wrap(f"DELTA({seed_expr}, {w})"),
        }
        template_key = f"seed:{theme}:wrap"
        wrap_options = list(enrichments.keys()) + ["default"]
        chosen = self._reasoning_mem.get_sampler(template_key, wrap_options).sample()
        out = enrichments.get(chosen, _wrap(seed_expr))
        self._register_candidate(out, {
            "origin": "seed",
            "theme": theme,
            "seed_mutation_type": mutation_type,
            "seed_wrap": chosen,
        })
        return out

    def _synthesize_from_composition(self, operators: List[str],
                                      variables: List[str], windows: List[int],
                                      regime: str = "NEUTRAL",
                                      single_asset_mode: bool = False,
                                      engine=None) -> str:
        # All selections learned per regime
        var1 = self._sample_variable(regime, variables)
        alt_vars = [v for v in variables if v != var1] or variables
        var2 = self._sample_variable(regime, alt_vars)
        w1 = self._sample_window(regime, windows)
        remaining = [w for w in windows if w != w1] or windows
        w2 = self._sample_window(regime, remaining)
        op = self._reasoning_mem.get_sampler(f"{regime}:comp_op", operators).sample() if operators else "DELTA"

        def _wrap(expr):
            if engine and _resolve_normalization_policy() == "forced":
                return engine.wrap_expression(expr)
            return _adaptive_wrap_expression(
                expr,
                single_asset_mode=single_asset_mode,
                preferred_mode="TS_RANK" if single_asset_mode else "RANK",
                window=max(20, w2),
            )

        def _zscore(x, w):
            if engine:
                return engine.zscore_expression(x, w)
            return f"(({x} - TS_MEAN({x}, {w})) / (TS_STD({x}, {w}) + 1e-8))"

        registry = FOREX_ALPHA_REGISTRY if single_asset_mode else STOCK_ALPHA_REGISTRY
        bs = registry.get("BINARY", [])
        ts = registry.get("TRINARY", [])
        is_binary = op in bs
        is_trinary = op in ts

        inner_templates = [
            f"DELTA({var1}, {w1})",
            f"TS_MEAN({var1}, {w1})",
            f"{var1} / DELAY({var1}, {w1})",
            f"{var1} - TS_MEAN({var1}, {w1})",
            f"TS_STD({var1}, {w1})"
        ]
        inner_key = f"{regime}:comp_inner"
        inner_names = ["delta", "mean", "roc_minus1", "demean", "std"]
        chosen_inner_name = self._reasoning_mem.get_sampler(inner_key, inner_names).sample()
        inner = dict(zip(inner_names, inner_templates))[chosen_inner_name]

        comp_options = {
            "op_binary": _wrap(f"{op}({inner}, {var2}, {w2})") if is_trinary else (_wrap(f"{op}({inner}, {w2})") if is_binary else _wrap(f"{op}({inner})")),
            "signed_cross": _wrap(f"{inner} * SIGN(DELTA({var2}, {w2}))"),
            "corr_cross": _wrap(f"TS_CORR({inner}, {var2}, {w2})") if var2 != var1 else _wrap(_zscore(inner, w2)),
        }
        comp_sampler_key = f"{regime}:comp_structure"
        chosen_comp = self._reasoning_mem.get_sampler(comp_sampler_key, list(comp_options.keys())).sample()
        return comp_options[chosen_comp]

    def _mutate(self, expression: str, operators: List[str],
                variables: List[str], windows: List[int],
                mutation_type: Optional[str] = None,
                single_asset_mode: bool = False,
                engine=None) -> str:
        if engine:
            registry = engine.get_gene_registry()
        else:
            registry = FOREX_ALPHA_REGISTRY if single_asset_mode else STOCK_ALPHA_REGISTRY

        if mutation_type == "variable_swap":
            for var in registry.get("VAR", []):
                if var in expression:
                    new_var = self._sample_variable("mutation", variables)
                    expression = expression.replace(var, new_var, 1)
                    break

        elif mutation_type == "window_shift":
            numbers = re.findall(r'(?<=,\s)\d+|(?<=,)\d+', expression)
            if numbers:
                # Replace the window with the highest-probability window from sampler
                new_window = self._sample_window("mutation", windows)
                old_num = numbers[0]  # Replace first occurrence deterministically
                expression = expression.replace(old_num, str(new_window), 1)

        elif mutation_type == "operator_wrap":
            if engine:
                wrap_ops = engine.get_mutation_wraps()
                if "RAW" not in wrap_ops:
                    wrap_ops = list(wrap_ops) + ["RAW"]
                wrap_op = self._reasoning_mem.get_sampler("mutation:wrap", wrap_ops).sample()
                if wrap_op == "RAW":
                    expression = _adaptive_wrap_expression(
                        expression,
                        single_asset_mode=single_asset_mode,
                        preferred_mode="RAW",
                        window=60,
                    )
                else:
                    expression = engine.mutate_wrap(expression, wrap_op)
            else:
                wrap_candidates = ["RAW", "TS_RANK", "ABS", "SIGN"] if single_asset_mode else ["RAW", "RANK", "ABS", "SIGN"]
                wrap_op = self._reasoning_mem.get_sampler("mutation:wrap", wrap_candidates).sample()
                if wrap_op in {"RAW", "RANK", "TS_RANK"}:
                    expression = _adaptive_wrap_expression(
                        expression,
                        single_asset_mode=single_asset_mode,
                        preferred_mode=wrap_op,
                        window=60,
                    )
                elif not expression.startswith(f"{wrap_op}("):
                    expression = f"{wrap_op}({expression})"

        elif mutation_type == "arithmetic_inject":
            arith_ops = ["+", "-", "*"]
            arith = self._reasoning_mem.get_sampler("mutation:arith", arith_ops).sample()
            var = self._sample_variable("mutation", variables)
            w = self._sample_window("mutation", windows)
            inject = f"DELTA({var}, {w})"
            expression = f"({expression}) {arith} ({inject})"

        return expression

    def _crossover(self, parent_a: str, parent_b: str,
                   strategy: Optional[str] = None) -> Optional[str]:
        tokens_a = self.tokenizer.tokenize(parent_a)
        tokens_b = self.tokenizer.tokenize(parent_b)
        if len(tokens_a) < 3 or len(tokens_b) < 3:
            return None
        strategy_map = {
            "segment_recombination": self._crossover_segment_recombination,
            "operator_substitution": self._crossover_operator_substitution,
            "structural_fusion": self._crossover_structural_fusion,
        }
        fn = strategy_map.get(strategy, self._crossover_structural_fusion)
        return fn(parent_a, parent_b, tokens_a, tokens_b)

    def _crossover_segment_recombination(self, pa: str, pb: str,
                                          ta: List[str], tb: List[str]) -> str:
        mid_a = len(ta) // 2
        mid_b = len(tb) // 2
        child = "".join(ta[:mid_a]) + "".join(tb[mid_b:])
        open_parens = child.count("(")
        close_parens = child.count(")")
        if open_parens > close_parens:
            child += ")" * (open_parens - close_parens)
        elif close_parens > open_parens:
            child = "(" * (close_parens - open_parens) + child
        single_asset = pa.startswith("TS_RANK")
        preferred = "TS_RANK" if single_asset else ("RANK" if pa.startswith("RANK") else "RAW")
        return _adaptive_wrap_expression(
            child,
            single_asset_mode=single_asset,
            preferred_mode=preferred,
            window=60,
        )

    def _crossover_operator_substitution(self, pa: str, pb: str,
                                          ta: List[str], tb: List[str]) -> str:
        ops_a = [t.rstrip('(') for t in ta if t.endswith('(')]
        ops_b = [t.rstrip('(') for t in tb if t.endswith('(')]
        result = pa
        if ops_a and ops_b:
            # Pick dominant op from A (first) and replace with dominant op from B (first)
            old_op = ops_a[0]
            new_op = ops_b[0]
            result = result.replace(old_op + "(", new_op + "(", 1)
        return result

    def _crossover_structural_fusion(self, pa: str, pb: str,
                                      ta: List[str], tb: List[str]) -> str:
        arith_ops = ["+", "-", "*"]
        op = self._reasoning_mem.get_sampler("crossover:arith", arith_ops).sample()
        single_asset = pa.startswith("TS_RANK")
        preferred = "TS_RANK" if single_asset else ("RANK" if pa.startswith("RANK") else "RAW")
        return _adaptive_wrap_expression(
            f"({pa}) {op} ({pb})",
            single_asset_mode=single_asset,
            preferred_mode=preferred,
            window=60,
        )

    def record_success(
        self,
        expression: str,
        regime: str = "NEUTRAL",
        ic: float = 0.0,
        icir: Optional[float] = None,
        ir: Optional[float] = None,
        arr: Optional[float] = None,
        mdd: Optional[float] = None,
        rank_ic: Optional[float] = None,
    ):
        """Feed positive reward into sampling decisions used for this expression."""
        self.successful_patterns.append(expression)
        if len(self.successful_patterns) > 100:
            self.successful_patterns = self.successful_patterns[-80:]
        meta_list = self.get_candidate_meta(expression)
        meta = meta_list[0] if meta_list else {}
        reward = self._reward_from_metrics(
            ic=ic,
            icir=icir if icir is not None else meta.get("icir"),
            ir=ir if ir is not None else meta.get("ir"),
            arr=arr if arr is not None else meta.get("arr"),
            mdd=mdd if mdd is not None else meta.get("mdd"),
            rank_ic=rank_ic if rank_ic is not None else meta.get("rank_ic"),
            success=True,
        )

        # Complexity regularization: prefer simpler expressions when performance is similar.
        if os.environ.get("SOVA_COMPLEXITY_PENALTY", "1").strip().lower() not in {"0", "false", "no"}:
            try:
                structure = self.tokenizer.extract_structure(expression)
                comp = float(structure.get("complexity_score", 0.0) or 0.0)
                weight = float(os.environ.get("SOVA_COMPLEXITY_PENALTY_WEIGHT", "0.08") or "0.08")
                penalty = weight * min(1.0, comp / 260.0)
                reward = float(np.clip(reward - penalty, -1.0, 1.0))
            except Exception:
                pass

        # Structural reinforcement
        ops = re.findall(r'[A-Z_]+', expression)
        for op in ops:
            self._reasoning_mem.update(f"{regime}:ops", op, reward)

        # Attribution reinforcement: template/mutation/crossover decisions
        origin = meta.get("origin")
        if origin == "template":
            theme = meta.get("theme")
            tpl = meta.get("template")
            if theme and tpl:
                self._reasoning_mem.update(f"{regime}:{theme}:template", str(tpl), reward)
        if origin == "mutation":
            mut = meta.get("mutation_type")
            if mut:
                self._reasoning_mem.update(f"{regime}:mutation", str(mut), reward)
        if origin == "crossover":
            strat = meta.get("crossover_strategy")
            if strat:
                self._reasoning_mem.update(f"{regime}:crossover", str(strat), reward)
        if origin == "seed":
            theme = meta.get("theme")
            sm = meta.get("seed_mutation_type")
            sw = meta.get("seed_wrap")
            if theme and sm:
                self._reasoning_mem.update(f"seed:{theme}:mutation", str(sm), reward * 0.6)
            if theme and sw:
                self._reasoning_mem.update(f"seed:{theme}:wrap", str(sw), reward * 0.4)
        if origin == "counterfactual":
            self._reasoning_mem.update(f"{regime}:counterfactual", "mean_reversion", reward * 0.25)
            self._reasoning_mem.update(f"{regime}:counterfactual", "momentum", reward * 0.25)
            self._reasoning_mem.update(f"{regime}:counterfactual", "flow", reward * 0.25)
            self._reasoning_mem.update(f"{regime}:counterfactual", "range", reward * 0.25)

        norm = self._normalize_expr(expression)
        self._expr_failure_streak[norm] = 0
        self._expr_cooldown_until.pop(norm, None)

        # Nudge arith samplers towards what worked
        arith_used = [a for a in ["+", "-", "*"] if f") {a} (" in expression]
        for a in arith_used:
            self._reasoning_mem.update("mutation:arith", a, reward * 0.3)
            self._reasoning_mem.update("crossover:arith", a, reward * 0.3)

        _append_jsonl(memory_path("evolution_experience.jsonl"), {
            "kind": "success",
            "regime": regime,
            "expression": expression,
            "reward": reward,
            "metrics": {
                "ic": float(ic or 0.0),
                "icir": None if icir is None else float(icir),
                "ir": None if ir is None else float(ir),
                "arr": None if arr is None else float(arr),
                "mdd": None if mdd is None else float(mdd),
                "rank_ic": None if rank_ic is None else float(rank_ic),
            },
            "meta": meta_list,
            "ts_utc": datetime.utcnow().isoformat() + "Z",
        })

    def record_failure(
        self,
        expression: str,
        regime: str = "NEUTRAL",
        ic: float = 0.0,
        icir: Optional[float] = None,
        ir: Optional[float] = None,
        arr: Optional[float] = None,
        mdd: Optional[float] = None,
        rank_ic: Optional[float] = None,
    ):
        """Feed negative reward into samplers for features seen in failed expression."""
        self.failed_patterns.append(expression)
        if len(self.failed_patterns) > 100:
            self.failed_patterns = self.failed_patterns[-80:]
        meta_list = self.get_candidate_meta(expression)
        meta = meta_list[0] if meta_list else {}
        reward = self._reward_from_metrics(
            ic=ic,
            icir=icir if icir is not None else meta.get("icir"),
            ir=ir if ir is not None else meta.get("ir"),
            arr=arr if arr is not None else meta.get("arr"),
            mdd=mdd if mdd is not None else meta.get("mdd"),
            rank_ic=rank_ic if rank_ic is not None else meta.get("rank_ic"),
            success=False,
        )

        if os.environ.get("SOVA_COMPLEXITY_PENALTY", "1").strip().lower() not in {"0", "false", "no"}:
            try:
                structure = self.tokenizer.extract_structure(expression)
                comp = float(structure.get("complexity_score", 0.0) or 0.0)
                weight = float(os.environ.get("SOVA_COMPLEXITY_PENALTY_WEIGHT", "0.08") or "0.08")
                penalty = weight * min(1.0, comp / 260.0)
                reward = float(np.clip(reward - penalty, -1.0, 1.0))
            except Exception:
                pass
        ops = re.findall(r'[A-Z_]+', expression)
        for op in ops:
            self._reasoning_mem.update(f"{regime}:ops", op, reward)

        origin = meta.get("origin")
        if origin == "template":
            theme = meta.get("theme")
            tpl = meta.get("template")
            if theme and tpl:
                self._reasoning_mem.update(f"{regime}:{theme}:template", str(tpl), reward)
        if origin == "mutation":
            mut = meta.get("mutation_type")
            if mut:
                self._reasoning_mem.update(f"{regime}:mutation", str(mut), reward)
        if origin == "crossover":
            strat = meta.get("crossover_strategy")
            if strat:
                self._reasoning_mem.update(f"{regime}:crossover", str(strat), reward)
        if origin == "seed":
            theme = meta.get("theme")
            sm = meta.get("seed_mutation_type")
            sw = meta.get("seed_wrap")
            if theme and sm:
                self._reasoning_mem.update(f"seed:{theme}:mutation", str(sm), reward * 0.6)
            if theme and sw:
                self._reasoning_mem.update(f"seed:{theme}:wrap", str(sw), reward * 0.4)

        norm = self._normalize_expr(expression)
        self._failure_event_idx += 1
        self._expr_failure_streak[norm] += 1
        # After repeated failures, avoid retrying the exact same expression for a short horizon.
        if self._expr_failure_streak[norm] >= int(os.environ.get("SOVA_FAIL_COOLDOWN_STREAK", "2") or "2"):
            horizon = int(os.environ.get("SOVA_FAIL_COOLDOWN_HORIZON", "6") or "6")
            self._expr_cooldown_until[norm] = self._failure_event_idx + max(1, horizon)

        _append_jsonl(memory_path("evolution_experience.jsonl"), {
            "kind": "failure",
            "regime": regime,
            "expression": expression,
            "reward": reward,
            "metrics": {
                "ic": float(ic or 0.0),
                "icir": None if icir is None else float(icir),
                "ir": None if ir is None else float(ir),
                "arr": None if arr is None else float(arr),
                "mdd": None if mdd is None else float(mdd),
                "rank_ic": None if rank_ic is None else float(rank_ic),
            },
            "meta": meta_list,
            "ts_utc": datetime.utcnow().isoformat() + "Z",
        })


class ChessTradeAnalyst:
    """
    Trade analysis engine with active pattern learning.
    - LessonMemory: persists trade errors and their frequency across sessions.
    - Dynamic thresholds: timing/capture/quality thresholds adapt based on regime history.
    - Pattern recurrence detection: repeated error types trigger increasingly urgent recommendations.
    - Positive pattern mining: successful trade conditions are codified into entry templates.
    """

    def __init__(self):
        self.error_taxonomy = TRADE_ERROR_TAXONOMY
        self.analysis_history: deque = deque(maxlen=500)
        self.pattern_counter: Counter = Counter()
        self.lesson_book: List[Dict[str, Any]] = []
        self._reasoning_mem = ReasoningMemory()
        # Per-regime success condition storage: {regime: {"avg_timing": ..., "avg_capture": ..., "n": ...}}
        self._regime_benchmarks: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"avg_timing": 0.5, "avg_capture": 0.4, "n": 0, "win_patterns": []}
        )
        self._load_lesson_memory()

    def _load_lesson_memory(self):
        """Load persisted pattern counts and regime benchmarks across sessions."""
        path = memory_path("trade_lessons.json")
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self.pattern_counter = Counter(data.get("pattern_counter", {}))
                self._regime_benchmarks.update(data.get("regime_benchmarks", {}))
                self.lesson_book = data.get("lesson_book", [])[-200:]  # keep latest 200
            except Exception:
                pass

    def _save_lesson_memory(self):
        path = memory_path("trade_lessons.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "pattern_counter": dict(self.pattern_counter),
                    "regime_benchmarks": dict(self._regime_benchmarks),
                    "lesson_book": self.lesson_book[-200:]
                }, f, indent=2)
        except Exception:
            pass

    def analyze_trade(self, trade: Dict[str, Any], dna: MarketDNA,
                     price_context: Optional[np.ndarray] = None) -> Dict[str, Any]:
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_pct', 0)
        direction = trade.get('direction', 'LONG')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        mfe = trade.get('mfe', 0)
        mfe_pct = trade.get('mfe_pct', 0)
        mae = trade.get('mae', 0)
        mae_pct = trade.get('mae_pct', 0)
        bars = trade.get('bars', 0)

        is_winner = pnl > 0
        quality = self._assess_move_quality(pnl_pct, mfe_pct, mae_pct, bars, dna)
        errors = [] if is_winner else self._diagnose_errors(trade, dna, price_context)
        efficiency = self._compute_efficiency(pnl_pct, mfe_pct, mae_pct)
        timing_score = self._assess_timing(trade, dna, price_context)
        direction_alignment = self._check_direction_alignment(direction, dna)
        lessons = self._extract_lessons(trade, quality, errors, dna)

        # Update regime benchmarks from this trade
        self._update_regime_benchmark(dna.regime, timing_score, efficiency["capture_ratio"], is_winner)

        # Feed error pattern to Thompson Sampler — learns which error types are most frequent in this regime
        for error in errors:
            self._reasoning_mem.update(f"{dna.regime}:error_freq", error["type"], 1.0)

        analysis = {
            "trade_id": trade.get('id', id(trade)),
            "quality": quality,
            "errors": errors,
            "efficiency": efficiency,
            "timing_score": timing_score,
            "direction_alignment": direction_alignment,
            "lessons": lessons,
            "regime_at_entry": dna.regime,
            "detailed_assessment": self._generate_assessment(
                trade, quality, errors, efficiency, timing_score, direction_alignment, dna
            )
        }

        self.analysis_history.append(analysis)
        for error in errors:
            self.pattern_counter[error["type"]] += 1
        for lesson in lessons:
            self.lesson_book.append(lesson)
            if len(self.lesson_book) > 500:
                self.lesson_book = self.lesson_book[-400:]

        self._save_lesson_memory()
        return analysis

    def _update_regime_benchmark(self, regime: str, timing: float, capture: float, win: bool):
        b = self._regime_benchmarks[regime]
        n = b["n"]
        # Exponential moving average of timing and capture per regime
        alpha = 0.1
        b["avg_timing"] = b["avg_timing"] * (1 - alpha) + timing * alpha
        b["avg_capture"] = b["avg_capture"] * (1 - alpha) + capture * alpha
        b["n"] = n + 1

    def _assess_move_quality(self, pnl_pct: float, mfe_pct: float,
                            mae_pct: float, bars: int, dna: MarketDNA) -> str:
        if pnl_pct > 0.05 and mae_pct < 0.01:
            return "BRILLIANT"
        elif pnl_pct > 0.03 and mae_pct < 0.02:
            return "EXCELLENT"
        elif pnl_pct > 0.01:
            return "GOOD"
        elif pnl_pct > 0:
            return "ACCEPTABLE"
        elif pnl_pct > -0.005:
            return "SCRATCH"
        elif pnl_pct > -0.015:
            return "INACCURACY"
        elif pnl_pct > -0.03:
            return "MISTAKE"
        else:
            return "BLUNDER"

    def _diagnose_errors(self, trade: Dict[str, Any], dna: MarketDNA,
                        price_context: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        errors = []
        direction = trade.get('direction', 'LONG')
        pnl_pct = trade.get('pnl_pct', 0)
        mfe_pct = trade.get('mfe_pct', 0)
        mae_pct = trade.get('mae_pct', 0)
        bars = trade.get('bars', 0)

        if direction == 'LONG' and dna.trend < -0.03:
            errors.append({
                "type": "WRONG_DIRECTION",
                "detail": f"Long trade in downtrend (trend={dna.trend:.3f})",
                "severity": self.error_taxonomy["WRONG_DIRECTION"]["severity"],
                "correction": self.error_taxonomy["WRONG_DIRECTION"]["correction"]
            })
        elif direction == 'SHORT' and dna.trend > 0.03:
            errors.append({
                "type": "WRONG_DIRECTION",
                "detail": f"Short trade in uptrend (trend={dna.trend:.3f})",
                "severity": self.error_taxonomy["WRONG_DIRECTION"]["severity"],
                "correction": self.error_taxonomy["WRONG_DIRECTION"]["correction"]
            })

        if dna.regime == "CHAOTIC_NOISE" or dna.spectral_entropy > 2.5:
            errors.append({
                "type": "TREATING_NOISE_AS_SIGNAL",
                "detail": f"Entry during high noise (entropy={dna.spectral_entropy:.2f})",
                "severity": self.error_taxonomy["TREATING_NOISE_AS_SIGNAL"]["severity"],
                "correction": self.error_taxonomy["TREATING_NOISE_AS_SIGNAL"]["correction"]
            })

        if mfe_pct > abs(pnl_pct) * 3 and pnl_pct < 0:
            errors.append({
                "type": "HELD_TOO_LONG",
                "detail": f"Had MFE={mfe_pct:.3f} but ended at PnL={pnl_pct:.3f}",
                "severity": self.error_taxonomy["HELD_TOO_LONG"]["severity"],
                "correction": self.error_taxonomy["HELD_TOO_LONG"]["correction"]
            })

        if abs(mae_pct) > dna.volatility * 0.5 and bars < 3:
            errors.append({
                "type": "CHASING_TREND_LATE",
                "detail": f"Quick adverse excursion MAE={mae_pct:.3f} in {bars} bars",
                "severity": self.error_taxonomy["CHASING_TREND_LATE"]["severity"],
                "correction": self.error_taxonomy["CHASING_TREND_LATE"]["correction"]
            })

        if mfe_pct > 0.02 and pnl_pct < mfe_pct * 0.3 and pnl_pct > 0:
            errors.append({
                "type": "PREMATURE_EXIT",
                "detail": f"Captured only {pnl_pct / mfe_pct * 100:.0f}% of available move",
                "severity": self.error_taxonomy["PREMATURE_EXIT"]["severity"],
                "correction": self.error_taxonomy["PREMATURE_EXIT"]["correction"]
            })

        if abs(mae_pct) < dna.volatility * 0.15 and pnl_pct < -0.005 and bars <= 2:
            errors.append({
                "type": "STOP_TOO_TIGHT",
                "detail": f"Stop triggered within noise band (MAE={mae_pct:.3f}, vol={dna.volatility:.3f})",
                "severity": self.error_taxonomy["STOP_TOO_TIGHT"]["severity"],
                "correction": self.error_taxonomy["STOP_TOO_TIGHT"]["correction"]
            })

        strategy_config = REGIME_STRATEGY_MAP.get(dna.regime, REGIME_STRATEGY_MAP["NEUTRAL"])
        if strategy_config["aggression"] < 0.3 and abs(pnl_pct) > 0.03:
            errors.append({
                "type": "OVERSIZED_POSITION",
                "detail": f"Large loss in defensive regime ({dna.regime})",
                "severity": self.error_taxonomy["OVERSIZED_POSITION"]["severity"],
                "correction": self.error_taxonomy["OVERSIZED_POSITION"]["correction"]
            })

        if not errors:
            errors.append({
                "type": "REGIME_MISMATCH",
                "detail": f"General loss in {dna.regime} regime",
                "severity": "MEDIUM",
                "correction": "Review regime classification accuracy and strategy alignment"
            })

        return errors

    def _compute_efficiency(self, pnl_pct: float, mfe_pct: float, mae_pct: float) -> Dict[str, float]:
        capture_ratio = pnl_pct / (mfe_pct + 1e-10) if mfe_pct > 0 else 0
        pain_ratio = abs(mae_pct) / (abs(pnl_pct) + 1e-10) if pnl_pct != 0 else float('inf')
        edge_ratio = (mfe_pct - abs(mae_pct)) / (mfe_pct + abs(mae_pct) + 1e-10)
        return {
            "capture_ratio": min(capture_ratio, 1.0),
            "pain_ratio": min(pain_ratio, 10.0),
            "edge_ratio": edge_ratio
        }

    def _assess_timing(self, trade: Dict[str, Any], dna: MarketDNA,
                      price_context: Optional[np.ndarray]) -> float:
        score = 0.5
        direction = trade.get('direction', 'LONG')
        pnl_pct = trade.get('pnl_pct', 0)

        if direction == 'LONG' and dna.trend > 0 and pnl_pct > 0:
            score += 0.2
        elif direction == 'SHORT' and dna.trend < 0 and pnl_pct > 0:
            score += 0.2

        if dna.regime_confidence > 0.6 and pnl_pct > 0:
            score += 0.15

        if dna.regime_transition_probability < 0.3 and pnl_pct > 0:
            score += 0.1

        if pnl_pct < 0:
            score -= 0.2
            if dna.regime_transition_probability > 0.5:
                score -= 0.1

        return max(0, min(1, score))

    def _check_direction_alignment(self, direction: str, dna: MarketDNA) -> Dict[str, Any]:
        aligned = True
        confidence = 0.5
        warnings = []

        if direction == 'LONG':
            if dna.trend < -0.02:
                aligned = False
                confidence = 0.2
                warnings.append(f"Long against downtrend ({dna.trend:.3f})")
            elif dna.trend > 0.02:
                confidence = 0.8
            if dna.regime in ("CAPITULATION_CRASH", "STABLE_EROSION"):
                aligned = False
                confidence = 0.1
                warnings.append(f"Long in bearish regime: {dna.regime}")
        elif direction == 'SHORT':
            if dna.trend > 0.02:
                aligned = False
                confidence = 0.2
                warnings.append(f"Short against uptrend ({dna.trend:.3f})")
            elif dna.trend < -0.02:
                confidence = 0.8
            if dna.regime in ("EXPONENTIAL_BULL", "STABLE_ACCUMULATION"):
                aligned = False
                confidence = 0.1
                warnings.append(f"Short in bullish regime: {dna.regime}")

        return {"aligned": aligned, "confidence": confidence, "warnings": warnings}

    def _extract_lessons(self, trade: Dict[str, Any], quality: str,
                        errors: List[Dict], dna: MarketDNA) -> List[Dict[str, Any]]:
        lessons = []

        if quality in ("BRILLIANT", "EXCELLENT"):
            lesson = {
                "type": "POSITIVE",
                "lesson": f"Successful {trade.get('direction', 'LONG')} in {dna.regime} with confidence {dna.regime_confidence:.2f}",
                "actionable": f"Replicate conditions: regime={dna.regime}, vol={dna.volatility:.3f}, trend={dna.trend:.3f}, hurst={dna.hurst:.3f}",
                "regime": dna.regime,
                "conditions": {
                    "regime": dna.regime, "vol": dna.volatility, "trend": dna.trend,
                    "hurst": dna.hurst, "confidence": dna.regime_confidence,
                    "direction": trade.get("direction", "LONG"), "quality": quality
                },
                "timestamp": datetime.now().isoformat()
            }
            lessons.append(lesson)
            # Record positive condition in Thompson Sampler for direction selection
            self._reasoning_mem.update(
                f"{dna.regime}:direction",
                trade.get("direction", "LONG"),
                trade.get("pnl_pct", 0.01)
            )

        for error in errors:
            severity_penalty = {"CRITICAL": 0.4, "HIGH": 0.3, "MEDIUM": 0.2, "LOW": 0.05}.get(error["severity"], 0.1)
            lessons.append({
                "type": "CORRECTIVE",
                "lesson": f"{error['type']}: {error['detail']}",
                "actionable": error["correction"],
                "regime": dna.regime,
                "recurrence_count": self.pattern_counter.get(error["type"], 0) + 1,
                "timestamp": datetime.now().isoformat()
            })

        return lessons

    def _generate_assessment(self, trade: Dict[str, Any], quality: str,
                            errors: List[Dict], efficiency: Dict,
                            timing: float, alignment: Dict, dna: MarketDNA) -> str:
        lines = [
            f"{'='*60}",
            f"TRADE ANALYSIS [{quality}]",
            f"{'='*60}",
            f"Direction: {trade.get('direction', 'N/A')} | PnL: {trade.get('pnl_pct', 0)*100:.2f}%",
            f"MFE: {trade.get('mfe_pct', 0)*100:.2f}% | MAE: {trade.get('mae_pct', 0)*100:.2f}%",
            f"Bars: {trade.get('bars', 0)} | Regime: {dna.regime}",
            f"",
            f"EFFICIENCY:",
            f"  Capture Ratio: {efficiency['capture_ratio']:.2f}",
            f"  Pain Ratio: {efficiency['pain_ratio']:.2f}",
            f"  Edge Ratio: {efficiency['edge_ratio']:.2f}",
            f"  Timing Score: {timing:.2f}",
            f"",
            f"DIRECTION ALIGNMENT: {'ALIGNED' if alignment['aligned'] else 'MISALIGNED'}",
        ]

        if alignment.get('warnings'):
            for w in alignment['warnings']:
                lines.append(f"  WARNING: {w}")

        if errors:
            lines.append(f"")
            lines.append(f"ERRORS DETECTED:")
            for i, err in enumerate(errors, 1):
                recurrence = self.pattern_counter.get(err["type"], 0)
                urgency = " ⚠️ RECURRING" if recurrence >= 3 else ""
                lines.append(f"  [{err['severity']}] {err['type']}{urgency} (seen {recurrence}x)")
                lines.append(f"    Detail: {err['detail']}")
                lines.append(f"    Fix: {err['correction']}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def generate_session_report(self, trades: List[Dict[str, Any]],
                               dna: MarketDNA) -> Dict[str, Any]:
        analyses = [self.analyze_trade(t, dna) for t in trades]

        winners = [a for a in analyses if a["quality"] in ("BRILLIANT", "EXCELLENT", "GOOD", "ACCEPTABLE")]
        losers = [a for a in analyses if a["quality"] in ("INACCURACY", "MISTAKE", "BLUNDER")]

        all_errors = []
        for a in analyses:
            all_errors.extend(a["errors"])

        error_frequency = Counter(e["type"] for e in all_errors)
        # Merge session frequency with cross-session historical counts
        combined_frequency = error_frequency + self.pattern_counter

        critical_patterns = [
            (error_type, count)
            for error_type, count in combined_frequency.most_common()
            if count >= 2
        ]

        avg_timing = np.mean([a["timing_score"] for a in analyses]) if analyses else 0
        avg_capture = np.mean([a["efficiency"]["capture_ratio"] for a in analyses]) if analyses else 0

        all_lessons = []
        for a in analyses:
            all_lessons.extend(a["lessons"])

        # Include relevant historical lessons for this regime
        historical = [l for l in self.lesson_book if l.get("regime") == dna.regime][-5:]
        recommendations = self._generate_recommendations(
            critical_patterns, dna, avg_timing, avg_capture, historical
        )

        return {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "quality_distribution": Counter(a["quality"] for a in analyses),
            "critical_patterns": critical_patterns,
            "avg_timing_score": avg_timing,
            "avg_capture_ratio": avg_capture,
            "lessons": all_lessons,
            "historical_lessons": historical,
            "recommendations": recommendations,
            "detailed_analyses": analyses
        }

    def _generate_recommendations(self, critical_patterns: List[Tuple],
                                  dna: MarketDNA, avg_timing: float,
                                  avg_capture: float,
                                  historical_lessons: Optional[List] = None) -> List[str]:
        recs = []
        
        # Use regime-adaptive benchmarks learned from past trades
        bench = self._regime_benchmarks[dna.regime]
        timing_threshold = bench["avg_timing"] * 0.9  # recommend if below 90% of regime average
        capture_threshold = bench["avg_capture"] * 0.9

        if avg_timing < timing_threshold:
            recs.append(
                f"TIMING: Entry timing ({avg_timing:.2f}) below regime average ({bench['avg_timing']:.2f}). "
                f"Consider waiting for regime confirmation: conf>{dna.regime_confidence:.2f}, "
                f"transition_prob<{dna.regime_transition_probability:.2f}."
            )

        if avg_capture < capture_threshold:
            recs.append(
                f"CAPTURE: Capturing {avg_capture:.2f} vs regime avg {bench['avg_capture']:.2f}. "
                f"Widen exit targets in {dna.regime} regime or implement trailing stops."
            )

        for error_type, count in critical_patterns:
            if count >= 2 and error_type in self.error_taxonomy:
                urgency = "CRITICAL" if count >= 5 else "PATTERN"
                recs.append(
                    f"[{urgency}] {error_type} (x{count}): "
                    f"{self.error_taxonomy[error_type]['correction']}"
                )

        aggression = REGIME_STRATEGY_MAP.get(dna.regime, {}).get("aggression", 0.5)
        if aggression < 0.4:
            recs.append(
                f"REGIME [{dna.regime}] is defensive (aggression={aggression:.1f}). "
                f"Reduce position sizes and frequency. Benchmark timing: {bench['avg_timing']:.2f}."
            )
        elif aggression > 0.7:
            recs.append(
                f"REGIME [{dna.regime}] favors aggression (={aggression:.1f}). "
                f"Scale up winning trades. Current capture: {avg_capture:.2f}."
            )

        # Mine positive patterns from historical lessons
        positive_lessons = [l for l in (historical_lessons or []) if l.get("type") == "POSITIVE"]
        if positive_lessons:
            latest = positive_lessons[-1]
            recs.append(
                f"REPEAT WINNER: {latest['actionable']}"
            )

        return recs


class SOVAExecutive:
    def __init__(self, vortex_path: Optional[str] = None):
        self.perception = NeuralPerceptionLayer()
        resolved_vortex_path = str(get_vortex_root()) if not vortex_path else str(Path(vortex_path))
        self.vortex = SynapticVortexMemory(resolved_vortex_path)
        self.oracle = GovernanceOracle()
        self.hypothesis_gen = HypothesisGenerator()
        self.evolver = RecursiveEvolver()
        self.analyst = ChessTradeAnalyst()
        self.judge = EnsembleJudge() 
        self.summarizer = StrategicSummarizer()
        self._composition_engine = InternalCompositionEngine()
        self.performance_track: deque = deque(maxlen=500)
        self.latest_feedback = None
        self.cycle_count = 0

    def run_cognitive_cycle(self, price: np.ndarray, volume: Optional[np.ndarray] = None,
                           high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None,
                           trade_history: Optional[List[Dict[str, Any]]] = None,
                           single_asset_mode: bool = False,
                           engine=None) -> Dict[str, Any]:
        self.cycle_count += 1
        logger.info(f"[CYCLE {self.cycle_count}] Starting cognitive cycle")
        
        dna = self.perception.sense(price, volume, high, low, single_asset_mode=single_asset_mode)
        logger.info(f"[PERCEPTION] Regime={dna.regime} (conf={dna.regime_confidence:.2f}) {'[SINGLE_ASSET]' if single_asset_mode else ''}")
        logger.info(f"[PERCEPTION] Vol={dna.volatility:.3f} Hurst={dna.hurst:.3f} Trend={dna.trend:.3f}")
        
        trade_analysis = None
        if trade_history:
            trade_analysis = self.analyst.generate_session_report(trade_history, dna)
            logger.info(f"[ADVISORY] Analyzed {trade_analysis['total_trades']} trades: "
                       f"W={trade_analysis['winners']} L={trade_analysis['losers']} "
                       f"Timing={trade_analysis['avg_timing_score']:.2f}")
            
            for rec in trade_analysis.get("recommendations", []):
                logger.info(f"[ADVISORY] {rec}")
        
        recalled = self.vortex.recall(dna.regime, top_n=10)
        
        feedback_data: Dict[str, Any] = {}
        if self.latest_feedback:
            feedback_data.update({
                "suggested_intent": self.latest_feedback.suggested_intent,
                "decision": self.latest_feedback.decision,
                "observations": self.latest_feedback.observations
            })
        if trade_analysis:
            if trade_analysis.get("critical_patterns"):
                feedback_data["dominant_errors"] = [
                    e for e, _ in trade_analysis["critical_patterns"][:3]
                ]
            feedback_data["avg_timing"] = trade_analysis.get("avg_timing_score", 0.5)
            feedback_data["avg_capture"] = trade_analysis.get("avg_capture_ratio", 0.5)
        
        hypothesis = self.hypothesis_gen.generate(dna.regime, dna, recalled, feedback_data, single_asset_mode=single_asset_mode, engine=engine)
        logger.info(f"[REASONING] Generated hypothesis: {hypothesis['primary_theme']} -> {hypothesis['secondary_theme']}")
        for step in hypothesis["reasoning_chain"]:
            logger.info(f"  {step}")
        
        candidates = self.evolver.evolve(hypothesis, recalled, self.cycle_count, single_asset_mode=single_asset_mode, engine=engine)
        logger.info(f"[EVOLUTION] Generated {len(candidates)} candidate alphas")
        
        valid_alphas = []
        for alpha in candidates:
            ok, msg = self.oracle.validate(alpha)
            if ok:
                valid_alphas.append(alpha)
                logger.info(f"  [OK] {alpha[:80]}...")
            else:
                logger.info(f"  [REJECT] {msg}: {alpha[:60]}...")
        
        if not valid_alphas:
            fallback = "RANK(DELTA($close, 5))"
            valid_alphas.append(fallback)
            logger.warning("[FALLBACK] All candidates rejected. Using baseline alpha.")
        
        ranked_alphas = []
        for alpha in valid_alphas:
            predicted_ic = self.judge.forecast_fitness(alpha, dna.regime, dna)
            # Multi-objective internal score:
            # - predicted_ic: ML estimate of quality
            # - synergy: novelty/complementarity vs memory
            # - origin_diversity_bonus: prefer ideas produced by multiple engines
            synergy = self._calculate_synergy(alpha, dna.regime)
            origin_diversity_bonus = 0.0
            try:
                origins = {
                    str(m.get("origin", ""))
                    for m in self.evolver.get_candidate_meta(alpha)
                    if isinstance(m, dict)
                }
                if len(origins) >= 2:
                    origin_diversity_bonus = 0.01
            except Exception:
                origin_diversity_bonus = 0.0

            score = (
                predicted_ic * 0.82
                + synergy * 0.12
                + origin_diversity_bonus
            )
            ranked_alphas.append((alpha, predicted_ic, score, synergy))

        ranked_alphas.sort(key=lambda x: x[2], reverse=True)
        best_alpha = ranked_alphas[0][0]
        best_predicted_ic = ranked_alphas[0][1]
        best_score = ranked_alphas[0][2]
        best_synergy = ranked_alphas[0][3]
        
        # --- INTERNAL SELECTION (no external API) ---
        logger.info(
            f"[SELECTION] Internal Selection: {best_alpha} "
            f"(predicted IC={best_predicted_ic:.4f}, score={best_score:.4f}, synergy={best_synergy:.3f})"
        )
        
        # --- INTERNAL STRATEGY ADVISORY ---
        market_summary = (f"Regime: {dna.regime}, Vol={dna.volatility:.2f}, Hurst={dna.hurst:.2f}, "
                          f"Trend={dna.trend:.2f}, Volume Surge={dna.volume_surge:.2f}")
        logger.info(f"[STRATEGY ADVISORY] {market_summary}")
        
        return {
            "alpha": best_alpha,
            "all_candidates": [a for a, _, _, _ in ranked_alphas],
            "predicted_ic": best_predicted_ic,
            "selection_score": best_score,
            "selection_synergy": best_synergy,
            "regime": dna.regime,
            "dna": dna,
            "hypothesis": hypothesis,
            "trade_analysis": trade_analysis,
            "cycle": self.cycle_count,
            "synergy_score": self._calculate_synergy(best_alpha, dna.regime)
        }

    def _calculate_synergy(self, alpha: str, regime: str) -> float:
        """
        ELITE UPGRADE: genetic complementarity check.
        Ensures new Alpha is not redundant with existing high-fitness factors in memory.
        """
        memories = self.vortex.recall(regime, top_n=5)
        if not memories:
            return 1.0 # Maximum synergy if no memory exists
            
        # Semantic synergy: penalize if AST signature matches too closely
        new_sig = set(re.findall(r'[A-Z_]+', alpha))
        synergy_scores = []
        
        for m in memories:
            m_sig = set(re.findall(r'[A-Z_]+', m.Expression))
            intersection = new_sig.intersection(m_sig)
            # Jaccard-like distance for semantic diversity
            sim = len(intersection) / len(new_sig.union(m_sig)) if new_sig.union(m_sig) else 1.0
            synergy_scores.append(1.0 - sim)
            
        return float(np.mean(synergy_scores))

    def record_experience(
        self,
        expression: str,
        regime: str,
        metrics: BacktestMetrics,
        dna: Optional[MarketDNA] = None,
    ):
        logger.info(f"[REINFORCEMENT] Recording {expression[:60]}... in {regime}")
        
        experiment_data = {
            "expression": expression,
            "metrics": {
                "ic": metrics.IC, "sharpe": metrics.Sharpe, "mdd": metrics.MDD,
                "rank_ic": metrics.RankIC, "ann_ret": metrics.AnnRet
            },
            "regime": regime
        }
        
        recalled = self.vortex.recall(regime, top_n=1)
        sota_metrics = None
        if recalled:
            s_m = recalled[0].Metrics
            sota_metrics = {"ic": s_m.IC, "sharpe": s_m.Sharpe, "mdd": s_m.MDD}
        
        feedback = self.summarizer.generate_feedback(experiment_data, sota_metrics)
        self.latest_feedback = feedback

        # Learn from both success and failure with adaptive sample weight.
        # This keeps ML grounded in real outcomes and speeds convergence.
        reward_like = (
            2.0 * float(metrics.IC)
            + 1.0 * float(metrics.RankIC)
            + 0.6 * float(metrics.ICIR)
            + 0.5 * max(0.0, float(metrics.Sharpe))
            + 0.4 * float(metrics.AnnRet)
            - 0.8 * abs(float(metrics.MDD))
        )
        sample_weight = float(np.clip(0.8 + abs(np.tanh(reward_like * 4.0)) * 2.2, 0.8, 3.0))
        try:
            self.judge.update(expression, regime, float(metrics.IC), dna=dna, sample_weight=sample_weight)
        except Exception:
            # Keep reinforcement loop resilient; failure here should not break cycle.
            pass
        
        logger.info(f"[FEEDBACK] Decision={feedback.decision} | Intent={feedback.suggested_intent}")
        if feedback.stability_warnings:
            for w in feedback.stability_warnings:
                logger.warning(f"[STABILITY] {w}")
        
        if feedback.decision:
            logger.info(f"[VORTEX] Imprinting superior alpha into {regime} memory")
            self.vortex.imprint(MemoryImpression(
                Regime=regime, Expression=expression, Metrics=metrics
            ))
            self.evolver.record_success(expression, regime=regime, ic=metrics.IC)
            self.hypothesis_gen.record_result(
                {"regime": regime, "primary_theme": "RECORDED", "secondary_theme": "RECORDED",
                 "reasoning_chain": [], "complexity_target": 20, "gene_selection": {},
                 "aggression": 0.5, "description": "recorded", "seed_tokens": [],
                 "timestamp": datetime.now().isoformat()},
                True, metrics
            )
        else:
            self.evolver.record_failure(expression, regime=regime, ic=metrics.IC)
        
        self.performance_track.append({
            "expression": expression,
            "regime": regime,
            "metrics": asdict(metrics),
            "decision": feedback.decision,
            "intent": feedback.suggested_intent,
            "timestamp": datetime.now().isoformat()
        })

    def analyze_trades(self, trades: List[Dict[str, Any]],
                      price: np.ndarray, volume: Optional[np.ndarray] = None,
                      high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None) -> Dict[str, Any]:
        dna = self.perception.sense(price, volume, high, low)
        return self.analyst.generate_session_report(trades, dna)

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.performance_track:
            return {"message": "No performance data available"}
        
        decisions = [p["decision"] for p in self.performance_track]
        ics = [p["metrics"]["IC"] for p in self.performance_track]
        
        return {
            "total_cycles": len(self.performance_track),
            "success_rate": sum(decisions) / len(decisions) if decisions else 0,
            "avg_ic": np.mean(ics),
            "max_ic": max(ics),
            "min_ic": min(ics),
            "recent_trend": np.mean(ics[-10:]) - np.mean(ics[-20:-10]) if len(ics) > 20 else 0,
            "memory_stats": {
                regime: self.vortex.get_statistics(regime)
                for regime in self.vortex.vortex.keys()
            }
        }


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("SOVA RIGHT BRAIN (Reasoning Matrix) - Standalone Test")
    logger.info("="*80)
    
    agent = SOVAExecutive()
    
    price = np.random.randn(300).cumsum() + 10000
    volume = np.random.rand(300) * 1000000
    high = price + np.random.rand(300) * 50
    low = price - np.random.rand(300) * 50
    
    result = agent.run_cognitive_cycle(price, volume, high, low)
    logger.info(f"\n[CYCLE RESULT] Alpha: {result['alpha']}")
    logger.info(f"[CYCLE RESULT] Regime: {result['regime']}")
    logger.info(f"[CYCLE RESULT] Predicted IC: {result['predicted_ic']:.4f}")
    logger.info(f"[CYCLE RESULT] Candidates: {len(result['all_candidates'])}")
    
    metrics = BacktestMetrics(IC=0.045, Sharpe=2.1, MDD=0.12)
    metrics.compute_fitness()
    agent.record_experience(result['alpha'], result['regime'], metrics)
    
    sample_trades = [
        {"direction": "LONG", "pnl": 500, "pnl_pct": 0.05, "entry_price": 10000,
         "exit_price": 10500, "mfe": 600, "mfe_pct": 0.06, "mae": -100,
         "mae_pct": -0.01, "bars": 20},
        {"direction": "SHORT", "pnl": -300, "pnl_pct": -0.03, "entry_price": 10200,
         "exit_price": 10500, "mfe": 100, "mfe_pct": 0.01, "mae": -400,
         "mae_pct": -0.04, "bars": 5},
        {"direction": "LONG", "pnl": -150, "pnl_pct": -0.015, "entry_price": 10100,
         "exit_price": 9950, "mfe": 200, "mfe_pct": 0.02, "mae": -200,
         "mae_pct": -0.02, "bars": 8},
    ]
    
    result2 = agent.run_cognitive_cycle(price, volume, high, low, trade_history=sample_trades)
    logger.info(f"\n[CYCLE 2 RESULT] Alpha: {result2['alpha']}")
    
    if result2.get("trade_analysis"):
        ta = result2["trade_analysis"]
        logger.info(f"\n[TRADE ANALYSIS SUMMARY]")
        logger.info(f"  Total: {ta['total_trades']}, Winners: {ta['winners']}, Losers: {ta['losers']}")
        logger.info(f"  Quality Distribution: {dict(ta['quality_distribution'])}")
        logger.info(f"  Avg Timing: {ta['avg_timing_score']:.2f}")
        logger.info(f"  Avg Capture: {ta['avg_capture_ratio']:.2f}")
        
        if ta.get("critical_patterns"):
            logger.info(f"  Critical Patterns: {ta['critical_patterns']}")
        
        for rec in ta.get("recommendations", []):
            logger.info(f"  RECOMMENDATION: {rec}")
    
    summary = agent.get_performance_summary()
    logger.info(f"\n[PERFORMANCE] {json.dumps(summary, indent=2, default=str)}")
    
    logger.info("="*80)
    logger.info("SOVA RIGHT BRAIN Test Complete")
    logger.info("="*80)
