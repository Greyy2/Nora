#!/usr/bin/env python3
"""
Stock Alpha Engine
Self-contained engine for cross-sectional stock alpha generation and evaluation.
Uses adaptive normalization (RANK by default), ICIR backtest, and stock-specific axioms.
"""
import os
import re
import random
import hashlib
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from Sova.engines.base_engine import AlphaEngine

logger = logging.getLogger("SOVA.StockEngine")

# ═══════════════════════════════════════════════════════════════
# STOCK-SPECIFIC AXIOMS (reasoning context for hypothesis generation)
# ═══════════════════════════════════════════════════════════════
STOCK_AXIOMS = {
    "MEAN_REVERSION": [
        "In high-entropy noise, price extremes revert to the synaptic baseline.",
        "Kurtosis peaks precede rapid mean-reversion snapbacks.",
        "Spectral energy concentration in low-frequency bands signals structural support.",
        "Multi-timeframe divergence between fast/slow oscillators triggers rebalancing.",
        "Fractal dimension collapse indicates oversold compression.",
        "Z-score extremes beyond 2.5 sigma have 78% reversion probability within 5 bars."
    ],
    "MOMENTUM_FLOW": [
        "In persistent regimes, volume surges confirm trend propagation.",
        "Hurst exponent > 0.6 validates momentum continuation strategies.",
        "Acceleration in volume-price correlation confirms institutional flow.",
        "Price-volume divergence at extremes signals exhaustion.",
        "Sequential breakout patterns amplify when fractal dimension < 1.3.",
        "Wavelet decomposition reveals hidden momentum at scale 4-8."
    ],
    "LIQUIDITY_DYNAMICS": [
        "Volume-weighted price divergence reveals institutional stealth activity.",
        "Order flow imbalance precedes price impact by 3-7 bars.",
        "Bid-ask spread expansion measures liquidity withdrawal velocity.",
        "Volume clustering at support-resistance reveals institutional intent.",
        "Microstructure efficiency drops precede regime transitions."
    ],
    "REGIME_TRANSITION": [
        "Volatility compression precedes explosive directional moves.",
        "Cross-timeframe divergence signals regime instability.",
        "Entropy peaks mark the boundary between order and chaos.",
        "Volume surge without price follow-through signals distribution.",
        "Hurst exponent crossing 0.5 boundary marks trend-reversal pivot.",
        "Fractal dimension spikes precede structural market shifts."
    ],
    "RISK_MANAGEMENT": [
        "Position sizing must adapt to realized volatility regime.",
        "Drawdown velocity exceeding historical norms signals tail risk.",
        "Correlation breakdown in stressed markets demands hedging escalation.",
        "Sequential losing trades indicate regime misclassification.",
        "Win-rate degradation below 40% triggers strategic reassessment.",
        "Maximum adverse excursion patterns reveal stop-loss optimization zones."
    ]
}

# ═══════════════════════════════════════════════════════════════
# STOCK GENE REGISTRY (cross-sectional operators for multi-instrument)
# ═══════════════════════════════════════════════════════════════
STOCK_GENE_REGISTRY = {
    "UNARY": ["LOG", "ABS", "SIGN", "SQRT", "INV", "EXP"],
    "BINARY": ["DELAY", "DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SKEW", "TS_KURT",
               "TS_SUM", "TS_MEDIAN", "TS_ARGMAX", "TS_ARGMIN", "EMA", "SMA", "WMA", "TS_RANK"],
    "TRINARY": ["TS_CORR", "TS_COVARIANCE"],
    "CROSS_SECTIONAL": ["RANK"],  # Stock-only! 
    "VAR": ["$close", "$open", "$high", "$low", "$volume", "$vwap"],
    "ARITHMETIC": ["+", "-", "*", "/"],
    "DERIVED": [
        "$close - $open", "$high - $low", "($close - $low) / ($high - $low + 1e-8)",
        "$close / $open - 1", "($high + $low) / 2", "$volume * $close"
    ]
}

# ═══════════════════════════════════════════════════════════════
# STOCK STRATEGY MAP (regime → strategy for cross-sectional alpha)
# ═══════════════════════════════════════════════════════════════
STOCK_STRATEGY_MAP = {
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
        "operators": ["RANK", "TS_STD", "TS_MEAN", "DELTA"],
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
        "description": "Balanced state: monitor both momentum and reversion cues"
    }
}


class StockAlphaEngine(AlphaEngine):
    """
    Cross-sectional stock alpha engine.
    Uses adaptive normalization with cross-sectional RANK() as default.
    Evaluates alphas via ICIR (Information Coefficient Information Ratio).
    """
    
    @property
    def name(self) -> str:
        return "STOCK"
    
    def get_axioms(self) -> Dict[str, List[str]]:
        return STOCK_AXIOMS
    
    def get_gene_registry(self) -> Dict[str, List[str]]:
        return STOCK_GENE_REGISTRY
    
    def get_strategy_map(self) -> Dict[str, Dict[str, Any]]:
        return STOCK_STRATEGY_MAP

    @staticmethod
    def _expression_scale_stable(expr: str) -> bool:
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

    @staticmethod
    def _norm_policy() -> str:
        policy = str(os.environ.get("SOVA_NORMALIZATION_POLICY", "adaptive") or "adaptive").strip().lower()
        return policy if policy in {"adaptive", "forced", "raw"} else "adaptive"
    
    def wrap_expression(self, expr: str) -> str:
        """Stock wrapping with adaptive normalization policy."""
        e = str(expr or "").strip()
        if not e:
            return e
        if e.startswith("RANK("):
            return e

        policy = self._norm_policy()
        if policy == "raw":
            return e
        if policy == "forced":
            return f"RANK({e})"

        # Adaptive policy: keep structurally normalized expressions raw.
        if self._expression_scale_stable(e):
            return e
        return f"RANK({e})"
    
    def zscore_expression(self, expr: str, window: int) -> str:
        return f"(({expr} - TS_MEAN({expr}, {window})) / (TS_STD({expr}, {window}) + 1e-8))"
    
    def get_mutation_wraps(self) -> List[str]:
        return ["RAW", "RANK", "ABS", "SIGN"]
    
    def mutate_wrap(self, expression: str, wrap_op: str) -> str:
        if wrap_op == "RAW":
            return expression
        if not expression.startswith(f"{wrap_op}("):
            return f"{wrap_op}({expression})"
        return expression
    
    def load_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Load CSI300 stock data via Qlib."""
        import qlib
        from qlib.data import D
        from dotenv import load_dotenv
        
        load_dotenv()
        project_root = Path(__file__).resolve().parents[2]
        provider_uri = os.environ.get("QLIB_DATA_DIR", 
                                       str(project_root / "QuantaAlpha/data/qlib_data/cn_data"))
        
        if not Path(provider_uri).exists():
            raise FileNotFoundError(f"Qlib data not found: {provider_uri}")
        
        qlib.init(provider_uri=provider_uri, region="cn")
        logger.info(f"[STOCK] Qlib initialized: {provider_uri}")
        
        try:
            instruments = D.instruments(market='csi300')
            # Expanded window: 2018-2021 for better IC differentiation
            price_df = D.features(instruments, ['$close', '$volume', '$high', '$low', '$open'], 
                                 start_time="2018-01-01", end_time="2021-12-31")
        except Exception:
            price_df = D.features(['SH600000'], ['$close', '$volume', '$high', '$low', '$open'], 
                                 start_time="2018-01-01", end_time="2021-12-31")
        
        n_instruments = len(price_df.index.get_level_values('instrument').unique())
        logger.info(f"[STOCK] Loaded {n_instruments} instruments, {len(price_df)} rows")
        
        return {
            "price": price_df['$close'].values,
            "volume": price_df['$volume'].values,
            "high": price_df['$high'].values,
            "low": price_df['$low'].values,
            "df": price_df
        }
    
    def run_backtest(self, factor: pd.Series, df: pd.DataFrame, expression: str) -> Any:
        """
        ICIR Backtest: Cross-sectional IC calculation with daily IC Sharpe.
        Integrated with GovernanceOracle for Semantic Deduplication.
        """
        from Sova.sova_matching_learning import BacktestMetrics, GovernanceOracle
        
        # SEMANTIC VALIDATION
        oracle = GovernanceOracle()
        valid, msg = oracle.validate(expression, factor.values)
        if not valid:
            logger.warning(f"[STOCK] Semantic rejection for {expression[:30]}: {msg}")
            return BacktestMetrics(IC=0.0, Sharpe=0.0, MDD=0.1, Novelty=0.0)
        
        labels = df['$open'].groupby('instrument').transform(lambda x: x.shift(-2) / x.shift(-1) - 1)
        combined = pd.concat([factor.rename('factor'), labels.rename('label')], axis=1).dropna()
        
        if combined.empty or len(combined) < 50:
            return BacktestMetrics(IC=0.0, Sharpe=0.0, MDD=0.1)
        
        # IC = Pearson correlation (NaN-safe)
        ic = float(combined['factor'].corr(combined['label']))
        if np.isnan(ic): ic = 0.0
        
        # Rank IC (more robust, non-parametric)
        rank_ic = float(combined['factor'].rank().corr(combined['label'].rank()))
        if np.isnan(rank_ic): rank_ic = 0.0
        
        # Daily IC series for ICIR (Sharpe of IC)
        if 'datetime' in combined.index.names:
            daily_ic = combined.groupby('datetime').apply(
                lambda x: x['factor'].corr(x['label']) if len(x) > 5 else np.nan
            ).dropna()
            ic_mean = float(daily_ic.mean()) if len(daily_ic) > 0 else ic
            ic_std = float(daily_ic.std()) if len(daily_ic) > 1 else 0.01
        else:
            ic_mean = ic
            ic_std = 0.01
        
        ic_sharpe = float(ic_mean / (ic_std + 1e-8) * np.sqrt(252))
        
        metrics = BacktestMetrics(
            ID=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            IC=ic, RankIC=rank_ic, 
            Sharpe=ic_sharpe, 
            MDD=0.1, 
            AnnRet=ic * 2,
            Complexity=len(re.findall(r'[A-Z_]+', str(expression)))
        )
        metrics.compute_fitness()
        return metrics
