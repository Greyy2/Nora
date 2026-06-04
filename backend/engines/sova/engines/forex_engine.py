#!/usr/bin/env python3
"""
Forex Alpha Engine (Unified Cross-Sectional)
Treats Forex markets using a Virtual Cross-Sectional approach (multi-timeframe).
Uses adaptive normalization (TS_RANK by default) and IC Rank evaluation across timeframes.
"""
import os
import re
import hashlib
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from Sova.engines.base_engine import AlphaEngine

logger = logging.getLogger("SOVA.ForexEngine")

# ═══════════════════════════════════════════════════════════════
# UNIFIED FOREX AXIOMS (Including Cross-Scale Dynamics)
# ═══════════════════════════════════════════════════════════════
FOREX_AXIOMS = {
    "MEAN_REVERSION": [
        "In high-entropy noise, price extremes revert to the synaptic baseline.",
        "Kurtosis peaks precede rapid mean-reversion snapbacks.",
        "Multi-timeframe divergence between fast/slow oscillators triggers rebalancing.",
        "Z-score extremes beyond 2.5 sigma have 78% reversion probability within 5 bars.",
        "Session VWAP acts as an intraday attractor for mean-reversion signals.",
        "Bollinger Band width compression precedes directional breakouts."
    ],
    "MOMENTUM_FLOW": [
        "In persistent regimes, volume surges confirm trend propagation.",
        "Hurst exponent > 0.6 validates momentum continuation strategies.",
        "Acceleration in volume-price correlation confirms institutional flow.",
        "Price-volume divergence at extremes signals exhaustion.",
        "Sequential breakout patterns amplify when fractal dimension < 1.3.",
        "Higher-timeframe trend alignment strengthens intrabar momentum signals."
    ],
    "VOLATILITY_DYNAMICS": [
        "Volatility clustering follows GARCH(1,1) dynamics in FX markets.",
        "ATR expansion during London open signals trending conditions.",
        "Realized vs implied volatility spread indicates market mispricing.",
        "Volatility compression below 20-period low precedes breakouts.",
        "Session transition (Asian→London→NY) volatility regime shifts are predictable.",
        "High-low range contraction for 3+ bars signals imminent expansion."
    ],
    "CROSS_SCALE_DYNAMICS": [
        "Lower timeframe trends (1H) are noise until validated by higher timeframe (4H) structure.",
        "Divergence between M15 momentum and H4 trend indicates imminent mean reversion.",
        "Fractional scale alignment (H1, 4H, 1D all trending) precedes target-rich liquidity zones.",
        "Volatility compression on D1 preceded by expansion on H1 signals structural breakout.",
        "Rank-extremes in 1H momentum within a D1 mean-reversion zone identify precision entries.",
        "Volume flow across scales reveals institutional accumulation levels."
    ],
    "RISK_MANAGEMENT": [
        "Position sizing must adapt to realized volatility regime.",
        "Spread widening during illiquid hours (Asian close) increases execution risk.",
        "Maximum adverse excursion patterns reveal stop-loss optimization zones.",
        "Drawdown velocity exceeding historical norms signals tail risk.",
        "Win-rate degradation below 40% triggers strategic reassessment."
    ]
}

# ═══════════════════════════════════════════════════════════════
# CROSS-SECTIONAL GENE REGISTRY (Includes RANK for multi-timeframe)
# ═══════════════════════════════════════════════════════════════
FOREX_GENE_REGISTRY = {
    "UNARY": ["LOG", "ABS", "SIGN"],
    "BINARY": ["DELAY", "DELTA", "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN",
               "TS_SKEW", "TS_KURT", "TS_SUM", "TS_RANK", "EMA", "SMA"],
    "TRINARY": ["TS_CORR", "TS_COVARIANCE"],
    "CROSS_SECTIONAL": ["RANK"],  # Enabled for virtual universe logic
    "VAR": ["$close", "$open", "$high", "$low", "$volume"],
    "ARITHMETIC": ["+", "-", "*", "/"],
    "DERIVED": [
        "$close - $open",
        "$high - $low",
        "($close - $low) / ($high - $low + 1e-8)",
        "$close / $open - 1"
    ]
}

FOREX_STRATEGY_MAP = {
    "CHAOTIC_NOISE": {
        "primary": "MEAN_REVERSION",
        "secondary": "CROSS_SCALE_DYNAMICS",
        "operators": ["RANK", "TS_RANK", "DELTA", "TS_STD"],
        "windows": [5, 10, 20, 60],
        "aggression": 0.3,
        "description": "Virtual Cross-Section: Using 1D/4H context to filter H1 noise."
    },
    "EXPONENTIAL_BULL": {
        "primary": "MOMENTUM_FLOW",
        "secondary": "CROSS_SCALE_DYNAMICS",
        "operators": ["RANK", "DELTA", "TS_CORR", "EMA"],
        "windows": [5, 10, 20, 60],
        "aggression": 0.7,
        "description": "Scale alignment: hunting for H1 breakouts validated by larger scales."
    },
    "NEUTRAL": {
        "primary": "CROSS_SCALE_DYNAMICS",
        "secondary": "MEAN_REVERSION",
        "operators": ["RANK", "TS_MEAN", "TS_STD"],
        "windows": [10, 20, 60],
        "aggression": 0.4,
        "description": "Relative scale strength analysis."
    }
}

FOREX_COST_MODEL = {
    "xau": 0.00005,
    "btc": 0.0001,
    "forex": 0.00003,
}

class ForexAlphaEngine(AlphaEngine):
    """
    Unified Forex Engine.
    Operates on a 'Virtual Universe' of multiple timeframes (H1, 4H, 1D).
    Uses adaptive normalization with TS_RANK() as default and IC Rank as core metric.
    """
    def __init__(self, instrument: str = "XAUUSD", market: str = "xau"):
        self.instrument = instrument
        self.market = market
        self.trading_resolution = "1H"

    @property
    def name(self) -> str:
        return f"FOREX_CS/{self.instrument}"

    def get_axioms(self) -> Dict[str, List[str]]:
        return FOREX_AXIOMS

    def get_gene_registry(self) -> Dict[str, List[str]]:
        return FOREX_GENE_REGISTRY

    def get_strategy_map(self) -> Dict[str, Dict[str, Any]]:
        return FOREX_STRATEGY_MAP

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
        """Forex wrapping with adaptive normalization policy."""
        e = str(expr or "").strip()
        if not e:
            return e
        if e.startswith("TS_RANK("):
            return e

        policy = self._norm_policy()
        prefer_mode = str(os.environ.get("SOVA_FOREX_WRAP_MODE", "TS_RANK") or "TS_RANK").strip().upper()
        if prefer_mode not in {"TS_RANK", "RANK"}:
            prefer_mode = "TS_RANK"

        if policy == "raw":
            return e

        if e.startswith("RANK("):
            if policy == "forced" and prefer_mode == "TS_RANK":
                inner = re.sub(r'^RANK\((.*)\)$', r'\1', e)
                return f"TS_RANK({inner}, 60)"
            return e

        if policy == "forced":
            return f"TS_RANK({e}, 60)" if prefer_mode == "TS_RANK" else f"RANK({e})"

        # Adaptive policy: keep structurally normalized expressions raw.
        if self._expression_scale_stable(e):
            return e
        return f"TS_RANK({e}, 60)" if prefer_mode == "TS_RANK" else f"RANK({e})"

    def zscore_expression(self, expr: str, window: int) -> str:
        return f"(({expr} - TS_MEAN({expr}, {window})) / (TS_STD({expr}, {window}) + 1e-8))"

    def get_mutation_wraps(self) -> List[str]:
        return ["RAW", "RANK", "TS_RANK", "ABS", "SIGN"]

    def mutate_wrap(self, expression: str, wrap_op: str) -> str:
        if wrap_op == "RAW":
            return expression
        if wrap_op == "TS_RANK":
            return f"TS_RANK({expression}, 60)" if not expression.startswith("TS_RANK(") else expression
        return f"{wrap_op}({expression})" if not expression.startswith(f"{wrap_op}(") else expression

    def load_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Loads and aligns H1, 4H, 1D data into a virtual universe."""
        if not data_path:
            raise ValueError("Forex engine requires data_path")
        
        base_dir = Path(data_path).parent
        # Look for standard resolution files based on the base file provided
        file_patterns = {"1H": "_1h.csv", "4H": "_4h.csv", "1D": "_1d.csv"}
        
        # Determine prefix (e.g., xauusd_mid)
        base_name = Path(data_path).name.lower()
        prefix = base_name.split('_1h.csv')[0] if '_1h.csv' in base_name else \
                 base_name.split('_4h.csv')[0] if '_4h.csv' in base_name else \
                 base_name.split('_1d.csv')[0] if '_1d.csv' in base_name else base_name.replace('.csv', '')

        all_dfs = []
        for tf, suffix in file_patterns.items():
            path = base_dir / (prefix + suffix)
            if not path.exists():
                # Try fallback: market prefix + resolution
                path = base_dir / (self.market + "usd_mid" + suffix)
                if not path.exists(): continue
                
            df_tf = pd.read_csv(path)
            df_tf.columns = [c.lower() for c in df_tf.columns]
            ts = df_tf['timestamp'].iloc[0] if 'timestamp' in df_tf.columns else None
            df_tf['datetime'] = pd.to_datetime(df_tf['timestamp'], unit='ms' if ts and ts > 1e11 else 's') if ts else pd.to_datetime(df_tf['date'])
            df_tf['instrument'] = tf
            df_tf.set_index(['datetime', 'instrument'], inplace=True)
            df_tf.columns = [f"${c}" if not c.startswith("$") else c for c in df_tf.columns]
            all_dfs.append(df_tf)
            
        if not all_dfs:
            # Last resort: load just the provided file
            df_base = pd.read_csv(data_path)
            df_base.columns = [f"${c.lower()}" if not c.startswith("$") else c.lower() for c in df_base.columns]
            df_base['datetime'] = pd.to_datetime(df_base['$timestamp'], unit='s') # heuristic
            df_base['instrument'] = '1H'
            df_base.set_index(['datetime', 'instrument'], inplace=True)
            all_dfs.append(df_base)

        combined_df = pd.concat(all_dfs).sort_index()
        
        # --- CRITICAL FIX: Align timeframes via ffill ---
        # Without this, RANK() across scales at H1 timestamps would only see H1 data
        # as 4H/1D timestamps only occur periodically.
        combined_df = combined_df.unstack().ffill().stack().sort_index()
        
        h1_data_list = combined_df.index.get_level_values('instrument').unique()
        primary_tf = '1H' if '1H' in h1_data_list else h1_data_list[0]
        h1_df = combined_df.xs(primary_tf, level='instrument')
        
        logger.info(f"[FOREX_CS] Unified Engine: Virtual universe {list(h1_data_list)} aligned")
        
        return {
            "price": h1_df['$close'].values,
            "volume": h1_df['$volume'].values if '$volume' in h1_df.columns else np.ones(len(h1_df)),
            "high": h1_df['$high'].values if '$high' in h1_df.columns else h1_df['$close'].values,
            "low": h1_df['$low'].values if '$low' in h1_df.columns else h1_df['$close'].values,
            "df": combined_df
        }

    def run_backtest(self, factor: pd.Series, df: pd.DataFrame, expression: str) -> Any:
        """
        Backtests on the Virtual Cross-Sectional universe.
        Integrated with GovernanceOracle for Semantic Deduplication.
        """
        from Sova.sova_matching_learning import BacktestMetrics, GovernanceOracle
        
        # SEMANTIC VALIDATION
        oracle = GovernanceOracle()
        valid, msg = oracle.validate(expression, factor.values)
        if not valid:
            logger.warning(f"[FOREX] Semantic rejection for {expression[:30]}: {msg}")
            return BacktestMetrics(IC=0.0, Sharpe=0.0, MDD=0.1, Novelty=0.0)
        
        df = df.copy()
        df['ret'] = df.groupby('instrument')['$open'].transform(lambda x: x.shift(-2) / x.shift(-1) - 1).fillna(0)
        
        # 1. Unified IC Rank across all scales
        valid_mask = factor.notna() & df['ret'].notna()
        if valid_mask.sum() < 100:
            return BacktestMetrics(IC=0.0, Sharpe=0.0, MDD=0.1)
            
        ic = float(factor[valid_mask].corr(df['ret'][valid_mask]))
        rank_ic = float(factor[valid_mask].rank().corr(df['ret'][valid_mask].rank()))
        
        # 2. PnL Simulation on Trading Scale
        tfs = df.index.get_level_values('instrument').unique()
        trading_tf = '1H' if '1H' in tfs else tfs[0]
        
        h1_factor = factor.xs(trading_tf, level='instrument')
        h1_df = df.xs(trading_tf, level='instrument')
        
        roll_mean, roll_std = h1_factor.rolling(50, min_periods=10).mean(), h1_factor.rolling(50, min_periods=10).std()
        signal = ((h1_factor - roll_mean) / (roll_std + 1e-6)).clip(-3, 3) / 3.0
        
        cost_per_trade = FOREX_COST_MODEL.get(self.market, 0.00005)
        strategy_returns = (signal * h1_df['ret'].values) - (signal.diff().abs().fillna(0) * cost_per_trade)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) < 100:
            return BacktestMetrics(IC=ic, Sharpe=0.0, MDD=0.1)
            
        ann_factor = 252 * 24 if trading_tf == '1H' else 252 * 6 if trading_tf == '4H' else 252
        ret_mean, ret_std = float(strategy_returns.mean()), float(strategy_returns.std())
        sharpe = float(ret_mean / (ret_std + 1e-8) * np.sqrt(ann_factor)) if ret_std > 1e-9 else 0.0
        sharpe = np.clip(sharpe, -10, 10)
        
        cum_ret = (1 + strategy_returns).cumprod()
        mdd = float(((cum_ret.expanding().max() - cum_ret) / (cum_ret.expanding().max() + 1e-8)).max())
        
        metrics = BacktestMetrics(
            ID=hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            IC=ic, RankIC=rank_ic,
            Sharpe=sharpe, MDD=mdd,
            AnnRet=float(strategy_returns.mean() * ann_factor),
            Complexity=len(re.findall(r'[A-Z_]+', str(expression)))
        )
        metrics.compute_fitness()
        return metrics
