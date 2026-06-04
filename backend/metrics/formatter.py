"""
Metrics Formatter - Centralized formatting and sanitization for all backtest results
Ensures 100% parity between Single Backtest and Optimizer metrics and handles JSON compatibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def sanitize_value(value):
    """Convert NaN/Inf to None for JSON compatibility"""
    try:
        # Preserve booleans (bool is a subclass of int in Python)
        if isinstance(value, bool):
            return value
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    except:
        return None

def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize dictionary values"""
    clean = {}
    for k, v in d.items():
        if isinstance(v, dict):
            clean[k] = sanitize_dict(v)
        elif isinstance(v, list):
            clean[k] = [sanitize_dict(i) if isinstance(i, dict) else sanitize_value(i) for i in v]
        else:
            if isinstance(v, bool):
                clean[k] = v
            else:
                clean[k] = sanitize_value(v) if isinstance(v, (int, float, np.number)) else v
    return clean

from typing import Dict, Any, Optional
from utils.helpers import safe_float, safe_int  # Centralized helpers

def format_comprehensive_metrics(params: Dict[str, Any], metrics: Dict[str, Any], initial_capital: float) -> Dict[str, Any]:
    """
    Format comprehensive metrics matching frontend expectations (camelCase).
    Ensures all metrics have valid values for stable display.
    """
    # Extract max drawdown values
    max_drawdown_abs = safe_float(metrics.get('max_drawdown', 0.0), 0.0)
    max_drawdown_pct = safe_float(metrics.get('max_drawdown_pct', 0.0), 0.0)
    
    # Final equity handling
    final_equity = safe_float(metrics.get('final_equity') or metrics.get('total_profit', initial_capital), initial_capital)
    
    # Strategy parameters extraction
    strategy_cfg = params.get('strategy', {})
    ps_cfg = strategy_cfg.get('ps', {})
    bse_cfg = strategy_cfg.get('bse', {})

    formatted = {
        # Core Performance
        'totalTrades': safe_int(metrics.get('total_trades', 0), 0),
        'winRate': safe_float(metrics.get('win_rate', 0.0), 0.0),
        'finalEquity': final_equity,
        'roi': safe_float(metrics.get('roi', 0.0), 0.0),
        'profit': safe_float(metrics.get('total_pnl') or metrics.get('profit', 0.0), 0.0),
        
        # Risk Metrics
        'sharpe': safe_float(metrics.get('sharpe') or metrics.get('sharpe_ratio', 0.0), 0.0),
        'sortino': safe_float(metrics.get('sortino') or metrics.get('sortino_ratio', 0.0), 0.0),
        'maxDrawdown': max_drawdown_abs,
        'mdd': abs(max_drawdown_pct), # Frontend expects 'mdd' for pct
        
        # Trade Statistics
        'profitFactor': safe_float(metrics.get('profit_factor', 0.0), 0.0),
        'expectancy': safe_float(metrics.get('expectancy', 0.0), 0.0),
        'cagr': safe_float(metrics.get('cagr', 0.0), 0.0),
        'winCount': safe_int(metrics.get('win_count', 0), 0),
        'lossCount': safe_int(metrics.get('loss_count', 0), 0),
        'avgWin': safe_float(metrics.get('avg_win', 0.0), 0.0),
        'avgLoss': safe_float(metrics.get('avg_loss', 0.0), 0.0),
        'largestWin': safe_float(metrics.get('largest_win', 0.0), 0.0),
        'largestLoss': safe_float(metrics.get('largest_loss', 0.0), 0.0),
        
        # Cost Metrics
        'commissionPaid': safe_float(metrics.get('commission_paid', 0.0), 0.0),
        'commissionPct': safe_float(params.get('commission_pct') or metrics.get('commission_pct', 0.0), 0.0),
        # Optimizer stores slippage in campaign-level broker config; we also inject it into params for parity.
        'skid': safe_float(params.get('slippage_pct') or params.get('skid') or metrics.get('skid', 0.0), 0.0),
        
        # Config Parameters
        'timeframe': params.get('timeframe') or params.get('frequency') or '',
        'frequency': params.get('frequency') or params.get('timeframe') or '',
        'contract': safe_float(params.get('multiple'), 0.0) if params.get('multiple') is not None else 0.0,
        
        # Strategy Parameters
        'ema': safe_int(params.get('length_ema'), 0),
        'atr': safe_int(params.get('length_atr'), 0),
        'highVf': safe_float(params.get('long_vol_factor') or params.get('high_vf'), 0.0),
        'lowVf': safe_float(params.get('short_vol_factor') or params.get('low_vf'), 0.0),
        'ir': safe_float(ps_cfg.get('ir'), 0.0) if ps_cfg.get('ir') is not None else 0.0,
        'er': safe_float(ps_cfg.get('er'), 0.0) if ps_cfg.get('er') is not None else 0.0,
        'or': safe_float(ps_cfg.get('or'), 0.0) if ps_cfg.get('or') is not None else 0.0,
        # Backward-compatible numeric field: represents OR value (not boolean)
        'onGoing': safe_float(ps_cfg.get('or'), 0.0) if ps_cfg.get('or') is not None else 0.0,

        # Explicit boolean fields (used by verification)
        'isOnGoing': bool(bse_cfg.get('is_on_going', False)),
        'useDelta': bool(params.get('use_delta', True)),
        'side': bse_cfg.get('side', 'long'),

        # Status
        'status': 'completed',
        'dateRange': metrics.get('date_range', ''),
        
        # Survival Metrics (New)
        'maxLeverage': safe_float(metrics.get('max_leverage', 0.0), 0.0),
        'consecutiveLosses': safe_int(metrics.get('max_consecutive_losses', 0), 0),
        'ddDuration': safe_int(metrics.get('max_drawdown_duration', 0), 0),
        'survivalRating': 0.0 # Placeholder for complex rating logic
    }
    
    # Merge RAW metrics with FORMATTED metrics to ensure 100% parity
    return {**metrics, **formatted}
