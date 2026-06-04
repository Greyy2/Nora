"""
Backtest Service - Business logic layer

Responsibilities:
1. Validate backtest parameters
2. Call broker
3. Format response for API (RAW DATA ONLY)
4. Error handling
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from core.broker import Broker
from metrics.formatter import format_comprehensive_metrics, sanitize_dict
from services.chart_service import _sanitize_value
from utils.helpers import safe_float, safe_int  # Centralized helpers


def run_backtest(
    asset: str,
    timeframe: str,
    initial_capital: float = 1000000.0,
    commission: float = 0.1,
    slippage_pct: float = 0.0,
    risk_per_trade_pct: float = 0.02,
    max_risk_equity_pct: float = 0.50,
    ema_length: int = 50,
    atr_length: int = 14,
    multiplier: float = 2.0,
    long_vol_factor: float = 2.0,
    short_vol_factor: float = 2.0,
    trade_option: str = 'Both',
    is_on_going: bool = True,
    on_going_risk: float = 0.95,  # Must match optimizer default
    use_delta: Optional[bool] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_type: str = "OKX"
) -> Dict[str, Any]:
    """
    Run backtest and return RAW results (trades + metrics + equity curve)
    
    Returns:
        {
            "success": true,
            "data": {
                "trades": [...],  # Raw trade objects
                "metrics": {...},  # Raw metrics from broker
                "equity_curve": [...]  # Raw equity curve
            }
        }
    """

    try:
        # 1. Validate params
        _validate_params(asset, timeframe, initial_capital, ema_length, atr_length)

        # Handle empty strings which might come from "Full Range" button
        if start_date == "": start_date = None
        if end_date == "": end_date = None
        
        # Construct strategy_params dict required by Broker
        strategy_params = {
            'length_ema': ema_length,
            'length_atr': atr_length,
            'timeframe': timeframe,  # Pass timeframe for metrics calculation
            'long_vol_factor': long_vol_factor if long_vol_factor is not None else multiplier,
            'short_vol_factor': short_vol_factor if short_vol_factor is not None else multiplier,
            'multiple': multiplier,
            'strategy': {
                'ps': {
                    'ir': risk_per_trade_pct,
                    'er': max_risk_equity_pct,
                    'or': on_going_risk
                },
                'bse': {
                    'is_on_going': is_on_going,
                    'side': {
                        'Both': 'both',
                        'Long Only': 'long',
                        'Short Only': 'short'
                    }.get(trade_option, 'both'),
                    'trade_option': trade_option 
                }
            },
            'use_delta': True if use_delta is None else bool(use_delta),
            'commission_pct': commission
        }

        # 2. Run backtest
        broker = Broker(
            initial_capital=initial_capital,
            commission_pct=commission,
            slippage_pct=slippage_pct,  # Pass dynamic slippage
            strategy_params=strategy_params,
            data_dir=data_type
        )

        result = broker.run_backtest(
            asset=asset,
            timeframe=timeframe,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date
        )

        # 3. Format response - RAW DATA ONLY
        trades_list = _format_trades_raw(broker.trades)
        equity_curve_list = _format_equity_curve_raw(result['equity_curve'])
        
        # Format detailed results (TradingView style)
        detailed_results = _format_detailed_results(broker, result['metrics'], broker.trades, strategy_params)
        
        # Comprehensive metrics (Centralized Formatter)
        comp_metrics = format_comprehensive_metrics(strategy_params, result['metrics'], initial_capital)
        
        # Sanitize metrics for JSON
        final_metrics = sanitize_dict(comp_metrics)
        
        return {
            'success': True,
            'data': {
                'trades': trades_list,
                'list_of_trades': trades_list,
                'metrics': final_metrics,
                'equity_curve': equity_curve_list,
                
                # detailed sections
                'overview': detailed_results['overview'],
                'performance': detailed_results['performance'],
                'trades_analyst': detailed_results['trades_analyst'],
                'capital_efficiency': detailed_results['capital_efficiency'],
                'runups_drawdowns': detailed_results['runups_drawdowns']
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }



# Removed local sanitization logic, using metrics.formatter.sanitize_dict

def _format_trades_raw(trades: List) -> List[Dict[str, Any]]:
    """
    Convert Trade objects to raw dict format for API
    """
    trades_list = []
    
    for i, trade in enumerate(trades, 1):
        t = trade.to_dict() if hasattr(trade, 'to_dict') else dict(trade)
        
        # Format timestamps to ISO 8601
        entry_time = t.get('entry_time')
        exit_time = t.get('exit_time')
        
        if hasattr(entry_time, 'isoformat'):
            entry_time = entry_time.isoformat().replace('+00:00', 'Z')
            if 'Z' not in entry_time and '+' not in entry_time:
                entry_time += 'Z'
        
        if hasattr(exit_time, 'isoformat'):
            exit_time = exit_time.isoformat().replace('+00:00', 'Z')
            if 'Z' not in exit_time and '+' not in exit_time:
                exit_time += 'Z'
            
        # Calculate position size in USDT
        entry_price = _sanitize_value(t.get('entry_price'))
        quantity = _sanitize_value(t.get('quantity'))
        pos_size = entry_price * quantity if entry_price and quantity else 0
        
        
        trades_list.append({
            'id': i,
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'direction': str(t.get('direction', '')).capitalize(),
            'entry_price': entry_price,
            'exit_price': _sanitize_value(t.get('exit_price')),
            'quantity': quantity,
            'pos_size': pos_size,
            'pnl': _sanitize_value(t.get('pnl')),
            'net_pnl': _sanitize_value(t.get('net_pnl', t.get('pnl'))),
            'pnl_pct': _sanitize_value(t.get('pnl_pct')),
            'commission': _sanitize_value(t.get('commission')),
            'mfe': _sanitize_value(t.get('mfe')),
            'mae': _sanitize_value(t.get('mae')),
            'mfe_pct': _sanitize_value(t.get('mfe_pct')),
            'mae_pct': _sanitize_value(t.get('mae_pct')),
            'cumulative_pnl': _sanitize_value(t.get('cumulative_pnl')),
            'exit_reason': str(t.get('exit_reason', '')),
            'bars': int(t.get('bars', 0)),
            
            # OR (On-going Risk) metadata
            'or_risk_before': _sanitize_value(t.get('or_risk_before')),
            'or_risk_after': _sanitize_value(t.get('or_risk_after')),
            'or_contracts_before': _sanitize_value(t.get('or_contracts_before')),
            'or_contracts_after': _sanitize_value(t.get('or_contracts_after')),
            'or_unrealized_pnl': _sanitize_value(t.get('or_unrealized_pnl')),
            'or_on_going_equity': _sanitize_value(t.get('or_on_going_equity')),
            'equity_after_exit': _sanitize_value(t.get('equity_after_exit'))
        })
    
    return trades_list


def _format_equity_curve_raw(equity_curve: List) -> List[Dict[str, Any]]:
    """
    Convert equity curve to raw dict format for API
    """
    equity_list = []
    
    for point in equity_curve:
        timestamp = point.get('timestamp')
        
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        
        equity_list.append({
            'time': str(timestamp),
            'equity': _sanitize_value(point.get('equity'))
        })
    
    return equity_list


def _calculate_long_short_capital_metrics(trades, initial_capital, metrics):
    """Calculate capital efficiency metrics separately for Long and Short trades"""
    from models.trade import TradeDirection
    
    # Filter trades by direction
    long_trades = [t for t in trades if t.direction == TradeDirection.LONG]
    short_trades = [t for t in trades if t.direction == TradeDirection.SHORT]
    
    def calc_metrics_for_direction(direction_trades):
        if not direction_trades:
            return {
                'cagr': 0,
                'return_on_initial_capital': 0,
                'net_profit_pct_of_largest_loss': 0
            }
        
        # Calculate total PnL for this direction
        total_pnl = sum(t.pnl for t in direction_trades)
        
        # Return on initial capital
        return_on_initial = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        
        # Find largest loss for this direction
        losses = [t.pnl for t in direction_trades if t.pnl < 0]
        largest_loss = min(losses) if losses else 0
        
        # Net profit as % of largest loss
        net_profit_pct_of_largest_loss = (total_pnl / abs(largest_loss) * 100) if largest_loss != 0 else 0
        
        # CAGR - use the same CAGR from overall metrics as approximation
        # (proper calculation would require tracking equity curve per direction)
        cagr = metrics.get('cagr', 0)
        
        return {
            'cagr': cagr,
            'return_on_initial_capital': return_on_initial,
            'net_profit_pct_of_largest_loss': net_profit_pct_of_largest_loss
        }
    
    return {
        'long': calc_metrics_for_direction(long_trades),
        'short': calc_metrics_for_direction(short_trades)
    }


def _format_detailed_results(broker, metrics, trades, strategy_params):
    """Format results into detailed sections (Overview, Performance, etc.)"""
    
    # Calculate additional metrics needed
    gross_profit = metrics.get('avg_win', 0) * metrics.get('win_count', 0) if metrics.get('win_count', 0) > 0 else 0
    gross_loss = abs(metrics.get('avg_loss', 0) * metrics.get('loss_count', 0)) if metrics.get('loss_count', 0) > 0 else 0
    
    largest_win_pct = (metrics.get('largest_win', 0) / broker.initial_capital * 100) if broker.initial_capital > 0 else 0
    largest_loss_pct = (metrics.get('largest_loss', 0) / broker.initial_capital * 100) if broker.initial_capital > 0 else 0
    largest_win_of_gross = (metrics.get('largest_win', 0) / gross_profit * 100) if gross_profit > 0 else 0
    largest_loss_of_gross = (abs(metrics.get('largest_loss', 0)) / gross_loss * 100) if gross_loss > 0 else 0
    ratio_win_loss = abs(metrics.get('avg_win', 0) / metrics.get('avg_loss', 0)) if metrics.get('avg_loss', 0) != 0 else 0
    
    # Calculate bars metrics
    avg_bars_in_trades = 0
    avg_bars_winning = 0
    avg_bars_losing = 0
    
    if trades:
        # Assuming 4h timeframe hardcoded for now or derived? 
        # strategy_params doesn't explicitly store timeframe seconds easily, but broker knows?
        # Broker has data_loader? No. 
        # We'll assume candles are 4h = 4*3600s if not specified, OR calculate raw candles if index available?
        # Trade object has entry_time (timestamp).
        # We can approximate bars = seconds / timeframe_seconds.
        # Default 4h = 14400s.
        # Calculate bars metrics dynamically based on timeframe
        timeframe_str = strategy_params.get('timeframe', '4h')
        
        # Helper to get seconds
        def _get_tf_seconds(tf_str):
            import re
            match = re.match(r'(\d+)([mhdw])', tf_str.lower())
            if not match: return 3600
            num, unit = int(match.group(1)), match.group(2)
            multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
            return num * multipliers.get(unit, 3600)

        bars_divisor = _get_tf_seconds(timeframe_str)
        
        # Calculate bars using correct divisor
        bars_in_trades = [(t.exit_time - t.entry_time).total_seconds() / bars_divisor for t in trades]
        avg_bars_in_trades = sum(bars_in_trades) / len(bars_in_trades)
        
        winning_trades_list = [t for t in trades if t.pnl > 0]
        losing_trades_list = [t for t in trades if t.pnl < 0]
        
        if winning_trades_list:
            bars_winning = [(t.exit_time - t.entry_time).total_seconds() / bars_divisor for t in winning_trades_list]
            avg_bars_winning = sum(bars_winning) / len(bars_winning)
        
        if losing_trades_list:
            bars_losing = [(t.exit_time - t.entry_time).total_seconds() / bars_divisor for t in losing_trades_list]
            avg_bars_losing = sum(bars_losing) / len(bars_losing)
    
    return_on_initial = (metrics.get('total_pnl', 0) / broker.initial_capital * 100) if broker.initial_capital > 0 else 0
    max_dd_pct_of_initial = (metrics.get('max_drawdown', 0) / broker.initial_capital * 100) if broker.initial_capital > 0 else 0
    return_of_max_dd = (metrics.get('total_pnl', 0) / metrics.get('max_drawdown', 0) * 100) if metrics.get('max_drawdown', 0) != 0 else 0
    net_profit_pct_of_largest_loss = (metrics.get('total_pnl', 0) / abs(metrics.get('largest_loss', 0)) * 100) if metrics.get('largest_loss', 0) != 0 else 0
    
    # Calculate Long/Short capital metrics
    long_short_capital = _calculate_long_short_capital_metrics(trades, broker.initial_capital, metrics)
    
    return {
        # 1. Overview
        'overview': {
            'total_pnl': _sanitize_value(metrics.get('total_pnl', 0)),
            'max_drawdown': _sanitize_value(metrics.get('max_drawdown', 0)),
            'max_drawdown_pct': _sanitize_value(metrics.get('max_drawdown_pct', 0)),
            'total_trades': metrics.get('total_trades', 0),
            'profitable_trades': metrics.get('win_count', 0),
            'win_rate': _sanitize_value(metrics.get('win_rate', 0)),
            'profit_factor': _sanitize_value(metrics.get('profit_factor', 0))
        },
        
        # 2. Performance
        'performance': {
            'return': {
                'initial_capital': _sanitize_value(broker.initial_capital),
                'open_pnl': 0.0,
                'net_pnl': _sanitize_value(metrics.get('total_pnl', 0)),
                'gross_profit': _sanitize_value(gross_profit),
                'gross_loss': _sanitize_value(gross_loss),
                'profit_factor': _sanitize_value(metrics.get('profit_factor', 0)),
                'commission_paid': _sanitize_value(metrics.get('commission_paid', 0)),
                'expected_payoff': _sanitize_value(metrics.get('expectancy', 0))
            },
            'benchmark_comparison': {
                # New standardized benchmark fields
                'buy_hold_return': _sanitize_value(metrics.get('benchmark', {}).get('buy_hold_return')),
                'buy_hold_max': _sanitize_value(metrics.get('benchmark', {}).get('buy_hold_max')),
                'buy_hold_min': _sanitize_value(metrics.get('benchmark', {}).get('buy_hold_min')),
                'strategy_return': _sanitize_value(metrics.get('benchmark', {}).get('strategy_return')),
                'strategy_max': _sanitize_value(metrics.get('benchmark', {}).get('strategy_max')),
                'strategy_min': _sanitize_value(metrics.get('benchmark', {}).get('strategy_min'))
            },
            'risk_adjusted': {
                'sharpe_ratio': _sanitize_value(metrics.get('sharpe', 0)),
                'sortino_ratio': _sanitize_value(metrics.get('sortino', 0))
            }
        },
        
        # 3. Trades Analyst
        'trades_analyst': {
            'details': {
                'total_trades': metrics.get('total_trades', 0),
                'winning_trades': metrics.get('win_count', 0),
                'losing_trades': metrics.get('loss_count', 0),
                'percent_profitable': _sanitize_value(metrics.get('win_rate', 0)),
                'avg_pnl': _sanitize_value(metrics.get('expectancy', 0)),
                'avg_winning_trade': _sanitize_value(metrics.get('avg_win', 0)),
                'avg_losing_trade': _sanitize_value(metrics.get('avg_loss', 0)),
                'ratio_avg_win_loss': _sanitize_value(ratio_win_loss),
                'largest_winning_trade': _sanitize_value(metrics.get('largest_win', 0)),
                'largest_winning_trade_pct': _sanitize_value(largest_win_pct),
                'largest_winner_pct_of_gross_profit': _sanitize_value(largest_win_of_gross),
                'largest_losing_trade': _sanitize_value(metrics.get('largest_loss', 0)),
                'largest_losing_trade_pct': _sanitize_value(largest_loss_pct),
                'largest_loser_pct_of_gross_loss': _sanitize_value(largest_loss_of_gross),
                'avg_bars_in_trades': _sanitize_value(avg_bars_in_trades),
                'avg_bars_in_winning_trades': _sanitize_value(avg_bars_winning),
                'avg_bars_in_losing_trades': _sanitize_value(avg_bars_losing)
            }
        },
        
        # 4. Capital Efficiency
        'capital_efficiency': {
            'capital_usage': {
                'cagr': _sanitize_value(metrics.get('cagr', 0)),
                'return_on_initial_capital': _sanitize_value(return_on_initial),
                'account_size_required': _sanitize_value(broker.initial_capital),
                'return_on_account_size_required': _sanitize_value(return_on_initial),
                'net_profit_pct_of_largest_loss': _sanitize_value(net_profit_pct_of_largest_loss),
                'long': {
                    'cagr': _sanitize_value(long_short_capital['long']['cagr']),
                    'return_on_initial_capital': _sanitize_value(long_short_capital['long']['return_on_initial_capital']),
                    'net_profit_pct_of_largest_loss': _sanitize_value(long_short_capital['long']['net_profit_pct_of_largest_loss'])
                },
                'short': {
                    'cagr': _sanitize_value(long_short_capital['short']['cagr']),
                    'return_on_initial_capital': _sanitize_value(long_short_capital['short']['return_on_initial_capital']),
                    'net_profit_pct_of_largest_loss': _sanitize_value(long_short_capital['short']['net_profit_pct_of_largest_loss'])
                }
            }
        },
        
        # 5. Run-ups and Drawdowns
        'runups_drawdowns': {
            'drawdowns': {
                'max_equity_drawdown': _sanitize_value(metrics.get('max_drawdown', 0)),
                'max_equity_drawdown_pct': _sanitize_value(metrics.get('max_drawdown_pct', 0)),
                'max_equity_drawdown_pct_of_initial': _sanitize_value(max_dd_pct_of_initial),
                'return_of_max_equity_drawdown': _sanitize_value(return_of_max_dd)
            }
        }
    }



def _validate_params(asset: str, timeframe: str, initial_capital: float, ema_length: int, atr_length: int):
    """Validate backtest parameters"""
    if not asset:
        raise ValueError("Asset is required")
    
    if timeframe not in ['1h', '4h', '1d']:
        # Also check if it matches minutes/hours/days pattern
        import re
        if not re.match(r'^\d+[mhdw]$', timeframe):
             raise ValueError(f"Invalid timeframe: {timeframe}")
    
    if initial_capital <= 0:
        raise ValueError("Initial capital must be > 0")
    
    if ema_length <= 0:
        raise ValueError("EMA length must be > 0") 
    
    if atr_length <= 0:
        raise ValueError("ATR length must be > 0")