"""
Monte Carlo Utilities - Helper functions for risk analysis

This module provides utility functions for:
- Trade data extraction and preprocessing
- Statistical analysis
- Report generation
- Data visualization helpers
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


def extract_trades_from_backtest(
    backtest_result: Dict[str, Any]
) -> Tuple[List[float], Dict[str, float]]:
    """
    Extract trade P&L list from backtest result
    
    Args:
        backtest_result: Complete backtest result document
        
    Returns:
        Tuple of (pnl_list, metrics)
    """
    result = backtest_result.get('result', {}).get('all', {})
    
    # Try to get from trades list
    trades = result.get('trades', [])
    if trades:
        pnl_list = [trade.get('pnl_pct', 0) for trade in trades]
    else:
        # Try to calculate from equity curve
        equity_curve = result.get('equity_curve', [])
        if equity_curve:
            pnl_list = calculate_returns_from_equity(equity_curve)
        else:
            raise ValueError("No trades or equity curve found in backtest result")
    
    # Extract metrics
    metrics = {
        'roi': result.get('roi', 0),
        'mdd': result.get('max_drawdown_pct', 0),
        'sharpe': result.get('sharpe_ratio', 0),
        'profit': result.get('profit', 0),
        'total_trades': len(pnl_list)
    }
    
    return pnl_list, metrics


def calculate_returns_from_equity(
    equity_curve: List[Dict]
) -> List[float]:
    """
    Calculate return series from equity curve
    
    Args:
        equity_curve: List of equity points [{'timestamp': ..., 'equity': ...}]
        
    Returns:
        List of percentage returns
    """
    if not equity_curve or len(equity_curve) < 2:
        return []
    
    returns = []
    for i in range(1, len(equity_curve)):
        prev_equity = equity_curve[i-1].get('equity', 0)
        curr_equity = equity_curve[i].get('equity', 0)
        
        if prev_equity > 0:
            ret = (curr_equity - prev_equity) / prev_equity
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def calculate_sharpe_from_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio from returns
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 52 for weekly, etc.)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
    
    # Annualized
    sharpe = (mean_return * periods_per_year - risk_free_rate) / (std_return * np.sqrt(periods_per_year))
    return float(sharpe)


def calculate_sortino_from_returns(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio (downside deviation only)
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    
    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside = infinite Sortino
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return 0.0
    
    # Annualized
    sortino = (mean_return * periods_per_year - risk_free_rate) / (downside_std * np.sqrt(periods_per_year))
    return float(sortino)


def calculate_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    years: float = 1.0
) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown)
    
    Args:
        total_return: Total return (e.g., 0.5 for 50%)
        max_drawdown: Max drawdown (absolute value, e.g., 0.2 for 20%)
        years: Investment period in years
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float('inf')
    
    cagr = (1 + total_return) ** (1 / years) - 1
    calmar = cagr / abs(max_drawdown)
    
    return float(calmar)


def generate_performance_summary(
    pnl_list: List[float],
    initial_capital: float = 1000000.0
) -> Dict[str, Any]:
    """
    Generate comprehensive performance summary
    
    Args:
        pnl_list: List of percentage returns
        initial_capital: Starting capital
        
    Returns:
        Dictionary with all performance metrics
    """
    pnl_array = np.array(pnl_list)
    
    # Equity calculation
    equity = initial_capital * np.cumprod(1 + pnl_array)
    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    
    # Drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = np.min(drawdown)
    
    # Win/Loss stats
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]
    
    win_rate = len(wins) / len(pnl_array) if len(pnl_array) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    # Risk metrics
    sharpe = calculate_sharpe_from_returns(pnl_array)
    sortino = calculate_sortino_from_returns(pnl_array)
    
    return {
        'total_trades': len(pnl_list),
        'win_rate': float(win_rate * 100),
        'total_return': float(total_return * 100),
        'final_equity': float(final_equity),
        'profit': float(final_equity - initial_capital),
        'max_drawdown': float(max_dd * 100),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'avg_win': float(avg_win * 100),
        'avg_loss': float(avg_loss * 100),
        'profit_factor': float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0,
        'num_wins': int(len(wins)),
        'num_losses': int(len(losses))
    }


def validate_pnl_data(pnl_list: List[float]) -> Tuple[bool, str]:
    """
    Validate P&L data quality
    
    Args:
        pnl_list: List of P&L percentages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not pnl_list:
        return False, "P&L list is empty"
    
    if len(pnl_list) < 10:
        return False, f"Insufficient data: only {len(pnl_list)} trades (minimum 10)"
    
    # Check for extreme values
    pnl_array = np.array(pnl_list)
    
    if np.any(np.isnan(pnl_array)):
        return False, "P&L list contains NaN values"
    
    if np.any(np.isinf(pnl_array)):
        return False, "P&L list contains infinite values"
    
    # Check for unrealistic returns (e.g., > 1000% per trade)
    if np.any(np.abs(pnl_array) > 10):
        max_val = np.max(np.abs(pnl_array))
        return False, f"Unrealistic returns detected: {max_val*100:.1f}%"
    
    return True, ""


def format_verdict_message(verdict: Dict[str, Any]) -> str:
    """
    Format verdict into human-readable message
    
    Args:
        verdict: Verdict dictionary from VerdictEngine
        
    Returns:
        Formatted message string
    """
    status = verdict.get('status')
    badge = verdict.get('badge', '')
    message = verdict.get('message', '')
    
    metrics = verdict.get('metrics', {})
    ror = metrics.get('risk_of_ruin', 0)
    worst_dd = metrics.get('worst_case_mdd_pct', 0)
    
    # Build detailed message
    parts = [f"{badge} {status}: {message}"]
    
    # Add key metrics
    parts.append(f"\nRisk of Ruin: {ror:.2f}%")
    parts.append(f"Worst Case Drawdown: {worst_dd:.1f}%")
    
    # Add warnings if any
    warnings = verdict.get('warnings', [])
    if warnings:
        parts.append("\n⚠️ Warnings:")
        for warning in warnings:
            parts.append(f"  - {warning}")
    
    # Add issues if any
    issues = verdict.get('issues', [])
    if issues:
        parts.append("\n❌ Critical Issues:")
        for issue in issues:
            parts.append(f"  - {issue}")
    
    return '\n'.join(parts)


def create_percentile_summary(
    values: np.ndarray,
    percentiles: List[int] = [5, 10, 25, 50, 75, 90, 95]
) -> Dict[str, float]:
    """
    Create percentile summary for any metric
    
    Args:
        values: Array of values
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary mapping percentile to value
    """
    result = {}
    
    for p in percentiles:
        result[f'p{p}'] = float(np.percentile(values, p))
    
    # Add mean and std
    result['mean'] = float(np.mean(values))
    result['std'] = float(np.std(values))
    result['min'] = float(np.min(values))
    result['max'] = float(np.max(values))
    
    return result


def compare_with_benchmark(
    strategy_return: float,
    strategy_mdd: float,
    benchmark_return: float,
    benchmark_mdd: float
) -> Dict[str, Any]:
    """
    Compare strategy with benchmark (e.g., Buy & Hold)
    
    Args:
        strategy_return: Strategy total return
        strategy_mdd: Strategy max drawdown
        benchmark_return: Benchmark total return
        benchmark_mdd: Benchmark max drawdown
        
    Returns:
        Comparison metrics
    """
    return_diff = strategy_return - benchmark_return
    mdd_diff = strategy_mdd - benchmark_mdd
    
    # Risk-adjusted comparison
    strategy_rar = strategy_return / abs(strategy_mdd) if strategy_mdd != 0 else 0
    benchmark_rar = benchmark_return / abs(benchmark_mdd) if benchmark_mdd != 0 else 0
    
    return {
        'return_difference': float(return_diff),
        'mdd_difference': float(mdd_diff),
        'outperforms': return_diff > 0,
        'strategy_risk_adjusted_return': float(strategy_rar),
        'benchmark_risk_adjusted_return': float(benchmark_rar),
        'risk_adjusted_outperformance': float(strategy_rar - benchmark_rar)
    }


def calculate_consecutive_losses(pnl_list: List[float]) -> int:
    """
    Calculate maximum consecutive losses
    
    Args:
        pnl_list: List of P&L values
        
    Returns:
        Maximum consecutive loss count
    """
    max_consecutive = 0
    current_consecutive = 0
    
    for pnl in pnl_list:
        if pnl < 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def create_backtest_report(
    results: Dict[str, Any],
    include_charts: bool = False
) -> str:
    """
    Create markdown report from Monte Carlo results
    
    Args:
        results: Complete results from MonteCarloSimulator
        include_charts: Whether to include chart links
        
    Returns:
        Markdown formatted report
    """
    summary = results.get('summary', {})
    verdict = results.get('verdict', {})
    mc = results.get('monte_carlo', {})
    bs = results.get('bootstrap', {})
    risk = results.get('risk_metrics', {})
    
    report = []
    
    # Header
    report.append("# Monte Carlo Risk Analysis Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Verdict
    report.append("## 📊 Verdict")
    report.append(format_verdict_message(verdict))
    report.append("")
    
    # Summary
    report.append("## 📈 Summary")
    report.append(f"- Total Trades: {summary.get('num_trades', 0)}")
    report.append(f"- Simulations: {summary.get('num_simulations', 0)}")
    report.append(f"- Confidence Level: {summary.get('confidence_level', 0.95)*100:.0f}%")
    report.append("")
    
    # Risk Metrics
    report.append("## ⚠️ Risk Metrics")
    report.append(f"- Risk of Ruin: {risk.get('risk_of_ruin', 0):.2f}%")
    report.append(f"- Value at Risk (5%): {risk.get('var', 0):.2f}%")
    report.append(f"- CVaR (5%): {risk.get('cvar', 0):.2f}%")
    report.append(f"- Probability of Profit: {risk.get('probability_of_profit', 0):.1f}%")
    report.append("")
    
    # Monte Carlo Results
    report.append("## 🎲 Monte Carlo (Stress Test)")
    worst_case = mc.get('worst_case', {})
    report.append(f"- Worst Case Drawdown: {worst_case.get('worst_drawdown', 0)*100:.1f}%")
    report.append(f"- Worst Case Return: {worst_case.get('worst_return', 0)*100:.1f}%")
    report.append("")
    
    # Bootstrap Results
    report.append("## 🔄 Bootstrap (Reliability Test)")
    report.append(f"- Mean Return: {bs.get('mean_return', 0):.2f}%")
    report.append(f"- Std Deviation: {bs.get('std_return', 0):.2f}%")
    
    ci = bs.get('confidence_interval', [0, 0])
    report.append(f"- 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    report.append("")
    
    return '\n'.join(report)


# Export all utility functions
__all__ = [
    'extract_trades_from_backtest',
    'calculate_returns_from_equity',
    'calculate_sharpe_from_returns',
    'calculate_sortino_from_returns',
    'calculate_calmar_ratio',
    'generate_performance_summary',
    'validate_pnl_data',
    'format_verdict_message',
    'create_percentile_summary',
    'compare_with_benchmark',
    'calculate_consecutive_losses',
    'create_backtest_report'
]
