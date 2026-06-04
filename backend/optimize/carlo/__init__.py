"""
Monte Carlo Simulation Module
Test strategy robustness through randomization
"""

from .carlo import (
    MonteCarloSimulator,
    MonteCarloEngine,
    BootstrapEngine,
    RiskAnalyzer,
    VerdictEngine,
    SimulationConfig
)

from .utils import (
    extract_trades_from_backtest,
    calculate_returns_from_equity,
    generate_performance_summary,
    validate_pnl_data,
    format_verdict_message,
    create_backtest_report
)

__all__ = [
    'MonteCarloSimulator',
    'MonteCarloEngine',
    'BootstrapEngine',
    'RiskAnalyzer',
    'VerdictEngine',
    'SimulationConfig',
    'extract_trades_from_backtest',
    'calculate_returns_from_equity',
    'generate_performance_summary',
    'validate_pnl_data',
    'format_verdict_message',
    'create_backtest_report'
]
