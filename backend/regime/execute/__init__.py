"""
Execute Module - The Final Gatekeeper

This module handles the final execution of trading orders from ExecutionPayload.
It only activates when Regime Mode is ON (dynamic sizing with regime analysis).

Core Components:
- ExecutionCommander: Main class for order execution
- ExecutionConfig: Configuration for execution parameters
- ExecutionReport: Detailed execution results
- ExecutionStatus: Enum for execution outcomes

Flow:
  REGIME OFF: Signal → Sizing → Broker (bypass Execute)
  REGIME ON:  Signal → Regime → Sizing → Execute → Broker

Key Features:
- Live market validation (spread, liquidity checks)
- Order sequencing (MARKET → STOP → LIMIT)
- Retry logic with exponential backoff
- Emergency cancel protocol
- Slippage tracking and reporting
- Broker-agnostic design (CCXT, MT5, custom APIs)

Usage Example:
    
    from regime.execute import ExecutionCommander, ExecutionConfig
    
    config = ExecutionConfig(max_spread_pct=0.002)
    
    commander = ExecutionCommander(
        exchange_api=ccxt_client,
        config=config,
        enable_live_checks=True
    )
    
    ticker = ccxt_client.fetch_ticker('BTC/USDT')
    
    report = commander.execute_payload(
        payload=execution_payload,
        symbol='BTC/USDT',
        current_ticker=ticker
    )
    
    if report.status == ExecutionStatus.SUCCESS:
        print(f"All orders filled! Avg slippage: {report.average_slippage_pct:.4%}")
"""

from .execute import (
    ExecutionCommander,
    ExecutionConfig,
    ExecutionReport,
    ExecutionStatus,
    create_execution_commander
)

__all__ = [
    'ExecutionCommander',
    'ExecutionConfig',
    'ExecutionReport',
    'ExecutionStatus',
    'create_execution_commander'
]

__version__ = '1.0.0'
