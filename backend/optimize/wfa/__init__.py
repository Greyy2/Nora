"""
Unified Walk-Forward Analysis Module

Combines IOS and WFA into one optimized system:
- IOS: Single window In-Out Sample validation
- WFA: Rolling window Walk-Forward Analysis

Entry Point:
    from optimize.wfa import run_unified_wfa
    
    result = run_unified_wfa(batch_id, config)
"""

from .wfa import UnifiedWalkForwardOptimizer, run_unified_wfa, check_running_job, stop_running_job
from .ios_engine import IOSEngine
from .wfa_engine import WFAEngine

__all__ = [
    'UnifiedWalkForwardOptimizer',
    'run_unified_wfa',
    'check_running_job',
    'stop_running_job',
    'IOSEngine',
    'WFAEngine'
]

