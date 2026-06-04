"""
Unified Walk-Forward Analysis Engine
Combines IOS (In-Out Sample) and WFA (Walk-Forward Analysis) into one optimized system

Key Insight: IOS = WFA with 1 window
Performance Target: 800-1000 configs/s (same as backtest optimizer)

Architecture:
- wfa.py: Main orchestrator (this file)
- ios_engine.py: IOS logic (single window validation)
- wfa_engine.py: WFA logic (rolling window validation)

Modes:
- 'ios': Run In-Out Sample (1 fixed window)
- 'wfa': Run Walk-Forward Analysis (N rolling windows)
- 'ios_then_wfa': Run IOS first, then WFA on selected strategies

Performance Optimizations:
1. Fork Copy-on-Write: Share data across workers without duplication
2. Precomputed Indicators: Matrix-based O(1) lookup
3. Dynamic Chunking: Optimal worker/chunk allocation
4. Async DB Writer: Non-blocking database writes
5. Batch Processing: Process configs in optimized batches

Background Job Support:
- Jobs can run in background (survive reload/navigation)
- Frontend can check running status and stop jobs
- Progress updates continue even when disconnected
"""

import os
import json
import time
import warnings
import psutil
import multiprocessing
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

warnings.filterwarnings('ignore')

from database.mongo_service import MongoService
from optimize.prepare import load_and_precompute
from .ios_engine import IOSEngine
from .wfa_engine import WFAEngine

import logging
logger = logging.getLogger(__name__)

# Verbose logging
VERBOSE_LOGGING = str(os.getenv('GREY_WFA_VERBOSE', '0')).lower() in {'1', 'true', 'yes'}

def log_unified(msg: str, level: str = 'info'):
    """Log with UNIFIED prefix"""
    if VERBOSE_LOGGING or level in {'warning', 'error'}:
        prefix = "[UNIFIED-WFA]"
        getattr(logger, level)(f"{prefix} {msg}")


# ============================================================================
# Background Job Management
# ============================================================================

JOB_STATUS_DIR = '/tmp/wfa_jobs'
CONTROL_DIR = '/tmp/wfa_control'

# Ensure directories exist
os.makedirs(JOB_STATUS_DIR, exist_ok=True)
os.makedirs(CONTROL_DIR, exist_ok=True)


def check_running_job(batch_id: str) -> Optional[Dict[str, Any]]:
    """
    Check if a job is currently running
    
    Args:
        batch_id: Batch ID to check
        
    Returns:
        Job status dict if running, None otherwise
        
    Example response:
        {
            'batch_id': 'batch_123',
            'status': 'running',
            'mode': 'ios',
            'progress': {'completed': 50, 'total': 100},
            'started_at': '2026-02-09T10:00:00',
            'pid': 12345
        }
    """
    status_file = os.path.join(JOB_STATUS_DIR, f'{batch_id}.json')
    
    if not os.path.exists(status_file):
        return None
    
    try:
        with open(status_file, 'r') as f:
            job_status = json.load(f)
        
        # Check if process is still alive
        pid = job_status.get('pid')
        if pid:
            try:
                os.kill(pid, 0)  # Check if process exists (doesn't actually kill)
                return job_status
            except OSError:
                # Process is dead, cleanup status file
                _cleanup_job_status(batch_id)
                return None
        
        return job_status
        
    except Exception as e:
        log_unified(f"Error checking job status: {e}", 'error')
        return None


def stop_running_job(batch_id: str) -> bool:
    """
    Stop a running job
    
    Args:
        batch_id: Batch ID to stop
        
    Returns:
        True if stop signal was sent, False otherwise
    """
    # Check if job is running
    job_status = check_running_job(batch_id)
    if not job_status:
        log_unified(f"No running job found for batch {batch_id}", 'warning')
        return False
    
    # Create stop signal file
    stop_file = os.path.join(CONTROL_DIR, f'{batch_id}.stop')
    try:
        with open(stop_file, 'w') as f:
            f.write(json.dumps({'stopped_at': datetime.utcnow().isoformat()}))
        
        log_unified(f"Stop signal sent for batch {batch_id}")
        return True
        
    except Exception as e:
        log_unified(f"Error sending stop signal: {e}", 'error')
        return False


def _save_job_status(batch_id: str, status: Dict[str, Any]):
    """Save job status to file for frontend to query"""
    status_file = os.path.join(JOB_STATUS_DIR, f'{batch_id}.json')
    
    try:
        # Add PID for process alive check
        status['pid'] = os.getpid()
        status['updated_at'] = datetime.utcnow().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
            
    except Exception as e:
        log_unified(f"Error saving job status: {e}", 'error')


def _cleanup_job_status(batch_id: str):
    """Cleanup job status file after completion"""
    status_file = os.path.join(JOB_STATUS_DIR, f'{batch_id}.json')
    stop_file = os.path.join(CONTROL_DIR, f'{batch_id}.stop')
    pause_file = os.path.join(CONTROL_DIR, f'{batch_id}.pause')
    
    for f in [status_file, stop_file, pause_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass


def _check_stop_signal(batch_id: str) -> bool:
    """Check if stop signal has been sent"""
    stop_file = os.path.join(CONTROL_DIR, f'{batch_id}.stop')
    return os.path.exists(stop_file)


class UnifiedWalkForwardOptimizer:
    """
    Unified optimizer combining IOS and WFA
    
    Resource Limits (same as backtest):
    - Max 40 workers
    - Max 90GB RAM
    """
    
    # Resource limits (MUST match optimizer.py)
    HARD_LIMIT_WORKERS = 40
    HARD_LIMIT_MEMORY_GB = 80  # 80GB not 90GB (consistency with backtest)
    
    def __init__(
        self, 
        batch_id: str,
        config: Dict[str, Any],
        mongo: Optional[MongoService] = None,
        progress_callback: Optional[Callable] = None,
        disable_db: bool = False
    ):
        """
        Initialize Unified WFA Optimizer
        
        Args:
            batch_id: Unique batch identifier
            config: Configuration dict containing mode and parameters
            mongo: MongoDB service instance
            progress_callback: Progress notification callback
            disable_db: Skip database operations (testing mode)
        """
        self.batch_id = batch_id
        self.config = config
        self.mongo = mongo or MongoService()
        self.progress_callback = progress_callback
        self.disable_db = disable_db
        
        # Extract mode
        self.mode = config.get('mode', 'ios')  # 'ios', 'wfa', or 'ios_then_wfa'
        
        # Extract common config
        self.asset = config.get('asset', 'BTCUSDT')
        self.timeframes = config.get('timeframes', ['1h'])
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_pct = config.get('commission_pct', 0.1)
        self.slippage_pct = config.get('slippage_pct', 0.0)
        
        # Source configs (if running from backtest)
        self.source_batch_id = config.get('source_batch_id')
        
        # Check resources
        self._check_resources()
        
        log_unified(f"Initialized with mode: {self.mode}")
    
    def _check_resources(self):
        """Check available system resources"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        if available_gb < 10:
            log_unified(f"⚠️  Low memory: {available_gb:.1f}GB available", 'warning')
        
        cpu_count = os.cpu_count() or 4
        if cpu_count < 4:
            log_unified(f"⚠️  Low CPU count: {cpu_count} cores", 'warning')
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution entry point
        Routes to appropriate engine based on mode
        Supports background execution with job tracking
        """
        start_time = time.time()
        log_unified(f"🚀 Starting optimization - Mode: {self.mode}")
        
        # Save initial job status
        _save_job_status(self.batch_id, {
            'batch_id': self.batch_id,
            'status': 'running',
            'mode': self.mode,
            'started_at': datetime.utcnow().isoformat(),
            'progress': {'completed': 0, 'total': 0}
        })
        
        try:
            # Check if stop signal exists before starting
            if _check_stop_signal(self.batch_id):
                log_unified("Stop signal detected before start, aborting")
                _cleanup_job_status(self.batch_id)
                return {'status': 'stopped', 'mode': self.mode}
            
            if self.mode == 'ios':
                result = self._run_ios_only()
            elif self.mode == 'wfa':
                result = self._run_wfa_only()
            elif self.mode == 'ios_then_wfa':
                result = self._run_ios_then_wfa()
            else:
                raise ValueError(f"Invalid mode: {self.mode}. Must be 'ios', 'wfa', or 'ios_then_wfa'")
            
            # Check if stopped during execution
            if _check_stop_signal(self.batch_id):
                log_unified("Stop signal detected, aborting")
                result['status'] = 'stopped'
                _cleanup_job_status(self.batch_id)
                return result
            
            duration = time.time() - start_time
            result['duration'] = duration
            result['mode'] = self.mode
            
            log_unified(f"✅ Optimization complete in {duration:.1f}s")
            
            # Cleanup job status on completion
            _cleanup_job_status(self.batch_id)
            
            return result
            
        except Exception as e:
            log_unified(f"❌ Error: {str(e)}", 'error')
            _cleanup_job_status(self.batch_id)
            raise
    
    def _run_ios_only(self) -> Dict[str, Any]:
        """
        Run IOS (In-Out Sample) only
        Single window validation
        """
        log_unified("📊 Running IOS (In-Out Sample) mode")
        
        # Create IOS engine
        ios_engine = IOSEngine(
            batch_id=self.batch_id,
            config=self.config,
            mongo=self.mongo,
            progress_callback=self._wrap_progress_callback('ios'),
            disable_db=self.disable_db
        )
        
        # Execute IOS pipeline
        result = ios_engine.run()
        
        return {
            'status': 'completed',
            'mode': 'ios',
            'ios_result': result
        }
    
    def _run_wfa_only(self) -> Dict[str, Any]:
        """
        Run WFA (Walk-Forward Analysis) only
        Multiple rolling windows
        """
        log_unified("📊 Running WFA (Walk-Forward Analysis) mode")
        
        # Create WFA engine
        wfa_engine = WFAEngine(
            batch_id=self.batch_id,
            config=self.config,
            mongo=self.mongo,
            progress_callback=self._wrap_progress_callback('wfa'),
            disable_db=self.disable_db
        )
        
        # Execute WFA pipeline
        result = wfa_engine.run()
        
        return {
            'status': 'completed',
            'mode': 'wfa',
            'wfa_result': result
        }
    
    def _run_ios_then_wfa(self) -> Dict[str, Any]:
        """
        Run IOS first, then WFA on selected strategies
        Sequential pipeline: IOS → WFA
        """
        log_unified("📊 Running IOS → WFA sequential pipeline")
        
        # Step 1: Run IOS
        log_unified("Step 1/2: Running IOS...")
        ios_engine = IOSEngine(
            batch_id=f"{self.batch_id}_ios",
            config=self.config,
            mongo=self.mongo,
            progress_callback=self._wrap_progress_callback('ios'),
            disable_db=self.disable_db
        )
        ios_result = ios_engine.run()
        
        # Check IOS success
        if ios_result.get('status') != 'completed':
            log_unified("⚠️  IOS failed, skipping WFA", 'warning')
            return {
                'status': 'failed',
                'mode': 'ios_then_wfa',
                'ios_result': ios_result,
                'error': 'IOS stage failed'
            }
        
        # Step 2: Extract selected configs from IOS for WFA
        log_unified("Step 2/2: Running WFA on IOS-selected strategies...")
        
        # Get selected component IDs from IOS
        component_ids = ios_result.get('stages', {}).get('mega', {}).get('component_ids', [])
        if not component_ids:
            log_unified("⚠️  No strategies selected by IOS, skipping WFA", 'warning')
            return {
                'status': 'completed',
                'mode': 'ios_then_wfa',
                'ios_result': ios_result,
                'wfa_result': None,
                'warning': 'No strategies passed IOS selection'
            }
        
        log_unified(f"IOS selected {len(component_ids)} strategies for WFA validation")
        
        # Create WFA config with IOS results
        wfa_config = {
            **self.config,
            'source_batch_id': self.source_batch_id,
            'ios_batch_id': f"{self.batch_id}_ios",
            'selected_config_ids': component_ids
        }
        
        # Run WFA on selected strategies
        wfa_engine = WFAEngine(
            batch_id=f"{self.batch_id}_wfa",
            config=wfa_config,
            mongo=self.mongo,
            progress_callback=self._wrap_progress_callback('wfa'),
            disable_db=self.disable_db
        )
        wfa_result = wfa_engine.run()
        
        return {
            'status': 'completed',
            'mode': 'ios_then_wfa',
            'ios_result': ios_result,
            'wfa_result': wfa_result
        }
    
    def _wrap_progress_callback(self, stage: str) -> Optional[Callable]:
        """
        Wrap progress callback with stage prefix and job status updates
        
        Args:
            stage: 'ios' or 'wfa'
        """
        def wrapped_callback(data: Dict[str, Any]):
            # Add stage prefix to progress data
            data['stage'] = stage
            
            # Save job status for background monitoring
            _save_job_status(self.batch_id, {
                'batch_id': self.batch_id,
                'status': 'running',
                'mode': self.mode,
                'current_stage': stage,
                'progress': {
                    'completed': data.get('completed', 0),
                    'total': data.get('total', 0),
                    'stage': stage,
                    'message': data.get('message', '')
                },
                'started_at': data.get('started_at') if hasattr(self, '_start_time') else datetime.utcnow().isoformat()
            })
            
            # Call original callback if provided
            if self.progress_callback:
                self.progress_callback(data)
        
        return wrapped_callback


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_unified_wfa(
    batch_id: str,
    config: Dict[str, Any],
    mongo: Optional[MongoService] = None,
    progress_callback: Optional[Callable] = None,
    disable_db: bool = False,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main entry point for unified WFA optimization
    
    Args:
        batch_id: Unique batch ID
        config: Configuration dict with mode and parameters
        mongo: MongoDB service
        progress_callback: Progress callback function
        disable_db: Skip database operations
        max_workers: Override max workers (for testing)
    
    Returns:
        Result dict with status and metrics
    """
    # Add max_workers to config if specified
    if max_workers:
        config['max_workers'] = max_workers
    
    optimizer = UnifiedWalkForwardOptimizer(
        batch_id=batch_id,
        config=config,
        mongo=mongo,
        progress_callback=progress_callback,
        disable_db=disable_db
    )
    
    return optimizer.run()

# Alias for backward compatibility with routes/backtest.py
run_wfa_optimization = run_unified_wfa
