"""
Grey Optimizer - Production Grade
Áp dụng Vinh patterns + Multiprocessing
Refactored: Uses STANDARD execution mode (parity with Single Core)
"""

import time
import math
import warnings
import os  # Required for control signals
from typing import Dict, List, Any, Optional
from datetime import datetime
from multiprocessing import Pool
import pandas as pd
import numpy as np
import atexit

# Tắt warnings
warnings.filterwarnings('ignore')

from database.mongo_service import MongoService
from optimize.prepare import load_and_precompute
from core.broker import Broker
from core.load_data import DataLoader
from optimize.post_process import save_results_batch, ProductionDBWriter
from metrics.formatter import format_comprehensive_metrics, sanitize_dict
from optimize.analyze import analyze_results
from utils.helpers import safe_float, safe_int  # Centralized helpers

# ============================================================================
# GLOBAL STATE (Fork Copy-on-Write)
# ============================================================================
GLOBAL_DATA_FRAMES = None  # Dict[timeframe, DataFrame]
GLOBAL_TS_MAPS = None      # Dict[timeframe, Dict[timestamp, idx]] - CACHED!
GLOBAL_TIMESTAMPS = None   # Dict[timeframe, np.ndarray] - CACHED!
GLOBAL_BROKER = None        # Singleton per worker
GLOBAL_BROKER_CONFIG = None

def init_worker():
    """
    Initialize worker (fork inherits global state)
    """
    global GLOBAL_BROKER
    GLOBAL_BROKER = None # Ensure fresh broker per worker
    # reset error logging
    if hasattr(process_config, '_error_logged'):
        delattr(process_config, '_error_logged')


def process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single config using shared global data
    Pattern: Call Broker Standard Mode to ensure 100% parity with Single Core
    """
    global GLOBAL_DATA_FRAMES, GLOBAL_BROKER, GLOBAL_BROKER_CONFIG
    
    try:
        # Lazy broker init (once per worker, reused across all configs)
        if GLOBAL_BROKER is None:
            GLOBAL_BROKER = Broker(
                initial_capital=GLOBAL_BROKER_CONFIG['initial_capital'],
                commission_pct=GLOBAL_BROKER_CONFIG['commission_pct'],
                slippage_pct=GLOBAL_BROKER_CONFIG.get('slippage_pct', 0.0)
            )
        
        # Get parameters
        params = config['params']
        params['commission_pct'] = GLOBAL_BROKER_CONFIG['commission_pct']
        # Include broker-level slippage in params so formatter/UI/verification can reuse exact value
        if GLOBAL_BROKER_CONFIG is not None and 'slippage_pct' in GLOBAL_BROKER_CONFIG:
            params['slippage_pct'] = GLOBAL_BROKER_CONFIG.get('slippage_pct', 0.0)
        timeframe = params['timeframe']
        
        # ensure strategy params are complete
        if 'strategy' not in params:
            params['strategy'] = {}
        if 'bse' not in params['strategy']:
            params['strategy']['bse'] = {}
        if 'ps' not in params['strategy']:
            params['strategy']['ps'] = {}
        
        # Ensure bse.side exists (default: long)
        if 'side' not in params['strategy']['bse']:
            params['strategy']['bse']['side'] = 'long'
        
        # Ensure bse.is_on_going exists
        if 'is_on_going' not in params['strategy']['bse']:
            params['strategy']['bse']['is_on_going'] = True
        
        # Ensure ps.ir, ps.er, ps.or exist
        if 'ir' not in params['strategy']['ps']:
            params['strategy']['ps']['ir'] = 0.02
        if 'er' not in params['strategy']['ps']:
            params['strategy']['ps']['er'] = 0.5
        if 'or' not in params['strategy']['ps']:
            params['strategy']['ps']['or'] = 0.95
        
        # Get preloaded data structure (shared via COW)
        if timeframe not in GLOBAL_DATA_FRAMES:
            # Debug: Show available timeframes for first error
            if not hasattr(process_config, '_tf_error_logged'):
                print(f"\n⚠️  Timeframe mismatch: '{timeframe}' not in {list(GLOBAL_DATA_FRAMES.keys())[:5]}...", flush=True)
                process_config._tf_error_logged = True
            
            return {
                'config_hash': config.get('config_hash') or config.get('_id'),
                'batch_id': config['batch_id'],
                'params': params,
                'metrics': {},
                'status': 'failed',
                'error': f'Timeframe {timeframe} not found'
            }
        
        # Retrieve cached data
        cached_data = GLOBAL_DATA_FRAMES[timeframe]
        df = cached_data['df']
        
        # CRITICAL OPTIMIZATION: Use precomputed indicator matrix!
        # Extract exact EMA and ATR for this config's params
        ema_length = params['length_ema']
        atr_length = params['length_atr']
        
        # Get from pre-computed matrix (O(1) lookup!)
        ema_array = cached_data['ema_matrix'][ema_length - 1]  # matrix[length-1]
        atr_array = cached_data['atr_matrix'][atr_length - 1]
        
        # FAST MODE: Use precomputed indicators + cached ts_map
        asset = config.get('asset') or config.get('metadata', {}).get('asset', 'BTCUSDT')
        
        result = GLOBAL_BROKER.run_backtest(
            asset=asset,
            timeframe=timeframe,
            strategy_params=params,
            df=df,  # Shared DataFrame (fork COW)
            start_date=GLOBAL_BROKER_CONFIG.get('start_date'),
            end_date=GLOBAL_BROKER_CONFIG.get('end_date'),
            
            # CRITICAL: Use precomputed indicators (3-5x faster!)
            precomputed_ema=ema_array,
            precomputed_atr=atr_array,
            fast_mode=True,  # Enable fast mode
            timestamps=cached_data.get('timestamps'),
            ts_map=cached_data.get('ts_map')
        )
        
        # Format comprehensive metrics (Centralized Formatter)
        raw_metrics = result.get('metrics', {})
        formatted_metrics = format_comprehensive_metrics(params, raw_metrics, GLOBAL_BROKER_CONFIG['initial_capital'])
        
        # Merged metrics (Raw + Formatted) for 100% parity with Single Core
        final_metrics = sanitize_dict(formatted_metrics)
        
        # Return standardized result with config_hash for deduplication
        return {
            'config_hash': config.get('config_hash') or config.get('_id'),
            'batch_id': config['batch_id'],
            'params': params,
            'metrics': final_metrics,
            'status': 'success'
        }
        
    except Exception as e:
        # Debug: Print first few errors to help diagnose
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        # Print first error for debugging (avoid spam)
        if not hasattr(process_config, '_error_logged'):
            print(f"\n⚠️  Worker Error (first occurrence): {error_msg}")
            traceback.print_exc()
            process_config._error_logged = True
        
        return {
            'config_hash': config.get('config_hash') or config.get('_id'),
            'batch_id': config.get('batch_id'),
            'params': config.get('params', {}),
            'metrics': {},
            'status': 'failed',
            'error': error_msg
        }

def calculate_optimal_chunking(total_configs: int, max_workers: int = 40, min_chunk_size: int = 100, max_chunk_size: int = 1000) -> tuple:
    """
    🧮 DYNAMIC CHUNKING LOGIC (PRODUCTION-GRADE)
    
    Tính toán số worker và chunk size tối ưu dựa trên workload:
    - Ít config → ít worker, chunk nhỏ
    - Nhiều config → nhiều worker, chunk lớn
    
    Args:
        total_configs: Tổng số config cần xử lý
        max_workers: Số worker tối đa (hard limit)
        min_chunk_size: Kích thước chunk tối thiểu
        max_chunk_size: Kích thước chunk tối đa
        
    Returns:
        (effective_workers, chunk_size)
    """
    # BƯỚC 1: Xác định số worker lý tưởng
    ideal_workers = min(
        max_workers,
        max(1, total_configs // min_chunk_size)
    )
    
    # BƯỚC 2: Tính chunk size động
    chunk_size = math.ceil(total_configs / ideal_workers)
    chunk_size = max(min_chunk_size, min(chunk_size, max_chunk_size))
    
    # BƯỚC 3: Chốt worker thực sự (MUST NOT EXCEED max_workers!)
    effective_workers = min(max_workers, math.ceil(total_configs / chunk_size))
    
    return effective_workers, chunk_size


def _notify_progress(progress_callback, completed: int, total: int, speed: float, status: str, db_saved: int = None, eta: float = None, silent: bool = False):
    """Helper: Notify progress callback with error handling"""
    if not progress_callback:
        return
    try:
        # Keep console clean by default; enable only when explicitly requested.
        if (not silent) and os.getenv('NORA_OPT_PROGRESS_LOG', '0') == '1':
            if completed % 100 == 0 or completed == total:
                eta_str = f" - ETA: {int(eta)}s" if eta is not None else ""
                print(f"🔔 [PROGRESS] {completed}/{total} ({speed:.1f} cfg/s){eta_str} - {status}")
        
        payload = {
            'completed': int(completed),
            'total': int(total),
            'speed': float(speed),
            'status': status
        }
        
        if eta is not None:
            payload['eta'] = float(eta)
            
        # 🔧 FIX: Include db_saved count so subprocess can report accurate saved count
        if db_saved is not None:
            payload['db_saved'] = int(db_saved)
            
        progress_callback(payload)
    except Exception as e:
        pass



def _check_control_signals(stop_file: str, pause_file: str, pbar, progress_callback,
                           results_count: int, total: int, start_time: float,
                           flush_callback: Optional[callable] = None, silent: bool = False) -> tuple:
    """
    Check pause/stop control signals
    Returns: (should_stop: bool, was_paused: bool)
    """
    from tqdm import tqdm  # Import here to avoid scope issues
    import sys

    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _log(msg: str) -> None:
        if silent:
            return
        if is_tty:
            tqdm.write(msg)
        else:
            print(msg)
    
    # Check stop signal
    if os.path.exists(stop_file):
        _log(f"\n⏹️  Stop signal received. Stopping optimization...")
        if flush_callback:
            _log(f"💾 Flushing buffer before stop...")
            flush_callback(force=True)
        _notify_progress(progress_callback, results_count, total, 0, 'stopped', db_saved=results_count, silent=silent)
        return True, False
    
    # Check pause signal
    was_paused = False
    while os.path.exists(pause_file):
        if not hasattr(pbar, 'paused_logged'):
            _log(f"\n⏸️  Pause signal received. Flushing buffer...")
            if flush_callback:
                flush_callback(force=True)  # Flush trước khi pause
                _log(f"✅ Buffer flushed. Waiting for resume...")
            pbar.paused_logged = True
            was_paused = True
            elapsed = time.time() - start_time
            rate = results_count / elapsed if elapsed > 0 else 0
            _notify_progress(progress_callback, results_count, total, rate, 'paused', db_saved=results_count, silent=silent)
        
        time.sleep(1)
        
        # Check stop while paused
        if os.path.exists(stop_file):
            _log(f"\n⏹️  Stop signal received while paused.")
            _notify_progress(progress_callback, results_count, total, 0, 'stopped', db_saved=results_count, silent=silent)
            return True, was_paused
    
    # Resumed from pause
    if hasattr(pbar, 'paused_logged'):
        _log(f"\n▶️  Resuming optimization...")
        delattr(pbar, 'paused_logged')
        if was_paused:
            elapsed = time.time() - start_time
            rate = results_count / elapsed if elapsed > 0 else 0
            _notify_progress(progress_callback, results_count, total, rate, 'running', db_saved=results_count, silent=silent)
    
    return False, was_paused


def run_optimization(batch_id: str, max_workers: Optional[int] = None, config: Optional[Dict] = None, collection_type: str = 'backtest', configs_override: Optional[List[Dict]] = None, progress_callback: Optional[callable] = None, disable_db: bool = False, silent: bool = False):
    """
    Main optimization entry point với DYNAMIC CHUNKING + ASYNC DB WRITER
    Resource Limits: Max 40 workers, Max 80GB RAM
    
    🎯 NEW FEATURES:
    - Dynamic worker/chunk allocation based on workload
    - Async DB writer (non-blocking compute)
    - Smooth progress bar (per result, not per batch)
    - Clean 1-line logs with tqdm.write()
    - Pause/Resume/Stop control via flag files
    
    Args:
        batch_id: Batch ID to process
        max_workers: Optional max workers override
        config: Optional config dict (for year-by-year runs with different date ranges)
        collection_type: 'backtest' or 'wfo' (determines which collection to use)
        configs_override: Optional list of config dicts (for in-memory processing, skips MongoDB read)
    """
    import os
    import sys
    import multiprocessing
    import psutil
    from tqdm import tqdm

    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    show_progress = (not silent) and is_tty
    debug_logs = (not silent) and os.getenv('NORA_OPT_DEBUG', '0') == '1'

    def _log(msg: str) -> None:
        if silent:
            return
        if show_progress:
            tqdm.write(msg)
        else:
            print(msg)

    if debug_logs:
        _log(f"🔧 [DEBUG] run_optimization called with progress_callback: {progress_callback is not None}")
        if progress_callback:
            _log(f"🔧 [DEBUG] Callback type: {type(progress_callback)}, callable: {callable(progress_callback)}")
    
    # Control flags directory
    CONTROL_DIR = '/tmp/backtest_control'
    os.makedirs(CONTROL_DIR, exist_ok=True)
    pause_file = os.path.join(CONTROL_DIR, f'{batch_id}.pause')
    stop_file = os.path.join(CONTROL_DIR, f'{batch_id}.stop')
    
    # Cleanup old control files
    if os.path.exists(pause_file):
        os.remove(pause_file)
    if os.path.exists(stop_file):
        os.remove(stop_file)
    
    # 0. Resource Limits - STRICT ENFORCEMENT
    # Server policy: Grey must not exceed 4 CPU workers.
    try:
        env_workers = int(os.getenv('GREY_MAX_WORKERS', '4'))
    except Exception:
        env_workers = 4
    env_workers = max(1, min(env_workers, 4))

    HARD_LIMIT_WORKERS = min(40, env_workers)
    HARD_LIMIT_MEMORY_GB = 80
    MIN_CHUNK_SIZE = 1
    MAX_CHUNK_SIZE = 50
    
    # Check available memory
    mem = psutil.virtual_memory()
    available_memory_gb = mem.available / (1024**3)
    
    # Prefer affinity-based CPU count when running under cpuset/taskset.
    try:
        cpu_count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        cpu_count = os.cpu_count() or 4
    mongo = MongoService() if not disable_db else None
    
    try:
        # 1. Load batch info (use provided config or load from MongoDB)
        if config is None:
            if disable_db:
                print(f"❌ [Error] disable_db=True requires config parameter")
                return
            history = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
            if not history:
                print(f"❌ [Error] Batch {batch_id} not found")
                return
            config = history.get('config', {})
        
        asset = config.get('asset', 'BTCUSDT')
        timeframes = config.get('timeframes', ['4h'])
        
        # 2. Load configs from correct collection based on type
        # 
        # 🔄 PRODUCTION FLOW:
        # 1. Generation Phase → Save params to backtest-config collection (field: params)
        # 2. Optimization Phase → Read params from backtest-config → Run backtest → Save results to backtest-result
        # 
        # Override option (configs_override) is for testing/debugging only
        if configs_override:
            configs = configs_override  # Testing/Debug mode - skip DB read
        elif disable_db:
            print(f"❌ [Error] disable_db=True requires configs_override parameter")
            return
        else:
            # PRODUCTION: Read from backtest-config collection
            if not silent:
                print(f"\n📥 STEP 2: Loading configs from database...")
            config_collection = mongo.wfo_config if collection_type == 'wfo' else mongo.backtest_config
            configs = list(config_collection.find({'batch_id': batch_id}))
            if not silent:
                print(f"   ✅ Loaded {len(configs):,} configs from DB")
        
        total_total = len(configs)
        initial_offset = 0  # Default: no resume offset
        
        # 🎯 Resume Check: Filter out already completed configs (only SUCCESS ones)
        # SKIP if disable_db=True
        if not disable_db:
            res_collection = mongo.wfo_result if collection_type == 'wfo' else mongo.backtest_result
            existing_hashes = set(res_collection.distinct('config_hash', {'batch_id': batch_id, 'status': 'success'}))
            
            initial_offset = len(existing_hashes)
            if existing_hashes:
                if not silent:
                    print(f"⏩ Resuming: Skipping {initial_offset} existing results for {batch_id}")
                configs = [c for c in configs if (c.get('config_hash') or str(c.get('_id'))) not in existing_hashes]
            
            if not configs:
                print(f"✅ All {int(total_total):,} configurations already completed for {batch_id}.")
                if progress_callback:
                    progress_callback({
                        'completed': int(total_total),
                        'total': int(total_total),
                        'speed': 0,
                        'status': 'completed',
                        'db_saved': int(total_total)
                    })
                return {'total': int(total_total), 'success': int(total_total), 'failed': 0, 'rate': 0}
        
        total_to_run = len(configs)
        if not silent:
            print(f"🚀 Starting optimization for {batch_id}: {int(total_to_run):,} configs to run ({int(initial_offset):,} already done)")
        
        if not configs:
            coll_name = 'wfo-config' if collection_type == 'wfo' else 'backtest-config'
            print(f"❌ No configs found for batch {batch_id} in {coll_name}")
            return None
        
        # 🔧 INITIAL PROGRESS: Notify frontend that we're starting (0 completed, total known)
        # This prevents the "15k/s spike" issue where initialization was being reported as progress
        if progress_callback:
            progress_callback({
                'completed': 0,
                'total': int(total_total),
                'speed': 0,
                'status': 'running',
                'db_saved': int(initial_offset),  # Already completed from previous runs
                'message': 'Đang khởi tạo...'
            })
        
        # 2.1 🧮 DYNAMIC CHUNKING: Calculate optimal workers + chunk size
        total_configs = len(configs)
        
        # Override max_workers if provided, but always enforce HARD_LIMIT_WORKERS (<= 4)
        max_workers_limit = min(max_workers or HARD_LIMIT_WORKERS, cpu_count, HARD_LIMIT_WORKERS)
        
        # Memory-based safety check
        max_workers_by_memory = int(available_memory_gb / 2.0)
        if max_workers_limit > max_workers_by_memory:
            if not silent:
                print(f"⚠️  [MEMORY LIMIT] Giảm max workers từ {max_workers_limit} xuống {max_workers_by_memory} (RAM khả dụng: {available_memory_gb:.1f}GB)")
            max_workers_limit = max(1, max_workers_by_memory)
        
        # Calculate optimal chunking
        worker_count, chunk_size = calculate_optimal_chunking(
            total_configs=total_configs,
            max_workers=max_workers_limit,
            min_chunk_size=MIN_CHUNK_SIZE,
            max_chunk_size=MAX_CHUNK_SIZE
        )
        
        # STEP 1: Calculate formula breakdown (AFTER worker_count is defined)
        if configs_override is None and not silent:
            try:
                # Show calculation formula (Log only, don't let it crash the process)
                ind_cfg = config.get('indicator', {})
                ema_cfg = ind_cfg.get('ema', {})
                atr_cfg = ind_cfg.get('atr', {})
                
                ema_range = range(int(ema_cfg.get('start', 5)), 
                                 int(ema_cfg.get('end', 200)) + 1,
                                 int(ema_cfg.get('step', 5)))
                atr_range = range(int(atr_cfg.get('start', 10)),
                                 int(atr_cfg.get('end', 30)) + 1, 
                                 int(atr_cfg.get('step', 5)))
                
                ema_count = len(list(ema_range))
                atr_count = len(list(atr_range))
                
                # Estimate other params (simplified)
                vf_count = 4  # typical high_vf × low_vf combinations
                ps_count = 3  # typical PS param combinations
                tf_count = len(timeframes)
                
                formula = f"{ema_count}(ema) × {atr_count}(atr) × {vf_count}(vf) × {ps_count}(ps) × {tf_count}(tf)"
                print(f"\n📊 STEP 1: Tính toán tổng configs")
                print(f"   Formula: {formula} = {int(total_configs):,} configs")
                
                # Estimate duration
                avg_speed = worker_count * 50  # Realistic assumption: each worker handles 50 cfg/s
                est_duration = total_configs / avg_speed / 60  # minutes
                print(f"   Dự tính: ~{est_duration:.1f} phút với tốc độ ~{avg_speed} cfg/s")
            except Exception as e:
                print(f"⚠️  [Log Warning] Could not calculate formula breakdown: {e}")
        
        # 3. Extract timeframes dynamically from actual configs to run FIRST
        # This prevents issues where config['timeframes'] is a dict (range config) instead of list
        derived_timeframes = set()
        for c in configs:
            if 'params' in c and 'timeframe' in c['params']:
                derived_timeframes.add(c['params']['timeframe'])

        if derived_timeframes:
            timeframes = list(derived_timeframes)
            if not silent:
                print(f"   ℹ️  Timeframes: {len(timeframes)} loaded (1h to 24h)")
        else:
            timeframes = config.get('timeframes', ['4h'])
            if isinstance(timeframes, dict):
                 # Handle dict case - try to construct single timeframe or fail gracefully
                 if not silent:
                     print(f"   ⚠️  Warning: timeframes is dict and no configs found. Using default ['4h']")
                 timeframes = ['4h']
        
        if not silent:
            # Fix logical numbering
            print(f"\n⚙️  STEP 4: Worker Preparation")
            print(f"   From {int(total_configs):,} configs → Creating {worker_count} workers")
            print(f"   Chunk size: {chunk_size} configs/batch")
            print(f"   + 1 Main Thread: Orchestration & DB Writing")
            print(f"   Available RAM: {available_memory_gb:.1f}GB / {mem.total / (1024**3):.1f}GB")
        
        # 4. Load data with prepare module (includes indicator precompute)
        if not silent:
            print(f"\n📊 STEP 3: Load Data & Precompute Indicators")
        start_load = time.time()
            
        
        # Get max lengths from config (convert to int)
        indicator_config = config.get('indicator', {})
        max_ema = int(indicator_config.get('ema', {}).get('end', 100))
        max_atr = int(indicator_config.get('atr', {}).get('end', 30))
        
        # Use prepare module to load data + precompute indicators
        data_frames = load_and_precompute(
            asset=asset,
            timeframes=timeframes,
            max_ema_length=max_ema,
            max_atr_length=max_atr,
            data_type=config.get('data_type', 'OKX')
        )
        
        # Count total dataframes
        total_dfs = sum(1 for tf_data in data_frames.values() if 'df' in tf_data)
        total_rows = sum(len(tf_data['df']) for tf_data in data_frames.values() if 'df' in tf_data)
            
        load_time = time.time() - start_load
        if not silent:
            print(f"   ✅ Đã load xong data {asset} full với {total_dfs} dataframes")
            print(f"      ({int(total_rows):,} rows total) trong {load_time:.1f}s")
        
        # Indicator computation (silent - already computed)
        if not silent:
            print(f"   ✅ Đã tính toán xong indicators (EMA: 1-{max_ema}, ATR: 1-{max_atr})")
        
        # 4. Prepare workers with dynamic pool
        # Pattern: Global variables + Fork (COW) = Zero IPC overhead for large data
        global GLOBAL_DATA_FRAMES, GLOBAL_BROKER_CONFIG
        GLOBAL_DATA_FRAMES = data_frames
        GLOBAL_BROKER_CONFIG = {
            'initial_capital': config.get('initial_capital', 1000000),
            'commission_pct': config.get('commission_pct', 0.1),
            'slippage_pct': config.get('slippage_pct', 0.0),
            'start_date': config.get('start_date'),
            'end_date': config.get('end_date')
        }
        
        ctx = multiprocessing.get_context('fork') if os.name != 'nt' else multiprocessing.get_context('spawn')
        pool = ctx.Pool(processes=worker_count, initializer=init_worker) # Inherit globals via Fork
        
        # 5. 🚀 Execute optimization with MAIN THREAD BUFFER (RAM)
        # tqdm already imported above
        
        if not silent:
            print(f"\n⚙️  STEP 5: Processing - Chạy backtest")
        
        # 5.1 Main thread buffer - TÁCH BIỆT PROCESS VÀ DATABASE FLUSH
        import threading
        results_ram_buffer = []  # Accumulate results in main thread RAM
        buffer_lock = threading.Lock()  # Protect buffer access
        last_flush_time = time.time()
        last_flush_check = time.time()
        FLUSH_BATCH_SIZE = 5000  # Flush every 5000 items
        FLUSH_INTERVAL_SEC = 10.0  # OR every 10 seconds
        FLUSH_CHECK_INTERVAL = 1.0  # Check flush conditions every 1s (not every result)
        
        # MongoDB connection (lazy init)
        mongo_service = None if disable_db else MongoService()
        flush_count = 0  # Track number of flushes
        flush_thread = None  # Background flush thread
        
        def flush_to_database(force=False, silent_flush=False):
            """
            Flush buffer to MongoDB in BACKGROUND THREAD
            
            CRITICAL: Non-blocking - launches thread and returns immediately
            Main loop continues updating progress while flush happens in background
            """
            nonlocal results_ram_buffer, last_flush_time, mongo_service, flush_count, flush_thread
            
            if disable_db:
                return 0
                
            with buffer_lock:
                if not results_ram_buffer:
                    return 0
                    
                current_time = time.time()
                buffer_size = len(results_ram_buffer)
                time_elapsed = current_time - last_flush_time
                
                should_flush = (
                    force or 
                    buffer_size >= FLUSH_BATCH_SIZE or 
                    time_elapsed >= FLUSH_INTERVAL_SEC
                )
                
                if not should_flush:
                    return 0
                    
                # Wait for previous flush to complete (if any)
                if flush_thread and flush_thread.is_alive():
                    if force:
                        flush_thread.join()  # Only wait if forced (final flush)
                    else:
                        return 0  # Skip this flush, previous one still running
                
                # Take snapshot and clear buffer (atomic operation)
                batch_to_flush = results_ram_buffer[:]
                results_ram_buffer.clear()
                last_flush_time = current_time
            
            # Launch background flush thread
            def _background_flush():
                nonlocal flush_count
                try:
                    stats = save_results_batch(
                        batch_to_flush, 
                        mongo_service, 
                        collection_type=collection_type
                    )
                    saved = stats.get('upserted', 0) + stats.get('modified', 0)
                    with buffer_lock:
                        flush_count += 1
                    
                    # Log flush completion (Dùng tqdm.write để hiển thị rõ ràng thông báo lưu batch)
                    if not silent_flush:
                        _log(f"💾 Flush #{flush_count}: Đã lưu {saved:,} kết quả vào MongoDB.")
                        
                except Exception as e:
                    # Always show DB flush errors
                    print(f"⚠️  Lỗi lưu DB: {e}")
            
            # Start thread (non-blocking)
            if force:
                _background_flush()  # Synchronous for final flush
                return len(batch_to_flush)
            else:
                flush_thread = threading.Thread(target=_background_flush, daemon=True)
                flush_thread.start()
                return 0  # Return immediately, don't wait
        
        if disable_db and not silent:
            print(f"⚠️  Database disabled - results will only be kept in memory")
        
        # Professional tqdm format with mininterval for smooth visual updates
        bar_format = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        
        start_opt = time.time()
        results_buffer = []  # Collect for stats only
        last_progress_update = 0.0  # Track last progress notification time
        progress_update_interval = 0.3  # Notify frontend every 0.3s
        
        try:
            with tqdm(
                total=total_total,
                initial=initial_offset,
                desc=f"🚀 Đang tối ưu hóa",
                bar_format=bar_format,
                colour='green',
                ncols=120,
                unit='cfg',
                dynamic_ncols=True,
                mininterval=0.1,  # Update display every 0.1s for smooth progress
                disable=not show_progress,
            ) as pbar:
                # Save pbar to global-ish reference for flush thread
                process_config._pbar = pbar
                
                # 🔥 Use dynamic chunk_size from calculation
                configs_list = list(configs)
                
                count = 0
                
                for result in pool.imap_unordered(process_config, configs_list, chunksize=chunk_size):
                    count += 1
                    
                    # Check control signals (pause/stop) - pass flush callback
                    should_stop, was_paused = _check_control_signals(
                        stop_file, pause_file, pbar, progress_callback,
                        len(results_buffer) + initial_offset, total_total, start_opt,
                        flush_callback=flush_to_database if not disable_db else None,
                        silent=silent,
                    )
                    if should_stop:
                        _log(f"⚠️  Stopping: {len(results_buffer):,} configs processed")
                        break
                    
                    # ✅ STEP 1: Update progress bar
                    pbar.update(1)
                    
                    if result:
                        results_buffer.append(result)
                        
                        # STEP 2-5: Single lock acquisition for all buffer operations
                        if not disable_db:
                            with buffer_lock:
                                # Append result
                                results_ram_buffer.append(result)
                                # Get buffer size
                                ram_buffer_size = len(results_ram_buffer)
                                # Check if should flush (>=5000)
                                should_flush = ram_buffer_size >= FLUSH_BATCH_SIZE
                            
                            # Calculate speed
                            elapsed = time.time() - start_opt
                            rate = len(results_buffer) / elapsed if elapsed > 0 else 0
                            
                            # STEP 4: Update display
                            pbar.set_postfix_str(
                                f"Buf:{ram_buffer_size:>4,} Flush:{flush_count:>2} | {rate:>5.0f}cfg/s",
                                refresh=False
                            )
                            
                            # STEP 5: Trigger flush if needed (non-blocking background)
                            if should_flush:
                                flush_to_database(force=False, silent_flush=(silent or (not is_tty)))
                        else:
                            # No DB mode
                            elapsed = time.time() - start_opt
                            rate = len(results_buffer) / elapsed if elapsed > 0 else 0
                            pbar.set_postfix_str(f"{rate:>5.0f}cfg/s", refresh=False)
                    else:
                        # ⚠️ Received None result (shouldn't happen)
                        if not silent:
                            _log(f"⚠️ Received None result from worker")
                    
                    # STEP 6: Time-based progress notification (every 0.3s to keep frontend updated)
                    current_time = time.time()
                    if progress_callback and (current_time - last_progress_update) >= progress_update_interval:
                        elapsed = current_time - start_opt
                        # use pbar stats for absolute accuracy
                        current_completed = pbar.n
                        current_total = pbar.total
                        
                        # rate calculation (overall speed)
                        # pbar.format_dict['rate'] is more stable for long runs
                        rate = pbar.format_dict.get('rate')
                        
                        # Fallback for small batches or beginning
                        if rate is None or rate == 0:
                            if elapsed > 0:
                                rate = (current_completed - initial_offset) / elapsed
                            else:
                                rate = 0
                        
                        # 🔧 SPEED SMOOTHING: Avoid jitter in first seconds
                        if elapsed < 2.0 and rate > 500: # Unrealistic burst
                             rate = 0
                        
                        # 🔧 FIX: Calculate ETA accurately
                        remaining = current_total - current_completed
                        eta = remaining / rate if rate > 0 else 0
                        
                        # 🔧 FIX: Only notify with DB-saved count, not processing count
                        # This prevents "completed" notification before DB flush finishes
                        with buffer_lock:
                            # Calculate approximate DB saved count based on flushes
                            approx_db_saved = flush_count * FLUSH_BATCH_SIZE
                            # Add buffered items that will be saved in next flush
                            buffer_size = len(results_ram_buffer)
                        
                        _notify_progress(
                            progress_callback,
                            current_completed,  # Processing count (for speed calculation)
                            current_total,
                            rate,
                            'running',
                            db_saved=approx_db_saved,  # Actual DB saved count
                            eta=eta,
                            silent=silent,
                        )
                        last_progress_update = current_time
                            
        except KeyboardInterrupt:
            _log(f"\n⚠️ Interrupted by user!")
            if not disable_db:
                with buffer_lock:
                    remaining = len(results_ram_buffer)
                _log(f"💾 Flushing {remaining:,} remaining results...")
                flush_to_database(force=True)
            _log(f"✅ Exited gracefully")
        finally:
            pool.close()
            pool.join()
            
            # 🛑 Final flush - wait for background thread and flush remaining
            if not disable_db:
                # Wait for any background flush to complete
                if flush_thread and flush_thread.is_alive():
                    _log(f"\n⏳ Waiting for background flush to complete...")
                    flush_thread.join()
                
                # Flush any remaining items
                with buffer_lock:
                    remaining = len(results_ram_buffer)
                if remaining > 0:
                    _log(f"\n💾 Final flush: {remaining:,} results → MongoDB...")
                    flush_to_database(force=True)
                _log(f"✅ All results saved to database")
            else:
                _log(f"\n✅ Database disabled - {len(results_buffer):,} results in memory")
        
        opt_time = time.time() - start_opt
        total_speed = len(configs) / opt_time if opt_time > 0 else 0
        
        # Count success/failed from results buffer
        success = sum(1 for r in results_buffer if r.get('status') == 'success')
        failed = sum(1 for r in results_buffer if r.get('status') == 'failed')
        total = len(results_buffer)
        
        # STEP 6: Final summary
        if not silent:
            print(f"\n" + "="*70)
            print(f"✅ OPTIMIZATION COMPLETE")
            print(f"="*70)
            
            if disable_db:
                # Database disabled mode
                print(f"✅ Đã chạy backtest: {int(total):,} configs")
                print(f"   - Tốc độ trung bình: {total_speed:.1f} configs/s")
                print(f"   - Kết quả: {int(success):,} success, {int(failed):,} failed")
                print(f"   - Thời gian hoàn thành: {opt_time/60:.1f} phút ({opt_time:.1f}s)")
                print(f"   - Database: DISABLED (results in memory only)")
            else:
                # Database enabled - all results flushed to MongoDB
                print(f"✅ Đã chạy backtest: {int(total):,} configs")
                print(f"   - Tốc độ trung bình: {total_speed:.1f} configs/s")
                print(f"   - Kết quả: {int(success):,} success, {int(failed):,} failed")
                print(f"   - Đã lưu vào database: {int(success):,}/{int(total):,} configs")
                print(f"   - Thời gian hoàn thành: {opt_time/60:.1f} phút ({opt_time:.1f}s)")
            
            print(f"="*70)
        
        # STEP 7: Post-processing - Analyze results (skip if DB disabled)
        analysis_result = None
        if not disable_db:
            try:
                if not silent:
                    print(f"\n📊 POST-PROCESSING - Analyzing results...")
                
                # 🔧 Get final DB count after all flushes
                if mongo_service:
                    result_collection = mongo_service.wfo_result if collection_type == 'wfo' else mongo_service.backtest_result
                    try:
                        final_db_count = result_collection.count_documents({'batch_id': batch_id, 'status': 'success'})
                    except:
                        final_db_count = success  # Fallback to success count from results_buffer
                else:
                    final_db_count = success
                
                # Create progress callback wrapper for analysis phase
                def analysis_progress(data):
                    if progress_callback:
                        progress_callback({
                            'completed': final_db_count,  # Use actual DB count
                            'total': total,
                            'speed': 0,
                            'status': 'processing',
                            'db_saved': final_db_count,  # Include db_saved
                            'message': data.get('message', 'Đang phân tích kết quả...')
                        })
                
                analysis_result = analyze_results(batch_id, mongo, progress_callback=analysis_progress, collection_type=collection_type)
            except Exception as e:
                print(f"⚠️  Post-processing error: {e}")
        else:
            if not silent:
                print(f"\n⏭️  POST-PROCESSING skipped (database disabled)")
        
        # Return stats + results for caller
        return {
            'success': success,
            'failed': failed,
            'total': total,
            'duration': opt_time,
            'rate': total_speed,
            'worker_count': worker_count,
            'chunk_size': chunk_size,
            'results': results_buffer,  # 🔥 Return results for Excel export
            'analysis': analysis_result  # 📊 Post-processing stats
        }
        
    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        
        # Flush remaining buffer on error
        if not disable_db and 'flush_to_database' in locals():
            try:
                print(f"   Flushing remaining buffer...")
                flush_to_database(force=True)
            except Exception as flush_err:
                print(f"   ⚠️  Flush error: {flush_err}")
        
        return None
    finally:
        # Always close MongoDB connection (if not disabled)
        if mongo is not None:
            try:
                mongo.close()
            except Exception as close_err:
                print(f"⚠️  MongoDB close warning: {close_err}")
