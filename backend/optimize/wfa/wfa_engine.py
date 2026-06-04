"""
WFA (Walk-Forward Analysis) Engine
Rolling window validation across multiple time periods

Architecture:
1. Generate walk-forward windows (IS + OS periods with step)
2. For each window: Run configs → Filter → Validate
3. Aggregate results across all windows
4. Select strategies that perform consistently

Performance: Optimized multiprocessing (target: 800-1000 cfg/s)
Pattern: Similar to IOS but with window iteration
"""

import os
import time
import math
import warnings
import psutil
import multiprocessing
from multiprocessing import Pool
from typing import Dict, Any, List, Optional, Callable, Tuple, Iterable, Iterator
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

from database.mongo_service import MongoService
from optimize.prepare import load_and_precompute
from optimize.generation import generate_params, generate_param_hash
from core.broker import Broker
from metrics.formatter import format_comprehensive_metrics, sanitize_dict
from metrics.calculator import MetricsCalculator
from optimize.post_process import save_results_batch

import logging
logger = logging.getLogger(__name__)

# Verbose logging
VERBOSE_LOGGING = str(os.getenv('GREY_WFA_VERBOSE', '0')).lower() in {'1', 'true', 'yes'}

def log_wfa(msg: str, level: str = 'info'):
    """Log with WFA prefix"""
    if VERBOSE_LOGGING or level in {'warning', 'error'}:
        prefix = "[WFA]"
        getattr(logger, level)(f"{prefix} {msg}")


# ============================================================================
# GLOBAL STATE (Fork Copy-on-Write)
# ============================================================================
GLOBAL_DATA_FRAMES = None
GLOBAL_BROKER = None
GLOBAL_BROKER_CONFIG = None

def init_wfa_worker():
    """Initialize worker process"""
    global GLOBAL_BROKER
    GLOBAL_BROKER = None


def process_wfa_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single WFA config
    Runs backtest on FULL range (IS + OS), then splits metrics
    """
    global GLOBAL_DATA_FRAMES, GLOBAL_BROKER, GLOBAL_BROKER_CONFIG
    
    try:
        # Lazy broker init
        if GLOBAL_BROKER is None:
            GLOBAL_BROKER = Broker(
                initial_capital=GLOBAL_BROKER_CONFIG['initial_capital'],
                commission_pct=GLOBAL_BROKER_CONFIG['commission_pct'],
                slippage_pct=GLOBAL_BROKER_CONFIG.get('slippage_pct', 0.0)
            )
        
        # Extract params
        params = config['params']
        params['commission_pct'] = GLOBAL_BROKER_CONFIG['commission_pct']
        params['slippage_pct'] = GLOBAL_BROKER_CONFIG.get('slippage_pct', 0.0)
        
        timeframe = params['timeframe']
        window_id = params.get('window_id', 'W0')
        
        # Get cached data
        if timeframe not in GLOBAL_DATA_FRAMES:
            return {
                'config_hash': config.get('config_hash'),
                'batch_id': config['batch_id'],
                'window_id': window_id,
                'params': params,
                'metrics': {},
                'status': 'failed',
                'error': f'Timeframe {timeframe} not found'
            }
        
        cached_data = GLOBAL_DATA_FRAMES[timeframe]
        df = cached_data['df']
        
        # Optimization: Precomputed indicators
        ema_length = params['length_ema']
        atr_length = params['length_atr']
        precomputed_ema = cached_data['ema_matrix'][ema_length - 1]
        precomputed_atr = cached_data['atr_matrix'][atr_length - 1]
        
        # WFA: Extract window dates from params
        start_date = params.get('start_date')  # IS start
        end_date = params.get('end_date')      # OS end (full range)
        is_end = params.get('is_end')          # Split point
        
        asset = params.get('asset', GLOBAL_BROKER_CONFIG.get('asset', 'BTCUSDT'))
        
        # Run backtest on FULL range (IS + OS combined)
        result = GLOBAL_BROKER.run_backtest(
            asset=asset,
            timeframe=timeframe,
            strategy_params=params,
            df=df,
            timestamps=cached_data.get('timestamps'),
            ts_map=cached_data.get('ts_map'),
            start_date=start_date,
            end_date=end_date,
            fast_mode=True,
            precomputed_ema=precomputed_ema,
            precomputed_atr=precomputed_atr
        )
        
        # Format full metrics
        raw_metrics = result.get('metrics', {})
        full_metrics = format_comprehensive_metrics(params, raw_metrics, GLOBAL_BROKER_CONFIG['initial_capital'])
        
        # Split metrics into IS and OS
        metrics_is = {}
        metrics_oos = {}
        equity_curve = result.get('equity_curve', [])
        trades = result.get('trades', [])
        
        try:
            # Find split index based on is_end date
            df_slice = result.get('df')
            
            # Debug check
            if df_slice is None:
                log_wfa(f"⚠️  {window_id}: df_slice is None, cannot split metrics", 'warning')
            if not is_end:
                log_wfa(f"⚠️  {window_id}: is_end missing ({is_end})", 'warning')
            if len(equity_curve) == 0:
                log_wfa(f"⚠️  {window_id}: equity_curve empty", 'warning')
            
            if df_slice is not None and is_end and len(equity_curve) > 0:
                # Ensure timezone consistency (normalize to UTC if needed)
                is_end_ts = pd.Timestamp(is_end)
                if df_slice.index.tz is not None:
                    # DataFrame has timezone, localize is_end_ts
                    if is_end_ts.tz is None:
                        is_end_ts = is_end_ts.tz_localize('UTC')
                    elif is_end_ts.tz != df_slice.index.tz:
                        is_end_ts = is_end_ts.tz_convert(df_slice.index.tz)
                else:
                    # DataFrame has no timezone, ensure is_end_ts has no timezone
                    if is_end_ts.tz is not None:
                        is_end_ts = is_end_ts.tz_localize(None)
                
                split_idx = df_slice.index.get_indexer([is_end_ts], method='pad')[0]
                
                # Split trades - ensure timezone consistency
                is_trades = []
                oos_trades = []
                for t in trades:
                    # Access Trade dataclass attribute directly (not .get())
                    entry_ts = pd.Timestamp(t.entry_time if hasattr(t, 'entry_time') else t.get('entry_time', 0))
                    # Normalize timezone to match is_end_ts
                    if is_end_ts.tz is not None:
                        if entry_ts.tz is None:
                            entry_ts = entry_ts.tz_localize('UTC')
                        elif entry_ts.tz != is_end_ts.tz:
                            entry_ts = entry_ts.tz_convert(is_end_ts.tz)
                    else:
                        if entry_ts.tz is not None:
                            entry_ts = entry_ts.tz_localize(None)
                    
                    if entry_ts <= is_end_ts:
                        is_trades.append(t)
                    else:
                        oos_trades.append(t)
                
                # Split equity
                is_equity = equity_curve[:split_idx+1] if isinstance(equity_curve, (list, np.ndarray)) else []
                oos_equity = equity_curve[split_idx:] if isinstance(equity_curve, (list, np.ndarray)) else []
                
                # Calculate IS metrics
                if len(is_equity) > 0:
                    is_equity_arr = np.array(is_equity) if not isinstance(is_equity, np.ndarray) else is_equity
                    calc_is = MetricsCalculator(is_trades, is_equity_arr, GLOBAL_BROKER_CONFIG['initial_capital'])
                    raw_is = calc_is.calculate_fast()
                    metrics_is = format_comprehensive_metrics({}, raw_is, GLOBAL_BROKER_CONFIG['initial_capital'])
                
                # Calculate OOS metrics
                if len(oos_equity) > 0:
                    oos_start_cap = is_equity[-1] if len(is_equity) > 0 else GLOBAL_BROKER_CONFIG['initial_capital']
                    oos_equity_arr = np.array(oos_equity) if not isinstance(oos_equity, np.ndarray) else oos_equity
                    calc_oos = MetricsCalculator(oos_trades, oos_equity_arr, oos_start_cap)
                    raw_oos = calc_oos.calculate_fast()
                    metrics_oos = format_comprehensive_metrics({}, raw_oos, oos_start_cap)
        except Exception as e:
            log_wfa(f"⚠️  Error splitting metrics for window {window_id}: {e}", 'warning')
        
        return {
            'config_hash': config.get('config_hash') or config.get('_id'),
            'config_id': config.get('config_id'),
            'batch_id': config['batch_id'],
            'window_id': window_id,
            'params': sanitize_dict(params),
            'metrics': sanitize_dict(full_metrics),
            'metrics_is': sanitize_dict(metrics_is),
            'metrics_oos': sanitize_dict(metrics_oos),
            'equity_curve': equity_curve,
            'trades': trades,
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'config_hash': config.get('config_hash'),
            'batch_id': config.get('batch_id'),
            'window_id': config.get('params', {}).get('window_id', 'W0'),
            'params': config.get('params', {}),
            'metrics': {},
            'status': 'failed',
            'error': str(e)
        }


def generate_wf_windows(
    start_date: str,
    end_date: str,
    is_val: int,
    oos_val: int,
    step_val: int,
    step_type: str = 'monthly'
) -> List[Dict[str, str]]:
    """
    Generate walk-forward windows
    
    Args:
        start_date: Overall start date (YYYY-MM-DD)
        end_date: Overall end date (YYYY-MM-DD)
        is_val: In-Sample period value
        oos_val: Out-Sample period value
        step_val: Step size value
        step_type: 'weekly', 'monthly', or 'yearly'
    
    Returns:
        List of windows with is_start, is_end, os_start, os_end
    """
    windows = []
    window_idx = 0
    
    current_start = datetime.strptime(start_date, '%Y-%m-%d')
    final_end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Define delta based on step_type
    def get_delta(val):
        if step_type == 'weekly':
            return relativedelta(weeks=val)
        elif step_type == 'yearly':
            return relativedelta(years=val)
        else:  # monthly (default)
            return relativedelta(months=val)
    
    while True:
        is_start = current_start
        is_end = is_start + get_delta(is_val)
        os_start = is_end
        os_end = os_start + get_delta(oos_val)
        
        # Check if we have enough data
        if os_end > final_end:
            break
        
        windows.append({
            'window_id': f'W{window_idx}',
            'is_start': is_start.strftime('%Y-%m-%d'),
            'is_end': is_end.strftime('%Y-%m-%d'),
            'os_start': os_start.strftime('%Y-%m-%d'),
            'os_end': os_end.strftime('%Y-%m-%d')
        })
        
        window_idx += 1
        current_start += get_delta(step_val)
    
    log_wfa(f"Generated {len(windows)} walk-forward windows ({step_type})")
    return windows


class WFAEngine:
    """
    WFA Engine - Rolling window validation
    
    Performance optimizations:
    - Multiprocessing with fork COW
    - Precomputed indicator matrices
    - Dynamic chunking (optimized for short data periods)
    - Batch processing per window
    
    ⚠️ SYSTEM-WIDE Resource Limits:
    - Max 40 workers (entire system, not per operation)
    - Max 80GB RAM (entire system, shared across all operations)
    - Physical resources ignored, only hard limits enforced
    """
    
    # Resource limits (SYSTEM-WIDE, shared across ALL operations)
    HARD_LIMIT_WORKERS = 40
    HARD_LIMIT_MEMORY_GB = 80  # System-wide cap, ignores physical RAM
    
    # Chunking params - WFA generates N×M configs (large batches)
    # Higher chunk sizes reduce scheduling overhead for 100k+ config batches
    MIN_CHUNK_SIZE = 10        # Minimum batch per worker
    MAX_CHUNK_SIZE = 500       # Large chunks for WFA workload (N configs × M windows)
    
    def __init__(
        self,
        batch_id: str,
        config: Dict[str, Any],
        mongo: Optional[MongoService] = None,
        progress_callback: Optional[Callable] = None,
        disable_db: bool = False
    ):
        self.batch_id = batch_id
        self.config = config
        self.mongo = mongo or MongoService()
        self.progress_callback = progress_callback
        self.disable_db = disable_db
        
        # Extract WFA config
        wfa_config = config.get('wfa', {})
        self.is_val = wfa_config.get('is_val', wfa_config.get('is_months', 24))
        self.oos_val = wfa_config.get('oos_val', wfa_config.get('os_months', 6))
        self.step_val = wfa_config.get('step_val', wfa_config.get('step_months', 1))
        self.step_type = wfa_config.get('step_type', 'monthly')
        
        # Date range
        self.start_date = config.get('start_date', '2018-01-01')
        self.end_date = config.get('end_date', '2024-12-31')
        
        # Source configs
        self.source_batch_id = config.get('source_batch_id')
        self.ios_batch_id = config.get('ios_batch_id')
        self.selected_config_ids = config.get('selected_config_ids')
        
        # Mode: 'fresh' or 'from_ios'
        self.mode = 'from_ios' if self.ios_batch_id else 'fresh'
        
        # Extract common broker config
        self.asset = config.get('asset', 'BTCUSDT')
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_pct = config.get('commission_pct', 0.1)
        self.slippage_pct = config.get('slippage_pct', 0.0)
        
        # Resource settings
        self._max_workers = self._compute_max_workers(config.get('max_workers'))
        
        # Source filter (pre-filter backtest results)
        source_filter_config = config.get('source_filter', {})
        self.source_filter_enabled = bool(source_filter_config)
        self.source_filter_expression = source_filter_config.get('expression', '') if source_filter_config else ''
        
        # WFA-specific filters (giống IOS pipeline)
        self.expression = config.get('expression', 'total_pnl > 0')  # IS filter expression
        self.oos_expression = config.get('oos_expression', '')       # OS filter expression
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.top_n = config.get('top_n', 20)  # Max configs after correlation filter
        
        log_wfa(f"Initialized WFA Engine - Mode: {self.mode}")
        log_wfa(f"Windows: IS={self.is_val}, OS={self.oos_val}, Step={self.step_val} ({self.step_type})")
    
    def _compute_max_workers(self, override: Optional[int] = None) -> int:
        """Calculate optimal worker count"""
        cpu_count = os.cpu_count() or 4
        hard_cap = max(1, self.HARD_LIMIT_WORKERS - 1)
        
        if override:
            return max(1, min(override, hard_cap))
        
        return max(1, min(cpu_count - 1, hard_cap))
    
    def run(self) -> Dict[str, Any]:
        """
        Execute WFA pipeline
        Returns: Result dict with per-window and aggregate metrics
        """
        start_time = time.time()
        log_wfa(f"🚀 Starting WFA analysis")
        
        try:
            # Generate windows
            windows = generate_wf_windows(
                self.start_date,
                self.end_date,
                self.is_val,
                self.oos_val,
                self.step_val,
                self.step_type
            )
            
            if not windows:
                return {'status': 'failed', 'error': 'No windows generated'}
            
            log_wfa(f"Processing {len(windows)} windows")
            
            # Load base configs
            base_configs = self._load_base_configs()
            if not base_configs:
                return {'status': 'failed', 'error': 'No configs to process'}
            
            log_wfa(f"Loaded {len(base_configs)} base configs")
            
            # Preload data for performance
            self._preload_data(base_configs)
            
            total_tasks = len(base_configs) * len(windows)
            print(f"\n🔄 Processing {len(windows)} windows sequentially (each with {len(base_configs):,} configs)")
            log_wfa(f"Total configs to process: {total_tasks:,} ({len(base_configs)} × {len(windows)})")

            # Process each window sequentially
            all_results = []
            window_summaries = []
            
            for i, window in enumerate(windows, 1):
                print(f"\n{'='*70}")
                print(f"📅 Window {i}/{len(windows)}: {window['window_id']}")
                print(f"   IS: {window['is_start']} → {window['is_end']}")
                print(f"   OS: {window['os_start']} → {window['os_end']}")
                print(f"{'='*70}")
                
                window_start = time.time()
                
                # Create configs for this window only
                window_configs = list(self._create_window_configs_single(base_configs, window))
                
                # Process this window with full pipeline
                global_offset = (i - 1) * len(base_configs)
                window_results = self._process_window(
                    window_configs,
                    window['window_id'],
                    global_offset=global_offset,
                    global_total=total_tasks,
                    window_index=i,
                    windows_total=len(windows),
                )
                all_results.extend(window_results)
                
                # Apply IOS pipeline: IS Filter → Correlation → Mega → OOS Validation
                pipeline_result = self._apply_window_pipeline(window_results, window)
                
                window_duration = time.time() - window_start
                
                # Window summary
                success_count = sum(1 for r in window_results if r.get('status') == 'success')
                window_speed = len(window_configs) / window_duration if window_duration > 0 else 0
                
                window_summaries.append({
                    'window_id': window['window_id'],
                    'window_index': i,
                    'success_count': success_count,
                    'duration': window_duration,
                    'speed': window_speed,
                    'pipeline': pipeline_result
                })
                
                # Display window result
                print(f"\n✅ Window {i} complete:")
                print(f"   Backtest: {success_count}/{len(window_configs)} success")
                print(f"   Duration: {window_duration:.1f}s ({window_speed:.1f} cfg/s)")
                print(f"   IS Filter: {pipeline_result.get('is_passed', 0)} passed")
                print(f"   Correlation: {pipeline_result.get('corr_selected', 0)} selected")
                print(f"   Mega ROI: IS={pipeline_result.get('mega_is_roi', 0):.2f}% | OOS={pipeline_result.get('mega_oos_roi', 0):.2f}%")
                print(f"   Window Status: {pipeline_result.get('final_status', 'N/A')}")
            
            # Final aggregate analysis (pass pipeline summaries for enriched output)
            print(f"\n📊 Aggregating results across all {len(windows)} windows...")
            agg_start = time.time()
            analysis = self._analyze_results(all_results, windows, window_summaries)
            agg_duration = time.time() - agg_start
            print(f"✅ Aggregation complete in {agg_duration:.1f}s")
            
            duration = time.time() - start_time
            success = sum(1 for r in all_results if r.get('status') == 'success')
            
            # Print summary table
            print(f"\n{'='*70}")
            print(f"📈 WFA SUMMARY")
            print(f"{'='*70}")
            
            # Count PASS/FAIL windows
            pass_count = sum(1 for ws in window_summaries if ws['pipeline'].get('final_status') == 'PASS')
            fail_count = len(window_summaries) - pass_count
            
            for ws in window_summaries:
                pipeline = ws['pipeline']
                status_icon = "✅" if pipeline.get('final_status') == 'PASS' else "❌"
                print(f"{status_icon} {ws['window_id']:6} | "
                      f"IS: {pipeline.get('mega_is_roi', 0):6.1f}% | "
                      f"OOS: {pipeline.get('mega_oos_roi', 0):6.1f}% | "
                      f"Correl: {pipeline.get('corr_selected', 0):2} | "
                      f"{pipeline.get('final_status', 'N/A'):4}")
            
            print(f"{'='*70}")
            print(f"📊 WFA Result: {pass_count} PASS / {fail_count} FAIL (out of {len(window_summaries)} windows)")
            print(f"{'='*70}")
            
            log_wfa(f"✅ WFA Complete: {success}/{len(all_results)} success in {duration:.1f}s")
            log_wfa(f"Rate: {len(all_results)/duration:.1f} configs/s")
            log_wfa(f"Windows: {pass_count} PASS / {fail_count} FAIL")
            
            return {
                'status': 'completed',
                'batch_id': self.batch_id,
                'window_count': len(windows),   # renamed from 'windows' (int) to avoid clash with analysis.windows (dict)
                'windows_pass': pass_count,
                'windows_fail': fail_count,
                'configs_per_window': len(base_configs),
                'total_configs': total_tasks,
                'success_count': success,
                'failed_count': len(all_results) - success,
                'duration': duration,
                'rate': len(all_results) / duration if duration > 0 else 0,
                'analysis': analysis,
                # Promote analysis fields to top level so frontend can read them as
                #   wfaAnalysis.windows  (dict)  and  wfaAnalysis.aggregate  (dict)
                # without needing to go through wfaAnalysis.analysis.*
                'windows': analysis.get('windows', {}),
                'aggregate': analysis.get('aggregate', {}),
                'results': all_results,
                'window_summaries': window_summaries
            }
            
        except Exception as e:
            import traceback
            log_wfa(f"❌ Error: {str(e)}", 'error')
            log_wfa(f"Traceback: {traceback.format_exc()}", 'error')
            return {'status': 'failed', 'error': str(e), 'batch_id': self.batch_id}
    
    def _load_base_configs(self) -> List[Dict]:
        """Load base configs (from IOS or backtest)"""
        if self.mode == 'from_ios' and self.selected_config_ids:
            # Load selected configs from IOS
            configs = list(self.mongo.backtest_result.find({
                'config_hash': {'$in': self.selected_config_ids}
            }))
            log_wfa(f"Loaded {len(configs)} configs from IOS selection")
        elif self.source_batch_id:
            # Load from source backtest
            configs = list(self.mongo.backtest_result.find({
                'batch_id': self.source_batch_id,
                'status': {'$ne': 'failed'}
            }))
            log_wfa(f"Loaded {len(configs)} configs from backtest")
        else:
            # Generate fresh (would need generation logic)
            log_wfa("Fresh generation mode not yet implemented", 'warning')
            configs = []
        
        # Apply source filter if enabled
        if self.source_filter_enabled and self.source_filter_expression:
            original_count = len(configs)
            filtered_configs = []
            for cfg in configs:
                metrics = cfg.get('metrics', {})
                if self._evaluate_source_filter(metrics):
                    filtered_configs.append(cfg)
            configs = filtered_configs
            log_wfa(f"Source filter: {len(configs)}/{original_count} passed (expression: {self.source_filter_expression})")
        
        return configs
    
    def _preload_data(self, configs: List[Dict]):
        """Preload market data + indicators"""
        # Detect timeframes
        timeframes = set()
        for cfg in configs[:100]:
            params = cfg.get('params', {})
            tf = params.get('timeframe', '1h')
            timeframes.add(tf)
        
        timeframes = list(timeframes) or ['1h']
        
        # Get max indicator lengths
        indicator_config = self.config.get('indicator', {})
        max_ema = int(indicator_config.get('ema', {}).get('end', 100))
        max_atr = int(indicator_config.get('atr', {}).get('end', 30))
        
        log_wfa(f"Preloading data: {self.asset} {timeframes}")
        
        # Load with indicators
        global GLOBAL_DATA_FRAMES
        GLOBAL_DATA_FRAMES = load_and_precompute(
            asset=self.asset,
            timeframes=timeframes,
            max_ema_length=max_ema,
            max_atr_length=max_atr,
            data_type=self.config.get('data_type', 'OKX')
        )
        
        log_wfa(f"Data preloaded: {len(GLOBAL_DATA_FRAMES)} timeframes")
    
    def _create_window_configs(self, base_configs: List[Dict], windows: List[Dict]) -> List[Dict]:
        """Create config for each (base_config, window) pair"""
        all_configs = []
        
        for window in windows:
            for base_cfg in base_configs:
                params = base_cfg.get('params', {}).copy()
                
                # Add window dates
                params['window_id'] = window['window_id']
                params['start_date'] = window['is_start']
                params['end_date'] = window['os_end']
                params['is_end'] = window['is_end']
                params['os_start'] = window['os_start']
                params['os_end'] = window['os_end']
                
                all_configs.append({
                    'config_hash': f"{base_cfg.get('config_hash')}_{window['window_id']}",
                    'config_id': base_cfg.get('config_hash'),
                    'batch_id': self.batch_id,
                    'params': params
                })
        
        return all_configs

    def _create_window_configs_single(self, base_configs: List[Dict], window: Dict) -> Iterator[Dict]:
        """Create configs for a single window."""
        wid = window['window_id']
        for base_cfg in base_configs:
            params = base_cfg.get('params', {}).copy()

            # Add window dates
            params['window_id'] = wid
            params['start_date'] = window['is_start']
            params['end_date'] = window['os_end']
            params['is_end'] = window['is_end']
            params['os_start'] = window['os_start']
            params['os_end'] = window['os_end']

            yield {
                'config_hash': f"{base_cfg.get('config_hash')}_{wid}",
                'config_id': base_cfg.get('config_hash'),
                'batch_id': self.batch_id,
                'params': params,
            }
    
    def _process_window(
        self,
        configs: List[Dict],
        window_id: str,
        *,
        global_offset: int = 0,
        global_total: Optional[int] = None,
        window_index: Optional[int] = None,
        windows_total: Optional[int] = None,
    ) -> List[Dict]:
        """Process configs for a single window with multiprocessing."""
        global GLOBAL_DATA_FRAMES, GLOBAL_BROKER_CONFIG
        
        # Prepare broker config
        GLOBAL_BROKER_CONFIG = {
            'initial_capital': self.initial_capital,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct,
            'asset': self.asset
        }
        
        # Dynamic chunking
        total_configs = len(configs)
        
        # STEP 1: Calculate optimal workers based on workload
        ideal_workers = min(
            self._max_workers,
            max(1, total_configs // self.MIN_CHUNK_SIZE)
        )
        
        # STEP 2: Calculate dynamic chunk size
        chunk_size = math.ceil(total_configs / ideal_workers)
        chunk_size = max(self.MIN_CHUNK_SIZE, min(chunk_size, self.MAX_CHUNK_SIZE))
        
        # STEP 3: Final worker count (ensure not exceeding max)
        worker_count = min(self._max_workers, max(1, math.ceil(total_configs / chunk_size)))
        
        # Memory safety check - enforce HARD_LIMIT_MEMORY_GB (80GB)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        # Cap at 80GB, 2GB per worker
        enforceable_gb = min(available_gb, self.HARD_LIMIT_MEMORY_GB)
        max_workers_by_memory = int(enforceable_gb / 2.0)
        
        if worker_count > max_workers_by_memory:
            log_wfa(f"⚠️  Memory limit: reducing workers from {worker_count} to {max_workers_by_memory} (capped at {self.HARD_LIMIT_MEMORY_GB}GB)", 'warning')
            worker_count = max(1, max_workers_by_memory)
        
        # Performance tracking
        process_start = time.time()
        results = []
        last_progress_time = process_start
        configs_since_last = 0
        
        #Multiprocessing
        with Pool(processes=worker_count, initializer=init_wfa_worker) as pool:
            # Use tqdm progress bar for this window (leave=False to clear)
            with tqdm(total=total_configs, desc=f"  {window_id}", unit="cfg", ncols=100, leave=False) as pbar:
                for result in pool.imap_unordered(process_wfa_config, configs, chunksize=chunk_size):
                    results.append(result)
                    pbar.update(1)
                    configs_since_last += 1
                    
                    # Progress callback (every 100 configs or 5 seconds)
                    current_time = time.time()
                    if self.progress_callback and (len(results) % 100 == 0 or current_time - last_progress_time >= 5.0):
                        elapsed = current_time - process_start
                        configs_per_sec = len(results) / elapsed if elapsed > 0 else 0

                        global_done = int(global_offset + len(results))
                        global_tot = int(global_total if global_total is not None else total_configs)
                        win_idx = int(window_index) if window_index is not None else None
                        win_tot = int(windows_total) if windows_total is not None else None

                        prefix = f"{window_id}"
                        if win_idx is not None and win_tot is not None:
                            prefix = f"Window {win_idx}/{win_tot} ({window_id})"
                        
                        self.progress_callback({
                            # Report GLOBAL progress so UI doesn't jump backwards between windows.
                            'completed': global_done,
                            'total': global_tot,
                            'status': 'processing',
                            'configs_per_sec': configs_per_sec,
                            'elapsed': elapsed,
                            # Extra context (stored as message by API layer)
                            'message': f"{prefix}: {len(results)}/{total_configs} (global {global_done}/{global_tot})",
                            'window_id': window_id,
                            'window_completed': len(results),
                            'window_total': total_configs,
                        })
                        
                        # Periodic speed log
                        if current_time - last_progress_time >= 5.0:
                            interval_speed = configs_since_last / (current_time - last_progress_time)
                            log_wfa(f"Progress: {len(results)}/{total_configs} ({configs_per_sec:.1f} cfg/s avg, {interval_speed:.1f} cfg/s current)")
                            last_progress_time = current_time
                            configs_since_last = 0
        
        return results
    
    def _apply_window_pipeline(self, window_results: List[Dict], window: Dict) -> Dict[str, Any]:
        """
        Apply IOS pipeline cho 1 window: IS Filter → Correlation → Mega → OOS Validation
        """
        # Stage 1: IS Filter
        is_passed = self._filter_by_expression(window_results)
        
        if not is_passed:
            return {
                'is_passed': 0,
                'corr_selected': 0,
                'mega_is_roi': 0,
                'mega_oos_roi': 0,
                'final_status': 'FAIL (No IS candidates)'
            }
        
        # Stage 2: Correlation Filter
        selected = self._correlation_filter_window(is_passed)
        
        if not selected:
            return {
                'is_passed': len(is_passed),
                'corr_selected': 0,
                'mega_is_roi': 0,
                'mega_oos_roi': 0,
                'final_status': 'FAIL (No correlation survivors)'
            }
        
        # Stage 3: Mega Assembly (combine IS metrics)
        mega_is_metrics = self._create_mega_portfolio(selected, metric_key='metrics_is')
        
        # Stage 4: Mega Assembly (combine OOS metrics)
        mega_oos_metrics = self._create_mega_portfolio(selected, metric_key='metrics_oos')
        
        mega_is_roi = mega_is_metrics.get('roi', 0)
        mega_oos_roi = mega_oos_metrics.get('roi', 0)
        
        # OOS Validation
        if self.oos_expression:
            # Create a dummy result for the evaluator
            dummy_result = {'status': 'success', 'metrics_oos': mega_oos_metrics}
            passed_oos = self._filter_by_expression([dummy_result], self.oos_expression, metric_key='metrics_oos')
            final_status = 'PASS' if passed_oos else 'FAIL'
        else:
            # Legacy/Fallback: 10% OOS ROI
            final_status = 'PASS' if mega_oos_roi > 10 else 'FAIL'

        # Best config params (highest IS ROI among corr_selected)
        best_in_selected = max(selected, key=lambda x: x.get('metrics_is', {}).get('roi', 0))
        best_params = best_in_selected.get('params', {})

        return {
            'is_passed': len(is_passed),
            'corr_selected': len(selected),
            'mega_is_roi': mega_is_roi,
            'mega_oos_roi': mega_oos_roi,
            'mega_is_metrics': mega_is_metrics,
            'mega_oos_metrics': mega_oos_metrics,
            'final_status': final_status,
            'component_ids': [r.get('config_hash') for r in selected],
            'best_params': best_params,
        }
    
    def _filter_by_expression(self, results: List[Dict], expression: str = None, metric_key: str = 'metrics_is') -> List[Dict]:
        """Lọc configs theo expression (Dùng cho cả IS và OS)"""
        if expression is None:
            expression = self.expression
        
        if not expression:
            return [r for r in results if r.get('status') == 'success']

        passed = []
        failed_count = 0
        error_sample = None
        
        for r in results:
            if r.get('status') != 'success':
                continue
            
            metrics = r.get('metrics_is', {})
            
            # Debug: Check structure
            if not metrics:
                failed_count += 1
                if not error_sample:
                    # Check if metrics_is exists but empty vs not exists
                    all_keys = list(r.keys())
                    error_sample = f"Empty metrics_is. Result keys: {all_keys[:10]}"
                continue
            
            # Log first result's metrics_is keys for debugging
            if len(passed) == 0 and failed_count == 0:
                log_wfa(f"🔍 DEBUG: First result metrics_is keys: {list(metrics.keys())[:15]}")
            
            # Simple expression eval
            try:
                # ... (rest of the safe_metrics logic remains same) ...
                # Use the passed metric_key
                metrics_to_eval = r.get(metric_key, {})
                if not metrics_to_eval:
                    failed_count += 1
                    continue

                safe_metrics = {}
                for k, v in metrics_to_eval.items():
                    if isinstance(v, (list, np.ndarray)):
                        safe_metrics[k] = v[-1] if len(v) > 0 else 0
                    else:
                        safe_metrics[k] = v
                
                # Replace logic fields for compatibility with rules
                if 'total_profit' not in safe_metrics and 'profit' in safe_metrics:
                    safe_metrics['total_profit'] = safe_metrics['profit']
                if 'profit' not in safe_metrics and 'total_profit' in safe_metrics:
                    safe_metrics['profit'] = safe_metrics['total_profit']
                if 'winrate' not in safe_metrics and 'positive_ratio' in safe_metrics:
                    safe_metrics['winrate'] = safe_metrics['positive_ratio']

                # Normalize expression for eval (AND -> and, OR -> or)
                normalized_expr = expression.replace(' AND ', ' and ').replace(' OR ', ' or ')

                if eval(normalized_expr, {"__builtins__": {}}, safe_metrics):
                    passed.append(r)
            except Exception as e:
                failed_count += 1
                if not error_sample:
                    error_sample = f"{str(e)} | Available keys: {list(metrics.keys())[:10]}"
        
        # Log if many failures
        if failed_count > 0 and not passed:
            log_wfa(f"⚠️  IS Filter: 0 passed, {failed_count} failed. Error: {error_sample}", 'warning')
        
        return passed
    
    def _correlation_filter_window(self, results: List[Dict]) -> List[Dict]:
        """Correlation Filter: Chọn top N configs không tương quan"""
        # Sort by IS ROI
        sorted_results = sorted(results, key=lambda x: x.get('metrics_is', {}).get('roi', 0), reverse=True)
        
        selected = []
        for candidate in sorted_results:
            if len(selected) >= self.top_n:
                break
            
            candidate_curve = candidate.get('equity_curve', [])
            # Handle numpy arrays properly
            if candidate_curve is None or (isinstance(candidate_curve, (list, np.ndarray)) and len(candidate_curve) == 0):
                continue
            
            # Check correlation with selected
            is_correlated = False
            for sel in selected:
                sel_curve = sel.get('equity_curve', [])
                # Handle numpy arrays properly
                if sel_curve is not None and len(sel_curve) > 0:
                    if self._calc_correlation(candidate_curve, sel_curve) > self.correlation_threshold:
                        is_correlated = True
                        break
            
            if not is_correlated:
                selected.append(candidate)
        
        return selected
    
    def _calc_correlation(self, curve1: List, curve2: List) -> float:
        """Calculate Pearson correlation"""
        try:
            arr1 = np.array(curve1, dtype=float)
            arr2 = np.array(curve2, dtype=float)
            min_len = min(len(arr1), len(arr2))
            if min_len < 2:
                return 0.0
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]
            return float(np.corrcoef(arr1, arr2)[0, 1])
        except:
            return 0.0
    
    def _create_mega_portfolio(self, strategies: List[Dict], metric_key: str = 'metrics_is') -> Dict:
        """Combine strategies into mega portfolio"""
        combined_equity = None
        
        for strategy in strategies:
            eq_curve = strategy.get('equity_curve', [])
            if isinstance(eq_curve, list) and len(eq_curve) > 0:
                eq_array = np.array(eq_curve, dtype=float)
                if combined_equity is None:
                    combined_equity = eq_array.copy()
                else:
                    min_len = min(len(combined_equity), len(eq_array))
                    combined_equity = combined_equity[:min_len]
                    eq_array = eq_array[:min_len]
                    combined_equity = (combined_equity + eq_array) / 2  # Equal weight
        
        # Calculate metrics from combined equity
        if combined_equity is not None and len(combined_equity) > 0:
            # Use first strategy's trades as reference (simplified)
            all_trades = []
            for s in strategies:
                all_trades.extend(s.get('trades', []))
            
            calc = MetricsCalculator(all_trades, combined_equity, self.initial_capital)
            raw = calc.calculate_fast()
            return format_comprehensive_metrics({}, raw, self.initial_capital)
        
        return {}
    
    def _analyze_results(self, results: List[Dict], windows: List[Dict], window_summaries: List[Dict] = None) -> Dict[str, Any]:
        """Analyze results across windows.

        Merges raw backtest stats with pipeline results (IS filter → Corr → Mega → OOS)
        so the frontend receives the same rich fields it needs:
          - is_metrics / oos_metrics / total_metrics  (full metric dicts from mega portfolio)
          - passed_filters  (number of corr-selected configs)
          - params          (best config params after corr filter)
          - period          (is_start/is_end/os_start/os_end)
          - pipeline_status (PASS / FAIL)
        """
        # Index window_summaries by window_id for O(1) lookup
        pipeline_by_wid: Dict[str, Dict] = {}
        if window_summaries:
            for ws in window_summaries:
                wid = ws.get('window_id', '')
                if wid:
                    pipeline_by_wid[wid] = ws.get('pipeline', {})

        # Group raw results by window
        window_results = defaultdict(list)
        for r in results:
            if r.get('status') == 'success':
                wid = r.get('window_id', 'W0')
                window_results[wid].append(r)
        
        # Calculate per-window statistics
        window_stats = {}
        for window in windows:
            wid = window['window_id']
            window_configs = window_results.get(wid, [])
            pipeline = pipeline_by_wid.get(wid, {})

            if not window_configs and not pipeline:
                window_stats[wid] = {'status': 'no_results', 'period': window}
                continue
            
            # ── Raw aggregate stats ──────────────────────────────────────────
            avg_is_roi = float(np.mean([c['metrics_is'].get('roi', 0) for c in window_configs if 'metrics_is' in c])) if window_configs else 0.0
            avg_oos_roi = float(np.mean([c['metrics_oos'].get('roi', 0) for c in window_configs if 'metrics_oos' in c])) if window_configs else 0.0
            best = max(window_configs, key=lambda x: x.get('metrics_is', {}).get('roi', 0)) if window_configs else {}

            # ── Pipeline-enriched fields (mega portfolio metrics) ─────────────
            # is_metrics / oos_metrics come from the mega portfolio built during
            # _apply_window_pipeline; fall back to best-config metrics when pipeline
            # result is absent (e.g., early-fail window).
            mega_is_metrics = pipeline.get('mega_is_metrics') or best.get('metrics_is', {})
            mega_oos_metrics = pipeline.get('mega_oos_metrics') or best.get('metrics_oos', {})

            # Combine IS + OOS for a "total" view (simple union, prefer OOS values on
            # conflict so callers can distinguish periods by field name)
            total_metrics = {**mega_is_metrics, **{f'oos_{k}': v for k, v in mega_oos_metrics.items()}}

            # Best params: from pipeline's best corr-selected config, else raw best
            params = pipeline.get('best_params') or best.get('params', {})

            # Counts
            passed_filters = int(pipeline.get('corr_selected', 0))

            window_stats[wid] = {
                'status': 'completed',
                'period': window,
                'configs_count': len(window_configs),
                # ── Raw averaged stats (kept for backwards compat) ──────────
                'avg_is_roi': avg_is_roi,
                'avg_oos_roi': avg_oos_roi,
                'best_config': best.get('config_hash'),
                'best_is_roi': float(best.get('metrics_is', {}).get('roi', 0)) if best else 0.0,
                'best_oos_roi': float(best.get('metrics_oos', {}).get('roi', 0)) if best else 0.0,
                # ── Frontend-required fields ────────────────────────────────
                'is_metrics': mega_is_metrics,
                'oos_metrics': mega_oos_metrics,
                'total_metrics': total_metrics,
                'passed_filters': passed_filters,
                'params': params,
                'pipeline_status': pipeline.get('final_status', 'N/A'),
                'is_passed_count': int(pipeline.get('is_passed', 0)),
            }
        
        # ── Aggregate across all windows ─────────────────────────────────────
        completed_windows = [w for w in window_stats.values() if w.get('status') == 'completed']
        pass_windows = [w for w in completed_windows if w.get('pipeline_status') == 'PASS']

        aggregate = {
            'total_windows': len(windows),
            'completed_windows': len(completed_windows),
            'pass_windows': len(pass_windows),
            'fail_windows': len(completed_windows) - len(pass_windows),
            'avg_is_roi_all_windows': float(np.mean([w['avg_is_roi'] for w in completed_windows])) if completed_windows else 0.0,
            'avg_oos_roi_all_windows': float(np.mean([w['avg_oos_roi'] for w in completed_windows])) if completed_windows else 0.0,
            'total_is_roi': float(sum(w['avg_is_roi'] for w in completed_windows)),
            'total_oos_roi': float(sum(w['avg_oos_roi'] for w in completed_windows)),
            # Mega-portfolio based aggregates (more accurate than raw avg)
            'avg_mega_is_roi': float(np.mean([w['is_metrics'].get('roi', 0) for w in completed_windows])) if completed_windows else 0.0,
            'avg_mega_oos_roi': float(np.mean([w['oos_metrics'].get('roi', 0) for w in completed_windows])) if completed_windows else 0.0,
        }
        
        log_wfa(f"Aggregate: {aggregate['total_oos_roi']:.2f}% total OOS ROI across {len(completed_windows)} windows "
                f"({len(pass_windows)} PASS / {len(completed_windows) - len(pass_windows)} FAIL)")
        
        return {
            'windows': window_stats,
            'aggregate': aggregate
        }
    
    def _evaluate_source_filter(self, metrics: Dict) -> bool:
        """Evaluate source filter expression on metrics"""
        if not self.source_filter_expression:
            return True
        
        # Field mapping for source filter
        field_map = {
            'profit': 'total_profit',
            'total_profit': 'total_profit',
            'mdd': 'mdd',
            'sharpe': 'sharpe_ratio',
            'roi': 'roi',
            'winrate': 'positive_ratio',
            'positive_ratio': 'positive_ratio',
            'trades': 'trades'
        }
        
        try:
            # Prepare safe metrics dict for eval
            safe_metrics = {}
            for field, metric_key in field_map.items():
                value = metrics.get(metric_key, 0)
                # Handle numpy arrays
                if isinstance(value, (list, np.ndarray)):
                    safe_metrics[field] = value[-1] if len(value) > 0 else 0
                else:
                    safe_metrics[field] = value
            
            # Support both lowercase and uppercase operators
            expr = self.source_filter_expression.replace(' AND ', ' and ').replace(' OR ', ' or ')
            return bool(eval(expr, {"__builtins__": {}}, safe_metrics))
        except Exception as e:
            log_wfa(f"⚠️  Source filter eval error: {e}", 'warning')
            return True  # Pass through on error
