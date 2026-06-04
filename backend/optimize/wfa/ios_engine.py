"""
IOS (In-Out Sample) Engine
Single window validation with multi-stage pipeline

Pipeline:
- Stage 0 (Optional): Yearly Pre-Filter
- Stage 1: In-Sample Filter (train period)
- Stage 2: Correlation Filter (remove correlated strategies)
- Stage 3: Mega Assembly (combine into portfolio)
- Stage 4: Out-Sample Validation (test period)

Performance: Optimized with multiprocessing (target: 800-1000 cfg/s)
Pattern: Follows backtest optimizer architecture for maximum speed
"""

import os
import time
import math
import warnings
import psutil
import multiprocessing
from multiprocessing import Pool
from typing import Dict, Any, List, Optional, Callable, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

from database.mongo_service import MongoService
from optimize.prepare import load_and_precompute
from core.broker import Broker
from metrics.formatter import format_comprehensive_metrics, sanitize_dict
from metrics.calculator import MetricsCalculator
from optimize.post_process import save_results_batch

import logging
logger = logging.getLogger(__name__)

# Verbose logging
VERBOSE_LOGGING = str(os.getenv('GREY_IOS_VERBOSE', '0')).lower() in {'1', 'true', 'yes'}

def log_ios(msg: str, level: str = 'info'):
    """Log with IOS prefix"""
    if VERBOSE_LOGGING or level in {'warning', 'error'}:
        prefix = "[IOS]"
        getattr(logger, level)(f"{prefix} {msg}")


# ============================================================================
# GLOBAL STATE (Fork Copy-on-Write for performance)
# ============================================================================
GLOBAL_DATA_FRAMES = None
GLOBAL_BROKER = None
GLOBAL_BROKER_CONFIG = None

def init_ios_worker():
    """Initialize worker process"""
    global GLOBAL_BROKER
    GLOBAL_BROKER = None


def process_ios_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process single config using shared global data
    Optimized: Reuse broker, precomputed indicators, cached data
    """
    global GLOBAL_DATA_FRAMES, GLOBAL_BROKER, GLOBAL_BROKER_CONFIG
    
    try:
        # Lazy broker init (once per worker)
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
        
        # Get cached data
        if timeframe not in GLOBAL_DATA_FRAMES:
            return {
                'config_hash': config.get('config_hash'),
                'batch_id': config['batch_id'],
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
        
        # Run backtest with date slicing
        asset = params.get('asset', GLOBAL_BROKER_CONFIG.get('asset', 'BTCUSDT'))
        start_date = params.get('start_date') or GLOBAL_BROKER_CONFIG.get('start_date')
        end_date = params.get('end_date') or GLOBAL_BROKER_CONFIG.get('end_date')
        
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
        
        # Format metrics
        raw_metrics = result.get('metrics', {})
        formatted_metrics = format_comprehensive_metrics(params, raw_metrics, GLOBAL_BROKER_CONFIG['initial_capital'])
        
        return {
            'config_hash': config.get('config_hash') or config.get('_id'),
            'batch_id': config['batch_id'],
            'params': sanitize_dict(params),
            'metrics': sanitize_dict(formatted_metrics),
            'equity_curve': result.get('equity_curve'),
            'trades': result.get('trades', []),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'config_hash': config.get('config_hash'),
            'batch_id': config.get('batch_id'),
            'params': config.get('params', {}),
            'metrics': {},
            'status': 'failed',
            'error': str(e)
        }


class IOSEngine:
    """
    IOS Engine - Single window validation
    
    Performance optimizations:
    - Multiprocessing with fork COW
    - Precomputed indicator matrices
    - Dynamic chunking (optimized for single window)
    - Batch database writes
    
    ⚠️ SYSTEM-WIDE Resource Limits:
    - Max 40 workers (entire system, not per operation)
    - Max 80GB RAM (entire system, shared across all operations)
    - Physical resources ignored, only hard limits enforced
    """
    
    # Resource limits (SYSTEM-WIDE, shared across ALL operations)
    HARD_LIMIT_WORKERS = 40
    HARD_LIMIT_MEMORY_GB = 80  # System-wide cap, ignores physical RAM
    
    # Chunking params (optimized for shorter periods)
    MIN_CHUNK_SIZE = 1         # Can process even 1 config
    MAX_CHUNK_SIZE = 100       # Larger chunks OK for single window
    
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
        
        # Extract IOS config
        ios_config = config.get('ios', {})
        self.in_sample_period = ios_config.get('in_sample_period', {})
        self.out_sample_period = ios_config.get('out_sample_period', {})
        self.expression = ios_config.get('expression', '')
        self.oos_expression = ios_config.get('oos_expression', '')
        self.correlation_threshold = ios_config.get('correlation_threshold', 0.8)
        self.top_n = ios_config.get('top_n', 20)
        
        # Yearly filter (Stage 0)
        yearly_config = ios_config.get('yearly_filter', {})
        self.yearly_filter_enabled = yearly_config.get('enabled', False)
        self.yearly_filter_expression = yearly_config.get('expression', '')
        self.yearly_filter_years = yearly_config.get('years')
        
        # Source filter (pre-filter backtest results)
        source_filter_config = config.get('source_filter', {})
        self.source_filter_enabled = bool(source_filter_config)
        self.source_filter_expression = source_filter_config.get('expression', '') if source_filter_config else ''
        
        # Source configs
        self.source_batch_id = config.get('source_batch_id')
        
        # Extract common broker config
        self.asset = config.get('asset', 'BTCUSDT')
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission_pct = config.get('commission_pct', 0.1)
        self.slippage_pct = config.get('slippage_pct', 0.0)
        
        # Resource settings
        self._max_workers = self._compute_max_workers(config.get('max_workers'))
        
        log_ios(f"Initialized IOS Engine for batch {batch_id}")
    
    def _compute_max_workers(self, override: Optional[int] = None) -> int:
        """Calculate optimal worker count"""
        cpu_count = os.cpu_count() or 4
        hard_cap = max(1, self.HARD_LIMIT_WORKERS - 1)
        
        if override:
            return max(1, min(override, hard_cap))
        
        return max(1, min(cpu_count - 1, hard_cap))
    
    def run(self) -> Dict[str, Any]:
        """
        Execute IOS pipeline
        Returns: Result dict with stage results and final status
        """
        start_time = time.time()
        stage_times = {}  # Track time per stage
        
        log_ios(f"🚀 Starting IOS analysis")
        log_ios(f"IS Period: {self.in_sample_period}")
        log_ios(f"OS Period: {self.out_sample_period}")
        
        try:
            # Load source configs
            stage_start = time.time()
            configs = self._load_source_configs()
            if not configs:
                return {'status': 'failed', 'error': 'No configs found', 'stage': 'load'}
            
            stage_times['load'] = time.time() - stage_start
            log_ios(f"Loaded {len(configs)} configs from source ({stage_times['load']:.1f}s)")
            
            # Preload data for performance
            stage_start = time.time()
            self._preload_data(configs)
            stage_times['preload'] = time.time() - stage_start
            log_ios(f"Preloaded data ({stage_times['preload']:.1f}s)")
            
            # Stage 0: Yearly Filter (optional)
            if self.yearly_filter_enabled:
                stage_start = time.time()
                log_ios("Stage 0: Yearly Filter")
                configs = self._stage0_yearly_filter(configs)
                stage_times['stage0'] = time.time() - stage_start
                if not configs:
                    return {'status': 'failed', 'error': 'No configs passed yearly filter', 'stage': '0'}
                log_ios(f"Stage 0 Complete: {len(configs)} configs passed ({stage_times['stage0']:.1f}s)")
            
            # Stage 1: In-Sample Filter
            stage_start = time.time()
            log_ios(f"Stage 1: In-Sample Filter ({len(configs)} configs)")
            is_results = self._stage1_in_sample_filter(configs)
            stage_times['stage1'] = time.time() - stage_start
            
            if not is_results:
                return {'status': 'failed', 'error': 'No configs passed IS filter', 'stage': '1'}
            
            # Calculate processing speed for Stage 1
            stage1_speed = len(configs) / stage_times['stage1'] if stage_times['stage1'] > 0 else 0
            log_ios(f"Stage 1 Complete: {len(is_results)} candidates ({stage_times['stage1']:.1f}s, {stage1_speed:.1f} cfg/s)")
            
            # Stage 2: Correlation Filter
            stage_start = time.time()
            log_ios(f"Stage 2: Correlation Filter ({len(is_results)} candidates)")
            selected_strategies = self._stage2_correlation_filter(is_results)
            stage_times['stage2'] = time.time() - stage_start
            
            if not selected_strategies:
                return {'status': 'failed', 'error': 'No configs passed correlation filter', 'stage': '2'}
            
            log_ios(f"Stage 2 Complete: {len(selected_strategies)} strategies ({stage_times['stage2']:.1f}s)")
            
            # Stage 3: Mega Assembly
            stage_start = time.time()
            log_ios(f"Stage 3: Mega Assembly ({len(selected_strategies)} strategies)")
            mega_result = self._stage3_mega_assembly(selected_strategies)
            stage_times['stage3'] = time.time() - stage_start
            log_ios(f"Stage 3 Complete ({stage_times['stage3']:.1f}s)")
            
            # Stage 4: Out-Sample Validation
            stage_start = time.time()
            log_ios("Stage 4: Out-Sample Validation")
            os_validation = self._stage4_out_sample_validation(mega_result)
            stage_times['stage4'] = time.time() - stage_start
            log_ios(f"Stage 4 Complete ({stage_times['stage4']:.1f}s)")
            
            duration = time.time() - start_time
            
            # Performance summary
            total_configs_processed = len(configs)  # Original config count for Stage 1
            avg_speed = total_configs_processed / stage_times.get('stage1', 1) if stage_times.get('stage1', 0) > 0 else 0
            
            # Log detailed performance
            log_ios("="*60)
            log_ios("Performance Summary:")
            for stage_name, stage_time in stage_times.items():
                log_ios(f"  {stage_name}: {stage_time:.1f}s")
            log_ios(f"  Total: {duration:.1f}s")
            log_ios(f"  Processing Speed (Stage 1): {avg_speed:.1f} cfg/s")
            log_ios("="*60)
            
            result = {
                'status': 'completed',
                'batch_id': self.batch_id,
                'source_batch_id': self.source_batch_id,
                'stages': {
                    'yearly_filter': {'enabled': self.yearly_filter_enabled},
                    'in_sample': {'candidates_count': len(is_results)},
                    'correlation': {'selected_count': len(selected_strategies)},
                    'mega': mega_result,
                    'out_sample': os_validation
                },
                'final_status': os_validation.get('status', 'FAIL'),
                'duration': duration,
                'stage_times': stage_times,
                'performance': {
                    'configs_processed': total_configs_processed,
                    'configs_per_sec': avg_speed
                },
                'created_at': start_time
            }
            
            log_ios(f"✅ IOS Complete: {result['final_status']} in {duration:.1f}s ({avg_speed:.1f} cfg/s)")
            return result
            
        except Exception as e:
            log_ios(f"❌ Error: {str(e)}", 'error')
            return {'status': 'failed', 'error': str(e), 'batch_id': self.batch_id}
    
    def _load_source_configs(self) -> List[Dict]:
        """Load configs from source backtest campaign"""
        if not self.source_batch_id:
            raise ValueError("source_batch_id required for IOS")
        
        configs = list(self.mongo.backtest_result.find({
            'batch_id': self.source_batch_id,
            'status': {'$ne': 'failed'}
        }))
        
        # Apply source filter if enabled
        if self.source_filter_enabled and self.source_filter_expression:
            original_count = len(configs)
            filtered_configs = []
            for cfg in configs:
                metrics = cfg.get('metrics', {})
                if self._evaluate_source_filter(metrics):
                    filtered_configs.append(cfg)
            configs = filtered_configs
            log_ios(f"Source filter: {len(configs)}/{original_count} passed (expression: {self.source_filter_expression})")
        
        return configs
    
    def _preload_data(self, configs: List[Dict]):
        """Preload market data + indicators"""
        # Detect timeframes from configs
        timeframes = set()
        for cfg in configs[:100]:  # Sample first 100
            params = cfg.get('params', {})
            tf = params.get('timeframe', '1h')
            timeframes.add(tf)
        
        timeframes = list(timeframes) or ['1h']
        
        # Get max indicator lengths
        indicator_config = self.config.get('indicator', {})
        max_ema = int(indicator_config.get('ema', {}).get('end', 100))
        max_atr = int(indicator_config.get('atr', {}).get('end', 30))
        
        log_ios(f"Preloading data: {self.asset} {timeframes}")
        
        # Load with indicators
        global GLOBAL_DATA_FRAMES
        GLOBAL_DATA_FRAMES = load_and_precompute(
            asset=self.asset,
            timeframes=timeframes,
            max_ema_length=max_ema,
            max_atr_length=max_atr,
            data_type=self.config.get('data_type', 'OKX')
        )
        
        log_ios(f"Data preloaded: {len(GLOBAL_DATA_FRAMES)} timeframes")
    
    def _stage0_yearly_filter(self, configs: List[Dict]) -> List[Dict]:
        """Stage 0: Yearly Pre-Filter (optional)"""
        # TODO: Implement yearly filter logic similar to in_out_sample/stages.py
        # For now, pass through
        log_ios(f"Yearly filter: {len(configs)} configs passed")
        return configs
    
    def _stage1_in_sample_filter(self, configs: List[Dict]) -> List[Dict]:
        """
        Stage 1: Filter configs on In-Sample period
        Uses multiprocessing for speed
        """
        global GLOBAL_DATA_FRAMES, GLOBAL_BROKER_CONFIG
        
        # Prepare broker config
        GLOBAL_BROKER_CONFIG = {
            'initial_capital': self.initial_capital,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct,
            'asset': self.asset,
            'start_date': self.in_sample_period.get('start'),
            'end_date': self.in_sample_period.get('end')
        }
        
        # Prepare configs for processing
        tasks = []
        for cfg in configs:
            params = cfg.get('params', {})
            params['start_date'] = self.in_sample_period.get('start')
            params['end_date'] = self.in_sample_period.get('end')
            
            tasks.append({
                'config_hash': cfg.get('config_hash') or cfg.get('_id'),
                'batch_id': self.batch_id,
                'params': params
            })
        
        # Dynamic chunking (same logic as optimizer)
        total_tasks = len(tasks)
        
        # Calculate optimal workers
        ideal_workers = min(
            self._max_workers,
            max(1, total_tasks // self.MIN_CHUNK_SIZE)
        )
        
        # Calculate dynamic chunk size
        chunk_size = math.ceil(total_tasks / ideal_workers)
        chunk_size = max(self.MIN_CHUNK_SIZE, min(chunk_size, self.MAX_CHUNK_SIZE))
        
        # Final worker count
        worker_count = min(self._max_workers, max(1, math.ceil(total_tasks / chunk_size)))
        
        # Memory safety check - enforce HARD_LIMIT_MEMORY_GB (80GB)
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        # Cap at 80GB, 2GB per worker
        enforceable_gb = min(available_gb, self.HARD_LIMIT_MEMORY_GB)
        max_workers_by_memory = int(enforceable_gb / 2.0)
        
        if worker_count > max_workers_by_memory:
            log_ios(f"⚠️  Memory limit: reducing workers from {worker_count} to {max_workers_by_memory} (capped at {self.HARD_LIMIT_MEMORY_GB}GB)", 'warning')
            worker_count = max(1, max_workers_by_memory)
        
        log_ios(f"Running IS filter: {total_tasks} configs, {worker_count} workers, chunk_size: {chunk_size}")
        
        # Multiprocessing
        results = []
        with Pool(processes=worker_count, initializer=init_ios_worker) as pool:
            with tqdm(total=len(tasks), desc="IS Filter", disable=not VERBOSE_LOGGING) as pbar:
                for result in pool.imap_unordered(process_ios_config, tasks, chunksize=chunk_size):
                    if result.get('status') == 'success':
                        # Filter by expression (In-Sample Filter)
                        if self._evaluate_expression(result['metrics'], self.expression):
                            results.append(result)
                    pbar.update(1)
        
        log_ios(f"IS Filter: {len(results)}/{len(tasks)} passed")
        return results
    
    def _stage2_correlation_filter(self, results: List[Dict]) -> List[Dict]:
        """
        Stage 2: Remove highly correlated strategies
        """
        if len(results) <= self.top_n:
            return results
        
        # Extract equity curves
        equity_curves = []
        for r in results:
            eq = r.get('equity_curve', [])
            if isinstance(eq, list) and len(eq) > 0:
                equity_curves.append(np.array(eq, dtype=float))
            else:
                equity_curves.append(np.array([self.initial_capital], dtype=float))
        
        # Greedy selection: Pick best uncorrelated strategies
        selected = []
        selected_curves = []
        
        # Sort by ROI descending
        sorted_results = sorted(results, key=lambda x: x['metrics'].get('roi', 0), reverse=True)
        
        for i, result in enumerate(sorted_results):
            if len(selected) >= self.top_n:
                break
            
            curve = equity_curves[results.index(result)]
            
            # Check correlation with already selected
            is_correlated = False
            for sel_curve in selected_curves:
                corr = self._calculate_correlation(curve, sel_curve)
                if abs(corr) > self.correlation_threshold:
                    is_correlated = True
                    break
            
            if not is_correlated:
                selected.append(result)
                selected_curves.append(curve)
        
        log_ios(f"Correlation filter: {len(selected)}/{len(results)} selected (threshold: {self.correlation_threshold})")
        return selected
    
    def _stage3_mega_assembly(self, strategies: List[Dict]) -> Dict[str, Any]:
        """
        Stage 3: Combine strategies into mega portfolio
        """
        # Calculate combined metrics on IS period
        combined_equity = None
        all_trades = []
        
        for strategy in strategies:
            eq_curve = strategy.get('equity_curve', [])
            if isinstance(eq_curve, list) and len(eq_curve) > 0:
                eq_array = np.array(eq_curve, dtype=float)
                if combined_equity is None:
                    combined_equity = eq_array.copy()
                else:
                    # Simple average (equal weight)
                    min_len = min(len(combined_equity), len(eq_array))
                    combined_equity = combined_equity[:min_len]
                    eq_array = eq_array[:min_len]
                    combined_equity = (combined_equity + eq_array) / 2
            
            trades = strategy.get('trades', [])
            all_trades.extend(trades)
        
        # Calculate portfolio metrics
        if combined_equity is not None and len(combined_equity) > 0:
            calc = MetricsCalculator(all_trades, combined_equity, self.initial_capital)
            raw_metrics = calc.calculate_fast()
            mega_metrics = format_comprehensive_metrics({}, raw_metrics, self.initial_capital)
        else:
            mega_metrics = {}
        
        component_ids = [s.get('config_hash') for s in strategies]
        
        log_ios(f"Mega assembly: {len(component_ids)} components, ROI: {mega_metrics.get('roi', 0):.2f}%")
        
        return {
            'component_ids': component_ids,
            'component_count': len(component_ids),
            'metrics': mega_metrics,
            'equity_curve': combined_equity.tolist() if combined_equity is not None else []
        }
    
    def _stage4_out_sample_validation(self, mega_result: Dict) -> Dict[str, Any]:
        """
        Stage 4: Validate mega portfolio on Out-Sample period
        """
        component_ids = mega_result.get('component_ids', [])
        
        if not component_ids:
            return {'status': 'FAIL', 'reason': 'No components in mega portfolio'}
        
        # Re-run components on OS period
        configs = list(self.mongo.backtest_result.find({
            'config_hash': {'$in': component_ids}
        }))
        
        global GLOBAL_BROKER_CONFIG
        GLOBAL_BROKER_CONFIG['start_date'] = self.out_sample_period.get('start')
        GLOBAL_BROKER_CONFIG['end_date'] = self.out_sample_period.get('end')
        
        # Prepare tasks
        tasks = []
        for cfg in configs:
            params = cfg.get('params', {})
            params['start_date'] = self.out_sample_period.get('start')
            params['end_date'] = self.out_sample_period.get('end')
            tasks.append({
                'config_hash': cfg.get('config_hash'),
                'batch_id': self.batch_id,
                'params': params
            })
        
        # Run on OS period
        log_ios(f"Running OS validation: {len(tasks)} configs")
        
        results = []
        worker_count = min(self._max_workers, max(1, len(tasks) // 50))
        with Pool(processes=worker_count, initializer=init_ios_worker) as pool:
            for result in pool.imap_unordered(process_ios_config, tasks):
                if result.get('status') == 'success':
                    results.append(result)
        
        # Combine OS results (simple average)
        combined_equity = None
        all_trades = []
        
        for result in results:
            eq = result.get('equity_curve', [])
            if isinstance(eq, list) and len(eq) > 0:
                eq_array = np.array(eq, dtype=float)
                if combined_equity is None:
                    combined_equity = eq_array.copy()
                else:
                    min_len = min(len(combined_equity), len(eq_array))
                    combined_equity[:min_len] = (combined_equity[:min_len] + eq_array[:min_len]) / 2
            
            all_trades.extend(result.get('trades', []))
        
        # Calculate OS metrics
        if combined_equity is not None and len(combined_equity) > 0:
            calc = MetricsCalculator(all_trades, combined_equity, self.initial_capital)
            raw_metrics = calc.calculate_fast()
            os_metrics = format_comprehensive_metrics({}, raw_metrics, self.initial_capital)
        else:
            os_metrics = {}
        
        # Check acceptance criteria
        is_metrics = mega_result.get('metrics', {})
        status = self._check_acceptance_criteria(is_metrics, os_metrics)
        
        log_ios(f"OS Validation: {status} | IS ROI: {is_metrics.get('roi', 0):.2f}% | OS ROI: {os_metrics.get('roi', 0):.2f}%")
        
        return {
            'status': status,
            'is_metrics': is_metrics,
            'os_metrics': os_metrics,
            'component_count': len(results)
        }
    
    def _evaluate_expression(self, metrics: Dict, expression: str = '') -> bool:
        """Evaluate filter expression on metrics"""
        if not expression:
            return True
        
        # Simple expression parser (supports: profit>100, mdd<15, etc.)
        field_map = {
            'profit': 'total_profit',
            'mdd': 'mdd',
            'sharpe': 'sharpe_ratio',
            'roi': 'roi',
            'winrate': 'positive_ratio',
            'trades': 'trades'
        }
        
        clauses = [c.strip() for c in expression.upper().split('AND')]
        
        for clause in clauses:
            for field, metric_key in field_map.items():
                if field.upper() in clause:
                    value = metrics.get(metric_key, 0)
                    
                    if '>=' in clause:
                        threshold = float(clause.split('>=')[1].strip())
                        if not (value >= threshold):
                            return False
                    elif '>' in clause:
                        threshold = float(clause.split('>')[1].strip())
                        if not (value > threshold):
                            return False
                    elif '<=' in clause:
                        threshold = float(clause.split('<=')[1].strip())
                        if not (value <= threshold):
                            return False
                    elif '<' in clause:
                        threshold = float(clause.split('<')[1].strip())
                        if not (value < threshold):
                            return False
                    break
        
        return True
    
    def _calculate_correlation(self, eq1: np.ndarray, eq2: np.ndarray) -> float:
        """Calculate Pearson correlation between two equity curves"""
        try:
            min_len = min(len(eq1), len(eq2))
            if min_len < 2:
                return 0.0
            
            a = eq1[:min_len]
            b = eq2[:min_len]
            
            a_norm = a - a.mean()
            b_norm = b - b.mean()
            
            denom = float(np.sqrt(np.dot(a_norm, a_norm) * np.dot(b_norm, b_norm)))
            if denom == 0:
                return 0.0
            
            return float(np.dot(a_norm, b_norm) / denom)
        except:
            return 0.0
    
    def _check_acceptance_criteria(self, is_metrics: Dict, os_metrics: Dict) -> str:
        """Check if OS performance meets acceptance criteria"""
        # If no OOS expression, fallback to legacy profit > 0 check
        if not self.oos_expression:
            return 'PASS' if os_metrics.get('total_profit', 0) > 0 else 'FAIL'
        
        # Evaluate dynamic OOS expression
        return 'PASS' if self._evaluate_expression(os_metrics, self.oos_expression) else 'FAIL'
    
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
            log_ios(f"⚠️  Source filter eval error: {e}", 'warning')
            return True  # Pass through on error
