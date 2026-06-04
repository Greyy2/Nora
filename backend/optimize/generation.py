"""
Generation Module
Generates parameter combinations matching Vinh's 12-param structure
"""
import itertools
import hashlib 
import json 
from typing import Dict, Any, Iterator, List
from datetime import datetime 

from database.mongo_service import MongoService

def frange(start: float, stop: float, step: float) -> List[float]:
    """
    Generate float range (matching Vinh's frange)
    
    Args:
        start: Start value
        stop: Stop value (inclusive)
        step: Step size
        
    Returns:
        List of float values
    """
    if step <= 0:
        return [round(start, 10)]
        
    result = []
    current = start
    # Small epsilon for float comparison
    while current <= stop + 1e-9:
        result.append(round(current, 10))
        current += step
    return result or [round(start, 10)]

def estimate_config_count(config: Dict[str, Any]) -> int:
    """
    Estimate total config count without generating them
    
    Used for early progress tracker initialization to avoid 504 timeouts
    """
    try:
        # Parse timeframes (supports dict, list, or scalar)
        tf_cfg = config.get('timeframes', {})
        if isinstance(tf_cfg, dict):
            start_val = _parse_hours(tf_cfg.get('start', '4h'))
            end_val = _parse_hours(tf_cfg.get('end', '4h'))
            step_val = _parse_hours(tf_cfg.get('step', '1h'))
            tf_count = max(1, len(frange(start_val, end_val, step_val)))
        elif isinstance(tf_cfg, list):
            tf_count = len(tf_cfg) if len(tf_cfg) > 0 else 1
        else:
            tf_count = 1

        # Parse indicator ranges
        ind = config.get('indicator', {})
        ema_count = len(frange(
            ind.get('ema', {}).get('start', 50),
            ind.get('ema', {}).get('end', 200),
            ind.get('ema', {}).get('step', 50)
        ))
        atr_count = len(frange(
            ind.get('atr', {}).get('start', 14),
            ind.get('atr', {}).get('end', 14),
            ind.get('atr', {}).get('step', 1)
        ))
        high_vf_count = len(frange(
            ind.get('high_vf', {}).get('start', 1.0),
            ind.get('high_vf', {}).get('end', 1.0),
            ind.get('high_vf', {}).get('step', 0.1)
        ))
        low_vf_count = len(frange(
            ind.get('low_vf', {}).get('start', 0.5),
            ind.get('low_vf', {}).get('end', 1.0),
            ind.get('low_vf', {}).get('step', 0.1)
        ))

        # Parse PS ranges
        ps = config.get('ps', {})
        ir_count = len(frange(
            ps.get('ir', {}).get('start', 0.01),
            ps.get('ir', {}).get('end', 0.03),
            ps.get('ir', {}).get('step', 0.01)
        ))
        er_count = len(frange(
            ps.get('er', {}).get('start', 0.5),
            ps.get('er', {}).get('end', 1.0),
            ps.get('er', {}).get('step', 0.2)
        ))

        # OR range - check if provided
        or_cfg = ps.get('or')
        if or_cfg and isinstance(or_cfg, dict) and or_cfg:
            or_count = len(frange(
                or_cfg.get('start', 0.0),
                or_cfg.get('end', 0.0),
                or_cfg.get('step', 0.01)
            ))
        else:
            or_count = 1

        # Calculate total combinations
        base_total = tf_count * ema_count * atr_count * high_vf_count * low_vf_count * ir_count * er_count * or_count

        multiple = ind.get('multiple', 1)
        side_count = 1  # Only 'long' for now
        total = base_total * multiple * side_count

        print(
            f"📊 Config estimate: {tf_count} TF × {ema_count} EMA × {atr_count} ATR × "
            f"{high_vf_count} HighVF × {low_vf_count} LowVF × {ir_count} IR × {er_count} ER × "
            f"{or_count} OR × {multiple} mult × {side_count} side = {total:,}"
        )
        return total
    except Exception as e:
        print(f"⚠️  Error estimating config count: {e}")
        return 10000

def _parse_hours(value) -> int:
    """Parse timeframe value to hours (e.g., '1h' -> 1, 24 -> 24, '5m' -> 5, '1d' -> 1)"""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().lower()
        # Remove any unit suffix (h, m, d) - we just extract the numeric part
        # The actual unit handling is done by _build_timeframes
        for suffix in ['h', 'm', 'd']:
            if value.endswith(suffix):
                value = value[:-1]
                break
        return int(value)
    raise ValueError(f"Unsupported timeframe value: {value}")

def _build_timeframes(timeframes_cfg) -> List[str]:
    """Build timeframe list from config (dict with start/end/step/unit or list)"""
    if isinstance(timeframes_cfg, dict):
        start = _parse_hours(timeframes_cfg.get('start', '1h'))
        end = _parse_hours(timeframes_cfg.get('end', '24h'))
        step = max(1, _parse_hours(timeframes_cfg.get('step', '1h')))
        unit = timeframes_cfg.get('unit', 'h')  # Get the unit: 'm', 'h', or 'd'
        
        if end < start:
            raise ValueError('timeframes.end must be >= timeframes.start')
        
        # Build list with proper unit suffix
        return [f"{hour}{unit}" for hour in range(start, end + 1, step)]
    if isinstance(timeframes_cfg, list):
        return timeframes_cfg
    return [str(timeframes_cfg)]

def generate_param_hash(params: Dict[str, Any]) -> str:
    """Generate unique hash for params"""
    # Sort keys for consistent hash
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def generate_params(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Generate parameter combinations using itertools.product
    Matching param structure exactly
    
    Args:
        config: Config dict with param ranges
    
    Yields:
        Dict with params for each combination (12 params)
    """
    try:
        # Extract config sections
        indicator = config.get('indicator', {})
        ps = config.get('ps', {})
        buy_sell_engine = config.get('buy_sell_engine', {})
        
        # Helper to get range safely
        def get_range_safely(section, key, default_start, default_end, default_step):
            cfg = section.get(key)
            if not isinstance(cfg, dict):
                cfg = {}
            start = cfg.get('start', default_start)
            end = cfg.get('end', default_end)
            step = cfg.get('step', default_step)
            return frange(start, end, step)

        # Helper to get int range safely
        def get_int_range_safely(section, key, default_start, default_end, default_step):
            cfg = section.get(key)
            if not isinstance(cfg, dict):
                cfg = {}
            start = int(cfg.get('start', default_start))
            end = int(cfg.get('end', default_end))
            step = max(1, int(cfg.get('step', default_step)))
            return sorted(set(range(start, end + 1, step)))

        # 1. EMA range
        ema_range = get_int_range_safely(indicator, 'ema', 10, 50, 10)
        
        # 2. ATR range
        atr_range = get_int_range_safely(indicator, 'atr', 14, 14, 1)
        
        # 3. High VF (long_vol_factor) range
        high_vf_range = sorted(set(get_range_safely(indicator, 'high_vf', 1.0, 1.0, 0.1)))
        
        # 4. Low VF (short_vol_factor) range
        low_vf_range = sorted(set(get_range_safely(indicator, 'low_vf', 1.0, 1.0, 0.1)))
        
        # 5. IR (initial_risk) range
        ir_range = sorted(set(get_range_safely(ps, 'ir', 0.01, 0.01, 0.01)))
        
        # 6. ER (equity_risk) range
        er_range = sorted(set(get_range_safely(ps, 'er', 0.5, 0.5, 0.1)))
        
        # 7. OR (ongoing_risk) range - default to single value [0] if not provided
        or_cfg = ps.get('or')
        if or_cfg is None or (isinstance(or_cfg, dict) and not or_cfg):
            # No OR config provided - use single value [0]
            or_range = [0.0]
        else:
            # Parse OR range from config
            or_range = sorted(set(get_range_safely(ps, 'or', 0.0, 0.0, 0.01)))
        
        # 8. is_on_going (boolean)
        is_on_going = [buy_sell_engine.get('is_on_going', False)]
        
        # 9. side (string)
        side = [indicator.get('side', buy_sell_engine.get('side', 'both'))]
        
        # 10. use_delta (boolean)
        use_delta = [config.get('use_delta', False)]
        
        # 11. multiple (int)
        multiple = [indicator.get('multiple', 1)]
        
        # 12. timeframes (list) - parse from config
        timeframes_cfg = config.get('timeframes', ['4h'])
        timeframes = _build_timeframes(timeframes_cfg)
        
        # Generate all combinations (12 params)
        for (tf, e, a, hvf, lvf, irv, erv, orv, 
             ongoing, s, delta, mul) in itertools.product(
            timeframes,
            ema_range,
            atr_range,
            high_vf_range,
            low_vf_range,
            ir_range,
            er_range,
            or_range,
            is_on_going,
            side,
            use_delta,
            multiple
        ):
            # Build param dict matching Vinh's structure
            param_dict = {
                'length_ema': e,
                'length_atr': a,
                'long_vol_factor': hvf,
                'short_vol_factor': lvf,
                'use_delta': delta,
                'multiple': mul,
                'timeframe': tf,
                'frequency': tf,  # Same as timeframe
                'strategy': {
                    'bse': {
                        'is_on_going': ongoing,
                        'side': s
                    },
                    'ps': {
                        'ir': irv,
                        'er': erv,
                        'or': orv
                    }
                }
            }
            
            yield param_dict
    except Exception as e:
        print(f"ERROR in generate_params: {e}")
        import traceback
        traceback.print_exc()
        raise e

def save_to_mongodb(batch_id: str, config: Dict[str, Any], mongo: MongoService, batch_size: int = 50000, collection_type: str = 'backtest', progress_callback: callable = None) -> int:
    """
    Generate params and batch insert to MongoDB (NEW STRUCTURE)
    Uses backtest-config or wfo-config collection based on collection_type
    
    Args:
        batch_id: Batch ID
        config: Config dict
        collection_type: 'backtest' or 'wfo' (determines which collection to use)
        mongo: MongoService instance
        batch_size: Batch size for bulk insert
        progress_callback: Optional callback for progress updates
    
    Returns:
        Total inserted count
    """
    asset = config.get('asset', 'BTCUSDT')
    
    # STEP 2: Generate strategies
    print(f"\n📋 STEP 2: Sinh chiến thuật")
    
    # Build batch name from config
    ind = config.get('indicator', {})
    ema_cfg = ind.get('ema', {})
    atr_cfg = ind.get('atr', {})
    highvf_cfg = ind.get('high_vf', {})
    lowvf_cfg = ind.get('low_vf', {})
    
    # Format: kema_ema_atr_vf
    batch_name = f"kema_{ema_cfg.get('start', 5)}_{atr_cfg.get('start', 10)}_{lowvf_cfg.get('start', 0.5)}"
    
    print(f"   Đang sinh chiến thuật...")
    start_time = datetime.utcnow()
    
    # CRITICAL: Parse timeframes config to list (handle dict with start/end/step)
    timeframes_cfg = config.get('timeframes', ['4h'])
    if isinstance(timeframes_cfg, dict):
        # Dict format: {start: '1h', end: '24h', step: '1h'} → ['1h', '2h', ..., '24h']
        timeframes_list = _build_timeframes(timeframes_cfg)
        config['timeframes'] = timeframes_list  # Override config with parsed list
    elif isinstance(timeframes_cfg, list):
        # Already a list, use as-is
        timeframes_list = timeframes_cfg
    else:
        # Single string: '4h' → ['4h']
        timeframes_list = [str(timeframes_cfg)]
        config['timeframes'] = timeframes_list
    
    batch = []
    total_inserted = 0
    
    # Estimate total for progress bar
    estimated_total = estimate_config_count(config)
    
    # Setup progress tracking
    from tqdm import tqdm
    import time
    
    configs_generated = 0
    last_update = time.time()
    update_interval = 0.3  # Update callback every 0.3s
    
    pbar = tqdm(total=estimated_total, desc="Generating configs", unit="cfg", mininterval=0.1)  # Visual update every 0.1s
    
    for params in generate_params(config):
        configs_generated += 1
        pbar.update(1)
        # Generate hash
        param_hash = generate_param_hash(params)
        
        # Sanitize params to convert NumPy types to Python native types
        clean_params = MongoService.sanitize_data(params)
        
        # Create document for backtest-config
        doc = {
            'config_hash': param_hash,
            'batch_id': batch_id,
            'asset': asset,
            'params': clean_params,  # JSON format with all 12 params (sanitized)
            'metadata': {
                'asset': asset,
                'timeframe': clean_params['timeframe'],
                'commission_pct': config.get('commission_pct', 0.001),
                'slippage_pct': config.get('slippage_pct', 0.0005),
            },
            'created_at': start_time
        }
        
        batch.append(doc)
        
        # Report progress periodically
        current_time = time.time()
        if progress_callback and (current_time - last_update) >= update_interval:
            speed = configs_generated / (current_time - start_time.timestamp())
            progress_callback({
                'status': 'generating',
                'completed': configs_generated,
                'total': estimated_total,
                'speed': speed
            })
            last_update = current_time
        
        # Batch insert when reaching batch_size
        if len(batch) >= batch_size:
            try:
                # Select collection based on type
                collection = mongo.wfo_config if collection_type == 'wfo' else mongo.backtest_config
                result = collection.insert_many(batch, ordered=False)
                total_inserted += len(result.inserted_ids)
            except Exception as e:
                print(f"   ⚠️  Batch insert error: {e}")
            
            batch.clear()
    
    # Insert remaining
    if batch:
        try:
            # Select collection based on type
            collection = mongo.wfo_config if collection_type == 'wfo' else mongo.backtest_config
            result = collection.insert_many(batch, ordered=False)
            total_inserted += len(result.inserted_ids)
        except Exception as e:
            print(f"   ⚠️  Final batch error: {e}")
    
    # Close progress bar
    pbar.close()
    
    # Send final progress update
    if progress_callback:
        duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        speed = total_inserted / duration_seconds if duration_seconds > 0 else 0
        progress_callback({
            'status': 'generated',
            'completed': total_inserted,
            'total': total_inserted,
            'speed': speed
        })
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Determine collection name for display
    collection_name = 'wfo_config' if collection_type == 'wfo' else 'backtest_config'
    
    print(f"   ✅ Đã sinh {total_inserted:,} chiến thuật và lưu vào collection '{collection_name}'")
    print(f"   Batch name: {batch_name}")
    print(f"   Thời gian: {duration:.1f}s")
    
    return total_inserted