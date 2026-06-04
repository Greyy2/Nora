"""
Prepare Module - Production-grade data loading & indicator precomputation

Architecture:
- Precompute EMA/ATR MATRIX for ALL lengths (not just max)
- Compute Keltner Bands on-the-fly in worker
- Handle NaN alignment (indicators start after warmup period)
- Enforce UTC timezone consistency

CRITICAL FIXES:
1. EMA/ATR matrix: Precompute for ALL lengths 1→max (not just max!)
2. NaN handling: Track valid_start_idx
3. Worker lookup: O(1) array indexing

Performance:
- Precompute once, reuse for all workers
- Workers get exact length via matrix[length-1]
- No indicator recalculation
- Memory efficient (fork COW)
"""

import pandas as pd
from typing import Dict, List, Tuple
import numpy as np

from core.load_data import DataLoader
from core.indicator import calculate_ema, calculate_atr


def calculate_ema_matrix(close: np.ndarray, max_length: int) -> np.ndarray:
    """
    Precompute EMA for ALL lengths from 1 to max_length
    
    CRITICAL: EMA(50) ≠ EMA(20), cannot slice!
    Must compute each length separately.
    
    Args:
        close: Close price array
        max_length: Maximum EMA length to compute
        
    Returns:
        matrix[length-1] = EMA(length)
        Shape: (max_length, len(close))
        
    Example:
        >>> ema_matrix = calculate_ema_matrix(close, 50)
        >>> ema_20 = ema_matrix[19]  # length 20 at index 19
        >>> ema_50 = ema_matrix[49]  # length 50 at index 49
    """
    n = len(close)
    max_length = int(max_length)  # Ensure integer for np.empty
    ema_matrix = np.empty((max_length, n))
    
    for length in range(1, max_length + 1):
        ema_matrix[length - 1] = calculate_ema(close, length)
    
    return ema_matrix


def calculate_atr_matrix(high: np.ndarray, low: np.ndarray, close: np.ndarray, max_length: int) -> np.ndarray:
    """
    Precompute ATR for ALL lengths from 1 to max_length
    
    CRITICAL: ATR(30) ≠ ATR(14), cannot slice!
    Must compute each length separately.
    
    Args:
        high: High price array
        low: Low price array
        close: Close price array
        max_length: Maximum ATR length to compute
        
    Returns:
        matrix[length-1] = ATR(length)
        Shape: (max_length, len(close))
    """
    n = len(close)
    max_length = int(max_length)  # Ensure integer for np.empty
    atr_matrix = np.empty((max_length, n))
    
    for length in range(1, max_length + 1):
        atr_matrix[length - 1] = calculate_atr(high, low, close, length)
    
    return atr_matrix


def load_and_precompute(
    asset: str,
    timeframes: List[str],
    max_ema_length: int,
    max_atr_length: int,
    data_type: str = "OKX"
) -> Dict[str, Dict]:
    """
    Load data and precompute indicator MATRIX for ALL lengths
    
    Strategy:
    - Precompute EMA matrix: ema_matrix[length-1] = EMA(length)
    - Precompute ATR matrix: atr_matrix[length-1] = ATR(length)
    - Workers get exact length via O(1) indexing
    - Workers compute bands on-the-fly
    
    Args:
        asset: Asset symbol (e.g., 'BTCUSDT')
        timeframes: List of timeframes (e.g., ['4h', '1d'])
        max_ema_length: Maximum EMA length to precompute
        max_atr_length: Maximum ATR length to precompute
        
    Returns:
        {
            timeframe: {
                'df': DataFrame (OHLCV only, read-only),
                'ema_matrix': np.ndarray (max_ema_length, n_candles),
                'atr_matrix': np.ndarray (max_atr_length, n_candles),
                'valid_start_idx': int (first valid index after NaN warmup)
            }
        }
        
    Example:
        >>> data = load_and_precompute('BTCUSDT', ['4h'], 50, 30)
        >>> # In worker:
        >>> ema_20 = data['4h']['ema_matrix'][19]  # Get EMA(20)
        >>> atr_14 = data['4h']['atr_matrix'][13]  # Get ATR(14)
        >>> upper = ema_20 + atr_14 * long_vol_factor
    """
    # Use DataLoader category/alias resolution (OKX, forex, vinh, grey, or absolute path).
    loader = DataLoader(data_dir=data_type)
    result = {}
    
    # Summary report (no spam, just start/end)
    import time
    start_time = time.time()
    print(f"📊 Loading {len(timeframes)} timeframes from {data_type} (EMA 1→{max_ema_length}, ATR 1→{max_atr_length})...")
    
    for tf in timeframes:
        # Load OHLCV data
        df = loader.load(asset, tf)
        
        # FIX: Enforce UTC timezone consistency
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"DataFrame index must be DatetimeIndex, got {type(df.index)}")
        
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        elif str(df.index.tz) != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        # Extract price arrays
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # FIX 1: Precompute EMA/ATR MATRIX for ALL lengths (silent mode)
        # Ensure int values for matrix calculations
        max_ema_int = int(max_ema_length)
        max_atr_int = int(max_atr_length)
        ema_matrix = calculate_ema_matrix(close, max_ema_int)
        atr_matrix = calculate_atr_matrix(high, low, close, max_atr_int)
        
        # FIX 2: Calculate valid_start_idx (first index after NaN warmup)
        # Indicators need warmup period, first max(ema_len, atr_len) bars may have NaN
        valid_start_idx = max(max_ema_int, max_atr_int)
        
        # Store data structure
        # CRITICAL OPTIMIZATION: Cache timestamps as numpy array (not list!)
        # .tolist() on datetime index is EXTREMELY slow (~0.05s per call)
        # Using numpy array + dict pre-cache saves 40-50% runtime!
        timestamps = df.index.to_numpy()  # Numpy array - MUCH faster
        ts_map = dict(zip(timestamps, range(len(timestamps))))  # Fast dict creation
        
        result[tf] = {
            'df': df,  # Read-only OHLCV
            'ema_matrix': ema_matrix,  # Shape: (max_ema_length, n_candles)
            'atr_matrix': atr_matrix,  # Shape: (max_atr_length, n_candles)
            'valid_start_idx': valid_start_idx,  # First valid index
            'close': close,
            'high': high,
            'low': low,
            'timestamps': timestamps,  # Cached numpy array (not list!)
            'ts_map': ts_map  # Cached dict
        }
    
    # Summary report (no spam, just result)
    elapsed = time.time() - start_time
    total_candles = sum(len(result[tf]['df']) for tf in result)
    print(f"✅ Loaded {len(result)} timeframes, {total_candles:,} total candles in {elapsed:.1f}s")
    
    return result


def get_indicators_for_strategy(
    ema_matrix: np.ndarray,
    atr_matrix: np.ndarray,
    ema_length: int,
    atr_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get EMA/ATR arrays for specific strategy parameters
    
    Args:
        ema_matrix: Precomputed EMA matrix (max_ema_length, n_candles)
        atr_matrix: Precomputed ATR matrix (max_atr_length, n_candles)
        ema_length: Desired EMA length
        atr_length: Desired ATR length
        
    Returns:
        (ema, atr) arrays for the specified lengths
        
    Example:
        >>> ema, atr = get_indicators_for_strategy(ema_matrix, atr_matrix, 20, 14)
        >>> # ema = ema_matrix[19], atr = atr_matrix[13]
    """
    # O(1) lookup via matrix indexing
    ema = ema_matrix[ema_length - 1]
    atr = atr_matrix[atr_length - 1]
    return ema, atr


def compute_keltner_bands(
    ema: np.ndarray,
    atr: np.ndarray,
    long_vol_factor: float,
    short_vol_factor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Keltner Bands on-the-fly (linear operation, very fast)
    
    FIX: Bands MUST be computed in worker, not precomputed
    Because vol_factor is a strategy parameter being optimized.
    
    Args:
        ema: EMA array
        atr: ATR array
        long_vol_factor: Multiplier for long side
        short_vol_factor: Multiplier for short side
        
    Returns:
        (upper_band, lower_band)
        
    Performance: O(N) vectorized NumPy, extremely fast
    """
    upper_band = ema + atr * long_vol_factor
    lower_band = ema - atr * short_vol_factor
    return upper_band, lower_band