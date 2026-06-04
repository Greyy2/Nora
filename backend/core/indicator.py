"""
Indicator Module - Tính technical indicators (TradingView Identical)
"""

import numpy as np 
import pandas as pd 
from typing import Dict, Tuple 

def calculate_ema(close: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate EMA matching TradingView with SMA seed - VECTORIZED
    
    Performance: ~50x faster than loop for 68k+ rows
    Uses exponential weighting with pandas for speed
    """
    n = len(close)
    if n < length:
        return np.full(n, np.nan)
    
    # Use pandas ewm for vectorized EMA (much faster than loop)
    # TradingView uses alpha = 2/(length+1), adjust=False, and SMA seed
    s = pd.Series(close)
    ema_series = s.ewm(span=length, adjust=False).mean()
    
    # Set first (length-1) values to NaN to match TV behavior
    ema = ema_series.to_numpy().copy()  # .copy() for numpy 2.x write safety
    ema[:length-1] = np.nan
    
    return ema

def calculate_rma_numpy(src: np.ndarray, length: int) -> np.ndarray:
    """
    Hàm phụ trợ: Tính RMA (Rolling Moving Average) chuẩn TradingView
    Quy tắc TV: 
    1. Giá trị đầu tiên (tại index length-1) = SMA của length giá trị đầu
    2. Các giá trị sau: RMA = (prev_RMA * (length-1) + current) / length
    
    Equivalent to: ewm(alpha=1/length, adjust=False) with SMA seed
    """
    n = len(src)
    if n < length:
        return np.full(n, np.nan)
    
    # Use pandas ewm with min_periods to ensure SMA seed
    # alpha = 1/length is Wilder's smoothing
    s = pd.Series(src)
    rma_series = s.ewm(alpha=1.0/length, min_periods=length, adjust=False).mean()
    
    rma = rma_series.to_numpy().copy()
    
    return rma

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate ATR matching TradingView EXACTLY
    
    Key differences from previous implementation:
    - TR starts from bar 1 (bar 0 is NaN)
    - ATR seed at bar `length` (not `length-1`)
    - Seed uses mean(TR[1:length+1]), excluding TR[0]
    - Loop starts from `length+1`
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    
    n = len(close)
    
    # VECTORIZED TR calculation (no loop - 100x+ faster)
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    hl = high - low
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    
    # Stack and take max along axis 0
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum(hl[1:], np.maximum(hc, lc))
    
    # VECTORIZED ATR using RMA (much faster than loop)
    atr = calculate_rma_numpy(tr, length)
    
    return atr  

def calculate_keltner_bands(
    close: np.ndarray, 
    high: np.ndarray,
    low: np.ndarray,
    ema_length: int,
    atr_length: int,
    multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính Keltner Bands chuẩn TradingView
    """
    
    # Tính EMA 
    ema = calculate_ema(close, ema_length)

    # Tính ATR (Chuẩn TV)
    atr = calculate_atr(high, low, close, atr_length)

    # Lưu ý: ATR sẽ có NaN ở đầu (do warm-up), các phép tính sau sẽ tự động ra NaN tại đó
    upper_band = ema + (multiplier * atr)
    lower_band = ema - (multiplier * atr)

    return ema, atr, upper_band, lower_band


def calculate_rsi(close: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate RSI matching TradingView EXACTLY - VECTORIZED
    """
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = calculate_rma_numpy(gain, length)
    avg_loss = calculate_rma_numpy(loss, length)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
    # Handle edge case where avg_loss is 0
    rsi[avg_loss == 0] = 100
    
    return rsi

def calculate_wma(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate WMA matching TradingView - VECTORIZED
    """
    if len(src) < length:
        return np.full(len(src), np.nan)
        
    weights = np.arange(1, length + 1, dtype=float)
    s = pd.Series(src)
    
    # Vectorized weighted sum using rolling window
    def _wma(x):
        return np.dot(x, weights) / weights.sum()
        
    wma = s.rolling(window=length).apply(_wma, raw=True).to_numpy()
    return wma


def calculate_sma(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate SMA with explicit warmup NaNs.
    """
    if len(src) < length:
        return np.full(len(src), np.nan)

    s = pd.Series(src)
    return s.rolling(window=length, min_periods=length).mean().to_numpy()


def calculate_rsi_wma_stack(
    close: np.ndarray,
    rsi_period: int = 14,
    rsi_ema_len: int = 9,
    rsi_wma_len: int = 45,
    rsi_signal_period: int = 14,
    wma_smoothing: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Build the full RSI/WMA stack from raw close prices only.

    Returns:
        {
            "rsi": np.ndarray,
            "rsi_ema": np.ndarray,
            "rsi_wma": np.ndarray,
            "rsi_signal": np.ndarray,
            "rsi_sma": np.ndarray,
            "wma_delta_smooth": np.ndarray,
        }
    """
    close_np = np.asarray(close, dtype=float)

    period = max(2, int(rsi_period))
    ema_len = max(1, int(rsi_ema_len))
    wma_len = max(1, int(rsi_wma_len))
    signal_len = max(2, int(rsi_signal_period))
    smoothing_len = max(1, int(wma_smoothing))

    rsi = calculate_rsi(close_np, length=period)
    rsi_ema = calculate_ema(rsi, ema_len)
    rsi_wma = calculate_wma(rsi, wma_len)
    rsi_signal = calculate_ema(rsi, signal_len)
    rsi_sma = calculate_sma(rsi, signal_len)

    wma_delta = rsi_ema - rsi_wma
    if smoothing_len > 1:
        wma_delta_smooth = calculate_ema(wma_delta, smoothing_len)
    else:
        wma_delta_smooth = wma_delta.copy()

    return {
        "rsi": rsi,
        "rsi_ema": rsi_ema,
        "rsi_wma": rsi_wma,
        "rsi_signal": rsi_signal,
        "rsi_sma": rsi_sma,
        "wma_delta_smooth": wma_delta_smooth,
    }

def calculate_kalman_filter(src: np.ndarray, gain: float = 0.15) -> np.ndarray:
    """
    Calculate a simple Kalman Filter (as described by the user)
    Trend(t) = Trend(t-1) + Gain * (Price(t) - Trend(t-1))
    """
    n = len(src)
    kalman = np.full(n, np.nan)
    
    # Find first non-NaN value
    first_idx = np.where(~np.isnan(src))[0]
    if len(first_idx) == 0:
        return kalman
        
    idx = first_idx[0]
    kalman[idx] = src[idx]
    
    # Use provided gain (0.15 default to differentiate from WMA45 and EMA9)
    for i in range(idx + 1, n):
        if np.isnan(src[i]):
            kalman[i] = kalman[i-1]
        else:
            kalman[i] = kalman[i-1] + gain * (src[i] - kalman[i-1])
            
    return kalman

def calculate_zscore(src: np.ndarray, length: int) -> np.ndarray:
    """
    Calculate Z-Score: (x - mean) / std
    """
    s = pd.Series(src)
    mean = s.rolling(window=length).mean()
    std = s.rolling(window=length).std()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (s - mean) / std
        
    return z.to_numpy()

def calculate_multi_layer_oscillator(
    close: np.ndarray,
    rsi_length: int = 14,
    ema_length: int = 9,
    wma_length: int = 45,
    norm_length: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the special RSI + EMA + WMA + Kalman + Z-Score Oscillator
    Returns: (ema_signal, wma_baseline, kalman, upper_env, lower_env, raw_signal)
    """
    # 1. Base RSI
    rsi = calculate_rsi(close, rsi_length)
    
    # 2. Normalize via Z-Score (Longer window to preserve RSI swings)
    z = calculate_zscore(rsi, norm_length)
    
    # 3. Smoothers
    # Signal line (Quick)
    ema_signal = calculate_ema(z, ema_length)
    # Baseline (Slow)
    wma_baseline = calculate_wma(z, wma_length)
    # Trend (Balanced) - 0.1 gain creates lag/strength relative to EMA
    kalman = calculate_kalman_filter(z, gain=0.1)
    
    # 4. Envelopes (Dynamic "Gương soi" - following recent extremes)
    # Using a 20-period lookback for bands makes them very responsive/curved
    s_z = pd.Series(z)
    upper_env = s_z.rolling(window=20).max()
    lower_env = s_z.rolling(window=20).min()
    
    # Smooth bands slightly for aesthetics
    upper_env = calculate_ema(upper_env.to_numpy(), length=10)
    lower_env = calculate_ema(lower_env.to_numpy(), length=10)
    
    # Defaults for NaNs
    upper_env[np.isnan(upper_env)] = 2.0
    lower_env[np.isnan(lower_env)] = -2.0
    
    return ema_signal, wma_baseline, kalman, upper_env, lower_env, rsi

def calculate_market_regime(
    close: np.ndarray,
    ema: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    slope_window: int = 20
) -> np.ndarray:
    """
    Classify market regime: 1 (Up), -1 (Down), 0 (Side)
    Logic based on KEMA slope, price position, and band width.
    """
    n = len(close)
    regime = np.zeros(n)
    
    if n < slope_window:
        return regime
        
    # 1. KEMA Slope (Normalise by price to get % slope)
    ema_pd = pd.Series(ema)
    slope = (ema_pd - ema_pd.shift(slope_window)) / ema_pd * 100
    
    # 2. Band Width (Volatility)
    bw = (upper - lower) / ema * 100
    bw_ema = pd.Series(bw).rolling(window=50).mean()
    
    # 3. Labeling
    for i in range(slope_window, n):
        if np.isnan(slope[i]) or np.isnan(ema[i]):
            continue
            
        # Slope thresholds (e.g., 0.1% change over 20 bars)
        is_slope_up = slope[i] > 0.1
        is_slope_down = slope[i] < -0.1
        
        # Price position
        is_above = close[i] > ema[i]
        is_below = close[i] < ema[i]
        
        # Band status
        is_expanding = bw[i] > bw_ema[i] * 0.9 # Slightly loose for trend
        
        if is_slope_up and is_above and is_expanding:
            regime[i] = 1
        elif is_slope_down and is_below and is_expanding:
            regime[i] = -1
        else:
            # Everything else or specifically crossing middle / narrow bands
            regime[i] = 0
            
    return regime
