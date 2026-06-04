"""
Chart data service.

Performance goals:
- Return chart payload quickly (default max bars)
- Keep Sova call optional to avoid slowing down the chart API

Regime goals:
- Detect SIDEWAYS with multi-factor logic (Hurst, ADX, Keltner squeeze, EMA slope)
- Return sideway zones for main-chart gray overlay
- Return per-bar regime for RSI background
"""

import asyncio
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.indicator import calculate_atr, calculate_ema, calculate_rsi_wma_stack
from core.load_data import DataLoader
from services.cache_service import cache_get, cache_set

try:
    from Sova.sova_brain_v4 import sova_brain
    SOVA_AVAILABLE = True
except ImportError:
    sova_brain = None
    SOVA_AVAILABLE = False

_CHART_CACHE_TTL_SECONDS = 30


def _chart_cache_key(**kwargs) -> str:
    return "|".join(f"{key}={kwargs[key]}" for key in sorted(kwargs))


def _chart_cache_get(cache_key: str):
    return cache_get('chart_service', cache_key)


def _chart_cache_set(cache_key: str, value: Dict[str, Any]):
    cache_set('chart_service', cache_key, value, _CHART_CACHE_TTL_SECONDS)


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating, np.integer)):
        return _sanitize_value(float(value))
    return value


def _sanitize_list(values: List[Any]) -> List[Any]:
    return [_sanitize_value(v) for v in values]


def _to_iso_z(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_candlestick(df: pd.DataFrame) -> Dict[str, List[Any]]:
    times = [_to_iso_z(t) for t in df.index]
    return {
        "time": times,
        "open": _sanitize_list(df["open"].tolist()),
        "high": _sanitize_list(df["high"].tolist()),
        "low": _sanitize_list(df["low"].tolist()),
        "close": _sanitize_list(df["close"].tolist()),
        "volume": _sanitize_list(df["volume"].tolist()),
    }


def _apply_date_filter(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    out = df
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        out = out[out.index >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[out.index <= end_ts]

    return out


def _rolling_hurst(close: np.ndarray, window: int = 96) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < window or window < 16:
        return out

    eps = 1e-8
    safe = np.maximum(close, eps)
    log_close = np.log(safe)

    for i in range(window - 1, n):
        segment = log_close[i - window + 1: i + 1]
        centered = segment - segment.mean()
        cum_dev = np.cumsum(centered)
        r = float(cum_dev.max() - cum_dev.min())
        s = float(np.std(centered))
        if s <= eps or r <= eps:
            out[i] = np.nan
            continue
        h = np.log((r / s) + eps) / np.log(window)
        out[i] = float(np.clip(h, 0.0, 1.0))

    return out


def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(high) < period + 2:
        return np.full(len(high), np.nan, dtype=float)

    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)

    up_move = high_s.diff()
    down_move = -low_s.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-8)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.to_numpy(dtype=float)


def _build_regime_features(
    df: pd.DataFrame,
    ema: np.ndarray,
    atr: np.ndarray,
    long_vol_factor: float,
    short_vol_factor: float,
) -> pd.DataFrame:
    out = df.copy()
    out["ema"] = ema
    out["atr"] = atr
    out["upper_band"] = out["ema"] + (float(long_vol_factor) * out["atr"])
    out["lower_band"] = out["ema"] - (float(short_vol_factor) * out["atr"])

    return out


def _classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = out["close"].to_numpy(dtype=float)
    ema = out["ema"].to_numpy(dtype=float)
    atr = out["atr"].to_numpy(dtype=float)
    adx = out["adx"].to_numpy(dtype=float)
    hurst = out["hurst"].to_numpy(dtype=float)
    ema_slope = out["ema_slope"].to_numpy(dtype=float)
    squeeze_ratio = out["keltner_squeeze_ratio"].to_numpy(dtype=float)

    adx_low = np.nan_to_num(adx < 25, nan=False)
    squeeze = np.nan_to_num(squeeze_ratio < 0.82, nan=False)
    ema_flat = np.nan_to_num(np.abs(ema_slope) < 0.10, nan=False)
    hurst_side = np.nan_to_num(hurst < 0.50, nan=False)

    side_score = (
        adx_low.astype(float) * 0.35
        + squeeze.astype(float) * 0.25
        + ema_flat.astype(float) * 0.20
        + hurst_side.astype(float) * 0.20
    )

    trend_strength = np.clip((np.nan_to_num(adx, nan=0.0) - 16.0) / 25.0, 0.0, 1.0)
    slope_pos = np.clip(np.nan_to_num(ema_slope, nan=0.0) / 0.30, 0.0, 1.0)
    slope_neg = np.clip(-np.nan_to_num(ema_slope, nan=0.0) / 0.30, 0.0, 1.0)
    price_displacement = (close - ema) / (np.maximum(atr, 1e-8))
    pos_displacement = np.clip(price_displacement / 2.0, 0.0, 1.0)
    neg_displacement = np.clip(-price_displacement / 2.0, 0.0, 1.0)

    bull_score = 0.45 * trend_strength + 0.30 * slope_pos + 0.25 * pos_displacement
    bear_score = 0.45 * trend_strength + 0.30 * slope_neg + 0.25 * neg_displacement

    # Penalize trend scores when side evidence is strong.
    trend_penalty = np.clip(side_score - 0.55, 0.0, 0.6)
    bull_score = np.clip(bull_score - trend_penalty, 0.0, 1.0)
    bear_score = np.clip(bear_score - trend_penalty, 0.0, 1.0)

    raw_sum = bull_score + bear_score + side_score + 1e-8
    bull_prob = (bull_score / raw_sum) * 100.0
    bear_prob = (bear_score / raw_sum) * 100.0
    side_prob = (side_score / raw_sum) * 100.0

    regime = np.full(len(out), "side", dtype=object)
    regime_raw = np.full(len(out), "sideways_consolidation", dtype=object)

    bull_mask = (bull_prob >= bear_prob) & (bull_prob >= side_prob) & (side_prob < 52.0)
    bear_mask = (bear_prob > bull_prob) & (bear_prob >= side_prob) & (side_prob < 52.0)
    side_mask = ~(bull_mask | bear_mask)

    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"
    regime[side_mask] = "side"

    strong_bull = bull_mask & (trend_strength > 0.60)
    strong_bear = bear_mask & (trend_strength > 0.60)
    weak_bull = bull_mask & ~strong_bull
    weak_bear = bear_mask & ~strong_bear

    regime_raw[strong_bull] = "bull_trend_strong"
    regime_raw[weak_bull] = "bull_trend_moderate"
    regime_raw[strong_bear] = "bear_trend_strong"
    regime_raw[weak_bear] = "bear_trend_moderate"
    regime_raw[side_mask & (squeeze == 1)] = "sideways_squeeze"
    regime_raw[side_mask & (squeeze != 1)] = "sideways_range"

    out["regime"] = regime
    out["regime_raw"] = regime_raw
    out["bull_prob"] = np.clip(bull_prob, 0.0, 100.0)
    out["bear_prob"] = np.clip(bear_prob, 0.0, 100.0)
    out["side_prob"] = np.clip(side_prob, 0.0, 100.0)
    out["side_score"] = np.clip(side_score, 0.0, 1.0)
    out["trend_strength_score"] = trend_strength

    return out


def _classify_kema_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    KEMA regime (TradingView-aligned intent):
    - LONG (bull): close > upper_band or strict crossover(close, upper_band)
    - SHORT (bear): close < lower_band or strict crossunder(close, lower_band)
    - SIDE: inside the channel [lower_band, upper_band]

    This is intentionally simpler and faster than the ensemble classifier,
    and matches the trading logic expected by the KEMA script.
    """
    out = df.copy()

    close = out["close"].to_numpy(dtype=float)
    upper = out["upper_band"].to_numpy(dtype=float)
    lower = out["lower_band"].to_numpy(dtype=float)

    close_prev = np.roll(close, 1)
    upper_prev = np.roll(upper, 1)
    lower_prev = np.roll(lower, 1)

    if len(close_prev) > 0:
        close_prev[0] = close[0]
        upper_prev[0] = upper[0]
        lower_prev[0] = lower[0]

    crossover_upper = (close_prev < upper_prev) & (close > upper)
    crossunder_lower = (close_prev > lower_prev) & (close < lower)

    is_long = (close > upper) | crossover_upper
    is_short = (close < lower) | crossunder_lower
    is_side = ~(is_long | is_short)

    regime = np.full(len(out), "side", dtype=object)
    regime[is_long] = "bull"
    regime[is_short] = "bear"

    regime_raw = np.full(len(out), "sideways_range", dtype=object)
    regime_raw[is_long] = "bull_band_break"
    regime_raw[is_short] = "bear_band_break"

    # Probabilities are deterministic and explicit for UI consistency.
    bull_prob = np.where(is_long, 100.0, 0.0)
    bear_prob = np.where(is_short, 100.0, 0.0)
    side_prob = np.where(is_side, 100.0, 0.0)

    out["regime"] = regime
    out["regime_raw"] = regime_raw
    out["bull_prob"] = bull_prob
    out["bear_prob"] = bear_prob
    out["side_prob"] = side_prob
    out["side_score"] = np.where(is_side, 1.0, 0.0)
    out["trend_strength_score"] = np.where(is_side, 0.0, 1.0)
    out["kema_cross_upper"] = crossover_upper
    out["kema_cross_lower"] = crossunder_lower

    return out


def _extract_sideway_zones(df: pd.DataFrame, min_bars: int = 4, max_zones: int = 48) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []
    if df.empty:
        return zones

    side_mask = df["regime"] == "side"
    idx = df.index

    start_i: Optional[int] = None
    for i in range(len(df)):
        is_side = bool(side_mask.iloc[i])
        if is_side and start_i is None:
            start_i = i

        is_last = i == len(df) - 1
        if start_i is not None and ((not is_side) or is_last):
            end_i = i if (is_side and is_last) else i - 1
            bars = end_i - start_i + 1

            if bars >= min_bars:
                segment = df.iloc[start_i:end_i + 1]
                side_prob_mean = float(np.nanmean(segment["side_prob"].to_numpy(dtype=float)))
                width_mean = float(np.nanmean((segment["high"] - segment["low"]).to_numpy(dtype=float)))
                atr_mean = float(np.nanmean(segment["atr"].to_numpy(dtype=float)))
                width_ratio = width_mean / (atr_mean + 1e-8) if atr_mean > 0 else 0.0

                if side_prob_mean >= 74:
                    quality = "high"
                elif side_prob_mean >= 62:
                    quality = "medium"
                else:
                    quality = "low"

                zones.append({
                    "start_time": _to_iso_z(idx[start_i]),
                    "end_time": _to_iso_z(idx[end_i]),
                    "upper": float(segment["high"].max()),
                    "lower": float(segment["low"].min()),
                    "quality": quality,
                    "bars": int(bars),
                    "side_probability": float(np.clip(side_prob_mean, 0.0, 100.0)),
                    "width_atr_ratio": float(width_ratio),
                })

            start_i = None

    if not zones:
        return zones

    return zones[-max_zones:]


def _to_regime_analysis(df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
    latest = df.iloc[-1]
    current = str(latest["regime"]).lower()

    if current == "bull":
        current_regime = "TRENDING_UP"
    elif current == "bear":
        current_regime = "TRENDING_DOWN"
    else:
        current_regime = "SIDEWAYS"

    trend_strength_score = float(latest.get("trend_strength_score", 0.0) or 0.0)
    if trend_strength_score < 0.25:
        strength = "weak"
    elif trend_strength_score < 0.50:
        strength = "moderate"
    elif trend_strength_score < 0.80:
        strength = "strong"
    else:
        strength = "extreme"

    confidence = float(np.clip(max(
        float(latest.get("bull_prob", 0.0)),
        float(latest.get("bear_prob", 0.0)),
        float(latest.get("side_prob", 0.0)),
    ) / 100.0, 0.0, 1.0))

    quality_score = float(np.clip(confidence * 100.0, 0.0, 100.0))

    return {
        "current_regime": current_regime,
        "strength": strength,
        "is_strong": strength in {"strong", "extreme"},
        "confidence": confidence,
        "quality_score": quality_score,
        "allow_long": current_regime != "TRENDING_DOWN",
        "allow_short": current_regime != "TRENDING_UP",
        "timeframe": timeframe,
        "probabilities": {
            "bull": float(np.clip(float(latest.get("bull_prob", 0.0)), 0.0, 100.0)),
            "bear": float(np.clip(float(latest.get("bear_prob", 0.0)), 0.0, 100.0)),
            "side": float(np.clip(float(latest.get("side_prob", 0.0)), 0.0, 100.0)),
        },
        "feature_snapshot": {
            "ema": float(_sanitize_value(float(latest.get("ema", np.nan)))) if not math.isnan(float(latest.get("ema", np.nan))) else None,
            "upper_band": float(_sanitize_value(float(latest.get("upper_band", np.nan)))) if not math.isnan(float(latest.get("upper_band", np.nan))) else None,
            "lower_band": float(_sanitize_value(float(latest.get("lower_band", np.nan)))) if not math.isnan(float(latest.get("lower_band", np.nan))) else None,
            "kema_cross_upper": bool(latest.get("kema_cross_upper", False)),
            "kema_cross_lower": bool(latest.get("kema_cross_lower", False)),
        },
        "logic": {
            "description": "KEMA band-state regime: close>upper => long, close<lower => short, inside channel => side",
            "thresholds": {
                "long_when": "close > upper_band",
                "short_when": "close < lower_band",
                "side_when": "lower_band <= close <= upper_band",
            },
        },
    }


def _generate_math_prediction(latest: pd.Series) -> str:
    raw = str(latest.get("regime_raw", "sideways_range"))
    side_prob = float(latest.get("side_prob", 0.0) or 0.0)
    bull_prob = float(latest.get("bull_prob", 0.0) or 0.0)
    bear_prob = float(latest.get("bear_prob", 0.0) or 0.0)

    if raw.startswith("sideways"):
        return (
            f"Sideways regime detected ({side_prob:.1f}%). "
            "Trend-following modules should be constrained and mean-reversion logic prioritized."
        )
    if raw.startswith("bull"):
        return f"Bullish trend regime ({bull_prob:.1f}%). Pullback entries are favored while downside risk is still monitored."
    if raw.startswith("bear"):
        return f"Bearish trend regime ({bear_prob:.1f}%). Rally fades are favored and long exposure should remain selective."
    return "Neutral market state. Observation mode."


async def get_chart_data(
    asset: str,
    timeframe: str,
    ema_length: int = 50,
    atr_length: int = 14,
    data_type: str = "OKX",
    **kwargs,
) -> Dict[str, Any]:
    try:
        max_bars = max(300, min(int(kwargs.get("max_bars", 2200)), 10000))
        include_sova = bool(kwargs.get("include_sova", False))
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")

        long_vol_factor = float(kwargs.get("long_vol_factor", 2.0))
        short_vol_factor = float(kwargs.get("short_vol_factor", 1.3))

        rsi_period = int(kwargs.get("rsi_period", 14))
        rsi_ema_len = int(kwargs.get("rsi_ema_len", 9))
        rsi_wma_len = int(kwargs.get("rsi_wma_len", 45))
        rsi_signal_period = int(kwargs.get("rsi_signal_period", 14))
        wma_smoothing = int(kwargs.get("wma_smoothing", 3))

        cache_key = _chart_cache_key(
            asset=asset,
            timeframe=timeframe,
            ema_length=ema_length,
            atr_length=atr_length,
            data_type=data_type,
            max_bars=max_bars,
            include_sova=include_sova,
            start_date=start_date or '',
            end_date=end_date or '',
            long_vol_factor=long_vol_factor,
            short_vol_factor=short_vol_factor,
            rsi_period=rsi_period,
            rsi_ema_len=rsi_ema_len,
            rsi_wma_len=rsi_wma_len,
            rsi_signal_period=rsi_signal_period,
            wma_smoothing=wma_smoothing,
        )
        cached = _chart_cache_get(cache_key)
        if cached is not None:
            return cached

        loader = DataLoader(data_dir=data_type)
        pdf = loader.load(asset, timeframe)
        if pdf.empty:
            raise ValueError(f"No data for {asset}")

        pdf = _apply_date_filter(pdf, start_date, end_date)
        if pdf.empty:
            raise ValueError("No data in selected date range")

        warmup = max(ema_length * 4, atr_length * 6, rsi_period * 6, 280)
        if len(pdf) > max_bars + warmup:
            pdf = pdf.iloc[-(max_bars + warmup):]

        close_np = pdf["close"].to_numpy(dtype=float)
        ema = calculate_ema(close_np, length=ema_length)
        atr = calculate_atr(
            pdf["high"].to_numpy(dtype=float),
            pdf["low"].to_numpy(dtype=float),
            close_np,
            length=atr_length,
        )

        rsi_stack = calculate_rsi_wma_stack(
            close_np,
            rsi_period=rsi_period,
            rsi_ema_len=rsi_ema_len,
            rsi_wma_len=rsi_wma_len,
            rsi_signal_period=rsi_signal_period,
            wma_smoothing=wma_smoothing,
        )

        df = _build_regime_features(
            pdf,
            ema=ema,
            atr=atr,
            long_vol_factor=long_vol_factor,
            short_vol_factor=short_vol_factor,
        )
        df = _classify_kema_regime(df)

        if len(df) > max_bars:
            df_out = df.iloc[-max_bars:]
        else:
            df_out = df

        latest = df_out.iloc[-1]
        math_prediction = _generate_math_prediction(latest)

        sova_advice = "Mathematical regime model active. Optional AI advisory is disabled for low-latency chart loading."
        if include_sova and SOVA_AVAILABLE and sova_brain:
            try:
                market_context = (
                    f"Asset: {asset} | Regime: {latest.get('regime_raw', 'unknown')} "
                    f"| SideProb: {float(latest.get('side_prob', 0.0)):.1f}%"
                )
                sova_advice = await asyncio.wait_for(
                    sova_brain.synthesize_alpha(
                        base_idea="Regime Context Brief",
                        diagnostics=[f"Regime: {latest.get('regime_raw', 'unknown')}"] ,
                        market_context=market_context,
                        current_metrics={"ic": 0.08},
                    ),
                    timeout=0.9,
                )
            except Exception:
                sova_advice = "Sova advisory unavailable. Using deterministic mathematical regime model."

        times = [_to_iso_z(t) for t in df_out.index]
        sideway_zones = _extract_sideway_zones(df_out)
        regime_analysis = _to_regime_analysis(df_out, timeframe=timeframe)
        regime_analysis["sideway_zones"] = sideway_zones

        response = {
            "success": True,
            "data": {
                "candlestick": _format_candlestick(df_out),
                "indicators": {
                    "ema": {"time": times, "values": _sanitize_list(df_out["ema"].tolist())},
                    "atr": {"time": times, "values": _sanitize_list(df_out["atr"].tolist())},
                    "upper_band": {"time": times, "values": _sanitize_list(df_out["upper_band"].tolist())},
                    "lower_band": {"time": times, "values": _sanitize_list(df_out["lower_band"].tolist())},
                    "rsi": {"time": times, "values": _sanitize_list(rsi_stack["rsi"][-len(df_out):].tolist())},
                    "osc_raw": {"time": times, "values": _sanitize_list(rsi_stack["rsi"][-len(df_out):].tolist())},
                    "rsi_ema": {"time": times, "values": _sanitize_list(rsi_stack["rsi_ema"][-len(df_out):].tolist())},
                    "rsi_wma": {"time": times, "values": _sanitize_list(rsi_stack["rsi_wma"][-len(df_out):].tolist())},
                    "rsi_signal": {"time": times, "values": _sanitize_list(rsi_stack["rsi_signal"][-len(df_out):].tolist())},
                    "rsi_sma": {"time": times, "values": _sanitize_list(rsi_stack["rsi_sma"][-len(df_out):].tolist())},
                    "wma_delta_smooth": {"time": times, "values": _sanitize_list(rsi_stack["wma_delta_smooth"][-len(df_out):].tolist())},
                    "regime": {"time": times, "values": df_out["regime"].tolist()},
                },
                "indicator_params": {
                    "source": "computed_from_ohlcv_close",
                    "rsi_period": rsi_period,
                    "rsi_ema_len": rsi_ema_len,
                    "rsi_wma_len": rsi_wma_len,
                    "rsi_signal_period": rsi_signal_period,
                    "wma_smoothing": wma_smoothing,
                    "ema_length": ema_length,
                    "atr_length": atr_length,
                    "long_vol_factor": long_vol_factor,
                    "short_vol_factor": short_vol_factor,
                },
                "sideway_zones": sideway_zones,
                "regime_analysis": regime_analysis,
                "sova_insight": {
                    "advice": sova_advice,
                    "prediction": math_prediction,
                    "raw_regime": str(latest.get("regime_raw", "sideways_range")),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            },
        }
        _chart_cache_set(cache_key, response)
        return response
    except Exception as e:
        return {"success": False, "error": str(e)}
