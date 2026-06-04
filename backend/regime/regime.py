from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
import time


class MarketState(Enum):
    LONG = 1
    SHORT = 2
    SIDEWAY = 0


class TrendStrength(Enum):
    STRONG = "strong"
    WEAK = "weak"
    NEUTRAL = "neutral"


class StrategyType(Enum):
    TREND_FOLLOW = "trend_follow"
    BREAKOUT = "breakout"
    MEAN_REVERT = "mean_revert"
    SCALP = "scalp"


@dataclass
class MarketZone:
    start_idx: int
    end_idx: int
    state: MarketState
    strength: float
    confidence: float


@dataclass
class RegimePrediction:
    current_state: MarketState
    next_state: MarketState
    confidence: float
    zones: List[MarketZone]
    
    trend_strength: TrendStrength
    is_strong: bool
    strength_score: float
    
    sideway_context: Optional[Dict] = None
    mtf_alignment: Optional[Dict] = None


@dataclass
class SignalValidation:
    is_valid: bool
    matches_regime: bool
    trust_score: float
    
    regime_says: MarketState
    signal_says: MarketState
    
    warnings: List[str] = field(default_factory=list)


@dataclass
class StrategyProbability:
    strategy: StrategyType
    probability: float
    expected_winrate: float
    risk_multiplier: float
    
    sl_multiplier: float
    tp_multiplier: float
    
    edge_score: float


@dataclass
class TransitionSignal:
    """
    Adaptive Transition Signal - Thích nghi động
    
    Phát hiện khi strength thay đổi đáng kể (WEAK → STRONG hoặc STRONG → WEAK)
    và cung cấp hướng dẫn chuyển đổi chiến lược.
    """
    transition_detected: bool
    from_strength: TrendStrength
    to_strength: TrendStrength
    
    confidence: float  # 0.0-1.0 (càng cao càng chắc chắn)
    confirmation_signals: int  # Số tín hiệu xác nhận (volume, ATR, RSI, etc.)
    
    can_execute: bool  # True nếu vượt qua tất cả điều kiện (cooldown, confirmation)
    reason: str  # Lý do (ví dụ: "Strength increased from 0.45 to 0.72")
    
    # Strategy transition instructions
    old_strategy: Optional[str] = None
    new_strategy: Optional[str] = None
    risk_adjustment: Optional[float] = None  # Tỷ lệ thay đổi risk (ví dụ: 0.4 → 1.2)


@dataclass
class RegimeOutput:
    prediction: RegimePrediction
    validation: Optional[SignalValidation]
    strategies: List[StrategyProbability]
    
    allow_long: bool
    allow_short: bool
    allow_sideway: bool
    
    recommended_strategy: Optional[StrategyProbability]
    metadata: Dict = field(default_factory=dict)
    
    # ADAPTIVE TRANSITION
    transition: Optional[TransitionSignal] = None


class RegimeEngineV3:
    
    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 100,
        strength_threshold: float = 0.65,  # PHASE 1: Raised from 0.55 (stricter)
        confidence_threshold: float = 0.65,  # PHASE 1: Raised from 0.55 (stricter)
        multi_tf_levels: int = 3,
        sideway_boundary_threshold: float = 0.3,
        volume_surge_threshold: float = 1.5,
        whipsaw_cross_threshold: int = 3,
        # MARKET-ADAPTIVE THRESHOLDS
        zone_band_pct: float = 0.01,           # ±1% for crypto, ±0.3% for forex
        sideway_slope_pct: float = 0.01,       # EMA slope for sideway confirm
        momentum_threshold: float = 0.02,      # next-state prediction threshold
        volume_weight: float = 0.25,           # strength_score volume component
        # ADAPTIVE TRANSITION PARAMETERS
        enable_adaptive_transition: bool = True,
        transition_confirmation_candles: int = 3,  # Số nến cần sustain
        transition_cooldown_seconds: float = 900.0,  # 15 phút cooldown
        transition_strength_delta: float = 0.15  # Delta tối thiểu để trigger (ví dụ: 0.45 → 0.60)
    ):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.strength_threshold = strength_threshold
        self.confidence_threshold = confidence_threshold
        self.multi_tf_levels = multi_tf_levels
        self.sideway_boundary_threshold = sideway_boundary_threshold
        self.volume_surge_threshold = volume_surge_threshold
        self.whipsaw_cross_threshold = whipsaw_cross_threshold
        self.zone_band_pct = zone_band_pct
        self.sideway_slope_pct = sideway_slope_pct
        self.momentum_threshold = momentum_threshold
        self.volume_weight = volume_weight
        
        # ADAPTIVE TRANSITION CONFIG
        self.enable_adaptive_transition = enable_adaptive_transition
        self.transition_confirmation_candles = transition_confirmation_candles
        self.transition_cooldown_seconds = transition_cooldown_seconds
        self.transition_strength_delta = transition_strength_delta
        
        # TRANSITION STATE TRACKING (stateful)
        # Lưu lại historical states để detect transitions
        self._historical_states = []  # List[(timestamp, state, is_strong, strength_score)]
        self._last_transition_time = 0.0  # Timestamp of last transition
        self._transition_confirmation_count = 0  # Số candles đã confirm
    
    
    def analyze(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        rsi: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        signal_long: bool = False,
        signal_short: bool = False,
        df_macro: Optional[pd.DataFrame] = None,
        ema_macro: Optional[np.ndarray] = None
    ) -> RegimeOutput:
        
        prediction = self._step0_classify_and_predict(
            df, ema, atr, upper_band, lower_band, rsi, volume, df_macro, ema_macro
        )
        
        validation = None
        if signal_long or signal_short:
            signal_state = MarketState.LONG if signal_long else MarketState.SHORT
            validation = self._step1_validate_signal(
                prediction, signal_state, df, ema, atr, rsi, volume
            )
        
        strategies = self._step2_calculate_probabilities(
            prediction, validation, df, ema, atr
        )
        
        allow_long = prediction.current_state == MarketState.LONG
        allow_short = prediction.current_state == MarketState.SHORT
        allow_sideway = prediction.current_state == MarketState.SIDEWAY
        
        if validation:
            if not validation.is_valid or not validation.matches_regime:
                allow_long = False
                allow_short = False
        
        recommended = None
        if strategies:
            recommended = max(strategies, key=lambda s: s.probability * s.edge_score)
        
        swing_lookback = 50
        ema_fast_value = float(ema[-1]) if len(ema) > 0 else float(df['close'].iloc[-1])
        ema_slow_value = float(ema[-1]) if len(ema) > 0 else float(df['close'].iloc[-1])
        swing_high_value = float(df['high'].iloc[-swing_lookback:].max()) if len(df) >= swing_lookback else float(df['high'].iloc[-1])
        swing_low_value = float(df['low'].iloc[-swing_lookback:].min()) if len(df) >= swing_lookback else float(df['low'].iloc[-1])
        
        # STATE SYNCHRONIZATION: Capture calculation timestamp and market state
        # Execute will use these to verify market hasn't changed dramatically
        calculated_price = float(df['close'].iloc[-1])
        calculated_atr = float(atr[-1]) if len(atr) > 0 else 0.0
        calculation_timestamp = time.time()
        
        # ALLOCATION MODE: Xác định "Đánh Du Kích" vs "Ma Trận 3 Chiều"
        # Chỉ áp dụng cho WEAK trends (LONG WEAK / SHORT WEAK)
        allocation_mode, stability_score = self._calculate_weak_stability(
            df, ema, atr, prediction.current_state, prediction.is_strong
        )
        
        # ADAPTIVE TRANSITION: Phát hiện chuyển đổi strength (WEAK → STRONG hoặc STRONG → WEAK)
        # Chỉ chuyển trong cùng 1 loại (LONG → LONG, SHORT → SHORT)
        transition_signal = self._detect_adaptive_transition(
            current_state=prediction.current_state,
            current_is_strong=prediction.is_strong,
            current_strength_score=prediction.strength_score,
            df=df,
            ema=ema,
            atr=atr,
            rsi=rsi,
            volume=volume
        )
        
        return RegimeOutput(
            prediction=prediction,
            validation=validation,
            strategies=strategies,
            allow_long=allow_long,
            allow_short=allow_short,
            allow_sideway=allow_sideway,
            recommended_strategy=recommended,
            transition=transition_signal,  # NEW: Adaptive transition signal
            metadata={
                'lookback_short': self.lookback_short,
                'lookback_long': self.lookback_long,
                'multi_tf_levels': self.multi_tf_levels,
                'ema_fast': ema_fast_value,
                'ema_slow': ema_slow_value,
                'swing_high': swing_high_value,
                'swing_low': swing_low_value,
                # Critical for Execute verification
                'calculated_price': calculated_price,
                'calculated_atr': calculated_atr,
                'calculation_timestamp': calculation_timestamp,
                # ALLOCATION MODE (WEAK trend strategy selection)
                'allocation_mode': allocation_mode,  # 'PARTIAL_ONLY' or 'FULL_MATRIX'
                'stability_score': stability_score   # 0.0-1.0
            }
        )
    
    
    def _step0_classify_and_predict(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray],
        df_macro: Optional[pd.DataFrame],
        ema_macro: Optional[np.ndarray]
    ) -> RegimePrediction:
        
        zones = self._identify_historical_zones(df, ema, atr)
        
        current_state = self._detect_current_state(df, ema, atr, upper_band, lower_band, rsi)
        
        next_state = self._predict_next_state(df, ema, atr, zones, rsi, volume)
        
        confidence = self._calculate_prediction_confidence(df, ema, zones, current_state, next_state)
        
        strength_score, trend_strength, is_strong = self._analyze_trend_strength(
            df, ema, atr, current_state, rsi, volume
        )
        
        mtf_alignment = None
        if df_macro is not None and ema_macro is not None:
            mtf_alignment = self._analyze_mtf_alignment(df, ema, df_macro, ema_macro, current_state)
        
        sideway_context = None
        if current_state == MarketState.SIDEWAY:
            sideway_context = self._activate_predator_mode(
                df, ema, atr, upper_band, lower_band, rsi, volume
            )
        
        return RegimePrediction(
            current_state=current_state,
            next_state=next_state,
            confidence=confidence,
            zones=zones,
            trend_strength=trend_strength,
            is_strong=is_strong,
            strength_score=strength_score,
            sideway_context=sideway_context,
            mtf_alignment=mtf_alignment
        )
    
    
    def _identify_historical_zones(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray
    ) -> List[MarketZone]:
        
        n = len(df)
        zones = []
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        states = np.where(
            close > ema * (1 + self.zone_band_pct),
            MarketState.LONG.value,
            np.where(
                close < ema * (1 - self.zone_band_pct),
                MarketState.SHORT.value,
                MarketState.SIDEWAY.value
            )
        )
        
        state_changes = states[1:] != states[:-1]
        boundary_indices = np.where(state_changes)[0] + 1
        all_boundaries = np.concatenate(([0], boundary_indices, [n]))
        
        for i in range(len(all_boundaries) - 1):
            start_idx = int(all_boundaries[i])
            end_idx = int(all_boundaries[i+1])
            zone_state = MarketState(states[start_idx])
            
            zone_strength = self._calculate_zone_strength(
                close[start_idx:end_idx], ema[start_idx:end_idx], atr[start_idx:end_idx]
            )
            zone_confidence = self._calculate_zone_confidence(
                high[start_idx:end_idx], low[start_idx:end_idx], ema[start_idx:end_idx]
            )
            
            zones.append(MarketZone(
                start_idx=start_idx,
                end_idx=end_idx-1,
                state=zone_state,
                strength=zone_strength,
                confidence=zone_confidence
            ))
        
        return zones
    
    
    def _calculate_zone_strength(
        self,
        close: np.ndarray,
        ema: np.ndarray,
        atr: np.ndarray
    ) -> float:
        
        if len(close) == 0:
            return 0.0
        
        distance = np.abs(close - ema)
        normalized_distance = distance / (atr + 1e-8)
        
        return float(np.clip(np.mean(normalized_distance), 0, 1))
    
    
    def _calculate_zone_confidence(
        self,
        high: np.ndarray,
        low: np.ndarray,
        ema: np.ndarray
    ) -> float:
        
        if len(high) == 0:
            return 0.0
        
        touches = 0
        for h, l, e in zip(high, low, ema):
            if l <= e <= h:
                touches += 1
        
        consistency = 1.0 - (touches / len(high))
        
        return float(np.clip(consistency, 0, 1))
    
    
    def _detect_current_state(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        rsi: Optional[np.ndarray]
    ) -> MarketState:
        """
        PHASE 2: Enhanced SIDEWAY detection với 4 confirmations
        - Range tightness: (H-L)/ATR < 3.0
        - EMA flatness: abs(slope) < 1%
        - Volume decline: current < 0.8x average
        - RSI neutral: 45 < RSI < 55
        
        SIDEWAY chỉ khi 3/4 confirmations pass (strict)
        """
        
        window = min(self.lookback_short, len(df))
        
        close = df['close'].values[-window:]
        high = df['high'].values[-window:]
        low = df['low'].values[-window:]
        ema_window = ema[-window:]
        
        above_ema = np.sum(close > ema_window) / window
        below_ema = np.sum(close < ema_window) / window
        
        # Initial classification (60% cases)
        if above_ema > 0.65:
            initial_state = MarketState.LONG
        elif below_ema > 0.65:
            initial_state = MarketState.SHORT
        else:
            initial_state = MarketState.SIDEWAY
        
        # PHASE 2: SIDEWAY CONFIRMATION (only check if initial = SIDEWAY)
        if initial_state == MarketState.SIDEWAY:
            confirmations = 0
            
            # 1. Range Tightness (20 bars trong 3 ATR range)
            range_high = np.max(high[-20:]) if len(high) >= 20 else high[-1]
            range_low = np.min(low[-20:]) if len(low) >= 20 else low[-1]
            current_atr = atr[-1] if len(atr) > 0 else 1.0
            range_atr = (range_high - range_low) / (current_atr + 1e-8)
            if range_atr < 3.0:
                confirmations += 1
            
            # 2. EMA Flatness (slope < threshold)
            if len(ema_window) >= 10:
                ema_slope = (ema_window[-1] - ema_window[-10]) / (ema_window[-10] + 1e-8)
                if abs(ema_slope) < self.sideway_slope_pct:
                    confirmations += 1
            
            # 3. Volume Decline (optional - only if volume available)
            if 'volume' in df.columns:
                volume = df['volume'].values[-window:]
                if len(volume) >= 2:
                    vol_ratio = volume[-1] / (np.mean(volume[-20:-1]) + 1e-8)
                    if vol_ratio < 0.8:
                        confirmations += 1
            
            # 4. RSI Neutral (optional - only if RSI available)
            if rsi is not None and len(rsi) > 0:
                current_rsi = rsi[-1]
                if 45 < current_rsi < 55:
                    confirmations += 1
            
            # Require 3/4 confirmations for SIDEWAY (strict)
            # Minimum 2 if volume/RSI not available
            required = 2 if ('volume' not in df.columns or rsi is None) else 3
            
            if confirmations >= required:
                return MarketState.SIDEWAY  # Confirmed SIDEWAY
            else:
                # Not really sideway, re-check with looser threshold
                if above_ema > 0.55:
                    return MarketState.LONG  # Early trend forming
                elif below_ema > 0.55:
                    return MarketState.SHORT
                else:
                    return MarketState.SIDEWAY  # Uncertain (default to sideway)
        
        return initial_state
    
    
    def _predict_next_state(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        zones: List[MarketZone],
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray]
    ) -> MarketState:
        
        if not zones:
            return MarketState.SIDEWAY
        
        recent_zone = zones[-1]
        
        if len(zones) >= 2:
            prev_zone = zones[-2]
            if recent_zone.state == prev_zone.state and recent_zone.strength > 0.6:
                return recent_zone.state
        
        window = min(self.lookback_short, len(df))
        close = df['close'].values[-window:]
        ema_window = ema[-window:]
        
        momentum = (close[-1] - close[0]) / (close[0] + 1e-8)
        ema_slope = (ema_window[-1] - ema_window[0]) / (ema_window[0] + 1e-8)
        
        mt = self.momentum_threshold
        st = self.sideway_slope_pct
        if momentum > mt and ema_slope > st:
            return MarketState.LONG
        elif momentum < -mt and ema_slope < -st:
            return MarketState.SHORT
        else:
            return MarketState.SIDEWAY
    
    
    def _calculate_prediction_confidence(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        zones: List[MarketZone],
        current_state: MarketState,
        next_state: MarketState
    ) -> float:
        
        if not zones:
            return 0.5
        
        recent_zone = zones[-1]
        zone_confidence = recent_zone.confidence
        
        state_alignment = 1.0 if current_state == next_state else 0.5
        
        window = min(self.lookback_short, len(df))
        close = df['close'].values[-window:]
        volatility = np.std(close) / (np.mean(close) + 1e-8)
        volatility_factor = np.clip(1.0 - volatility * 10, 0.3, 1.0)
        
        confidence = (zone_confidence * 0.4 + state_alignment * 0.4 + volatility_factor * 0.2)
        
        return float(np.clip(confidence, 0, 1))
    
    
    def _analyze_trend_strength(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        current_state: MarketState,
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray]
    ) -> Tuple[float, TrendStrength, bool]:
        
        window = min(self.lookback_short, len(df))
        
        close = df['close'].values[-window:]
        high = df['high'].values[-window:]
        low = df['low'].values[-window:]
        ema_window = ema[-window:]
        atr_window = atr[-window:]
        
        price_momentum = (close[-1] - close[0]) / (close[0] + 1e-8)
        ema_slope = (ema_window[-1] - ema_window[0]) / (ema_window[0] + 1e-8)
        
        distance_from_ema = np.abs(close - ema_window) / (atr_window + 1e-8)
        avg_distance = np.mean(distance_from_ema)
        
        if len(close) > 1:
            price_changes = close[1:] - close[:-1]
            if current_state == MarketState.LONG:
                directional_moves = np.sum(price_changes > 0)
            elif current_state == MarketState.SHORT:
                directional_moves = np.sum(price_changes < 0)
            else:
                directional_moves = 0
            consistency = directional_moves / len(price_changes)
        else:
            consistency = 0
        
        rsi_alignment = 0.5
        if rsi is not None and len(rsi) >= window:
            rsi_window = rsi[-window:]
            if current_state == MarketState.LONG:
                rsi_alignment = np.clip((np.mean(rsi_window) - 50) / 50, 0, 1)
            elif current_state == MarketState.SHORT:
                rsi_alignment = np.clip((50 - np.mean(rsi_window)) / 50, 0, 1)
        
        vol_strength = 0.5
        if volume is not None and len(volume) >= window:
            vol_window = volume[-window:]
            vol_avg = np.mean(vol_window[:-1]) if len(vol_window) > 1 else vol_window[0]
            vol_current = vol_window[-1]
            vol_strength = np.clip(vol_current / (vol_avg + 1e-8), 0, 2) / 2
        
        sustained_distance = avg_distance
        
        # Adaptive weights: volume_weight is configurable (crypto=0.25, forex=0.05)
        vw = self.volume_weight
        # Redistribute remaining weight proportionally
        remaining = 1.0 - vw
        base_non_vol = 0.75  # original: mom=0.20 + ema=0.10 + dist=0.15 + cons=0.20 + rsi=0.10
        scale = remaining / base_non_vol
        
        strength_score = (
            abs(price_momentum) * 0.20 * scale +
            abs(ema_slope) * 0.10 * scale +
            sustained_distance * 0.15 * scale +
            consistency * 0.20 * scale +
            rsi_alignment * 0.10 * scale +
            vol_strength * vw
        )
        
        strength_score = float(np.clip(strength_score, 0, 1))
        
        if strength_score >= self.strength_threshold:
            trend_strength = TrendStrength.STRONG
            is_strong = True
        elif strength_score >= 0.4:
            trend_strength = TrendStrength.WEAK
            is_strong = False
        else:
            trend_strength = TrendStrength.NEUTRAL
            is_strong = False
        
        return strength_score, trend_strength, is_strong
    
    
    def _analyze_mtf_alignment(
        self,
        df_micro: pd.DataFrame,
        ema_micro: np.ndarray,
        df_macro: pd.DataFrame,
        ema_macro: np.ndarray,
        micro_state: MarketState
    ) -> Dict:
        
        close_macro = df_macro['close'].values
        
        if len(close_macro) < 20:
            return {'aligned': False, 'macro_state': MarketState.SIDEWAY, 'alignment_score': 0.5}
        
        above_ema_macro = np.sum(close_macro[-20:] > ema_macro[-20:]) / 20
        
        if above_ema_macro > 0.65:
            macro_state = MarketState.LONG
        elif above_ema_macro < 0.35:
            macro_state = MarketState.SHORT
        else:
            macro_state = MarketState.SIDEWAY
        
        aligned = (micro_state == macro_state)
        
        alignment_score = 1.0 if aligned else 0.3
        if macro_state == MarketState.SIDEWAY:
            alignment_score = 0.6
        
        return {
            'aligned': aligned,
            'macro_state': macro_state,
            'alignment_score': alignment_score
        }
    
    
    def _calculate_weak_stability(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        current_state: MarketState,
        is_strong: bool
    ) -> Tuple[str, float]:
        """
        Tính "Độ Ổn Định" của thị trường YẾU để quyết định allocation mode:
        - PARTIAL_ONLY: Yếu + Bất ổn (nhiễu cao) → Chỉ vào phần đầu, không nhồi
        - FULL_MATRIX: Yếu + Ổn định (pullback nhịp nhàng) → Scale-in đầy đủ
        
        Công thức xét 4 yếu tố:
        1. Whipsaw (Dao động EMA): Càng nhiều cross → Càng bất ổn
        2. ATR Volatility: ATR tăng đột ngột → Bất ổn
        3. Directional Consistency: Giá lên xuống lung tung → Bất ổn
        4. Volume Spike: Volume đột ngột tăng → Bất ổn (có thể gãy trend)
        
        Returns:
            allocation_mode: 'PARTIAL_ONLY' hoặc 'FULL_MATRIX'
            stability_score: 0.0-1.0 (càng cao càng ổn định)
        """
        # Nếu không phải WEAK, return FULL_MATRIX (không cần đánh giá)
        if is_strong or current_state == MarketState.SIDEWAY:
            return ('FULL_MATRIX', 1.0)
        
        window = min(20, len(df))
        close = df['close'].values[-window:]
        ema_window = ema[-window:]
        atr_window = atr[-window:]
        
        # =====================================================================
        # 1. WHIPSAW SCORE (Dao động qua lại EMA)
        # =====================================================================
        above = close > ema_window
        crosses = int(np.sum(above[1:] != above[:-1]))
        whipsaw_score = crosses / window if window > 0 else 0
        
        # Nhiều cross (>0.3) → Bất ổn
        whipsaw_penalty = np.clip(whipsaw_score / 0.3, 0, 1)  # 0-1, càng cao càng tệ
        
        # =====================================================================
        # 2. ATR VOLATILITY EXPANSION
        # =====================================================================
        if len(atr_window) >= 5:
            atr_slope = (atr_window[-1] - atr_window[-5]) / (atr_window[-5] + 1e-8)
        else:
            atr_slope = 0
        
        # ATR tăng >10% → Bất ổn
        atr_penalty = np.clip(atr_slope / 0.10, 0, 1) if atr_slope > 0 else 0
        
        # =====================================================================
        # 3. DIRECTIONAL CONSISTENCY
        # =====================================================================
        if len(close) > 1:
            price_changes = close[1:] - close[:-1]
            
            if current_state == MarketState.LONG:
                # LONG WEAK: Pullback (giảm giá) là bình thường
                # Nếu giá tăng giảm lung tung → Bất ổn
                negative_moves = np.sum(price_changes < 0)
                positive_moves = np.sum(price_changes > 0)
                # Expect pullback: 60-70% negative moves
                expected_negative = 0.65 * len(price_changes)
                consistency_error = abs(negative_moves - expected_negative) / len(price_changes)
            
            elif current_state == MarketState.SHORT:
                # SHORT WEAK: Pullback (tăng giá) là bình thường
                negative_moves = np.sum(price_changes < 0)
                positive_moves = np.sum(price_changes > 0)
                expected_positive = 0.65 * len(price_changes)
                consistency_error = abs(positive_moves - expected_positive) / len(price_changes)
            
            else:
                consistency_error = 0
            
            consistency_penalty = np.clip(consistency_error, 0, 1)
        else:
            consistency_penalty = 0
        
        # =====================================================================
        # 4. VOLUME SPIKE (Khối lượng đột ngột tăng)
        # =====================================================================
        volume_penalty = 0
        if 'volume' in df.columns:
            volume = df['volume'].values[-window:]
            if len(volume) >= 10:
                avg_volume = np.mean(volume[:-1])
                current_volume = volume[-1]
                volume_surge = current_volume / (avg_volume + 1e-8)
                
                # Volume tăng >2x → Bất ổn (có thể bị gãy trend)
                volume_penalty = np.clip((volume_surge - 1.5) / 1.0, 0, 1) if volume_surge > 1.5 else 0
        
        # =====================================================================
        # TỔNG HỢP STABILITY SCORE
        # =====================================================================
        # Stability = 1.0 - weighted penalties
        total_penalty = (
            whipsaw_penalty * 0.35 +      # Whipsaw quan trọng nhất
            atr_penalty * 0.25 +           # Volatility expansion
            consistency_penalty * 0.25 +   # Directional consistency
            volume_penalty * 0.15          # Volume spike
        )
        
        stability_score = float(np.clip(1.0 - total_penalty, 0, 1))
        
        # =====================================================================
        # QUYẾT ĐỊNH ALLOCATION MODE
        # =====================================================================
        # Ngưỡng: stability < 0.5 → PARTIAL_ONLY (bất ổn, đánh du kích)
        #         stability >= 0.5 → FULL_MATRIX (ổn định, rải lưới scale-in)
        if stability_score < 0.5:
            allocation_mode = 'PARTIAL_ONLY'
        else:
            allocation_mode = 'FULL_MATRIX'
        
        return (allocation_mode, stability_score)
    
    def _detect_adaptive_transition(
        self,
        current_state: MarketState,
        current_is_strong: bool,
        current_strength_score: float,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray]
    ) -> TransitionSignal:
        """
        ADAPTIVE TRANSITION DETECTION (Phát hiện chuyển đổi thích nghi)
        
        Phát hiện khi strength thay đổi đáng kể và cung cấp hướng dẫn chuyển đổi chiến lược.
        
        Logic:
        1. Chỉ chuyển trong cùng một loại (LONG → LONG, SHORT → SHORT)
        2. Yêu cầu sustained change (N candles confirm)
        3. Kết hợp nhiều tín hiệu: Volume, ATR, RSI
        4. Cooldown period để tránh flip-flop
        5. Delta đủ lớn (ví dụ: 0.45 → 0.60 = +0.15)
        
        Returns:
            TransitionSignal với can_execute = True nếu đủ điều kiện
        """
        # Default: No transition
        default_signal = TransitionSignal(
            transition_detected=False,
            from_strength=TrendStrength.NEUTRAL,
            to_strength=TrendStrength.NEUTRAL,
            confidence=0.0,
            confirmation_signals=0,
            can_execute=False,
            reason="No transition detected"
        )
        
        # Check if adaptive transition is enabled
        if not self.enable_adaptive_transition:
            return default_signal
        
        # Cập nhật historical states
        current_time = time.time()
        current_trend_strength = TrendStrength.STRONG if current_is_strong else (
            TrendStrength.WEAK if current_strength_score >= 0.4 else TrendStrength.NEUTRAL
        )
        
        self._historical_states.append((
            current_time,
            current_state,
            current_is_strong,
            current_strength_score
        ))
        
        # Chỉ giữ lại N candles gần nhất
        max_history = self.transition_confirmation_candles + 5
        if len(self._historical_states) > max_history:
            self._historical_states = self._historical_states[-max_history:]
        
        # =====================================================================
        # ĐIỀU KIỆN 1: COOLDOWN PERIOD (Tránh flip-flop)
        # =====================================================================
        time_since_last_transition = current_time - self._last_transition_time
        if time_since_last_transition < self.transition_cooldown_seconds:
            return TransitionSignal(
                transition_detected=False,
                from_strength=current_trend_strength,
                to_strength=current_trend_strength,
                confidence=0.0,
                confirmation_signals=0,
                can_execute=False,
                reason=f"Cooldown active ({time_since_last_transition:.0f}s / {self.transition_cooldown_seconds:.0f}s)"
            )
        
        # =====================================================================
        # ĐIỀU KIỆN 2: Cần đủ historical data
        # =====================================================================
        # Need at least (confirmation_candles + 1) to compare old vs recent states
        if len(self._historical_states) <= self.transition_confirmation_candles:
            return default_signal
        
        # =====================================================================
        # PHÁT HIỆN TRANSITION (WEAK → STRONG hoặc STRONG → WEAK)
        # =====================================================================
        # Lấy state cũ (N candles trước)
        old_state_data = self._historical_states[-(self.transition_confirmation_candles + 1)]
        old_time, old_state, old_is_strong, old_strength_score = old_state_data
        
        # Chỉ chuyển trong cùng 1 loại (LONG → LONG, SHORT → SHORT)
        if old_state != current_state:
            return TransitionSignal(
                transition_detected=False,
                from_strength=current_trend_strength,
                to_strength=current_trend_strength,
                confidence=0.0,
                confirmation_signals=0,
                can_execute=False,
                reason=f"State changed from {old_state.name} to {current_state.name} (không hỗ trợ cross-transition)"
            )
        
        # Tính delta strength
        strength_delta = current_strength_score - old_strength_score
        
        # Phát hiện transition
        transition_detected = False
        from_strength = TrendStrength.STRONG if old_is_strong else TrendStrength.WEAK
        to_strength = TrendStrength.STRONG if current_is_strong else TrendStrength.WEAK
        
        # WEAK → STRONG
        if not old_is_strong and current_is_strong and strength_delta >= self.transition_strength_delta:
            transition_detected = True
            transition_type = "WEAK → STRONG"
        
        # STRONG → WEAK
        elif old_is_strong and not current_is_strong and strength_delta <= -self.transition_strength_delta:
            transition_detected = True
            transition_type = "STRONG → WEAK"
        
        else:
            # No significant transition
            return default_signal
        
        # =====================================================================
        # ĐIỀU KIỆN 3: SUSTAINED CHANGE (Confirm trong N candles)
        # =====================================================================
        # Kiểm tra xem trend có sustained trong N candles gần nhất không
        recent_states = self._historical_states[-self.transition_confirmation_candles:]
        sustained = all(
            (is_strong == current_is_strong) for (_, _, is_strong, _) in recent_states
        )
        
        if not sustained:
            num_matching = len([1 for (_, _, is_strong, _) in recent_states if is_strong == current_is_strong])
            return TransitionSignal(
                transition_detected=True,
                from_strength=from_strength,
                to_strength=to_strength,
                confidence=0.3,
                confirmation_signals=0,
                can_execute=False,
                reason=f"{transition_type} detected but not sustained ({num_matching}/{self.transition_confirmation_candles} candles)"
            )
        
        # =====================================================================
        # ĐIỀU KIỆN 4: MULTI-SIGNAL CONFIRMATION
        # =====================================================================
        confirmation_signals = 0
        confirmation_details = []
        
        window = min(10, len(df))
        
        # Signal 1: Volume Confirmation
        if volume is not None and len(volume) >= window:
            vol_window = volume[-window:]
            vol_avg = np.mean(vol_window[:-1]) if len(vol_window) > 1 else vol_window[0]
            vol_current = vol_window[-1]
            vol_surge = vol_current / (vol_avg + 1e-8)
            
            # WEAK → STRONG: Volume tăng (>1.2x) → Tích cực
            # STRONG → WEAK: Volume giảm (<0.8x) → Tiêu cực
            if transition_type == "WEAK → STRONG" and vol_surge > 1.2:
                confirmation_signals += 1
                confirmation_details.append(f"Volume surge {vol_surge:.2f}x")
            elif transition_type == "STRONG → WEAK" and vol_surge < 0.8:
                confirmation_signals += 1
                confirmation_details.append(f"Volume drop {vol_surge:.2f}x")
        
        # Signal 2: ATR Direction
        if len(atr) >= window:
            atr_window = atr[-window:]
            atr_slope = (atr_window[-1] - atr_window[0]) / (atr_window[0] + 1e-8)
            
            # WEAK → STRONG: ATR tăng nhưng giá theo hướng (tích cực)
            # STRONG → WEAK: ATR giảm (hết động lực)
            if transition_type == "WEAK → STRONG" and atr_slope > 0.05:
                confirmation_signals += 1
                confirmation_details.append(f"ATR expanding {atr_slope:.2%}")
            elif transition_type == "STRONG → WEAK" and atr_slope < -0.05:
                confirmation_signals += 1
                confirmation_details.append(f"ATR contracting {atr_slope:.2%}")
        
        # Signal 3: RSI Confirmation
        if rsi is not None and len(rsi) >= window:
            rsi_window = rsi[-window:]
            rsi_current = rsi_window[-1]
            
            # LONG: RSI tăng → Tích cực
            # SHORT: RSI giảm → Tích cực
            if current_state == MarketState.LONG:
                if transition_type == "WEAK → STRONG" and rsi_current > 55:
                    confirmation_signals += 1
                    confirmation_details.append(f"RSI bullish {rsi_current:.1f}")
                elif transition_type == "STRONG → WEAK" and rsi_current < 45:
                    confirmation_signals += 1
                    confirmation_details.append(f"RSI weakening {rsi_current:.1f}")
            
            elif current_state == MarketState.SHORT:
                if transition_type == "WEAK → STRONG" and rsi_current < 45:
                    confirmation_signals += 1
                    confirmation_details.append(f"RSI bearish {rsi_current:.1f}")
                elif transition_type == "STRONG → WEAK" and rsi_current > 55:
                    confirmation_signals += 1
                    confirmation_details.append(f"RSI weakening {rsi_current:.1f}")
        
        # Signal 4: Trend Consistency (Directional moves)
        close = df['close'].values[-window:]
        if len(close) > 1:
            price_changes = close[1:] - close[:-1]
            
            if current_state == MarketState.LONG:
                positive_ratio = np.sum(price_changes > 0) / len(price_changes)
                if transition_type == "WEAK → STRONG" and positive_ratio > 0.7:
                    confirmation_signals += 1
                    confirmation_details.append(f"Trend consistency {positive_ratio:.1%}")
            
            elif current_state == MarketState.SHORT:
                negative_ratio = np.sum(price_changes < 0) / len(price_changes)
                if transition_type == "WEAK → STRONG" and negative_ratio > 0.7:
                    confirmation_signals += 1
                    confirmation_details.append(f"Trend consistency {negative_ratio:.1%}")
        
        # =====================================================================
        # QUYẾT ĐỊNH CAN_EXECUTE
        # =====================================================================
        # Yêu cầu ít nhất 2/4 signals confirm
        min_confirmations = 2
        can_execute = (confirmation_signals >= min_confirmations)
        
        # Calculate confidence (0.0-1.0)
        confidence = (
            (confirmation_signals / 4) * 0.4 +  # Signal confirmation weight
            min(abs(strength_delta) / 0.3, 1.0) * 0.3 +  # Delta magnitude
            (1.0 if sustained else 0.0) * 0.3  # Sustained confirmation
        )
        confidence = float(np.clip(confidence, 0, 1))
        
        # =====================================================================
        # STRATEGY TRANSITION INSTRUCTIONS
        # =====================================================================
        old_strategy = None
        new_strategy = None
        risk_adjustment = None
        
        if can_execute:
            # Update last transition time
            self._last_transition_time = current_time
            
            # Determine strategy changes
            if current_state == MarketState.LONG:
                if transition_type == "WEAK → STRONG":
                    old_strategy = "ScaleOut_4_Tranches_LONG / Accumulate_3_3_4"
                    new_strategy = "Pyramid_4_3_3"
                    risk_adjustment = 1.2 / 0.4  # 40% → 120% = 3x increase
                elif transition_type == "STRONG → WEAK":
                    old_strategy = "Pyramid_4_3_3"
                    new_strategy = "Accumulate_3_3_4"
                    risk_adjustment = 0.8 / 1.2  # 120% → 80%
            
            elif current_state == MarketState.SHORT:
                if transition_type == "WEAK → STRONG":
                    old_strategy = "ScaleOut_3_Tranches_SHORT / Accumulate_SHORT_3_3_4"
                    new_strategy = "Strict_ScaleIn_2_3_3_2"
                    risk_adjustment = 0.8 / 0.3  # 30% → 80%
                elif transition_type == "STRONG → WEAK":
                    old_strategy = "Strict_ScaleIn_2_3_3_2"
                    new_strategy = "Accumulate_SHORT_3_3_4"
                    risk_adjustment = 0.7 / 0.8  # 80% → 70%
        
        # Compose reason
        reason = f"{transition_type}: Strength {old_strength_score:.2f} → {current_strength_score:.2f} (Δ={strength_delta:+.2f}). "
        reason += f"Confirmations: {confirmation_signals}/4 ({', '.join(confirmation_details)}). "
        if can_execute:
            reason += "✅ APPROVED for execution"
        else:
            reason += f"❌ Need {min_confirmations - confirmation_signals} more confirmations"
        
        return TransitionSignal(
            transition_detected=transition_detected,
            from_strength=from_strength,
            to_strength=to_strength,
            confidence=confidence,
            confirmation_signals=confirmation_signals,
            can_execute=can_execute,
            reason=reason,
            old_strategy=old_strategy,
            new_strategy=new_strategy,
            risk_adjustment=risk_adjustment
        )
    
    
    def _activate_predator_mode(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray]
    ) -> Dict:
        
        close = df['close'].values
        
        dist_to_upper = (upper_band[-1] - close[-1]) / (atr[-1] + 1e-8)
        dist_to_lower = (close[-1] - lower_band[-1]) / (atr[-1] + 1e-8)
        
        is_at_boundary = (
            dist_to_upper < self.sideway_boundary_threshold or 
            dist_to_lower < self.sideway_boundary_threshold
        )
        
        at_upper = dist_to_upper < self.sideway_boundary_threshold
        at_lower = dist_to_lower < self.sideway_boundary_threshold
        
        window = min(5, len(atr))
        atr_slope = (atr[-1] - atr[-window]) / (atr[-window] + 1e-8)
        atr_expanding = atr_slope > 0.05
        
        volume_surge = 1.0
        if volume is not None and len(volume) >= 20:
            avg_volume = np.mean(volume[-20:-1]) if len(volume) > 1 else volume[-1]
            volume_surge = volume[-1] / (avg_volume + 1e-8)
        
        band_width = (upper_band[-1] - lower_band[-1]) / (ema[-1] + 1e-8)
        band_squeezing = band_width < 0.02
        
        return {
            'is_at_boundary': is_at_boundary,
            'at_upper_band': at_upper,
            'at_lower_band': at_lower,
            'dist_to_upper': dist_to_upper,
            'dist_to_lower': dist_to_lower,
            'atr_expanding': atr_expanding,
            'atr_slope': atr_slope,
            'volume_surge': volume_surge,
            'band_squeezing': band_squeezing
        }
    
    
    def _calculate_whipsaw_score(
        self,
        df: pd.DataFrame,
        ema: np.ndarray,
        lookback: int = 20
    ) -> Tuple[int, float]:
        
        window = min(lookback, len(df))
        close = df['close'].values[-window:]
        ema_window = ema[-window:]
        
        above = close > ema_window
        crosses = int(np.sum(above[1:] != above[:-1]))
        
        whipsaw_score = crosses / window if window > 0 else 0
        
        return crosses, float(whipsaw_score)
    
    
    def _step1_validate_signal(
        self,
        prediction: RegimePrediction,
        signal_state: MarketState,
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray,
        rsi: Optional[np.ndarray],
        volume: Optional[np.ndarray]
    ) -> SignalValidation:
        
        regime_state = prediction.current_state
        matches = (regime_state == signal_state)
        
        warnings = []
        
        trust_components = np.zeros(7, dtype=np.float32)
        comp_idx = 0
        
        if prediction.confidence < self.confidence_threshold:
            trust_components[comp_idx] = 0.3
            warnings.append(f"Low prediction confidence: {prediction.confidence:.2f}")
        else:
            trust_components[comp_idx] = 0.8
        comp_idx += 1
        
        if prediction.is_strong:
            trust_components[comp_idx] = 0.9
        else:
            trust_components[comp_idx] = 0.5
            warnings.append(f"Weak trend strength: {prediction.strength_score:.2f}")
        comp_idx += 1
        
        if regime_state == MarketState.SIDEWAY and prediction.sideway_context:
            ctx = prediction.sideway_context
            
            if not ctx['is_at_boundary']:
                trust_components[comp_idx] = 0.2
                warnings.append(f"SIDEWAY: Price floating mid-range (Upper: {ctx['dist_to_upper']:.2f}ATR, Lower: {ctx['dist_to_lower']:.2f}ATR) - High risk")
            else:
                trust_components[comp_idx] = 0.9
                boundary_type = "UPPER" if ctx['at_upper_band'] else "LOWER"
                warnings.append(f"SIDEWAY: Price at {boundary_type} boundary - Prime entry zone")
        else:
            if matches:
                trust_components[comp_idx] = 1.0
            else:
                trust_components[comp_idx] = 0.2
                warnings.append(f"Signal-Regime mismatch: Signal={signal_state.name}, Regime={regime_state.name}")
        comp_idx += 1
        
        # PHASE 1: Stricter volatility threshold (0.05 → 0.03)
        window = min(self.lookback_short, len(df))
        close = df['close'].values[-window:]
        volatility = np.std(close) / (np.mean(close) + 1e-8)
        
        if volatility > 0.03:  # Was 0.05 (stricter threshold)
            trust_components[comp_idx] = 0.3  # Was 0.4 (harsher penalty)
            warnings.append(f"High volatility: {volatility:.4f}")
        else:
            trust_components[comp_idx] = 0.8
        comp_idx += 1
        
        # PHASE 1: Volume REQUIRED (not optional) - minimum 1.2x for confirmation
        if volume is not None and len(volume) >= 2:
            vol_spike = volume[-1] / (np.mean(volume[-20:-1]) + 1e-8)
            if vol_spike > 1.5:
                trust_components[comp_idx] = 0.9
            elif vol_spike > 1.2:  # NEW: Minimum threshold
                trust_components[comp_idx] = 0.6
            else:
                trust_components[comp_idx] = 0.3  # Was 0.6 (penalty for low volume)
                warnings.append(f"Low volume confirmation: {vol_spike:.2f}x (need >1.2x)")
        else:
            trust_components[comp_idx] = 0.5  # Was 0.6
        comp_idx += 1
        
        # PHASE 1: Stricter whipsaw penalty (0.3 → 0.1)
        crosses, whipsaw_score = self._calculate_whipsaw_score(df, ema, self.lookback_short)
        if crosses >= self.whipsaw_cross_threshold:
            trust_components[comp_idx] = 0.1  # Was 0.3 (stricter penalty)
            warnings.append(f"WHIPSAW ALERT: {crosses} EMA crosses in {self.lookback_short} candles (score: {whipsaw_score:.2f})")
        else:
            trust_components[comp_idx] = 0.8
        comp_idx += 1
        
        if prediction.mtf_alignment and not prediction.mtf_alignment['aligned']:
            trust_components[comp_idx] = 0.4
            warnings.append(f"MTF Misalignment: Micro={regime_state.name}, Macro={prediction.mtf_alignment['macro_state'].name}")
        elif prediction.mtf_alignment:
            trust_components[comp_idx] = 0.9
        
        trust_score = float(np.mean(trust_components))
        
        is_valid = (trust_score >= 0.65 and matches)
        
        return SignalValidation(
            is_valid=is_valid,
            matches_regime=matches,
            trust_score=trust_score,
            regime_says=regime_state,
            signal_says=signal_state,
            warnings=warnings
        )
    
    
    def _step2_calculate_probabilities(
        self,
        prediction: RegimePrediction,
        validation: Optional[SignalValidation],
        df: pd.DataFrame,
        ema: np.ndarray,
        atr: np.ndarray
    ) -> List[StrategyProbability]:
        """
        MULTIVERSE SIMULATOR:
        Giả định TẤT CẢ các chiến lược có thể cho trạng thái thị trường,
        tính toán xác suất thống kê cho từng chiến lược,
        trả về DANH SÁCH ĐẦY ĐỦ để Execute quyết định.
        
        Không chọn sẵn 1 chiến lược duy nhất,
        mà luôn mở rộng toàn bộ không gian khả năng (possibility space).
        """
        strategies = []
        
        current_state = prediction.current_state
        is_strong = prediction.is_strong
        strength_score = prediction.strength_score
        
        base_trust = validation.trust_score if validation else 0.5
        
        mtf_boost = 1.0
        if prediction.mtf_alignment and prediction.mtf_alignment['aligned']:
            mtf_boost = 1.2
        elif prediction.mtf_alignment and not prediction.mtf_alignment['aligned']:
            mtf_boost = 0.7
        
        # Calculate dynamic TP based on volatility (ATR) and trust score
        # Volatility thấp → TP nhỏ (2-3%)
        # Volatility cao → TP lớn (4-5%)
        # Trust score điều chỉnh fine-tune
        current_atr = float(atr[-1]) if len(atr) > 0 else 0.0
        current_price = float(df['close'].iloc[-1])
        
        # Volatility measure: ATR as % of price
        volatility_pct = (current_atr / current_price) if current_price > 0 else 0.02
        
        # Base TP: 2-5% depending on volatility
        # Low volatility (<1.5%) → TP 2-3%
        # High volatility (>3%) → TP 4-5%
        if volatility_pct < 0.015:
            tp_pct_base = 0.02 + (base_trust * 0.01)  # 2-3%
        elif volatility_pct > 0.03:
            tp_pct_base = 0.04 + (base_trust * 0.01)  # 4-5%
        else:
            tp_pct_base = 0.02 + (volatility_pct * 100) + (base_trust * 0.01)  # 2-5% adaptive
        
        # =====================================================================
        # LONG: Strategies theo is_strong
        # =====================================================================
        if current_state == MarketState.LONG:
            
            if is_strong:
                # Chiến lược 1: PYRAMID (Đuổi đà mạnh - "ĂN TO")
                # PHASE 1: Recalibrated WR 0.68 → 0.45 (match backtest reality ~30% conservative)
                pyramid_prob = 0.70 * base_trust * mtf_boost  # Reduce prob slightly
                pyramid_win_rate = 0.45  # CALIBRATED: Was 0.68 (overpredicted by 38%!)
                pyramid_rr = 3.0  # Increase R:R (ăn TO khi đúng)
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.TREND_FOLLOW,
                    probability=min(pyramid_prob, 0.95),
                    expected_winrate=pyramid_win_rate,
                    risk_multiplier=1.2,  # Aggressive, risk 1.2x base
                    sl_multiplier=1.5,    # Progressive SL
                    tp_multiplier=4.0,    # Max TP
                    edge_score=strength_score * pyramid_win_rate * pyramid_rr
                ))
                
                # Chiến lược 2: BREAKOUT (Chờ xác nhận breakout - CHỈ DÀNH CHO STRONG)
                # PHASE 1: Recalibrated WR 0.60 → 0.35 (breakout success rate thấp)
                breakout_prob = 0.60 * base_trust * mtf_boost
                breakout_win_rate = 0.35  # CALIBRATED: Was 0.60 (breakout khó hơn)
                breakout_rr = 4.0  # Increase R:R compensation
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.BREAKOUT,
                    probability=min(breakout_prob, 0.90),
                    expected_winrate=breakout_win_rate,
                    risk_multiplier=1.0,  # Normal risk
                    sl_multiplier=2.0,    # Standard SL
                    tp_multiplier=5.0,    # Extended TP
                    edge_score=strength_score * breakout_win_rate * breakout_rr
                ))
                
            else:  # LONG WEAK
                # Chiến lược 1: ACCUMULATE (Gom hàng từ từ 3-3-4)
                # Dùng FULL risk, nhồi dần 30%-30%-40%
                # PHASE 1: Recalibrated WR 0.65 → 0.50 (conservative)
                accumulate_prob = 0.65 * base_trust * mtf_boost
                accumulate_win_rate = 0.50  # CALIBRATED: Was 0.65 (more realistic)
                accumulate_rr = 2.0  # Slight increase
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.TREND_FOLLOW,
                    probability=min(accumulate_prob, 0.92),
                    expected_winrate=accumulate_win_rate,
                    risk_multiplier=0.8,  # Conservative (80% risk)
                    sl_multiplier=1.0,    # Tight SL at EMA
                    tp_multiplier=2.5,    # Realistic TP at swing high
                    edge_score=base_trust * accumulate_win_rate * accumulate_rr
                ))
                
                # Chiến lược 2: SCALE-OUT (Chốt lời 4 phần - LONG WEAK)
                # KEY: Chỉ dùng 40% vốn (4 phần trong 10 phần)
                # Ví dụ: risk = 1% → CHỈ DÙNG 0.4%
                # Entry 1 lần duy nhất, không nhồi thêm
                # Chốt lời trên 4 phần: 25%-25%-25%-25%
                # PHASE 1: Recalibrated WR 0.72 → 0.50 (scalp reality ~28%, leave safety margin)
                scaleout_prob = 0.75 * base_trust * mtf_boost
                scaleout_win_rate = 0.50  # CALIBRATED: Was 0.72 (overpredicted by 44%!)
                scaleout_rr = 1.5  # Increase R:R slightly
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.SCALP,  # Use SCALP type for scale-out
                    probability=min(scaleout_prob, 0.92),
                    expected_winrate=scaleout_win_rate,
                    risk_multiplier=0.4,  # CHỈ 40% vốn! (LONG WEAK)
                    sl_multiplier=0.8,    # Very tight SL
                    tp_multiplier=tp_pct_base / 0.01,  # Dynamic TP 2-5% theo volatility
                    edge_score=base_trust * scaleout_win_rate * scaleout_rr * 1.2
                ))
                
                # KHÔNG CÓ BREAKOUT CHO LONG WEAK!
                # Breakout chỉ dành cho STRONG hoặc SIDEWAY
        
        # =====================================================================
        # SHORT: Giả định 2-3 chiến lược khác nhau
        # =====================================================================
        elif current_state == MarketState.SHORT:
            
            if is_strong:
                # Chiến lược 1: STRICT SCALE-IN ("ĂN TRỌN" khi trend mạnh)
                # PHASE 1: Recalibrated WR 0.62 → 0.40 (match backtest ~45%)
                strict_prob = 0.68 * base_trust * mtf_boost
                strict_win_rate = 0.40  # CALIBRATED: Was 0.62 (SHORT harder than LONG)
                strict_rr = 2.8  # Increase R:R (ăn TRỌN khi đúng)
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.TREND_FOLLOW,
                    probability=min(strict_prob, 0.92),
                    expected_winrate=strict_win_rate,
                    risk_multiplier=0.8,  # Conservative (SHORT riskier than LONG)
                    sl_multiplier=1.5,    # Progressive SL
                    tp_multiplier=3.0,    # Realistic TP
                    edge_score=strength_score * strict_win_rate * strict_rr
                ))
                
                # Chiến lược 2: BREAKOUT (Breakdown confirmation)
                # PHASE 1: Recalibrated WR 0.58 → 0.35 (conservative)
                breakdown_prob = 0.55 * base_trust * mtf_boost
                breakdown_win_rate = 0.35  # CALIBRATED: Was 0.58
                breakdown_rr = 3.0  # Increase R:R
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.BREAKOUT,
                    probability=min(breakdown_prob, 0.88),
                    expected_winrate=breakdown_win_rate,
                    risk_multiplier=0.7,
                    sl_multiplier=2.0,
                    tp_multiplier=4.0,
                    edge_score=strength_score * breakdown_win_rate * breakdown_rr
                ))
                
            else:  # SHORT WEAK
                # Chiến lược 1: ACCUMULATE SHORT (Bán từ từ 3-3-4 - "Ăn BÉ")
                # Dùng FULL risk, bán dần
                # PHASE 1: Recalibrated WR 0.63 → 0.45 (conservative)
                accumulate_short_prob = 0.65 * base_trust * mtf_boost
                accumulate_short_win_rate = 0.45  # CALIBRATED: Was 0.63
                accumulate_short_rr = 1.8
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.TREND_FOLLOW,
                    probability=min(accumulate_short_prob, 0.90),
                    expected_winrate=accumulate_short_win_rate,
                    risk_multiplier=0.7,  # Conservative (SHORT riskier)
                    sl_multiplier=1.0,    # Tight SL
                    tp_multiplier=2.0,    # Realistic TP
                    edge_score=base_trust * accumulate_short_win_rate * accumulate_short_rr
                ))
                
                # Chiến lược 2: SCALE-OUT (Chốt lời 3 phần - SHORT WEAK)
                # KEY: Chỉ dùng 30% vốn (3 phần trong 10 phần)
                # Ví dụ: risk = 0.5% → CHỈ DÙNG 0.15%
                # Entry 1 lần duy nhất, không nhồi thêm
                # Chốt lời trên 3 phần: 33%-34%-33%
                # PHASE 1: Recalibrated WR 0.70 → 0.45
                scaleout_short_prob = 0.70 * base_trust * mtf_boost
                scaleout_short_win_rate = 0.45  # CALIBRATED: Was 0.70
                scaleout_short_rr = 1.5  # Increase slightly
                
                strategies.append(StrategyProbability(
                    strategy=StrategyType.SCALP,
                    probability=min(scaleout_short_prob, 0.90),
                    expected_winrate=scaleout_short_win_rate,
                    risk_multiplier=0.3,  # CHỈ 30% vốn! (SHORT WEAK)
                    sl_multiplier=0.5,    # Very tight SL at swing high
                    tp_multiplier=tp_pct_base / 0.01,  # Dynamic TP 2-5% theo volatility
                    edge_score=base_trust * scaleout_short_win_rate * scaleout_short_rr * 1.3
                ))
                
                # KHÔNG CÓ BREAKDOWN CHO SHORT WEAK!
                # Breakdown chỉ dành cho STRONG hoặc có xác nhận rõ ràng
        
        # =====================================================================
        # SIDEWAY: Giả định 2-3 chiến lược tùy context
        # =====================================================================
        elif current_state == MarketState.SIDEWAY:
            
            if prediction.sideway_context:
                ctx = prediction.sideway_context
                
                volume_surge = ctx['volume_surge']
                atr_expanding = ctx['atr_expanding']
                at_boundary = ctx['is_at_boundary']
                band_squeezing = ctx['band_squeezing']
                
                # Tình huống 1: Volume surge + ATR expanding → Sắp breakout ("1 CÚ HOOK TO")
                # PHASE 1+2: Recalibrated + User philosophy (breakout = first candle only)
                if volume_surge > self.volume_surge_threshold and atr_expanding:
                    
                    breakout_prob = 0.75 * base_trust  # Reduce slightly (conservative)
                    breakout_win_rate = 0.35  # CALIBRATED: Was 0.58 (breakout success rate ~21% actual)
                    breakout_rr = 4.0  # Increase R:R (1 cú hook TO - đánh nặng)
                    
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.BREAKOUT,
                        probability=min(breakout_prob, 0.88),
                        expected_winrate=breakout_win_rate,
                        risk_multiplier=0.8,  # Reduce from 1.0 (SIDEWAY breakout riskier)
                        sl_multiplier=1.5,    # Reduce from 2.0 (tighter SL)
                        tp_multiplier=5.0,    # Increase from 4.0 (maximize if continues → TREND)
                        edge_score=volume_surge * breakout_win_rate * breakout_rr * 0.5
                    ))
                    
                    # Backup: Mean revert (phòng trường hợp fake breakout)
                    revert_backup_prob = 0.20 * base_trust
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.MEAN_REVERT,
                        probability=revert_backup_prob,
                        expected_winrate=0.40,  # Conservative backup
                        risk_multiplier=0.3,
                        sl_multiplier=1.0,
                        tp_multiplier=1.5,
                        edge_score=0.2
                    ))
                
                # Tình huống 2: Ở boundary + Volume thấp → Mean revert ("ĂN NHỎ 1 LẦN")
                # PHASE 1+2: Recalibrated + User philosophy (quick scalp, 1 lần)
                elif at_boundary and volume_surge < 1.2 and not atr_expanding:
                    
                    # Chiến lược 1: SNIPER (Mean revert ở boundary - SCALP NHANH)
                    sniper_prob = 0.85 * base_trust
                    sniper_win_rate = 0.50  # CALIBRATED: Was 0.70 (more conservative)
                    sniper_rr = 1.8  # Increase from 1.3 (ăn nhỏ nhưng R:R tốt hơn)
                    
                    # TP động: Trust cao → 3%, Trust thấp → 2% (ĂN NHỎ)
                    tp_dynamic_sniper = 0.02 + (base_trust * 0.01)
                    
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.MEAN_REVERT,
                        probability=min(sniper_prob, 0.92),
                        expected_winrate=sniper_win_rate,
                        risk_multiplier=0.3,  # Very low risk (ăn nhỏ)
                        sl_multiplier=0.5,    # Super tight SL at band
                        tp_multiplier=tp_dynamic_sniper / 0.01,
                        edge_score=base_trust * sniper_win_rate * sniper_rr * 1.4
                    ))
                    
                    # Chiến lược 2: SCALP (Quick in-out - 1 LẦN DUY NHẤT)
                    scalp_prob = 0.70 * base_trust
                    scalp_win_rate = 0.45  # CALIBRATED: Was 0.65 (scalp reality ~28%, conservative)
                    scalp_rr = 1.0
                    
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.SCALP,
                        probability=min(scalp_prob, 0.90),
                        expected_winrate=scalp_win_rate,
                        risk_multiplier=0.3,
                        sl_multiplier=0.8,
                        tp_multiplier=1.0,
                        edge_score=base_trust * scalp_win_rate * scalp_rr
                    ))
                
                # Tình huống 3: Band squeezing + Volume surge → Breakout sắp xảy ra
                # PHASE 1: Recalibrated
                elif band_squeezing and volume_surge > 1.3:
                    
                    squeeze_breakout_prob = 0.75 * base_trust
                    squeeze_win_rate = 0.38  # CALIBRATED: Was 0.62 (squeeze breakout ~35% actual)
                    squeeze_rr = 3.5  # Increase R:R compensation
                    
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.BREAKOUT,
                        probability=min(squeeze_breakout_prob, 0.90),
                        expected_winrate=squeeze_win_rate,
                        risk_multiplier=0.8,
                        sl_multiplier=1.5,
                        tp_multiplier=4.0,
                        edge_score=volume_surge * squeeze_win_rate * squeeze_rr * 0.6
                    ))
                    
                    # Backup: Scalp (nếu breakout fail)
                    scalp_backup_prob = 0.40 * base_trust
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.SCALP,
                        probability=scalp_backup_prob,
                        expected_winrate=0.42,  # CALIBRATED: Was 0.58
                        risk_multiplier=0.3,
                        sl_multiplier=1.0,
                        tp_multiplier=1.5,
                        edge_score=0.4
                    ))
                
                # Tình huống 4: Default (floating mid-range - RỦI RO CAO)
                # PHASE 2: User philosophy ("ăn nhỏ" tránh mid-range noise)
                else:
                    
                    # Chiến lược 1: Mean revert (CONSERVATIVE - low probability)
                    revert_default_prob = 0.50 * base_trust  # Reduce from 0.60
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.MEAN_REVERT,
                        probability=revert_default_prob,
                        expected_winrate=0.30,  # CALIBRATED: Was 0.60 (mid-range very hard)
                        risk_multiplier=0.4,
                        sl_multiplier=1.0,
                        tp_multiplier=1.8,
                        edge_score=0.5
                    ))
                    
                    # Chiến lược 2: Scalp
                    scalp_default_prob = 0.55 * base_trust
                    strategies.append(StrategyProbability(
                        strategy=StrategyType.SCALP,
                        probability=scalp_default_prob,
                        expected_winrate=0.58,
                        risk_multiplier=0.3,
                        sl_multiplier=0.8,
                        tp_multiplier=1.2,
                        edge_score=0.45
                    ))
            
            else:
                # No sideway context → Default strategies
                
                revert_prob = 0.62 * base_trust
                strategies.append(StrategyProbability(
                    strategy=StrategyType.MEAN_REVERT,
                    probability=revert_prob,
                    expected_winrate=0.62,
                    risk_multiplier=0.5,
                    sl_multiplier=1.0,
                    tp_multiplier=2.0,
                    edge_score=0.7
                ))
                
                scalp_prob = 0.58 * base_trust
                strategies.append(StrategyProbability(
                    strategy=StrategyType.SCALP,
                    probability=scalp_prob,
                    expected_winrate=0.58,
                    risk_multiplier=0.3,
                    sl_multiplier=1.0,
                    tp_multiplier=1.5,
                    edge_score=0.6
                ))
        
        strategies.sort(key=lambda s: s.probability * s.edge_score, reverse=True)
        
        return strategies


def analyze_regime_v3(
    df: pd.DataFrame,
    ema: np.ndarray,
    atr: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray,
    rsi: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
    signal_long: bool = False,
    signal_short: bool = False,
    df_macro: Optional[pd.DataFrame] = None,
    ema_macro: Optional[np.ndarray] = None,
    **kwargs
) -> RegimeOutput:
    
    engine = RegimeEngineV3(**kwargs)
    return engine.analyze(
        df, ema, atr, upper_band, lower_band, rsi, volume, 
        signal_long, signal_short, df_macro, ema_macro
    )


def format_regime_output(output: RegimeOutput) -> Dict:
    
    pred = output.prediction
    val = output.validation
    
    result = {
        'current_state': pred.current_state.name,
        'next_state': pred.next_state.name,
        'confidence': round(pred.confidence, 4),
        'trend_strength': pred.trend_strength.value,
        'is_strong': pred.is_strong,
        'strength_score': round(pred.strength_score, 4),
        
        'zones': [
            {
                'start': z.start_idx,
                'end': z.end_idx,
                'state': z.state.name,
                'strength': round(z.strength, 4),
                'confidence': round(z.confidence, 4)
            }
            for z in pred.zones
        ],
        
        'allow_long': output.allow_long,
        'allow_short': output.allow_short,
        'allow_sideway': output.allow_sideway,
    }
    
    if pred.sideway_context:
        result['sideway_context'] = {
            'is_at_boundary': pred.sideway_context['is_at_boundary'],
            'at_upper_band': pred.sideway_context['at_upper_band'],
            'at_lower_band': pred.sideway_context['at_lower_band'],
            'dist_to_upper': round(pred.sideway_context['dist_to_upper'], 4),
            'dist_to_lower': round(pred.sideway_context['dist_to_lower'], 4),
            'atr_expanding': pred.sideway_context['atr_expanding'],
            'volume_surge': round(pred.sideway_context['volume_surge'], 4),
            'band_squeezing': pred.sideway_context['band_squeezing']
        }
    
    if pred.mtf_alignment:
        result['mtf_alignment'] = {
            'aligned': pred.mtf_alignment['aligned'],
            'macro_state': pred.mtf_alignment['macro_state'].name,
            'alignment_score': round(pred.mtf_alignment['alignment_score'], 4)
        }
    
    if val:
        result['validation'] = {
            'is_valid': val.is_valid,
            'matches_regime': val.matches_regime,
            'trust_score': round(val.trust_score, 4),
            'regime_says': val.regime_says.name,
            'signal_says': val.signal_says.name,
            'warnings': val.warnings
        }
    
    if output.strategies:
        result['strategies'] = [
            {
                'strategy': s.strategy.value,
                'probability': round(s.probability, 4),
                'expected_winrate': round(s.expected_winrate, 4),
                'risk_multiplier': round(s.risk_multiplier, 2),
                'sl_multiplier': round(s.sl_multiplier, 2),
                'tp_multiplier': round(s.tp_multiplier, 2),
                'edge_score': round(s.edge_score, 4)
            }
            for s in output.strategies
        ]
    
    if output.recommended_strategy:
        rec = output.recommended_strategy
        result['recommended'] = {
            'strategy': rec.strategy.value,
            'probability': round(rec.probability, 4),
            'edge_score': round(rec.edge_score, 4)
        }
    
    if output.metadata:
        result['ema_fast'] = output.metadata.get('ema_fast')
        result['ema_slow'] = output.metadata.get('ema_slow')
        result['swing_high'] = output.metadata.get('swing_high')
        result['swing_low'] = output.metadata.get('swing_low')
        # State synchronization metadata for Execute
        result['calculated_price'] = output.metadata.get('calculated_price')
        result['calculated_atr'] = output.metadata.get('calculated_atr')
        result['calculation_timestamp'] = output.metadata.get('calculation_timestamp')
    
    return result
