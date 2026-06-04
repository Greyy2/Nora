import math
import time
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class MarketState(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    SIDEWAY = "SIDEWAY"


class TriggerType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class SLType(Enum):
    """Stop Loss types: Fixed, ATR-adaptive, Trailing, Breakeven, Structure-based"""
    FIXED_SL = "FIXED_SL"
    ATR_ADAPTIVE_SL = "ATR_ADAPTIVE"
    TRAILING_SL = "TRAILING_SL"
    BREAKEVEN_SL = "BREAKEVEN_SL"
    STRUCTURE_SL = "STRUCTURE_SL"


class TPType(Enum):
    """Take Profit types: Hard (Z-Score), Dynamic (%-based), Trailing (EMA-based)"""
    HARD_TP = "HARD_TP"
    DYNAMIC_OR = "DYNAMIC_OR"
    TRAILING_TP = "TRAILING_TP"


@dataclass
class OrderTranche:
    tranche_id: int
    volume_lots: float
    entry_price: float
    sl_price: float
    tp_price: float
    trigger_type: str = "MARKET"
    weight: float = 0.0
    tp_type: str = "HARD_TP"
    sl_type: str = "FIXED_SL"
    
    # Dynamic SL/TP state tracking (Execute module)
    original_sl: Optional[float] = None
    breakeven_triggered: bool = False
    trailing_reference: Optional[float] = None
    structure_reference: Optional[str] = None
    
    # Dynamic Exit tracking
    profit_in_r: float = 0.0  # Current profit in R units
    trailing_activated: bool = False  # Trailing SL unlocked?
    dynamic_tp_enabled: bool = False  # Dynamic TP active?
    entry_timestamp: Optional[float] = None  # Entry time for stall detection
    candles_since_entry: int = 0  # Candle count since entry


@dataclass
class ExecutionPayload:
    strategy_name: str
    total_risk_pct: float
    total_lots: float
    tranches: List[OrderTranche] = field(default_factory=list)
    market_state: Optional[str] = None
    is_strong: Optional[bool] = None
    edge_score: Optional[float] = None
    regime_metadata: Optional[Dict] = None  # For Execute state synchronization verification


class PositionSizer:
    
    def __init__(self, 
                 initial_risk_pct: float = 0.02,
                 equity_risk_pct: float = 0.5,
                 on_going_risk_pct: float = 0.0,
                 min_qty: float = 0.0001,
                 tradingview_percent_of_equity: bool = False,
                 tradingview_fixed_qty: float = 0.0,
                 enable_regime_mode: bool = False,
                 account_balance: float = 10000.0,
                 # True Position Sizing: Fee & Slippage
                 taker_fee_pct: float = 0.0006,  # 0.06% Binance taker
                 slippage_atr_multiplier: float = 0.05,  # 5% of ATR
                 # Dynamic Exit
                 enable_dynamic_exit: bool = True,
                 trailing_activation_r: float = 1.0,  # Activate after 1R profit
                 stall_candles: int = 5,  # Kill if stalled for N candles
                 stall_threshold_atr: float = 0.2):
        
        self.initial_risk = initial_risk_pct
        self.equity_risk = equity_risk_pct
        self.on_going_risk = on_going_risk_pct
        self.min_qty = min_qty
        self.tradingview_percent_of_equity = tradingview_percent_of_equity
        self.tradingview_fixed_qty = tradingview_fixed_qty
        self.enable_regime_mode = enable_regime_mode
        self.balance = account_balance
        
        # Fee & Slippage (True Position Sizing)
        self.taker_fee_pct = taker_fee_pct
        self.slippage_atr_multiplier = slippage_atr_multiplier
        
        # Dynamic Exit Config
        self.enable_dynamic_exit = enable_dynamic_exit
        self.trailing_activation_r = trailing_activation_r
        self.stall_candles = stall_candles
        self.stall_threshold_atr = stall_threshold_atr

        self.RISK_RANGE = {
            'LONG': {
                'strong': {'min': 0.010, 'max': 0.018},  
                'weak':   {'min': 0.009, 'max': 0.014}, 
            },
            'SHORT': {
                'strong': {'min': 0.004, 'max': 0.006}, 
                'weak':   {'min': 0.003, 'max': 0.005},  
            },
            'SIDEWAY': {
                'strong': {'min': 0.003, 'max': 0.004},  
                'weak':   {'min': 0.003, 'max': 0.004},  
            },
        }

        # Legacy (kept for backward compatibility)
        self.MAX_RISK = {
            'LONG': 0.015,
            'SHORT': 0.008,
            'SIDEWAY': 0.005
        }
        
        # ABSOLUTE LIMITS for Master Equation (0.3% - 1.8%)
        self.ABSOLUTE_MIN_RISK = 0.003  # 0.3% floor
        self.ABSOLUTE_MAX_RISK = 0.018  # 1.8% ceiling (FTMO daily-loss safe)
        
        # DEFAULT Trading Cost & Leverage Config
        self.DEFAULT_FEE_PCT = 0.0006  # 0.06% Binance Taker
        self.DEFAULT_SLIPPAGE = 0.0  # Will be calculated from ATR dynamically
        self.DEFAULT_MAX_LEVERAGE = 3.0  # Max 3x leverage (notional value limit)
        
        # DEFAULT Quantity Precision (Live Exchange)
        self.DEFAULT_QTY_PRECISION = {
            'coin': 5,  # BTC, ETH: 0.00001
            'forex': 2,  # Gold, EUR: 0.01
            'stock': 0   # Stocks: 1 (integer)
        }
        
        self.DEFAULT_SL_ATR = 2.0
        self.DEFAULT_TP_ATR_LONG = 4.0
        self.DEFAULT_TP_ATR_SHORT = 4.0
        
        self.BASIC_SL_ATR_LONG = 3.0   # LONG SL = 3x ATR dưới entry
        self.BASIC_SL_ATR_SHORT = 2.0  # SHORT SL = 2x ATR trên entry
        self.BASIC_TP_ATR_LONG = 6.0   # LONG TP = 6x ATR (R/R = 2:1)
        self.BASIC_TP_ATR_SHORT = 4.0  # SHORT TP = 4x ATR (R/R = 2:1)
    
    def _calculate_hard_tp(
        self,
        signal_state: str,
        entry_price: float,
        atr: float,
        ema_value: float,
        std_dev: Optional[float] = None,
        num_tranches: int = 3
    ) -> List[float]:
        """Hard TP: Z-Score based targets (Z=1.0/2.0/3.0 for 68%/95%/99.7% probability)"""
        if std_dev is None or std_dev <= 0:
            std_dev = atr * 0.8
        
        baseline = ema_value
        if abs(ema_value - entry_price) > (2 * atr):
            baseline = entry_price
        
        tp_prices = []
        
        if num_tranches <= 3:
            z_scores = [1.0, 2.0, 3.0][:num_tranches]
        else:
            z_scores = [1.0 + (i * (2.0 / (num_tranches - 1))) for i in range(num_tranches)]
        
        if signal_state == 'LONG':
            for i, z in enumerate(z_scores):
                tp = baseline + (z * std_dev)
                if i > 0 and tp <= tp_prices[-1]:
                    tp = tp_prices[-1] + (0.5 * atr)
                tp_prices.append(max(tp, entry_price + (0.5 * atr)))
        
        elif signal_state == 'SHORT':
            for i, z in enumerate(z_scores):
                tp = baseline - (z * std_dev)
                if i > 0 and tp >= tp_prices[-1]:
                    tp = tp_prices[-1] - (0.5 * atr)
                tp_prices.append(min(tp, entry_price - (0.5 * atr)))
        
        else:
            for i in range(num_tranches):
                tp = ema_value + ((i + 1) * atr * 0.5)
                tp_prices.append(tp)
        
        return tp_prices
    
    def _calculate_dynamic_or_tp(
        self,
        signal_state: str,
        entry_price: float,
        total_lots: float,
        target_profit_pct: float,
        weights: List[float]
    ) -> List[Dict]:
        """Dynamic OR TP: Close based on unrealized profit % (2% -> 3% -> 5% targets)"""
        dynamic_tps = []
        base_target = target_profit_pct
        
        for i, weight in enumerate(weights):
            tranche_lots = total_lots * weight
            target_multiplier = 1.0 + (i * 0.5)
            tranche_target_pct = base_target * target_multiplier
            
            dynamic_tps.append({
                'tranche_id': i + 1,
                'lots': tranche_lots,
                'target_profit_pct': tranche_target_pct,
                'entry_price': entry_price,
                'signal_state': signal_state
            })
        
        return dynamic_tps
    
    def _calculate_dynamic_sl(
        self,
        signal_state: str,
        entry_price: float,
        atr: float,
        current_atr: float,
        ema_value: float,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        sl_type: str = "FIXED_SL",
        base_sl_multiplier: float = 2.0
    ) -> Dict:
        result = {
            'sl_price': 0.0,
            'sl_type': sl_type,
            'original_sl': 0.0,
            'trailing_reference': None,
            'structure_reference': None,
            'atr_ratio': None
        }
        
        if sl_type == SLType.FIXED_SL.value or sl_type == "FIXED_SL":
            if signal_state == 'LONG':
                sl_price = entry_price - (base_sl_multiplier * atr)
            else:
                sl_price = entry_price + (base_sl_multiplier * atr)
            
            result['sl_price'] = sl_price
            result['original_sl'] = sl_price
            return result
        
        elif sl_type == SLType.ATR_ADAPTIVE_SL.value or sl_type == "ATR_ADAPTIVE":
            atr_ratio = current_atr / atr if atr > 0 else 1.0
            adaptive_multiplier = base_sl_multiplier * atr_ratio
            
            if signal_state == 'LONG':
                sl_price = entry_price - (adaptive_multiplier * atr)
                original_sl = entry_price - (base_sl_multiplier * atr)
            else:
                sl_price = entry_price + (adaptive_multiplier * atr)
                original_sl = entry_price + (base_sl_multiplier * atr)
            
            result['sl_price'] = sl_price
            result['original_sl'] = original_sl
            result['atr_ratio'] = atr_ratio
            return result
        
        elif sl_type == SLType.TRAILING_SL.value or sl_type == "TRAILING_SL":
            if signal_state == 'LONG':
                if swing_low is not None:
                    sl_price = swing_low - (0.5 * atr)
                    trailing_ref = swing_low
                else:
                    sl_price = entry_price - (base_sl_multiplier * atr)
                    trailing_ref = entry_price - (base_sl_multiplier * atr)
            else:
                if swing_high is not None:
                    sl_price = swing_high + (0.5 * atr)
                    trailing_ref = swing_high
                else:
                    sl_price = entry_price + (base_sl_multiplier * atr)
                    trailing_ref = entry_price + (base_sl_multiplier * atr)
            
            result['sl_price'] = sl_price
            result['original_sl'] = entry_price - (base_sl_multiplier * atr) if signal_state == 'LONG' else entry_price + (base_sl_multiplier * atr)
            result['trailing_reference'] = trailing_ref
            return result
        
        elif sl_type == SLType.STRUCTURE_SL.value or sl_type == "STRUCTURE_SL":
            if signal_state == 'LONG':
                sl_from_ema   = ema_value - (2.0 * atr)      # EMA minus meaningful buffer
                sl_from_entry = entry_price - (3.0 * atr)    # Absolute floor: 3 ATR below entry
                sl_price = min(sl_from_ema, sl_from_entry)   # Most protective initial SL
                structure_ref = "EMA_30"
            else:  # SHORT
                sl_from_ema   = ema_value + (2.0 * atr)
                sl_from_entry = entry_price + (3.0 * atr)
                sl_price = max(sl_from_ema, sl_from_entry)   # Most protective initial SL
                structure_ref = "EMA_30"
            
            result['sl_price'] = sl_price
            result['original_sl'] = entry_price - (base_sl_multiplier * atr) if signal_state == 'LONG' else entry_price + (base_sl_multiplier * atr)
            result['structure_reference'] = structure_ref
            return result
        
        else:
            if signal_state == 'LONG':
                sl_price = entry_price - (base_sl_multiplier * atr)
            else:
                sl_price = entry_price + (base_sl_multiplier * atr)
            
            result['sl_price'] = sl_price
            result['original_sl'] = sl_price
            return result
    
    def _calculate_trailing_tp(
        self,
        signal_state: str,
        entry_price: float,
        atr: float,
        ema_value: float,
        num_tranches: int = 3,
        trail_ema_buffer: float = 0.3
    ) -> List[Dict]:
        tp_configs = []
        
        if num_tranches == 3:
            weights = [0.4, 0.3, 0.3]
            fixed_tps = 2
        elif num_tranches == 4:
            weights = [0.3, 0.3, 0.2, 0.2]
            fixed_tps = 3
        else:
            weights = [1.0 / num_tranches] * num_tranches
            fixed_tps = num_tranches - 1
        
        for i, weight in enumerate(weights):
            if i < fixed_tps:
                if signal_state == 'LONG':
                    tp_price = entry_price + ((i + 1) * 2.0 * atr)
                elif signal_state == 'SHORT':
                    tp_price = entry_price - ((i + 1) * 2.0 * atr)
                else:
                    tp_price = ema_value
                
                tp_configs.append({
                    'tranche_id': i + 1,
                    'tp_price': tp_price,
                    'tp_type': TPType.HARD_TP.value,
                    'weight': weight,
                    'ema_reference': None,
                    'trail_buffer': None
                })
            
            else:
                if signal_state == 'LONG':
                    initial_tp = ema_value + (trail_ema_buffer * atr)
                elif signal_state == 'SHORT':
                    initial_tp = ema_value - (trail_ema_buffer * atr)
                else:
                    initial_tp = ema_value
                
                tp_configs.append({
                    'tranche_id': i + 1,
                    'tp_price': initial_tp,
                    'tp_type': TPType.TRAILING_TP.value,
                    'weight': weight,
                    'ema_reference': ema_value,
                    'trail_buffer': trail_ema_buffer * atr
                })
        
        return tp_configs
    
    def _calculate_dynamic_risk_allocation(
        self,
        signal_state: str,
        is_strong: bool,
        strength_score: float,
        strategy_risk_multiplier: float = 1.0,
        trust_score: float = 0.7
    ) -> float:

        # 1. Select oscillation range by State × Strength
        state_ranges = self.RISK_RANGE.get(signal_state, self.RISK_RANGE['SIDEWAY'])
        sub_range = state_ranges['strong'] if is_strong else state_ranges['weak']
        range_floor = sub_range['min']
        range_ceiling = sub_range['max']
        
        # 2. Base risk oscillates within the selected band driven by strength_score
        base_risk = range_floor + (strength_score * (range_ceiling - range_floor))
        
        # 3. Strategy multiplier — trade type scales risk within the band
        strategy_adjusted = base_risk * strategy_risk_multiplier
        
        # 4. Trust factor — conviction gate, steeper for high-risk (strong) setups
        # Strong: only high-trust signals justify allocating in the 1-2% band
        # Weak: more forgiving since the band is already conservative
        if is_strong:
            # Steep: trust=0.5→0.40x, trust=0.7→0.75x, trust=0.9→1.15x
            trust_factor = float(np.clip(trust_score / 0.75, 0.40, 1.15))
        else:
            # Gentle: trust=0.5→0.50x, trust=0.7→0.88x, trust=0.9→1.10x
            trust_factor = float(np.clip(trust_score / 0.80, 0.50, 1.10))
        
        final_risk = strategy_adjusted * trust_factor
        
        # 5. ABSOLUTE CLIPPING (0.3% floor, 2.0% ceiling — exchange risk limit)
        return float(np.clip(final_risk, self.ABSOLUTE_MIN_RISK, self.ABSOLUTE_MAX_RISK))
    
    def _calculate_true_position_size(
        self,
        equity: float,
        risk_pct: float,
        entry_price: float,
        sl_price: float,
        fee_pct: float,
        slippage: float,
        contract_size: float = 1.0
    ) -> float:
        """
        True Position Sizing: Include fees + slippage for exact R% control.
        
        Formula:
        Q = (E × R%) / (C × (|P_entry - P_sl| + Fee% × (P_entry + P_sl) + Slippage))
        
        Args:
            fee_pct: Taker fee percentage (e.g., 0.0006 for 0.06%)
            slippage: Expected slippage in price units (not ATR multiplier)
        """
        price_distance = abs(entry_price - sl_price)
        fee_cost = fee_pct * (entry_price + sl_price)
        total_cost_per_unit = price_distance + fee_cost + slippage
        
        if total_cost_per_unit <= 0:
            return 0.0
        
        position_size = (equity * risk_pct) / (contract_size * total_cost_per_unit)
        return position_size
    
    def check_trailing_sl_activation(
        self,
        entry_price: float,
        current_price: float,
        sl_price: float,
        signal_state: str
    ) -> bool:
        """Check if Trailing SL should activate (profit >= 1R)"""
        initial_risk = abs(entry_price - sl_price)
        
        if signal_state == 'LONG':
            current_profit = current_price - entry_price
        else:
            current_profit = entry_price - current_price
        
        profit_in_r = current_profit / initial_risk if initial_risk > 0 else 0
        return profit_in_r >= self.trailing_activation_r
    
    def check_stall_exit(
        self,
        entry_price: float,
        current_price: float,
        atr: float,
        candles_since_entry: int
    ) -> bool:
        """Check if position stalled (no momentum after N candles)"""
        if candles_since_entry < self.stall_candles:
            return False
        
        price_movement = abs(current_price - entry_price)
        stall_threshold = self.stall_threshold_atr * atr
        return price_movement <= stall_threshold
    
    def check_dynamic_tp_conditions(
        self,
        is_strong: bool,
        volume_surge: Optional[float] = None
    ) -> bool:
        """Check if Dynamic TP (trailing) should be enabled"""
        if not is_strong:
            return False
        if volume_surge is not None and volume_surge < 1.5:
            return False
        return True
    
    def process_signal(self, 
                      signal_state: str,
                      current_price: float,
                      atr: float,
                      upper_band: Optional[float] = None,
                      lower_band: Optional[float] = None,
                      regime_data: Optional[Dict] = None,
                      contract_size: float = 1.0,
                      market: str = 'coin') -> Optional[ExecutionPayload]:
        
        if signal_state.upper() not in ['LONG', 'SHORT', 'SIDEWAY']:
            return None
        
        signal_state = signal_state.upper()
        
        if not self.enable_regime_mode or regime_data is None:
            if signal_state == 'SIDEWAY':
                return None
            
            return self._static_mode_calculator(
                signal_state=signal_state,
                current_price=current_price,
                atr=atr,
                upper_band=upper_band,
                lower_band=lower_band,
                contract_size=contract_size,
                market=market
            )
        
        return self._dynamic_mode_calculator(
            signal_state=signal_state,
            regime_data=regime_data,
            current_price=current_price,
            atr=atr,
            upper_band=upper_band,
            lower_band=lower_band,
            contract_size=contract_size,
            market=market
        )
    
    def _static_mode_calculator(self,
                                signal_state: str,
                                current_price: float,
                                atr: float,
                                upper_band: Optional[float],
                                lower_band: Optional[float],
                                contract_size: float,
                                market: str) -> Optional[ExecutionPayload]:
        
        if signal_state == 'LONG':
            sl_price = current_price - (self.DEFAULT_SL_ATR * atr)
            tp_price = current_price + (self.DEFAULT_TP_ATR_LONG * atr)
            risk_pct = self.MAX_RISK['LONG']
            strategy_pattern = 'Scale_In_3_4_3'
            weights = [0.3, 0.4, 0.3]
        else:
            sl_price = current_price + (self.DEFAULT_SL_ATR * atr)
            tp_price = current_price - (self.DEFAULT_TP_ATR_SHORT * atr)
            risk_pct = self.MAX_RISK['SHORT']
            strategy_pattern = 'Scale_In_2_3_3_2'
            weights = [0.2, 0.3, 0.3, 0.2]
        
        risk_distance = abs(current_price - sl_price)
        if risk_distance <= 0:
            return None
        
        total_lots = (self.balance * risk_pct) / (risk_distance * contract_size)
        
        max_position_value = self.balance * self.equity_risk
        max_lots = max_position_value / current_price
        total_lots = min(total_lots, max_lots)
        
        # Quantity precision (default for static mode)
        precision = self.DEFAULT_QTY_PRECISION.get(market, 5)
        total_lots = self._round_down(total_lots, precision=precision)
        
        if total_lots <= 0:
            return None
        
        tranches = self._create_tranches_static(
            signal_state=signal_state,
            weights=weights,
            total_lots=total_lots,
            entry_price=current_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr
        )
        
        return ExecutionPayload(
            strategy_name=strategy_pattern,
            total_risk_pct=risk_pct,
            total_lots=total_lots,
            tranches=tranches,
            market_state=signal_state,
            is_strong=None,
            edge_score=None
        )
    
    def _create_tranches_static(self,
                               signal_state: str,
                               weights: List[float],
                               total_lots: float,
                               entry_price: float,
                               sl_price: float,
                               tp_price: float,
                               atr: float) -> List[OrderTranche]:
        
        tranches = []
        
        for i, weight in enumerate(weights):
            tranche_lots = total_lots * weight
            
            if signal_state == 'LONG':
                entry_adj = entry_price - (i * 0.5 * atr)
                trigger = "MARKET" if i == 0 else "LIMIT"
            else:
                entry_adj = entry_price + (i * 0.5 * atr)
                trigger = "MARKET" if i == 0 else "LIMIT"
            
            tranches.append(OrderTranche(
                tranche_id=i + 1,
                volume_lots=tranche_lots,
                entry_price=entry_adj,
                sl_price=sl_price,
                tp_price=tp_price,
                trigger_type=trigger,
                weight=weight
            ))
        
        return tranches
    
    def _dynamic_mode_calculator(self,
                                 signal_state: str,
                                 regime_data: Dict,
                                 current_price: float,
                                 atr: float,
                                 upper_band: Optional[float],
                                 lower_band: Optional[float],
                                 contract_size: float,
                                 market: str) -> Optional[ExecutionPayload]:
        
        current_state = regime_data.get('current_state', signal_state)
        is_strong = regime_data.get('is_strong', False)
        strategies = regime_data.get('strategies', [])
        
        best_strategy = strategies[0] if strategies else None
        
        if best_strategy:
            risk_multiplier = best_strategy.get('risk_multiplier', 1.0)
            sl_multiplier = best_strategy.get('sl_multiplier', self.DEFAULT_SL_ATR)
            tp_multiplier = best_strategy.get('tp_multiplier', 
                self.DEFAULT_TP_ATR_LONG if signal_state == 'LONG' else self.DEFAULT_TP_ATR_SHORT)
            edge_score = best_strategy.get('edge_score', 0.5)
        else:
            risk_multiplier = 0.6 if not is_strong else 1.0
            sl_multiplier = self.DEFAULT_SL_ATR
            tp_multiplier = self.DEFAULT_TP_ATR_LONG if signal_state == 'LONG' else self.DEFAULT_TP_ATR_SHORT
            edge_score = 0.5
        
        # DYNAMIC RISK ALLOCATION (Master Equation)
        # Combine: base_risk × strategy_multiplier × trust_factor
        strength_score = regime_data.get('strength_score', 0.5)
        
        # Extract trust_score from validation (if regime ran signal validation)
        trust_score = 0.7  # default: neutral trust
        validation_data = regime_data.get('validation')
        if validation_data and isinstance(validation_data, dict):
            trust_score = validation_data.get('trust_score', 0.7)
        
        # Master Equation: includes strategy_risk_multiplier from Regime
        actual_risk = self._calculate_dynamic_risk_allocation(
            signal_state=signal_state,
            is_strong=is_strong,
            strength_score=strength_score,
            strategy_risk_multiplier=risk_multiplier,  # FROM REGIME STRATEGIES
            trust_score=trust_score                     # WHIPSAW FILTER
        )
        
        # Extract real trading costs from Regime or use defaults
        fee_pct = regime_data.get('fee_pct', self.DEFAULT_FEE_PCT)
        slippage_atr_mult = regime_data.get('slippage_atr_multiplier', self.slippage_atr_multiplier)
        expected_slippage = slippage_atr_mult * atr  # Convert to price units
        max_leverage = regime_data.get('max_leverage', self.DEFAULT_MAX_LEVERAGE)
        
        if signal_state == 'LONG':
            sl_price = current_price - (sl_multiplier * atr)
            tp_price = current_price + (tp_multiplier * atr)
        elif signal_state == 'SHORT':
            sl_price = current_price + (sl_multiplier * atr)
            tp_price = current_price - (tp_multiplier * atr)
        else:
            if upper_band and lower_band:
                sl_price = upper_band if signal_state == 'LONG' else lower_band
                tp_price = lower_band if signal_state == 'LONG' else upper_band
            else:
                sl_price = current_price - (sl_multiplier * atr)
                tp_price = current_price + (tp_multiplier * atr)
        
        risk_distance = abs(current_price - sl_price)
        if risk_distance <= 0:
            return None
        
        # --- TRUE POSITION SIZING ---
        # Calculate Q absorbing all costs (Slippage + Fee)
        total_lots = self._calculate_true_position_size(
            equity=self.balance,
            risk_pct=actual_risk,
            entry_price=current_price,
            sl_price=sl_price,
            fee_pct=fee_pct,
            slippage=expected_slippage,
            contract_size=contract_size
        )
        
        if total_lots <= 0:
            return None
        
        # --- POSITION LIMIT (NOT RISK LIMIT) ---
        # Limit leverage/notional value to prevent over-borrowing
        max_notional_value = self.balance * max_leverage
        max_capacity_lots = max_notional_value / (current_price * contract_size)
        
        # Take min of (risk-based lots) and (account capacity)
        total_lots = min(total_lots, max_capacity_lots)
        
        # Quantity precision (from exchange API or default)
        step_size = regime_data.get('step_size', None)
        qty_precision = regime_data.get('qty_precision', None)
        
        if step_size is None and qty_precision is None:
            # Fallback to default precision
            qty_precision = self.DEFAULT_QTY_PRECISION.get(market, 5)
        
        total_lots = self._round_down(total_lots, precision=qty_precision, step_size=step_size)
        
        if total_lots <= 0:
            return None
        
        allocation_mode = regime_data.get('allocation_mode', 'FULL_MATRIX')
        stability_score = regime_data.get('stability_score', 1.0)
        
        if allocation_mode == 'PARTIAL_ONLY':
            if signal_state == 'LONG':
                partial_ratio = 0.4
            else:
                partial_ratio = 0.3
            
            total_lots = total_lots * partial_ratio
            total_lots = self._round_down(total_lots, precision=qty_precision, step_size=step_size)
            
            if total_lots <= 0:
                return None
            
            if signal_state == 'LONG':
                strategy_name = 'PARTIAL_ONLY_ScaleOut_4_LONG'
                weights = [0.25, 0.25, 0.25, 0.25]
            else:
                strategy_name = 'PARTIAL_ONLY_ScaleOut_3_SHORT'
                weights = [0.33, 0.34, 0.33]
        
        else:
            strategy_name, weights = self._select_strategy_pattern(current_state, is_strong, best_strategy)
        
        metadata = {
            'ema_fast': regime_data.get('ema_fast', current_price),
            'ema_slow': regime_data.get('ema_slow', current_price),
            'swing_high': regime_data.get('swing_high', current_price + atr),
            'swing_low': regime_data.get('swing_low', current_price - atr),
            'allocation_mode': allocation_mode,
            'stability_score': stability_score
        }
        
        tranches = self._create_tranches_dynamic(
            signal_state=signal_state,
            strategy_name=strategy_name,
            weights=weights,
            total_lots=total_lots,
            entry_price=current_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr=atr,
            is_strong=is_strong,
            metadata=metadata
        )
        
        regime_metadata = {
            'calculated_price': regime_data.get('calculated_price'),
            'calculated_atr': regime_data.get('calculated_atr'),
            'calculation_timestamp': regime_data.get('calculation_timestamp')
        } if regime_data else None
        
        return ExecutionPayload(
            strategy_name=strategy_name,
            total_risk_pct=actual_risk,
            total_lots=total_lots,
            tranches=tranches,
            market_state=current_state,
            is_strong=is_strong,
            edge_score=edge_score,
            regime_metadata=regime_metadata
        )
    
    def _select_strategy_pattern(self, current_state: str, is_strong: bool, best_strategy: Optional[Dict] = None) -> tuple:
        """Select strategy pattern based on Regime data or fallback to state + strength"""
        if best_strategy:
            strategy_type = best_strategy.get('strategy', '')
            
            if strategy_type == 'scalp':
                if current_state == 'LONG':
                    return ('ScaleOut_4_Tranches_LONG', [0.25, 0.25, 0.25, 0.25])
                elif current_state == 'SHORT':
                    return ('ScaleOut_3_Tranches_SHORT', [0.33, 0.34, 0.33])
            
            elif strategy_type == 'trend_follow':
                if current_state == 'LONG':
                    return ('Pyramid_4_3_3', [0.4, 0.3, 0.3]) if is_strong else ('Accumulate_3_3_4', [0.3, 0.3, 0.4])
                elif current_state == 'SHORT':
                    return ('Strict_ScaleIn_2_3_3_2', [0.2, 0.3, 0.3, 0.2]) if is_strong else ('Accumulate_SHORT_3_3_4', [0.3, 0.3, 0.4])
            
            elif strategy_type == 'breakout':
                if current_state == 'LONG':
                    return ('Breakout_LONG_Aggressive', [0.5, 0.3, 0.2])
                elif current_state == 'SHORT':
                    return ('Breakdown_SHORT_Aggressive', [0.5, 0.3, 0.2])
                else:
                    return ('Breakout_SIDEWAY_Wait', [1.0])
        
        if current_state == 'LONG':
            return ('Pyramid_4_3_3', [0.4, 0.3, 0.3]) if is_strong else ('Accumulate_3_3_4', [0.3, 0.3, 0.4])
        elif current_state == 'SHORT':
            return ('Strict_ScaleIn_2_3_3_2', [0.2, 0.3, 0.3, 0.2]) if is_strong else ('ScaleOut_TP_Split', [0.33, 0.34, 0.33])
        else:
            return ('Sniper_Boundary', [1.0])
    
    def _create_tranches_dynamic(self,
                                signal_state: str,
                                strategy_name: str,
                                weights: List[float],
                                total_lots: float,
                                entry_price: float,
                                sl_price: float,
                                tp_price: float,
                                atr: float,
                                is_strong: bool,
                                metadata: Optional[Dict] = None) -> List[OrderTranche]:
        """Create tranches with dynamic SL/TP. Scale-Out: single MARKET entry + progressive TPs."""
        meta = metadata or {}
        ema_fast = meta.get('ema_fast', entry_price)
        ema_slow = meta.get('ema_slow', entry_price)
        swing_high = meta.get('swing_high', entry_price + atr)
        swing_low = meta.get('swing_low', entry_price - atr)
        
        tranches = []
        
        is_scaleout = 'ScaleOut' in strategy_name
        is_partial_only = 'PARTIAL_ONLY' in strategy_name
        allocation_mode = meta.get('allocation_mode', 'FULL_MATRIX')
        
        if is_partial_only or allocation_mode == 'PARTIAL_ONLY':
            sl_type_to_use = SLType.TRAILING_SL.value
            use_trailing_tp = True
        elif is_strong:
            sl_type_to_use = SLType.STRUCTURE_SL.value
            use_trailing_tp = True
        else:
            sl_type_to_use = SLType.ATR_ADAPTIVE_SL.value
            use_trailing_tp = False
        
        dynamic_sl_result = self._calculate_dynamic_sl(
            signal_state=signal_state,
            entry_price=entry_price,
            atr=atr,
            current_atr=atr,
            ema_value=ema_fast,
            swing_high=swing_high,
            swing_low=swing_low,
            sl_type=sl_type_to_use,
            base_sl_multiplier=2.0
        )
        
        dynamic_sl_price = dynamic_sl_result['sl_price']
        original_sl = dynamic_sl_result['original_sl']
        trailing_reference = dynamic_sl_result['trailing_reference']
        structure_reference = dynamic_sl_result['structure_reference']
        
        trailing_tp_configs = None
        if use_trailing_tp:
            num_tranches = len(weights)
            trailing_tp_configs = self._calculate_trailing_tp(
                signal_state=signal_state,
                entry_price=entry_price,
                atr=atr,
                ema_value=ema_fast,
                num_tranches=num_tranches,
                trail_ema_buffer=0.3
            )
        
        # Check dynamic TP conditions from Regime
        volume_surge = meta.get('volume_surge', None)
        enable_dynamic_tp = self.enable_dynamic_exit and self.check_dynamic_tp_conditions(
            is_strong=is_strong,
            volume_surge=volume_surge
        )
        
        if is_scaleout:
            num_tranches = len(weights)
            
            if is_partial_only or allocation_mode == 'PARTIAL_ONLY':
                tp_type = TPType.HARD_TP.value
                std_dev = atr * 0.8
                hard_tps = self._calculate_hard_tp(
                    signal_state=signal_state,
                    entry_price=entry_price,
                    atr=atr,
                    ema_value=ema_fast,
                    std_dev=std_dev,
                    num_tranches=num_tranches
                )
            else:
                tp_type = TPType.HARD_TP.value
                hard_tps = None
            
            for i, weight in enumerate(weights):
                tranche_lots = total_lots * weight
                entry_adj = entry_price
                trigger = "MARKET"
                sl_adj = dynamic_sl_price
                
                if trailing_tp_configs and use_trailing_tp:
                    tp_config = trailing_tp_configs[i]
                    tp_adj = tp_config['tp_price']
                    tp_type = tp_config['tp_type']
                elif hard_tps:
                    tp_adj = hard_tps[i]
                    tp_type = TPType.HARD_TP.value
                else:
                    tp_type = TPType.HARD_TP.value
                    if signal_state == 'LONG':
                        tp_spread = 3.0 * atr
                        tp_increment = tp_spread / num_tranches
                        tp_adj = entry_price + ((i + 1) * tp_increment)
                    elif signal_state == 'SHORT':
                        tp_spread = 2.5 * atr
                        tp_increment = tp_spread / num_tranches
                        tp_adj = entry_price - ((i + 1) * tp_increment)
                    else:
                        tp_adj = tp_price
                
                tranches.append(OrderTranche(
                    tranche_id=i + 1,
                    volume_lots=tranche_lots,
                    entry_price=entry_adj,
                    sl_price=sl_adj,
                    tp_price=tp_adj,
                    trigger_type=trigger,
                    weight=weight,
                    tp_type=tp_type,
                    sl_type=sl_type_to_use,
                    original_sl=original_sl,
                    breakeven_triggered=False,
                    trailing_reference=trailing_reference,
                    structure_reference=structure_reference,
                    # Dynamic Exit tracking
                    profit_in_r=0.0,
                    trailing_activated=False,
                    dynamic_tp_enabled=enable_dynamic_tp,
                    entry_timestamp=time.time(),
                    candles_since_entry=0
                ))
            
            return tranches
        
        for i, weight in enumerate(weights):
            tranche_lots = total_lots * weight
            
            if signal_state == 'LONG':
                if is_strong:
                    if i == 0:
                        entry_adj = entry_price
                        trigger = "MARKET"
                        sl_adj = entry_price - (1.5 * atr)
                    elif i == 1:
                        entry_adj = swing_high + (0.1 * atr)
                        trigger = "STOP"
                        sl_adj = entry_price
                    else:
                        entry_adj = tranches[i-1].entry_price + (0.5 * atr)
                        trigger = "STOP"
                        sl_adj = tranches[i-1].entry_price
                    tp_adj = tp_price
                else:
                    if i == 0:
                        entry_adj = entry_price
                        trigger = "MARKET"
                    elif i == 1:
                        entry_adj = entry_price - (1.0 * atr)   # Buy on first dip
                        trigger = "LIMIT"
                    else:
                        entry_adj = entry_price - (2.0 * atr)   # Buy on deeper dip
                        trigger = "LIMIT"
                    sl_adj = entry_price - (3.5 * atr)          # Wide unified SL
                    tp_adj = swing_high
                    
            elif signal_state == 'SHORT':
                if is_strong:
                    if i == 0:
                        entry_adj = entry_price
                        trigger = "MARKET"
                        sl_adj = entry_price + (1.5 * atr)
                    elif i == 1:
                        entry_adj = swing_low - (0.2 * atr)
                        trigger = "STOP"
                        sl_adj = entry_price
                    else:
                        entry_adj = tranches[i-1].entry_price - (0.8 * atr)
                        trigger = "STOP"
                        sl_adj = tranches[i-1].entry_price
                    tp_adj = tp_price
                else:
                    entry_adj = entry_price
                    trigger = "MARKET"
                    sl_adj = swing_high + (0.5 * atr)
                    tp_adj = entry_price - ((i + 1) * 1.0 * atr)
                    
            else:
                entry_adj = entry_price
                sl_adj = sl_price
                tp_adj = ema_fast
                trigger = "LIMIT"
            
            # Apply dynamic TP if trailing enabled
            if trailing_tp_configs and use_trailing_tp and i < len(trailing_tp_configs):
                tp_config = trailing_tp_configs[i]
                tp_adj = tp_config['tp_price']
                tp_type_final = tp_config['tp_type']
            else:
                tp_type_final = TPType.HARD_TP.value
            
            if i == 0:
                sl_final = dynamic_sl_price
            else:
                if sl_type_to_use == SLType.STRUCTURE_SL.value and trigger == 'STOP':
                    sl_final = sl_adj
                elif sl_type_to_use == SLType.STRUCTURE_SL.value:
                    sl_final = dynamic_sl_price
                else:
                    sl_final = sl_adj
            
            tranches.append(OrderTranche(
                tranche_id=i + 1,
                volume_lots=tranche_lots,
                entry_price=entry_adj,
                sl_price=sl_final,
                tp_price=tp_adj,
                trigger_type=trigger,
                weight=weight,
                tp_type=tp_type_final,
                sl_type=sl_type_to_use,  # All tranches use same SL type for merged trailing
                original_sl=original_sl,
                breakeven_triggered=False,
                trailing_reference=trailing_reference if i == 0 else None,
                structure_reference=structure_reference if i == 0 else None,
                # Dynamic Exit tracking
                profit_in_r=0.0,
                trailing_activated=False,
                dynamic_tp_enabled=enable_dynamic_tp,
                entry_timestamp=time.time(),
                candles_since_entry=0
            ))
        
        return tranches
    
    def calculate(self,
                  equity: float,
                  price: float,
                  atr: float,
                  upper: float,
                  lower: float,
                  contract_size: float = 1.0,
                  multiple_atr: Optional[float] = None,
                  use_delta: bool = True,
                  market: str = 'coin',
                  signal_state: str = 'LONG',
                  leverage: Optional[float] = None) -> dict:

        if self.tradingview_fixed_qty > 0:
            sl_price, tp_price = self._calc_basic_sl_tp(price, atr, signal_state)
            return {
                'quantity': self.tradingview_fixed_qty,
                'position_value_usdt': self.tradingview_fixed_qty * price,
                'sl_price': sl_price,
                'tp_price': tp_price
            }

        if self.tradingview_percent_of_equity:
            if price <= 0:
                return {'quantity': 0.0, 'position_value_usdt': 0.0, 'sl_price': 0.0, 'tp_price': 0.0}
            
            contracts_equity = (self.equity_risk * equity) / price
            
            delta = upper - lower
            risk_dist = delta if delta > 0 else (atr * 2 if atr > 0 else price * 0.01)
            
            contracts_risk = float('inf')
            if risk_dist > 0:
                contracts_risk = (self.initial_risk * equity) / risk_dist
            
            contracts = min(contracts_equity, contracts_risk)
            
            precision = self.DEFAULT_QTY_PRECISION.get(market, 5)
            contracts = self._round_down(contracts, precision=precision)
            contracts = max(0.0, contracts)
            
            sl_price, tp_price = self._calc_basic_sl_tp(price, atr, signal_state)
            return {
                'quantity': contracts,
                'position_value_usdt': contracts * price,
                'sl_price': sl_price,
                'tp_price': tp_price
            }
        
        if equity <= 0 or price <= 0:
            return {'quantity': 0.0, 'position_value_usdt': 0.0, 'sl_price': 0.0, 'tp_price': 0.0}
        
        precision = self.DEFAULT_QTY_PRECISION.get(market, 5)
        # Risk-based sizing: contracts = (risk × equity) / risk_distance
        # PnL = contracts × price_delta → no need to divide by price for any market
        # (Broker computes PnL as (exit-entry)*qty for both coin and forex)
        price_for_delta_atr = 1.0
        atr_value = multiple_atr if multiple_atr is not None else atr
        
        contracts_delta = None
        delta = upper - lower
        if delta > 0:
            try:
                contracts_delta = self._round_down(
                    (self.initial_risk * equity) / (delta * contract_size * price_for_delta_atr), precision=precision)
            except:
                contracts_delta = None
        
        contracts_atr = None
        if atr_value > 0:
            try:
                contracts_atr = self._round_down(
                    (self.initial_risk * equity) / (atr_value * contract_size * price_for_delta_atr), precision=precision)
            except:
                contracts_atr = None
        
        # --- CAPACITY CONSTRAINT ---
        # Leverage (Forex) vs Equity Risk (Coin)
        effective_leverage = leverage if leverage is not None else self.equity_risk
        
        contracts_equity = self._round_down(
            (effective_leverage * equity) / (price * contract_size), precision=precision)
        
        if contracts_atr is not None and contracts_delta is not None:
            contracts = min(contracts_delta, contracts_atr, contracts_equity) if use_delta else min(contracts_atr, contracts_equity)
        else:
            contracts = contracts_equity
        
        contracts = max(0.0, contracts)
        sl_price, tp_price = self._calc_basic_sl_tp(price, atr, signal_state)
        return {
            'quantity': contracts,
            'position_value_usdt': contracts * price,
            'sl_price': sl_price,
            'tp_price': tp_price
        }
    
    def _calc_basic_sl_tp(self, price: float, atr: float, signal_state: str) -> tuple:
        if atr <= 0 or price <= 0:
            return 0.0, 0.0
        
        state = signal_state.upper() if signal_state else 'LONG'
        
        if state == 'LONG':
            sl = price - (self.BASIC_SL_ATR_LONG * atr)
            tp = price + (self.BASIC_TP_ATR_LONG * atr)
        elif state == 'SHORT':
            sl = price + (self.BASIC_SL_ATR_SHORT * atr)
            tp = price - (self.BASIC_TP_ATR_SHORT * atr)
        else:  # SIDEWAY
            sl = price - (self.BASIC_SL_ATR_SHORT * atr)
            tp = price + (self.BASIC_SL_ATR_SHORT * atr)
        
        return sl, tp
    
    def _round_down(self, value: float, precision: Optional[int] = None, step_size: Optional[float] = None) -> float:
        if value <= 0:
            return 0.0
        
        # Step-based rounding (from exchange API)
        if step_size is not None and step_size > 0:
            result = math.floor(value / step_size) * step_size
            # Fix floating-point precision errors
            decimal_places = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
            return round(result, decimal_places)
        
        # Precision-based rounding (decimal places)
        if precision is not None:
            multiplier = 10 ** precision
            return math.floor(value * multiplier) / multiplier
        
        # Fallback: no rounding (should not happen)
        return value
    
    def check_on_going_risk(self,
                           contracts: float,
                           current_price: float,
                           entry_price: float,
                           on_going_equity: float,
                           contract_size: float = 1.0,
                           market: str = 'coin') -> Optional[float]:

        if self.on_going_risk <= 0 or contracts <= 0 or on_going_equity <= 0 or current_price == entry_price:
            return None
        
        precision = self.DEFAULT_QTY_PRECISION.get(market, 5)
        price_diff = abs(entry_price - current_price)
        unrealized_profit = contracts * price_diff * contract_size
        risk = (unrealized_profit / on_going_equity) * 100
        on_going_limit = self.on_going_risk * 100

        if risk <= on_going_limit:
            return None  # Within limit, no reduction needed

        # Reduce position to bring risk back to limit
        target_profit = (on_going_limit / 100) * on_going_equity
        if price_diff > 0:
            new_contracts = target_profit / (price_diff * contract_size)
            new_contracts = self._round_down(new_contracts, precision=precision)
            new_contracts = max(0.0, new_contracts)
            return new_contracts

        return None

    # ── Chandelier / Structure SL Ratchet ────────────────────────────────────

    @staticmethod
    def chandelier_sl(
        signal_state: str,
        highest_high: float,
        lowest_low: float,
        atr: float,
        current_sl: float,
        k: float = 3.5,
    ) -> float:
        """
        Chandelier Exit — ratchet-only SL update for open positions.

        Formula:
            LONG  →  SL = max(current_sl,  highest_high - k × ATR)   [only moves UP]
            SHORT →  SL = min(current_sl,  lowest_low  + k × ATR)    [only moves DOWN]

        The caller must track highest_high (rolling max of bar highs since entry)
        and lowest_low (rolling min of bar lows since entry).

        Args:
            k: ATR multiplier. 3.5–4.0 for trend-following; 2.5–3.0 for tighter protection.
        """
        if signal_state == 'LONG':
            candidate = highest_high - k * atr
            return max(current_sl, candidate)          # ratchet UP only
        else:  # SHORT
            candidate = lowest_low + k * atr
            if current_sl <= 0:
                return candidate
            return min(current_sl, candidate)           # ratchet DOWN only

    @staticmethod
    def update_structure_sl_ratchet(
        signal_state: str,
        ema_value: float,
        atr: float,
        current_sl: float,
        ema_buffer: float = 2.0,
    ) -> float:
        """
        Structure SL ratchet — trail SL relative to EMA (for live Execute module).

        Fixes the direction-inversion bug: re-applying min(ema-2ATR, entry-3ATR) on
        subsequent bars would clamp the LONG SL at entry-3ATR forever.
        Correct form: SL may only move toward profit (ratchet).

            LONG  →  SL = max(current_sl,  ema - ema_buffer × ATR)
            SHORT →  SL = min(current_sl,  ema + ema_buffer × ATR)
        """
        if signal_state == 'LONG':
            candidate = ema_value - ema_buffer * atr
            return max(current_sl, candidate)
        else:  # SHORT
            candidate = ema_value + ema_buffer * atr
            if current_sl <= 0:
                return candidate
            return min(current_sl, candidate)
