import pandas as pd 
import numpy as np 
from typing import Dict, List, Optional, Any, Tuple 
from datetime import datetime 
import sys
from pathlib import Path

# import các modules
from models.signal import Signal, SignalType
from models.trade import Trade, TradeDirection
from core.load_data import DataLoader
from core.indicator import calculate_ema, calculate_atr, calculate_keltner_bands
from core.position_sizer import PositionSizer
from strategies.kema import KEMAStrategy
from metrics.calculator import MetricsCalculator  

class Broker:
    # Class variable to track if mode has been logged (once per worker)
    _mode_logged = False
    
    def __init__(self, initial_capital: float, commission_pct: float, slippage_pct: float = 0.0, strategy_params: Optional[Dict] = None, data_dir: Optional[str] = None):
        """
        Init Broker
        
        Args:
            initial_capital: Vốn ban đầu
            commission_pct: % phí giao dịch (0.1 = 0.1%)
            slippage_pct: % slippage (0.4 = 0.4%)
            strategy_params: Optional strategy params (for caching)
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100.0  # Convert % to decimal
        self.base_slippage_pct = slippage_pct / 100.0  # Base slippage for 1h TF
        self.slippage_pct = self.base_slippage_pct  # Current slippage (will adjust per TF)
        self.strategy_params = strategy_params or {}
        self.contract_size = strategy_params.get('contract_size', 1.0) if strategy_params else 1.0
        
        # Components
        self.data_dir = data_dir # Save for usage in _execute_backtest
        self.data_loader = DataLoader(data_dir=data_dir)
        # Pass strategy params to KEMAStrategy (including debug, filters, etc.)
        strategy_config = self.strategy_params.get('strategy', {}).get('bse', {})
        self.strategy = KEMAStrategy(**strategy_config)

        # State tracking
        self.current_equity = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_position: Optional[Dict] = None
        
        # Backtest period tracking (for accurate CAGR)
        self.backtest_start_date = None
        self.backtest_end_date = None
        
        # On-going risk management attributes (initialized with defaults)
        self.is_on_going = False
        self.on_going_limit = 0.0
        self.pending_on_going_order = None
        # Stop Out management (Forex realism) - Only active if leverage/stop_out provided
        self.margin_stop_out_pct = strategy_params.get('margin_stop_out_pct', 0.50)
        self.leverage = strategy_params.get('leverage', None) # None = Use Core Equity Risk (Safe)
        self.on_going_checks = 0  # Count checks
        self.on_going_triggers = 0  # Count triggers

    def _extract_timeframe_hours(self, timeframe: str) -> float:
        """
        Extract hours from timeframe string.
        Examples: '1h'->1, '4h'->4, '1d'->24, '1w'->168
        """
        import re
        match = re.match(r'(\d+)([mhdw])', timeframe.lower())
        if not match:
            return 1.0  # Default to 1h
        
        num, unit = int(match.group(1)), match.group(2)
        multipliers = {'m': 1/60, 'h': 1, 'd': 24, 'w': 168}
        return num * multipliers.get(unit, 1)
    
    def _reset_state(self):
        """
        Reset tất cả state trước mỗi backtest
        
        Tại sao cần: Tránh state leak khi chạy nhiều backtest liên tiếp

        Mutil-threading Mỗi worker = instance riêng
        """
        self.current_equity = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.current_position = None 

    def run_backtest(self, 
                     asset: str, 
                     timeframe: str, 
                     strategy_params: Dict[str, Any],
                     start_date: Optional[str] = None, 
                     end_date: Optional[str] = None, 
                     df: Optional[pd.DataFrame] = None,
                     precomputed_ema: Optional[np.ndarray] = None,
                     precomputed_atr: Optional[np.ndarray] = None,
                     fast_mode: bool = False,
                     timestamps: Optional[List] = None,
                     ts_map: Optional[Dict] = None,
                     signals_override: Optional[List[Signal]] = None) -> Dict[str, Any]:

        # Reset state trước mỗi backtest
        self._reset_state()
        
        # Cập nhật contract_size từ params (nếu có)
        self.contract_size = strategy_params.get('contract_size', self.contract_size)

        # Formula: slippage = base_slippage / sqrt(tf_hours)
        # 1h: 0.5% / 1.0 = 0.50%, 4h: 0.5% / 2.0 = 0.25%, 24h: 0.5% / 4.9 = 0.10%
        tf_hours = self._extract_timeframe_hours(timeframe)
        import math
        self.slippage_pct = self.base_slippage_pct / math.sqrt(tf_hours)
        
        # Extract params từ strategy_params dict
        length_ema = strategy_params['length_ema']
        length_atr = strategy_params['length_atr']
        long_vol_factor = float(strategy_params.get('long_vol_factor', 2.0))
        short_vol_factor = float(strategy_params.get('short_vol_factor', 2.0))
        
        # Strategy params loaded silently (no debug output)
        # use_delta extracted below (line 121) - single source of truth
        multiple = strategy_params.get('multiple', 1)
        
        # Extract strategy sub-params
        strategy_config = strategy_params.get('strategy', {})
        bse_config = strategy_config.get('bse', {})
        ps_config = strategy_config.get('ps', {})
        
        is_on_going = bse_config.get('is_on_going', False)
        side = bse_config.get('side', 'long')
        debug = bse_config.get('debug', False)  # Debug flag from BSE config
        
        ir = ps_config.get('ir', 0.02)
        er = ps_config.get('er', 0.5)
        or_val = ps_config.get('or', 0.0)  # OR = 0 means disabled

        # Re-initialize strategy with current bse_config (including side)
        from strategies.kema import KEMAStrategy
        self.strategy = KEMAStrategy(**bse_config)

        # tmp/position_sizer.py
        use_strategies_v2 = ps_config.get('use_v2', False)
        aggression = ps_config.get('aggression_factor', 0.5)
        max_daily_loss = ps_config.get('max_daily_loss', 0)

        # Store tmp/or lại
        self.aggression_factor = aggression
        self.max_daily_loss = max_daily_loss
        
        # Store on-going parameters for partial exit logic
        self.is_on_going = is_on_going
        self.on_going_limit = or_val * 100  # OR = on-going risk limit (convert to %)
        self.debug_on_going = debug  # Enable debug from config
        self.pending_on_going_order = None  # Pending partial exit order

        # logic chọn PositionSizer 
        # Always use V1 (Classic PositionSizer) for standard test
        SizerClass = PositionSizer
        
        if self.debug_on_going and or_val > 0:
            print(f"\n🔧 On-going Risk Config:")
            print(f"   is_on_going: {self.is_on_going}")
            print(f"   OR: {or_val * 100:.1f}%")
            print(f"   Mode: V1 (Classic)")
        
        # FIX BUG 2: Store use_delta from strategy params (not from signal.metadata)
        self.use_delta = strategy_params.get('use_delta', True)
        
        # Init PositionSizer with OR integrated from the start (use selected class)
        self.position_sizer = SizerClass(
            initial_risk_pct=ir,
            equity_risk_pct=er,
            on_going_risk_pct=or_val,  # OR is core parameter now
            tradingview_percent_of_equity=ps_config.get('tradingview_percent_of_equity', False),
            tradingview_fixed_qty=ps_config.get('fixed_qty', 0.0)
        )

        # 1. Load data
        if df is None:
            # Load full data to ensure we have enough history for indicators
            df = self.data_loader.load(asset, timeframe)
            
        if df.empty:
            raise ValueError(f"No data found for {asset}")
        
        # Two modes:
        # 1. OPTIMIZER MODE: timestamps + ts_map pre-cached and passed in
        #    → Use directly, NO recomputation (FAST!)
        # 2. SINGLE CORE MODE: No cache provided
        #    → Compute once here if fast_mode=True
        #    → Or compute in _execute_backtest if fast_mode=False
        
        if timestamps is not None and ts_map is not None:
            # OPTIMIZER MODE: Reuse pre-cached (no computation!)
            timestamps_full = timestamps
            ts_map_full = ts_map
        elif fast_mode:
            # SINGLE CORE with fast_mode: Compute once now
            timestamps_full = df.index.to_numpy()
            ts_map_full = dict(zip(timestamps_full, range(len(timestamps_full))))
        else:
            # SINGLE CORE standard: Will compute in _execute_backtest
            timestamps_full, ts_map_full = None, None

        # 2. Indicator Strategy: Check if pre-computed, else calculate
        # 
        # Two modes:
        # 1. OPTIMIZER MODE: EMA/ATR pre-computed from matrix and passed in
        #    → Use directly, NO recalculation (HUGE speedup!)
        # 2. SINGLE CORE MODE: No pre-computed indicators
        #    → Calculate from scratch using params
        
        if precomputed_ema is not None and precomputed_atr is not None:
             # OPTIMIZER MODE: Use pre-computed from matrix (no calculation!)
             ema = precomputed_ema
             atr = precomputed_atr
        else:
             # SINGLE CORE MODE: Calculate indicators from scratch
             from core.indicator import calculate_keltner_bands
             
             close = df['close'].to_numpy().copy()
             high = df['high'].to_numpy().copy()
             low = df['low'].to_numpy().copy()

             ema, atr, _, _ = calculate_keltner_bands(
                 close=close,
                 high=high,
                 low=low,
                 ema_length=length_ema,
                 atr_length=length_atr,
                 multiplier=long_vol_factor # Dummy multiplier, we recalc bands below
             )
             
        
        # Compute Keltner Bands (on full data)
        # Vectorized calculation on full arrays
        upper_band = ema + (long_vol_factor * atr)
        lower_band = ema - (short_vol_factor * atr)

        # ═══════════════════════════════════════════════════════════════════
        # Store ATR for position sizing
        self.atr_values = atr

        # 3. Filter Data for Signal Generation & Backtest
        # We need to slice giving enough buffer for strategy to look back 1 bar for cross logic.
        
        start_dt = pd.to_datetime(start_date, utc=True) if start_date else df.index[0]
        end_dt = pd.to_datetime(end_date, utc=True) if end_date else df.index[-1]
        
        # Store backtest period for accurate CAGR calculation
        self.backtest_start_date = start_dt
        self.backtest_end_date = end_dt
        
        # Find integer index of start_dt
        start_loc = df.index.searchsorted(start_dt)
        # Always ensure at least 1 buffer bar (for cross logic), but don't go negative
        start_loc = max(0, start_loc - 1)
            
        end_loc = df.index.searchsorted(end_dt, side='right')
        
        # Slice DataFrame and Arrays
        df_slice = df.iloc[start_loc:end_loc]
        
        if df_slice.empty:
             raise ValueError(f"No data found in range {start_date} - {end_date}")

        ema_slice = ema[start_loc:end_loc]
        atr_slice = atr[start_loc:end_loc]
        upper_slice = upper_band[start_loc:end_loc]
        lower_slice = lower_band[start_loc:end_loc]
            
        # 4. Generate Signals on SLICED data (or accept externally provided signals)
        if signals_override is not None:
            signals_list = list(signals_override)
        else:
            signals_list = self.strategy.generate_signals(df_slice, upper_slice, lower_slice)
        
        # 5. Execute backtest on the SLICED period
        # CRITICAL: Reuse ts_map_full directly - no dict recreation!
        if timestamps_full is not None and ts_map_full is not None:
            # Fast mode: Slice timestamps, pass full ts_map with offset
            timestamps_slice = timestamps_full[start_loc:end_loc]
            # Pass full map + offset to avoid dict recreation
            self._execute_backtest(
                df_slice, 
                signals_list,
                atr_slice,
                upper_slice, 
                lower_slice,
                timestamps_cached=timestamps_slice,
                ts_map_cached=ts_map_full,  # Full map!
                offset=start_loc  # Tell _execute where slice starts
            )
        else:
            # Standard mode
            timestamps_slice = df_slice.index.to_numpy()
            ts_map_slice = dict(zip(timestamps_slice, range(len(timestamps_slice))))
            self._execute_backtest(
                df_slice, 
                signals_list,
                atr_slice,
                upper_slice, 
                lower_slice,
                timestamps_cached=timestamps_slice,
                ts_map_cached=ts_map_slice,
                offset=0
            )
        
        # Calculate metrics
        metrics = self._calculate_metrics(df_slice['close'].values if 'close' in df_slice.columns else None, fast_mode=fast_mode)
        
        curve_to_return = self.equity_curve
        if not fast_mode and isinstance(self.equity_curve, np.ndarray):
            curve_to_return = [{'timestamp': timestamps_slice[k], 'equity': float(self.equity_curve[k])} for k in range(len(timestamps_slice))]

        return {
            'trades': self.trades,
            'equity_curve': curve_to_return,
            'metrics': metrics,
            'df': df_slice
        }

    def _execute_backtest(self, df: pd.DataFrame, signals: List[Signal], atr: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, timestamps_cached: Optional[np.ndarray] = None, ts_map_cached: Optional[Dict] = None, offset: int = 0):
        """
        ULTRA-FAST JUMP LOGIC (Unified Version)
        Complexity: O(S + N_vectorized) where S is number of signals.
        Accuracy: 100% TradingView-matched (Vectorized bar-by-bar equity)
        
        Args:
            offset: When using full ts_map with sliced data, this is the slice start index
        """
        close_prices = df['close'].to_numpy()
        open_prices = df['open'].to_numpy()
        high_prices = df['high'].to_numpy()
        low_prices = df['low'].to_numpy()
        
        # 🔧 FOREX/XAU LOGIC: Check category and bid/ask columns
        # - Only enable specialized Bid/Ask execution if category is 'forex' AND columns exist
        is_forex = self.data_dir == 'forex'
        has_bid_ask = 'open_bid' in df.columns and 'open_ask' in df.columns
        
        if is_forex and has_bid_ask:
            open_bid_prices = df['open_bid'].to_numpy()
            open_ask_prices = df['open_ask'].to_numpy()
        else:
            # Fallback: Use MID price for both bid and ask (standard crypto/stock mode)
            open_bid_prices = open_prices
            open_ask_prices = open_prices
        
        if timestamps_cached is not None and ts_map_cached is not None:
            timestamps = timestamps_cached
            ts_map = ts_map_cached
        else:
            # Fallback: create once (should not happen in optimizer mode)
            timestamps = df.index.to_numpy()
            ts_map = dict(zip(timestamps, range(len(timestamps))))
            offset = 0
        
        n = len(timestamps)
        self.equity_arr = np.full(n, self.initial_capital, dtype=np.float64)
        
        # Signal lookups with offset support
        processed_signals = []
        for s in signals:
            idx = ts_map.get(s.timestamp)
            if idx is None and hasattr(s.timestamp, 'to_datetime64'):
                 idx = ts_map.get(s.timestamp.to_datetime64())
            if idx is not None:
                # Apply offset to convert global index to local index
                local_idx = idx - offset
                if 0 <= local_idx < n - 1:
                    processed_signals.append((local_idx, s))
        
        processed_signals.sort(key=lambda x: x[0])
        
        self.current_equity = self.initial_capital
        self.trades, self.cumulative_pnl, self.current_position = [], 0.0, None
        last_filled_idx = 0
        
        # Create signal index map for fast lookup
        signal_at_idx = {idx: sig for idx, sig in processed_signals}
        
        # CRITICAL: Loop through ALL candles when OR enabled or has position
        # Jump mode only when no position AND OR disabled
        if self.is_on_going or len(processed_signals) > 0:
            # Full candle iteration mode for OR checking
            for i in range(n - 1):  # Stop at n-1 to have next bar for execution
                signal = signal_at_idx.get(i)
                exec_idx = i + 1
                exec_ts = timestamps[exec_idx]
                
                # 🔧 EXECUTION LOGIC: Determine execution price based on signal direction
                # - If 'forex' mode: Use Bid/Ask
                #   * LONG: Use ASK price (buy at higher price)
                #   * SHORT: Use BID price (sell at lower price)
                # - If 'OKX' mode: Use MID price (bid=ask=mid)
                # - OR exit: Use appropriate price based on position direction
                if signal is not None and signal.is_entry():
                    # Có signal mới - dùng price phù hợp với direction
                    exec_price = open_ask_prices[exec_idx] if signal.is_long() else open_bid_prices[exec_idx]
                elif self.current_position is not None:
                    # Có position - dùng price phù hợp với exit direction (ngược lại entry)
                    # LONG position exit = bán = dùng BID
                    # SHORT position exit = mua = dùng ASK
                    is_long_pos = self.current_position['direction'] == TradeDirection.LONG
                    exec_price = open_bid_prices[exec_idx] if is_long_pos else open_ask_prices[exec_idx]
                else:
                    # Không có signal và không có position - dùng MID
                    exec_price = open_prices[exec_idx]
                
                # Vectorized equity fill for current bar
                if self.current_position:
                    pos = self.current_position
                    direction = 1 if pos['direction'] == TradeDirection.LONG else -1
                    self.equity_arr[i] = self.current_equity + (close_prices[i] - pos['entry_price']) * pos['quantity'] * direction
                else:
                    self.equity_arr[i] = self.current_equity
                
                # Check on-going risk at EVERY candle OPEN when position exists
                if self.is_on_going and self.current_position and i > 0:
                    # OR check dùng MID price (để consistent với indicator logic)
                    or_triggered = self._check_on_going_risk(open_prices[i], i)
                    if or_triggered and self.pending_on_going_order:
                        # Execute OR partial exit với price phù hợp
                        is_long_pos = self.current_position['direction'] == TradeDirection.LONG
                        or_exec_price = open_bid_prices[i] if is_long_pos else open_ask_prices[i]
                        self._process_on_going_order(timestamps[i], or_exec_price, i)
                
                # SL/TP check on current bar's high/low (before signal processing)
                if self.current_position is not None:
                    pos = self.current_position
                    sl_p = pos.get('sl_price', 0.0)
                    tp_p = pos.get('tp_price', 0.0)
                    if sl_p > 0 and tp_p > 0:
                        is_long_pos = pos['direction'] == TradeDirection.LONG
                        bar_high = high_prices[i]
                        bar_low = low_prices[i]
                        sl_hit = (is_long_pos and bar_low <= sl_p) or (not is_long_pos and bar_high >= sl_p)
                        tp_hit = (is_long_pos and bar_high >= tp_p) or (not is_long_pos and bar_low <= tp_p)
                        if sl_hit or tp_hit:
                            # SL has priority over TP (worst case first)
                            if sl_hit:
                                exit_price = sl_p
                                reason = "SL Hit"
                            else:
                                exit_price = tp_p
                                reason = "TP Hit"
                            slice_h = high_prices[pos['entry_index']:i+1]
                            slice_l = low_prices[pos['entry_index']:i+1]
                            max_h = float(slice_h.max()) if slice_h.size > 0 else exit_price
                            min_l = float(slice_l.min()) if slice_l.size > 0 else exit_price
                            self._close_position(timestamps[i], exit_price, reason, i, max_h, min_l)
                            # Skip signal processing for this bar since position was closed by SL/TP
                            continue

                # 🔧 MARGIN CALL CHECK (Forex Realism)
                if self.current_position is not None:
                    # Check if account equity fell below Stop Out level (Margin Call)
                    # Margin Level = (Equity / Used Margin) * 100
                    # Used Margin = Notional Value / Leverage
                    pos = self.current_position
                    notional_value = pos['quantity'] * open_prices[i]
                    effective_lev = self.leverage if self.leverage is not None else 1.0
                    used_margin = notional_value / effective_lev
                    
                    # Estimate current equity using bar low (worst case within bar)
                    is_long_pos = pos['direction'] == TradeDirection.LONG
                    worst_price = low_prices[i] if is_long_pos else high_prices[i]
                    unrealized_pnl = (worst_price - pos['entry_price']) * pos['quantity'] * (1 if is_long_pos else -1)
                    est_equity = self.current_equity + unrealized_pnl
                    
                    if used_margin > 0:
                        margin_level = est_equity / used_margin
                        if margin_level < self.margin_stop_out_pct:
                            # BOOM! Margin Call / Stop Out
                            slice_h = high_prices[pos['entry_index']:i+1]
                            slice_l = low_prices[pos['entry_index']:i+1]
                            max_h = float(slice_h.max()) if slice_h.size > 0 else worst_price
                            min_l = float(slice_l.min()) if slice_l.size > 0 else worst_price
                            self._close_position(timestamps[i], worst_price, f"STOP OUT (Margin: {margin_level:.1%})", i, max_h, min_l)
                            continue

                # Process signal if exists at this bar
                if signal is not None:
                    if self.current_position is None:
                        if signal.is_entry():
                            signal_ts = timestamps[i]
                            self._open_position(signal, signal_ts, exec_ts, exec_price, atr[i], upper_band[i], lower_band[i], exec_idx)
                    else:
                        if self._should_exit(signal):
                            pos = self.current_position
                            slice_h, slice_l = high_prices[pos['entry_index']:exec_idx], low_prices[pos['entry_index']:exec_idx]
                            max_h = max(float(slice_h.max()), exec_price) if slice_h.size > 0 else exec_price
                            min_l = min(float(slice_l.min()), exec_price) if slice_l.size > 0 else exec_price
                            
                            self._close_position(exec_ts, exec_price, f"Rev {signal.signal_type.value}", exec_idx, max_h, min_l)
                            if signal.is_entry():
                                signal_ts = timestamps[i]
                                # Recalculate exec_price cho signal mới
                                exec_price = open_ask_prices[exec_idx] if signal.is_long() else open_bid_prices[exec_idx]
                                self._open_position(signal, signal_ts, exec_ts, exec_price, atr[i], upper_band[i], lower_band[i], exec_idx)
                
                last_filled_idx = exec_idx
        else:
            # Original jump logic (when no OR and no position)
            for i, signal in processed_signals:
                exec_idx = i + 1
                exec_ts = timestamps[exec_idx]
                
                # 🔧 XAU LOGIC: Determine execution price (same logic as full iteration mode)
                if signal.is_entry():
                    exec_price = open_ask_prices[exec_idx] if signal.is_long() else open_bid_prices[exec_idx]
                elif self.current_position is not None:
                    is_long_pos = self.current_position['direction'] == TradeDirection.LONG
                    exec_price = open_bid_prices[exec_idx] if is_long_pos else open_ask_prices[exec_idx]
                else:
                    exec_price = open_prices[exec_idx]
                
                # Vectorized equity fill for gap
                if self.current_position:
                    pos = self.current_position
                    direction = 1 if pos['direction'] == TradeDirection.LONG else -1
                    self.equity_arr[last_filled_idx:exec_idx] = self.current_equity + (close_prices[last_filled_idx:exec_idx] - pos['entry_price']) * pos['quantity'] * direction
                else:
                    self.equity_arr[last_filled_idx:exec_idx] = self.current_equity
                
                # Logic branch
                if self.current_position is None:
                    if signal.is_entry():
                        signal_ts = timestamps[i]
                        self._open_position(signal, signal_ts, exec_ts, exec_price, atr[i], upper_band[i], lower_band[i], exec_idx)
                else:
                    if self._should_exit(signal):
                        pos = self.current_position
                        slice_h, slice_l = high_prices[pos['entry_index']:exec_idx], low_prices[pos['entry_index']:exec_idx]
                        max_h = max(float(slice_h.max()), exec_price) if slice_h.size > 0 else exec_price
                        min_l = min(float(slice_l.min()), exec_price) if slice_l.size > 0 else exec_price
                        
                        self._close_position(exec_ts, exec_price, f"Rev {signal.signal_type.value}", exec_idx, max_h, min_l)
                        if signal.is_entry():
                            signal_ts = timestamps[i]
                            # Recalculate exec_price cho signal mới
                            exec_price = open_ask_prices[exec_idx] if signal.is_long() else open_bid_prices[exec_idx]
                            self._open_position(signal, signal_ts, exec_ts, exec_price, atr[i], upper_band[i], lower_band[i], exec_idx)
                last_filled_idx = exec_idx
            
        # Final cleanup
        if self.current_position:
            pos = self.current_position
            direction = 1 if pos['direction'] == TradeDirection.LONG else -1
            self.equity_arr[last_filled_idx:] = self.current_equity + (close_prices[last_filled_idx:] - pos['entry_price']) * pos['quantity'] * direction
            
            last_idx = n-1
            # 🔧 XAU LOGIC: Final exit dùng price phù hợp
            is_long_pos = pos['direction'] == TradeDirection.LONG
            final_exit_price = open_bid_prices[last_idx] if is_long_pos else open_ask_prices[last_idx]
            
            slice_h, slice_l = high_prices[pos['entry_index']:], low_prices[pos['entry_index']:] 
            max_h = float(slice_h.max()) if slice_h.size > 0 else pos['entry_price']
            min_l = float(slice_l.min()) if slice_l.size > 0 else pos['entry_price']
            self._close_position(timestamps[last_idx], final_exit_price, "End", last_idx, max_h, min_l)
        else:
            self.equity_arr[last_filled_idx:] = self.current_equity
            
        self.equity_curve = self.equity_arr

    def _calculate_metrics(self, close_prices: np.ndarray = None, fast_mode: bool = False) -> Dict[str, Any]:
        """
        Tính metrics - Delegate to MetricsCalculator
        Passes internal equity_arr for ZERO-conversion speed
        """
        # If fast_mode = True, self.equity_curve IS our numpy array
        calculator = MetricsCalculator(
            self.trades, 
            equity_curve=self.equity_curve, 
            initial_capital=self.initial_capital,
            benchmark_prices=close_prices,
            start_date=self.backtest_start_date,
            end_date=self.backtest_end_date,
            contract_size=self.contract_size
        )
        
        return calculator.calculate_fast() if fast_mode else calculator.calculate_all()
    
    # REMOVED: _find_next_signal_index (replaced by bisect in execute_backtest)
    
    def _should_exit(self, signal: Signal) -> bool:
        """
        Check có nên exit position không

        Logic (matching TradingView):
        - LONG position: Exit on Exit_LONG signal OR SHORT entry signal
        - SHORT position: Exit on Exit_SHORT signal OR LONG entry signal
        
        TradingView processes exit signals before entry signals
        """
        if self.current_position is None:
            return False 
        
        pos_direction = self.current_position['direction']

        # Check explicit exit signals first
        if pos_direction == TradeDirection.LONG and signal.signal_type == SignalType.Exit_LONG:
            return True
        if pos_direction == TradeDirection.SHORT and signal.signal_type == SignalType.Exit_SHORT:
            return True
        
        # Then check opposite entry signals (for flipping)
        # LONG position, nhận SHORT entry signal -> Exit + flip
        if pos_direction == TradeDirection.LONG and signal.is_short():
            return True 
        
        # SHORT position, nhận LONG entry signal -> Exit + flip
        if pos_direction == TradeDirection.SHORT and signal.is_long():
            return True 
        
        return False

    def _open_position(self, signal: Signal, signal_time: datetime, entry_time: datetime, price: float, atr: float, upper: float, lower: float, index: int):
        """
        Mở position with index tracking for bars duration

        Flow:
        1. Tính quantity (3 constraints: Delta, ATR, Equity)
        2. Tính commission
        3. Apply slippage (worsen entry price)
        4. Tính equity
        5. Lưu position state
        """
        
        is_long = signal.is_long()
        slipped_price = price * (1 + self.slippage_pct) if is_long else price * (1 - self.slippage_pct)
        
        # 1. Tính quantity với 3 constraints (matching Vinh logic)
        # FIX BUG 2: Use use_delta from self (stored in run_backtest)
        _market = 'forex' if self.data_dir == 'forex' else 'coin'
        _signal_state = 'LONG' if is_long else 'SHORT'
        position_result = self.position_sizer.calculate(
            equity=self.current_equity,
            price=slipped_price,
            atr=atr,
            upper=upper,
            lower=lower,
            use_delta=self.use_delta,  # From strategy params (stored in __init__)
            market=_market,
            signal_state=_signal_state,
            leverage=self.leverage
        )
        
        quantity = position_result['quantity']
        position_value_usdt = position_result['position_value_usdt']

        if quantity <= 0:
            return # Không đủ vốn hoặc risk quá cao
        
        # 2. Tính commission (on slipped price)
        position_value = quantity * slipped_price
        commission = position_value * self.commission_pct

        # 3. Deduct commission từ equity
        self.current_equity -= commission

        # 4. Store position (with slipped entry price and signal time)
        self.current_position = {
            'direction': TradeDirection.LONG if signal.is_long() else TradeDirection.SHORT,
            'signal_time': signal_time,  # When signal was detected
            'entry_time': entry_time,    # When trade was entered (next bar open)
            'entry_price': slipped_price,  # Store slipped price
            'entry_price_no_slip': price,  # Store original price for slippage tracking
            'entry_index': index,  # Store index for bars calc
            'quantity': quantity,
            'original_quantity': quantity,  # Store original quantity for partial exits
            'commission_paid': commission,
            'on_going_equity': self.current_equity,  # Store equity at entry for on-going risk check (E₀)
            'on_going_profit': 0.0,  # Track profit from partial exits
            'atr': atr,  # 🆕 Store ATR for adaptive OR logic
            'sl_price': position_result['sl_price'],  # SL from position_sizer
            'tp_price': position_result['tp_price'],  # TP from position_sizer
        }
        
        if self.debug_on_going and self.is_on_going:
            direction = "LONG" if signal.is_long() else "SHORT"
            print(f"\n📊 ENTRY {direction}:")
            print(f"   Equity at entry (E₀): ${self.current_equity:,.2f}")
            print(f"   Entry price: ${slipped_price:.2f}")
            print(f"   Contracts: {quantity:.6f}")
    
    def _close_position(self, timestamp: datetime, price: float, reason: str, index: int, max_h: float = 0.0, min_l: float = 0.0):
        """
        Đóng position và tạo Trade
        Supports MFE/MAE inputs
        
        Flow: 
        1. Apply slippage to exit price
        2. Tính PnL (gross)
        3. Tính commission exit
        4. Tính net PnL
        5. Update equity
        6. Tạo Trade object
        7. Clear position
        """
        if self.current_position is None:
            return 
        
        pos = self.current_position
        
        # 1. Apply slippage to exit price (worsen price for trader)
        # Long exit (sell): receive less (slippage decreases price)
        # Short exit (buy): pay more (slippage increases price)
        is_long = (pos['direction'] == TradeDirection.LONG)
        slipped_price = price * (1 - self.slippage_pct) if is_long else price * (1 + self.slippage_pct)

        # 2. Tính gross PnL (using slipped exit price)
        if pos['direction'] == TradeDirection.LONG:
            # LONG: profit = (exit - entry) * quantity
            gross_pnl = (slipped_price - pos['entry_price']) * pos['quantity']
            # MFE/MAE: Apply slippage to max_h/min_l (exit worst case)
            # CRITICAL: max_h/min_l are raw OHLC, need slippage for realistic MFE/MAE
            slipped_max_h = max_h * (1 - self.slippage_pct)  # Sell at high (receive less)
            slipped_min_l = min_l * (1 - self.slippage_pct)  # Sell at low (receive less)
            mfe = (slipped_max_h - pos['entry_price']) * pos['quantity']
            mae = (slipped_min_l - pos['entry_price']) * pos['quantity']
        else:
            # SHORT: profit = (entry - exit) * quantity
            gross_pnl = (pos['entry_price'] - slipped_price) * pos['quantity']
            # MFE/MAE: Apply slippage to max_h/min_l (exit worst case)
            # CRITICAL: max_h/min_l are raw OHLC, need slippage for realistic MFE/MAE
            slipped_max_h = max_h * (1 + self.slippage_pct)  # Buy at high (pay more)
            slipped_min_l = min_l * (1 + self.slippage_pct)  # Buy at low (pay more)
            mfe = (pos['entry_price'] - slipped_min_l) * pos['quantity']
            mae = (pos['entry_price'] - slipped_max_h) * pos['quantity']

        # 3. Tính exit commission (on slipped price)
        exit_value = pos['quantity'] * slipped_price
        exit_commission = exit_value * self.commission_pct


        # 3. Net PnL = gross - entry commission - exit commission
        total_commission = pos['commission_paid'] + exit_commission
        net_pnl = gross_pnl - total_commission 
        
        # Note: on_going_profit already recorded as separate OR trade, don't add again

        # 4. Update equity
        self.current_equity += gross_pnl - exit_commission 
        
        # Update Cumulative PnL
        self.cumulative_pnl += net_pnl
        
        # Calculate percentages (use current quantity, not original)
        notional = pos['entry_price'] * pos['quantity']
        pnl_pct = (net_pnl / notional) * 100 if notional > 0 else 0
        mfe_pct = (mfe / notional) * 100 if notional > 0 else 0
        mae_pct = (mae / notional) * 100 if notional > 0 else 0

        # 5. Tạo Trade object (with slipped exit price)
        bars_count = index - pos['entry_index'] + 1
        # print(f"DEBUG: Trade Closed - Entry: {pos['entry_index']}, Exit: {index}, Bars: {bars_count}, PnL: {net_pnl}")
        
        # Calculate slippage cost
        entry_slippage = abs(pos['entry_price'] - pos.get('entry_price_no_slip', pos['entry_price'])) * pos['quantity']
        exit_slippage = abs(slipped_price - price) * pos['quantity']
        total_slippage = entry_slippage + exit_slippage
        
        trade = Trade(
            signal_time=pos.get('signal_time'),  # Signal detection time
            entry_time=pos['entry_time'],         # Trade entry time (next bar)
            exit_time=timestamp,
            direction=pos['direction'],
            entry_price=pos['entry_price'],  # Already slipped from _open_position
            exit_price=slipped_price,  # Slipped exit price
            quantity=pos['quantity'],  # Current quantity (after partial exits)
            pnl=net_pnl,  # Only this final exit PnL (OR trades recorded separately)
            pnl_pct=pnl_pct,
            commission=total_commission,
            mfe=mfe, 
            mfe_pct=mfe_pct,
            mae=mae, 
            mae_pct=mae_pct,
            cumulative_pnl=self.cumulative_pnl,
            exit_reason=reason,
            bars=bars_count,
            slippage_pct=self.slippage_pct,
            entry_price_no_slip=pos.get('entry_price_no_slip'),
            exit_price_no_slip=price,
            slippage_cost=total_slippage
        )

        self.trades.append(trade)

        # 6. Clear position
        self.current_position = None

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Tính unrealized PnL của position hiện tại

        Logic:
        - LONG: (current - entry) * quantity
        - SHORT: (entry - current) * quantity
        - Không trừ exit commission (vì chưa close)
        """
        if self.current_position is None:
            return 0.0
        pos = self.current_position

        # Tính gross PnL
        if pos['direction'] == TradeDirection.LONG:
            unrealized_pnl = (current_price - pos['entry_price']) * pos['quantity']
        else: 
            unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']

        return unrealized_pnl

    def _check_on_going_risk(self, open_price: float, index: int) -> bool:
        """
        Check if position profit exceeds on-going risk limit at bar OPEN.
        If yes, create partial exit order to execute IMMEDIATELY at current bar open.
        
        Returns:
            True if OR triggered, False otherwise
        """
        self.on_going_checks += 1
        
        if not self.current_position:
            return False
        
        # CRITICAL: If limit is 0, disable on-going
        if self.on_going_limit <= 0:
            if self.debug_on_going and self.on_going_checks == 1:
                print(f"\n⚠️  On-going DISABLED (limit={self.on_going_limit}%)")
            return False
        
        pos = self.current_position
        is_long = (pos['direction'] == TradeDirection.LONG)
        
        # Check if profitable using OPEN price
        unrealized_pnl = 0
        if is_long:
            unrealized_pnl = (open_price - pos['entry_price']) * pos['quantity']
        else:
            unrealized_pnl = (pos['entry_price'] - open_price) * pos['quantity']
        
        is_profitable = unrealized_pnl > 0
        
        if not is_profitable:
            if self.debug_on_going and self.on_going_checks <= 5:
                direction = "LONG" if is_long else "SHORT"
                pnl_pct = (unrealized_pnl / (pos['entry_price'] * pos['quantity'])) * 100
                print(f"\n📍 Check #{self.on_going_checks}: Not profitable yet")
                print(f"   Position: {direction} {pos['quantity']:.6f} @ ${pos['entry_price']:.2f}")
                print(f"   Current:  ${open_price:.2f}")
                print(f"   Unrealized PnL: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%)")
            return False  # Not profitable, no risk to manage
        
        # Position is profitable - check if risk exceeds limit
        if self.debug_on_going:
            direction = "LONG" if is_long else "SHORT"
            pnl_pct = (unrealized_pnl / (pos['entry_price'] * pos['quantity'])) * 100
            risk_pct = (unrealized_pnl / pos['on_going_equity']) * 100
            print(f"\n✅ Check #{self.on_going_checks}: PROFITABLE!")
            print(f"   Position: {direction} {pos['quantity']:.6f} @ ${pos['entry_price']:.2f}")
            print(f"   Current:  ${open_price:.2f}")
            print(f"   Unrealized PnL: ${unrealized_pnl:.2f} ({pnl_pct:.2f}%)")
            print(f"   Risk: {risk_pct:.2f}% (limit: {self.on_going_limit:.0f}%)")
        
        # ═══════════════════════════════════════════════════════════════════
        # OR STRATEGY - Classic V1 (Always uses PositionSizer)
        # ═══════════════════════════════════════════════════════════════════
        # V1: Classic OR (fixed limit)
        _market = 'forex' if self.data_dir == 'forex' else 'coin'
        new_contracts = self.position_sizer.check_on_going_risk(
            contracts=pos['quantity'],
            current_price=open_price,
            entry_price=pos['entry_price'],
            on_going_equity=pos['on_going_equity'],
            market=_market
        )
        # ═══════════════════════════════════════════════════════════════════
        
        # Create pending order if reduction needed
        if new_contracts is not None and new_contracts < pos['quantity']:
            self.on_going_triggers += 1
            
            # Calculate risk before and after for metadata
            risk_before = (unrealized_pnl / pos['on_going_equity']) * 100
            
            # Calculate risk after (estimated with new contracts at OPEN price)
            if is_long:
                new_unrealized = (open_price - pos['entry_price']) * new_contracts
            else:
                new_unrealized = (pos['entry_price'] - open_price) * new_contracts
            
            # Estimate equity after partial exit
            contracts_to_reduce = pos['quantity'] - new_contracts
            if is_long:
                profit_from_reduce = (open_price - pos['entry_price']) * contracts_to_reduce
            else:
                profit_from_reduce = (pos['entry_price'] - open_price) * contracts_to_reduce
            
            estimated_equity_after = pos['on_going_equity'] + profit_from_reduce
            risk_after = (new_unrealized / estimated_equity_after) * 100 if estimated_equity_after > 0 else 0
            
            self.pending_on_going_order = {
                'new_contracts': new_contracts,
                'check_index': index,
                'risk_before': risk_before,
                'risk_after': risk_after,
                'contracts_before': pos['quantity'],
                'contracts_after': new_contracts,
                'unrealized_pnl': unrealized_pnl,  
                'on_going_equity': pos['on_going_equity']  
            }
            
            if self.debug_on_going:
                direction = "LONG" if is_long else "SHORT"
                pnl_pct = ((open_price/pos['entry_price']-1)*100 if is_long else (1-open_price/pos['entry_price'])*100)
                print(f"\n🚨 TRIGGER #{self.on_going_triggers}: On-going risk exceeded!")
                print(f"   Position:     {direction} @ ${pos['entry_price']:.2f}")
                print(f"   Current:      ${open_price:.2f}")
                print(f"   PnL:          {pnl_pct:.2f}%")
                print(f"   Contracts:    {pos['quantity']:.6f} → {new_contracts:.6f}")
                print(f"   Reduction:    {(pos['quantity']-new_contracts):.6f} ({(1-new_contracts/pos['quantity'])*100:.1f}%)")
                print(f"   OR limit:     {self.on_going_limit:.1f}%")
            return True  
        
        return False  
    
    def _process_on_going_order(self, exec_time, exec_price: float, exec_idx: int):
        """
        Process pending on-going order (partial exit).
        Execute at next bar open after risk check.
        
        Flow:
        1. Calculate contracts to reduce
        2. Apply slippage to exit price
        3. Calculate profit from partial exit
        4. Update equity and position quantity
        5. Clear pending order
        """
        if not self.pending_on_going_order or not self.current_position:
            return
        
        pos = self.current_position
        new_contracts = self.pending_on_going_order['new_contracts']
        contracts_to_reduce = pos['quantity'] - new_contracts
        
        if contracts_to_reduce <= 0:
            self.pending_on_going_order = None
            return
        
        if self.debug_on_going:
            print(f"\n💰 EXECUTING Partial Exit:")
            print(f"   Contracts to reduce: {contracts_to_reduce:.6f}")
            print(f"   Exit price: ${exec_price:.2f}")
        
        # Apply slippage to exit price (same as regular exit)
        is_long = (pos['direction'] == TradeDirection.LONG)
        slipped_exit = exec_price * (1 - self.slippage_pct) if is_long else exec_price * (1 + self.slippage_pct)
        
        # Calculate profit from partial exit
        if is_long:
            gross_profit = (slipped_exit - pos['entry_price']) * contracts_to_reduce
        else:
            gross_profit = (pos['entry_price'] - slipped_exit) * contracts_to_reduce
        
        # Commission on partial exit
        exit_value = contracts_to_reduce * slipped_exit
        commission = exit_value * self.commission_pct
        net_profit = gross_profit - commission
        
        # Update equity
        self.current_equity += gross_profit - commission
        
        # Update Cumulative PnL
        self.cumulative_pnl += net_profit
        
        # Create separate trade for partial exit (OR trade)
        bars_count = exec_idx - pos['entry_index']
        pnl_pct = (net_profit / (pos['entry_price'] * contracts_to_reduce)) * 100 if contracts_to_reduce > 0 else 0
        
        # Get OR metadata from pending order
        or_metadata = self.pending_on_going_order
        
        or_trade = Trade(
            signal_time=pos.get('signal_time'),
            entry_time=pos['entry_time'],
            exit_time=exec_time,
            direction=pos['direction'],
            entry_price=pos['entry_price'],
            exit_price=slipped_exit,
            quantity=contracts_to_reduce,
            pnl=net_profit,
            pnl_pct=pnl_pct,
            commission=commission,
            mfe=0.0,  # MFE/MAE not tracked for partial exits
            mfe_pct=0.0,
            mae=0.0,
            mae_pct=0.0,
            cumulative_pnl=self.cumulative_pnl,
            exit_reason=f"On-going Risk (PnL: {((slipped_exit/pos['entry_price']-1)*100 if is_long else (1-slipped_exit/pos['entry_price'])*100):.2f}%, limit: {self.on_going_limit:.1f}%)",
            bars=bars_count,
            equity_after_exit=self.current_equity,
            or_risk_before=or_metadata.get('risk_before'),
            or_risk_after=or_metadata.get('risk_after'),
            or_contracts_before=or_metadata.get('contracts_before'),
            or_contracts_after=or_metadata.get('contracts_after'),
            or_unrealized_pnl=or_metadata.get('unrealized_pnl'),  # Add unrealized PnL
            or_on_going_equity=or_metadata.get('on_going_equity')  # Add equity at trigger
        )
        
        self.trades.append(or_trade)
        
        # Track on-going profit (still needed for final trade)
        pos['on_going_profit'] += net_profit
        # NOTE: Do NOT update on_going_equity here!
  
        # Reduce position quantity (use floor rounding, not nearest rounding)
        import math
        decimal = 5  # Crypto default - could be 2 for forex if needed
        multiplier = 10 ** decimal
        pos['quantity'] = math.floor(new_contracts * multiplier) / multiplier
        
        if self.debug_on_going:
            print(f"   Gross profit:        ${gross_profit:,.2f}")
            print(f"   Commission:          ${commission:,.2f}")
            print(f"   Net profit:          ${net_profit:,.2f}")
            print(f"   New equity:          ${self.current_equity:,.2f}")
            print(f"   Remaining contracts: {pos['quantity']:.6f}")
        
        # Clear pending order
        self.pending_on_going_order = None
        
        # If quantity too small, force close
        if pos['quantity'] <= 0.0001:
            self.current_position = None
    
    def _check_margin_call(self, current_price: float) -> Tuple[bool, float]:
        """
        Check if current position triggers a Margin Call (Stop Out).
        
        Formula:
        Used Margin = (Quantity * Price) / Leverage
        Margin Level = Equity / Used Margin
        Trigger if Margin Level < margin_stop_out_pct (e.g., 0.5 for 50%)
        """
        if not self.current_position:
            return False, 1.0
            
        pos = self.current_position
        notional_value = pos['quantity'] * current_price
        effective_lev = self.leverage if self.leverage is not None else 1.0
        used_margin = notional_value / effective_lev
        
        if used_margin <= 0:
            return False, 1.0
            
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        total_equity = self.current_equity + unrealized_pnl
        margin_level = total_equity / used_margin
        
        return (margin_level < self.margin_stop_out_pct), margin_level
    
    # NOTE: _calculate_metrics defined above (line ~538) — single definition only