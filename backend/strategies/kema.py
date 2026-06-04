"""
KEMA Strategy - Keltner Channel + EMA crossover logic

Signal Generation Logic (matching TradingView Pine Script):
- LONG: Close CROSSES OVER Upper Band (strict inequality: close > upper)
  → Previous bar: close < upper
  → Current bar: close > upper (MUST cross, not just touch!)
  
- SHORT: Close CROSSES UNDER Lower Band (strict inequality: close < lower)
  → Previous bar: close > lower
  → Current bar: close < lower (MUST cross, not just touch!)

CRITICAL: Uses STRICT inequality (< and >) to match TradingView's ta.crossover() and ta.crossunder()
Touching band (close = band) does NOT generate signal!
  
EXIT: Handled by Broker (flip position on opposite signal)
"""

import pandas as pd 
import numpy as np 
from typing import List, Dict, Optional
from datetime import datetime 

from models.signal import Signal, SignalType 

class KEMAStrategy:
    def __init__(self, **kwargs):
        self.params = kwargs

    def generate_signals(
        self,
        df: pd.DataFrame, 
        upper_band: np.ndarray,
        lower_band: np.ndarray,
    ) -> List[Signal]:
        """
            List[Signal]
        """
        # Extract arrays
        close = df['close'].to_numpy()
        upper = upper_band
        lower = lower_band
        index = df.index
        n = len(close)

        close_prev = np.roll(close, 1)
        close_prev[0] = close[0]
        lower_prev = np.roll(lower, 1)
        lower_prev[0] = lower[0]
        upper_prev = np.roll(upper, 1)
        upper_prev[0] = upper[0]
        
        long_entry = (close_prev < upper_prev) & (close > upper)
        short_exit = long_entry  
        
        short_entry = (close_prev > lower_prev) & (close < lower)
        long_exit = short_entry  

        signal_long_entry = np.zeros(n, dtype=np.int8)
        signal_short_entry = np.zeros(n, dtype=np.int8)
        signal_long_exit = np.zeros(n, dtype=np.int8)
        signal_short_exit = np.zeros(n, dtype=np.int8)
        
        signal_long_entry[long_entry] = 1
        signal_short_entry[short_entry] = 1
        signal_long_exit[long_exit] = 1
        signal_short_exit[short_exit] = 1
        
        signal = np.zeros(n, dtype=np.int8)
        signal[long_entry] = 1   # LONG entry
        signal[short_entry] = -1  # SHORT entry

        import pandas as pd
        signal_series = pd.Series(signal, index=df.index, dtype=np.int8)
        

        valid_signals = signal_series[signal_series != 0]
        
        # Keep only signals where direction CHANGES (1 → -1 or -1 → 1)
        # Compare with previous valid signal, keep if different
        deduped_mask = (valid_signals != valid_signals.shift())
        deduped_signals = valid_signals[deduped_mask]


        
        # 4. Create Signal objects - Filter by 'side' parameter
        side_filter = self.params.get('side', 'both').lower()
        close_array = df['close'].to_numpy()
        
        signals = []
        
        # Process deduped entry signals only
        for ts, sig_val in deduped_signals.items():
            try:
                idx = df.index.get_loc(ts)
                price = float(close_array[idx])
            except KeyError:
                continue
            
            if sig_val == 1:  # LONG entry
                if side_filter in ['both', 'long']:
                    signals.append(Signal(
                        timestamp=ts,
                        signal_type=SignalType.Long,
                        price=price,
                        reason=f'Long Entry: Close > Upper @ {price:.2f}'
                    ))
                elif side_filter == 'short':
                    # In Short Only mode, a Long signal serves as an Exit for Short positions
                    signals.append(Signal(
                        timestamp=ts,
                        signal_type=SignalType.Exit_SHORT,
                        price=price,
                        reason=f'Short Exit (Flip): Close > Upper @ {price:.2f}'
                    ))
            elif sig_val == -1:  # SHORT entry
                if side_filter in ['both', 'short']:
                    signals.append(Signal(
                        timestamp=ts,
                        signal_type=SignalType.Short,
                        price=price,
                        reason=f'Short Entry: Close < Lower @ {price:.2f}'
                    ))
                elif side_filter == 'long':
                    # In Long Only mode, a Short signal serves as an Exit for Long positions
                    signals.append(Signal(
                        timestamp=ts,
                        signal_type=SignalType.Exit_LONG,
                        price=price,
                        reason=f'Long Exit (Flip): Close < Lower @ {price:.2f}'
                    ))
        
        # DEBUG: Dump data to CSV for analysis (if debug mode enabled)
        if self.params.get('debug', False):
            try:
                debug_df = df.copy()
                debug_df['upper'] = upper
                debug_df['lower'] = lower
                debug_df['long_entry'] = long_entry
                debug_df['short_entry'] = short_entry
                debug_df['long_exit'] = long_exit
                debug_df['short_exit'] = short_exit
                debug_df['signal_raw'] = signal_series
                debug_df['signal_deduped'] = deduped_signals.reindex(df.index, fill_value=0)
                dump_path = '/home/ubuntu/vinh/noraquantengine/Grey/tmp/nope/kema_signals_debug.csv'
                debug_df.to_csv(dump_path)
                print(f"✅ DEBUG SIGNALS DUMPED TO: {dump_path}")
            except Exception as e:
                print(f"❌ Failed to dump debug data: {e}")

        return signals