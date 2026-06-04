"""
Metrics Calculator - Performance metrics với NumPy vectorization

Architecture:
- Read-only analytics layer
- Không mutate trades/equity_curve
- Không phụ thuộc strategy/broker
- Vectorize 1 pass duy nhất cho trade metrics

Optimization:
- Convert trades → numpy arrays 1 lần
- Tất cả trade metrics dùng vectorized ops
- Equity metrics đã vectorized từ đầu
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from models.trade import Trade


class MetricsCalculator:
    """
    Calculate performance metrics - Production-grade
    
    Vectorization strategy:
    1. Convert trades → numpy arrays once (init)
    2. All trade metrics use vectorized ops (no loops)
    3. Equity metrics use np.maximum.accumulate
    """
    
    def __init__(self, trades: List[Trade], equity_curve: Any, initial_capital: float, 
                 benchmark_prices: List[float] = None, start_date = None, end_date = None,
                 contract_size: float = 1.0):
        """
        Initialize calculator
        
        Args:
            trades: List of completed trades
            equity_curve: List of equity snapshots OR Numpy array of bar-by-bar equity
            initial_capital: Starting capital
            benchmark_prices: Optional list/array of asset closing prices for benchmark comparison
            start_date: Backtest start date (for accurate CAGR calculation)
            end_date: Backtest end date (for accurate CAGR calculation)
            contract_size: Size of 1 contract (e.g. 100 for Gold, 1 for BTC)
        """
        self.trades = trades
        self.equity_curve = equity_curve # Can be List[Dict] or np.ndarray
        self._equity_arr_cache = None # Lazy cache for the array version
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.contract_size = contract_size
        
        # Convert benchmark prices to numpy array if provided
        self.benchmark_prices = np.array(benchmark_prices) if benchmark_prices is not None and len(benchmark_prices) > 0 else np.array([])
        
        # OPTIMIZATION: Vectorize trades once (not 15 times)
        self._vectorized_trades = self._vectorize_trades()

    def _get_equity_arr(self) -> np.ndarray:
        """Helper to get equity as numpy array regardless of input format"""
        if self._equity_arr_cache is not None:
            return self._equity_arr_cache
            
        if isinstance(self.equity_curve, np.ndarray):
            self._equity_arr_cache = self.equity_curve
        elif isinstance(self.equity_curve, list) and len(self.equity_curve) > 0:
            # Handle List[Dict] format from Standard Mode
            if isinstance(self.equity_curve[0], dict):
                self._equity_arr_cache = np.array([e['equity'] for e in self.equity_curve], dtype=np.float64)
            else:
                self._equity_arr_cache = np.array(self.equity_curve, dtype=np.float64)
        else:
            self._equity_arr_cache = np.array([self.initial_capital], dtype=np.float64)
            
        return self._equity_arr_cache
    
    def _vectorize_trades(self) -> Dict[str, np.ndarray]:
        """
        Convert trades to numpy arrays ONCE
        
        Returns:
            Dict with vectorized trade data
        """
        if not self.trades:
            return {
                'pnl': np.array([]),
                'commission': np.array([]),
                'quantity': np.array([]),
                'entry_price': np.array([]),
                'exit_time': np.array([]),
                'bars': np.array([]),
            }
        
        return {
            'pnl': np.array([t.pnl for t in self.trades]),
            'commission': np.array([t.commission for t in self.trades]),
            'quantity': np.array([t.quantity for t in self.trades]),
            'entry_price': np.array([t.entry_price for t in self.trades]),
            'exit_time': np.array([t.exit_time for t in self.trades]),
            'bars': np.array([t.bars for t in self.trades]),
            'mfe': np.array([getattr(t, 'mfe', 0) or 0 for t in self.trades]),
            'mae': np.array([getattr(t, 'mae', 0) or 0 for t in self.trades]),
        }
    
    def calculate_fast(self) -> Dict[str, Any]:
        """Fast calculation of essential metrics for optimization loop (🚀 Optimized)"""
        if not self.trades and len(self._vectorized_trades['pnl']) == 0:
            return self._empty_metrics()

        pnl = self._vectorized_trades['pnl']
        wins = pnl > 0
        losses = pnl < 0
        
        # Core stats (Vectorized)
        total_trades = len(pnl)
        win_count = int(np.sum(wins))
        loss_count = int(np.sum(losses))
        # FIXED: Win rate should exclude breakeven trades (wins + losses only)
        decided_trades = win_count + loss_count
        win_rate = float((win_count / decided_trades) * 100) if decided_trades > 0 else 0.0
        total_pnl = float(np.sum(pnl))
        
        return {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': self._roi(),
            'initial_capital': self.initial_capital,
            'final_equity': self._final_equity(),
            'max_drawdown': self._max_drawdown(),
            'max_drawdown_pct': self._max_drawdown_pct(),
            'sharpe': self._sharpe_ratio(),
            'sharpe_ratio': self._sharpe_ratio(), # For backward compatibility
            'sortino': self._sortino_ratio(),
            'sortino_ratio': self._sortino_ratio(), # For backward compatibility
            'profit_factor': self._profit_factor(pnl, wins, losses),
            'expectancy': self._expectancy(pnl),
            'avg_win': float(np.mean(pnl[wins])) if np.any(wins) else 0.0,
            'avg_loss': float(np.mean(pnl[losses])) if np.any(losses) else 0.0,
            'largest_win': float(np.max(pnl)) if total_trades > 0 else 0.0,
            'largest_loss': float(np.min(pnl)) if total_trades > 0 else 0.0,
            'commission_paid': float(np.sum(self._vectorized_trades['commission'])),
            'cagr': self._cagr(),
            'calmar_ratio': self._calmar_ratio(),
            'recovery_factor': self._recovery_factor(),
            'total_return': self._total_return(),
            'max_consecutive_wins': self._max_consecutive_wins(wins),
            'max_consecutive_losses': self._max_consecutive_losses(losses)
        }

    def calculate_all(self) -> Dict[str, Any]:
        """
        Calculate all metrics - matching Vinh's 29 metrics
        
        Returns:
            Dict with all performance metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        # Get vectorized data
        pnl = self._vectorized_trades['pnl']
        
        # Vectorized masks (O(1) operations)
        wins = pnl > 0
        losses = pnl < 0  # Breakeven (pnl == 0) excluded
        
        # FIXED: Win rate should exclude breakeven trades
        win_count = int(np.sum(wins))
        loss_count = int(np.sum(losses))
        decided_trades = win_count + loss_count
        
        return {
            # Core trading metrics
            'total_trades': len(self.trades),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': float((win_count / decided_trades) * 100) if decided_trades > 0 else 0.0,
            
            # PnL metrics
            'total_pnl': float(np.sum(pnl)),
            'total_pnl_pct': self._total_pnl_pct(),
            'roi': self._roi(),
            'initial_capital': self.initial_capital,
            
            # Win/Loss stats
            'avg_win': float(np.mean(pnl[wins])) if np.any(wins) else 0.0,
            'avg_loss': float(np.mean(pnl[losses])) if np.any(losses) else 0.0,
            'largest_win': float(np.max(pnl)) if len(pnl) > 0 else 0.0,
            'largest_loss': float(np.min(pnl)) if len(pnl) > 0 else 0.0,
            
            # Duration metrics (Bars)
            'avg_bars': float(np.mean(self._vectorized_trades['bars'])) if len(pnl) > 0 else 0.0,
            'avg_bars_win': float(np.mean(self._vectorized_trades['bars'][wins])) if np.any(wins) else 0.0,
            'avg_bars_loss': float(np.mean(self._vectorized_trades['bars'][losses])) if np.any(losses) else 0.0,
            
            # Risk metrics
            'profit_factor': self._profit_factor(pnl, wins, losses),
            'expectancy': self._expectancy(pnl),
            
            # Cost metrics
            'commission_paid': float(np.sum(self._vectorized_trades['commission'])),
            
            # Equity metrics
            'final_equity': self._final_equity(),
            'max_drawdown': self._max_drawdown(),
            'max_drawdown_pct': self._max_drawdown_pct(),
            
            # Risk-adjusted returns (canonical keys only)
            'sharpe': self._sharpe_ratio(),
            'sortino': self._sortino_ratio(),
            'cagr': self._cagr(),
            
            # Additional metrics
            'frequency': self._frequency(),
            'total_turnover': self._total_turnover(),
            'tvr': self._tvr(),
            'ppc': self._ppc(),
            'total_days': self._total_days(),
            'monthly_returns': self._monthly_returns(),
            
            # Deprecated keys (for backward compatibility)
            'sharpe_ratio': self._sharpe_ratio(),
            'sortino_ratio': self._sortino_ratio(),
            'positive_ratio': float(win_count / decided_trades) if decided_trades > 0 else 0.0,
            'positive_month_ratio': self._positive_month_ratio(),
            
            # Placeholders
            'bench_max': {},
            'vs_bench_max': {},
            'lower': 0.0,
            
            # Benchmark Comparison (Calculated from price data)
            'benchmark': self._benchmark_metrics(),
            
            # Capital Efficiency (New)
            'capital_efficiency': self._capital_efficiency(pnl, losses),
            
            # Survival Metrics (Dynamic)
            'max_leverage': self._max_leverage(),
            'max_consecutive_losses': self._max_consecutive_losses(losses),
            'max_drawdown_duration': self._max_drawdown_duration()
        }
    
    def _profit_factor(self, pnl: np.ndarray, wins: np.ndarray, losses: np.ndarray) -> float:
        """Profit factor = total wins / abs(total losses)"""
        total_wins = np.sum(pnl[wins]) if np.any(wins) else 0.0
        total_losses = abs(np.sum(pnl[losses])) if np.any(losses) else 0.0
        
        if total_losses == 0:
            return 0.0 if total_wins == 0 else float('inf')
        
        return float(total_wins / total_losses)
    
    def _expectancy(self, pnl: np.ndarray) -> float:
        """Average PnL per trade"""
        return float(np.mean(pnl)) if len(pnl) > 0 else 0.0
    
    def _total_pnl_pct(self) -> float:
        """Total PnL as % of initial capital"""
        total_pnl = np.sum(self._vectorized_trades['pnl'])
        return float((total_pnl / self.initial_capital) * 100) if self.initial_capital > 0 else 0.0
    
    def _roi(self) -> float:
        """Return on Investment (%)"""
        return self._total_pnl_pct()
    
    def _final_equity(self) -> float:
        """Final equity"""
        equity = self._get_equity_arr()
        return float(equity[-1])
    
    def _max_drawdown(self) -> float:
        """Maximum drawdown (absolute)"""
        equity = self._get_equity_arr()
        if equity.size == 0: return 0.0
        
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        
        return float(np.max(drawdown))
    
    def _max_drawdown_pct(self) -> float:
        """Maximum drawdown (%)"""
        equity = self._get_equity_arr()
        if equity.size == 0: return 0.0
        
        peak = np.maximum.accumulate(equity)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown_pct = np.where(peak > 0, ((peak - equity) / peak) * 100, 0)
        
        return float(np.max(drawdown_pct))
    
    def _sharpe_ratio(self) -> float:
        """Sharpe ratio (annualized)"""
        equity = self._get_equity_arr()
        if equity.size < 2: return 0.0
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualize (252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        return float(sharpe)
    
    def _sortino_ratio(self) -> float:
        """Sortino ratio (annualized, downside deviation only)"""
        equity = self._get_equity_arr()
        if equity.size < 2: return 0.0
        
        returns = np.diff(equity) / equity[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        sortino = (mean_return / downside_std) * np.sqrt(252)
        
        return float(sortino)
    
    def _cagr(self) -> float:
        """Compound Annual Growth Rate (%) - Based on backtest period, not trade period"""
        # Use backtest period if available (professional approach)
        if self.start_date is not None and self.end_date is not None:
            time_diff = self.end_date - self.start_date
            years = time_diff.total_seconds() / (365.25 * 24 * 3600)
        # Fallback to trade period if no backtest dates provided (backward compatibility)
        elif self.trades and len(self.trades) > 0:
            first_trade = self.trades[0]
            last_trade = self.trades[-1]
            time_diff = last_trade.exit_time - first_trade.entry_time
            years = time_diff.total_seconds() / (365.25 * 24 * 3600)
        else:
            return 0.0
        
        if years <= 0:
            return 0.0
        
        final_equity = self._final_equity()
        
        if final_equity <= 0:
            return -100.0
        
        # CAGR = (Final / Initial)^(1/years) - 1
        cagr = (pow(final_equity / self.initial_capital, 1 / years) - 1) * 100
        
        return float(cagr)
    
    def _frequency(self) -> float:
        """Average trades per day (using backtest period if available)"""
        if not self.trades:
            return 0.0
        
        # Use backtest period if available (more accurate)
        if self.start_date is not None and self.end_date is not None:
            time_diff = self.end_date - self.start_date
            total_days = time_diff.total_seconds() / (24 * 3600)
        else:
            # Fallback to trade period
            total_days = self._total_days()
        
        if total_days <= 0:
            return 0.0
        
        return float(len(self.trades) / total_days)
    
    def _total_turnover(self) -> float:
        """Total turnover (sum of all trade notional values)"""
        pnl = self._vectorized_trades['pnl']
        qty = self._vectorized_trades['quantity']
        entry_price = self._vectorized_trades['entry_price']
        
        # Turnover = sum(quantity * entry_price) for all trades
        turnover = np.sum(qty * entry_price)
        
        return float(turnover)
    
    def _tvr(self) -> float:
        """Turnover ratio (turnover / initial_capital)"""
        turnover = self._total_turnover()
        
        if self.initial_capital <= 0:
            return 0.0
        
        return float(turnover / self.initial_capital)
    
    def _ppc(self) -> float:
        """Profit per capital (total_pnl / initial_capital)"""
        return self._roi() / 100  # ROI as ratio, not %
    
    def _total_days(self) -> float:
        """Total days in backtest period"""
        if not self.trades or len(self.trades) < 2:
            return 0.0
        
        first_trade = self.trades[0]
        last_trade = self.trades[-1]
        
        time_diff = last_trade.exit_time - first_trade.entry_time
        days = time_diff.total_seconds() / (24 * 3600)
        
        return float(days)
    
    def _monthly_returns(self) -> Dict[str, float]:
        """Monthly returns (vectorized with pandas groupby)"""
        if not self.trades:
            return {}
        
        # Convert to pandas for efficient groupby
        exit_times = self._vectorized_trades['exit_time']
        pnls = self._vectorized_trades['pnl']
        
        df = pd.DataFrame({
            'exit_time': exit_times,
            'pnl': pnls
        })
        
        # Group by month
        df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
        monthly = df.groupby('month')['pnl'].sum()
        
        # Convert to dict
        return {str(month): float(pnl) for month, pnl in monthly.items()}
    
    def _positive_month_ratio(self) -> float:
        """Ratio of positive months"""
        monthly_returns = self._monthly_returns()
        
        if not monthly_returns:
            return 0.0
        
        positive_months = sum(1 for pnl in monthly_returns.values() if pnl > 0)
        total_months = len(monthly_returns)
        
        return float(positive_months / total_months) if total_months > 0 else 0.0
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when no trades"""
        empty = {
            'total_trades': 0,
            'win_rate': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'positive_ratio': 0.0,
            'positive_month_ratio': 0.0,
            'total_pnl': 0.0,
            'total_pnl_pct': 0.0,
            'roi': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_bars': 0.0,
            'avg_bars_win': 0.0,
            'avg_bars_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'commission_paid': 0.0,
            'commission_pct': 0.0,
            'final_equity': self.initial_capital,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sharpe': 0.0,
            'sortino_ratio': 0.0,
            'sortino': 0.0,
            'cagr': 0.0,
            'frequency': 0.0,
            'total_turnover': 0.0,
            'tvr': 0.0,
            'ppc': 0.0,
            'total_days': 0.0,
            'monthly_returns': {},
            'bench_max': {},
            'vs_bench_max': {},
            'param_window': 0,
            'upper': 0.0,
            'lower': 0.0,
            'benchmark': self._benchmark_metrics() # Ensure benchmark calc works even with no trades (using prices)
        }
        return empty

    def _benchmark_metrics(self) -> Dict[str, float]:
        """
        Calculate Buy & Hold Benchmark Performance
        
        Logic:
        - B&H Return = (Last Price - First Price) / First Price * 100
        - B&H Max = (Max Price - First Price) / First Price * 100
        - B&H Min = (Min Price - First Price) / First Price * 100
        """
        if len(self.benchmark_prices) < 2:
            return {
                'buy_hold_return': 0.0,
                'buy_hold_max': 0.0,
                'buy_hold_min': 0.0,
                'strategy_return': 0.0,
                'strategy_max': 0.0,
                'strategy_min': 0.0
            }
            
        first_price = self.benchmark_prices[0]
        if first_price == 0:
            return {
                'buy_hold_return': 0.0,
                'buy_hold_max': 0.0,
                'buy_hold_min': 0.0,
                'strategy_return': 0.0,
                'strategy_max': 0.0,
                'strategy_min': 0.0
            }

        # Benchmark Stats
        current_price = self.benchmark_prices[-1]
        max_price = np.max(self.benchmark_prices)
        min_price = np.min(self.benchmark_prices)
        
        bh_return = ((current_price - first_price) / first_price) * 100
        bh_max = ((max_price - first_price) / first_price) * 100
        bh_min = ((min_price - first_price) / first_price) * 100
        
        # Strategy Stats (Equity Curve based - % Return relative to Initial Capital)
        # Using same scale: (Current Equity - Initial) / Initial * 100
        strat_return = 0.0
        strat_max = 0.0
        strat_min = 0.0
        
        if self.initial_capital > 0:
            equities = self._get_equity_arr()
            
            # Start logic: equity curve usually starts with initial capital.
            # We compare entire curve against initial.
            current_equity = equities[-1]
            max_equity = np.max(equities)
            min_equity = np.min(equities)
            
            strat_return = ((current_equity - self.initial_capital) / self.initial_capital) * 100
            strat_max = ((max_equity - self.initial_capital) / self.initial_capital) * 100
            strat_min = ((min_equity - self.initial_capital) / self.initial_capital) * 100
            
        return {
            'buy_hold_return': float(bh_return),
            'buy_hold_max': float(bh_max),
            'buy_hold_min': float(bh_min),
            'strategy_return': float(strat_return),
            'strategy_max': float(strat_max),
            'strategy_min': float(strat_min)
        }

    def _capital_efficiency(self, pnl: np.ndarray, losses: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Capital Efficiency metrics for frontend
        """
        roi = self._roi()
        total_pnl = float(np.sum(pnl)) if len(pnl) > 0 else 0.0
        
        largest_loss = float(np.min(pnl)) if len(pnl) > 0 else 0.0
        net_profit_largest_loss = 0.0
        if largest_loss < 0:
            net_profit_largest_loss = (total_pnl / abs(largest_loss)) * 100
            
        return {
            'capital_usage': {
                'return_on_initial_capital': roi,
                'return_on_account_size_required': roi, # Placeholder/Simplified
                'net_profit_pct_of_largest_loss': net_profit_largest_loss,
                
                # Sub-keys for Long/Short (Could be calculated separately if needed, passing blank for now)
                'long': {},
                'short': {}
            }
        }


    def _runup_drawdown_metrics(self, pnl: np.ndarray, mfe: np.ndarray) -> Dict[str, float]:
        """
        Calculate extensive Run-up and Drawdown metrics
        
        Returns keys matching frontend expectations:
        - avg_runup_duration
        - avg_runup
        - max_runup
        - maxRunupVal (intrabar)
        - maxRunupPct (intrabar)
        - avg_drawdown_duration
        - avg_drawdown
        """
        metrics = {
            'avg_runup_duration': 0.0,
            'avg_runup': 0.0,
            'max_runup': 0.0,
            'maxRunupVal': 0.0,
            'maxRunupPct': 0.0,
            'avg_drawdown_duration': 0.0,
            'avg_drawdown': 0.0
        }
        
        # 1. Close-to-Close Metrics (Equity Curve Analysis)
        equity = self._get_equity_arr()
        if equity.size > 1:
            
            # --- Drawdown Analysis ---
            peak = np.maximum.accumulate(equity)
            drawdown = peak - equity
            
            # Find Drawdown Periods (DD > 0)
            is_dd = drawdown > 0
            # Identify starts/ends of periods
            # Pad with False to detect transitions at edges
            padded_dd = np.concatenate(([False], is_dd, [False]))
            transitions = np.diff(padded_dd.astype(int))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            
            dd_durations = []
            dd_magnitudes = []
            
            if len(starts) > 0 and len(ends) > 0:
                for i in range(min(len(starts), len(ends))):
                    start = starts[i]
                    end = ends[i]
                    duration = end - start
                    # Max drawdown in this specific period
                    period_dd = np.max(drawdown[start:end])
                    
                    dd_durations.append(duration)
                    dd_magnitudes.append(period_dd)
                    
            if dd_durations:
                metrics['avg_drawdown_duration'] = float(np.mean(dd_durations))
                metrics['avg_drawdown'] = float(np.mean(dd_magnitudes))
                
            # --- Run-up Analysis (Symmetric) ---
            # Run-up measured from lowest point since start (Valley)
            valley = np.minimum.accumulate(equity)
            runup = equity - valley
            
            is_ru = runup > 0
            padded_ru = np.concatenate(([False], is_ru, [False]))
            transitions_ru = np.diff(padded_ru.astype(int))
            starts_ru = np.where(transitions_ru == 1)[0]
            ends_ru = np.where(transitions_ru == -1)[0]
            
            ru_durations = []
            ru_magnitudes = []
            
            if len(starts_ru) > 0 and len(ends_ru) > 0:
                for i in range(min(len(starts_ru), len(ends_ru))):
                    start = starts_ru[i]
                    end = ends_ru[i]
                    duration = end - start
                    period_ru = np.max(runup[start:end])
                    
                    ru_durations.append(duration)
                    ru_magnitudes.append(period_ru)
                    
            if ru_durations:
                metrics['avg_runup_duration'] = float(np.mean(ru_durations))
                metrics['avg_runup'] = float(np.mean(ru_magnitudes))
                metrics['max_runup'] = float(np.max(runup))
                
        # 2. Intrabar Metrics (Trade MFE Analysis)
        if len(mfe) > 0:
            max_mfe = np.max(mfe)
            metrics['maxRunupVal'] = float(max_mfe)
            
            if self.initial_capital > 0:
                metrics['maxRunupPct'] = float((max_mfe / self.initial_capital) * 100)
                
        return metrics

    def _max_consecutive_losses(self, losses: np.ndarray) -> int:
        """Max consecutive losses calculation (Vectorized)"""
        if len(losses) == 0:
            return 0
        
        # Identify blocks of consecutive losses
        loss_int = losses.astype(int)
        # Pad to detect transitions at ends
        is_loss = np.concatenate(([0], loss_int, [0]))
        diff = np.diff(is_loss)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
            
        durations = ends - starts
        return int(np.max(durations))
    
    def _max_consecutive_wins(self, wins: np.ndarray) -> int:
        """Max consecutive wins calculation (Vectorized)"""
        if len(wins) == 0:
            return 0
        
        # Identify blocks of consecutive wins
        win_int = wins.astype(int)
        # Pad to detect transitions at ends
        is_win = np.concatenate(([0], win_int, [0]))
        diff = np.diff(is_win)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
            
        durations = ends - starts
        return int(np.max(durations))
    
    def _total_return(self) -> float:
        """Total return % = (final_equity - initial) / initial * 100"""
        final = self._final_equity()
        if self.initial_capital <= 0:
            return 0.0
        return float(((final - self.initial_capital) / self.initial_capital) * 100)
    
    def _calmar_ratio(self) -> float:
        """Calmar ratio = CAGR / Max Drawdown %"""
        cagr = self._cagr()
        max_dd = self._max_drawdown_pct()
        
        if max_dd == 0:
            return 0.0
        
        return float(cagr / max_dd)
    
    def _recovery_factor(self) -> float:
        """Recovery factor = Total PnL / Max Drawdown (absolute)"""
        total_pnl = np.sum(self._vectorized_trades['pnl']) if len(self._vectorized_trades['pnl']) > 0 else 0
        max_dd = self._max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return float(total_pnl / max_dd)

    def _max_drawdown_duration(self) -> int:
        """Maximum number of bars/snapshots in a drawdown period"""
        equity = self._get_equity_arr()
        if equity.size < 2:
            return 0
            
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        
        # Drawdown exists where drawdown > 0
        is_dd = drawdown > 0
        padded = np.concatenate(([False], is_dd, [False]))
        diff = np.diff(padded.astype(int))
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
            
        return int(np.max(ends - starts))

    def _max_leverage(self) -> float:
        """
        Highest leverage used: (Contracts * Price * ContractSize) / Equity_at_entry
        """
        if not self.trades:
            return 0.0
            
        quantities = self._vectorized_trades['quantity']
        prices = self._vectorized_trades['entry_price']
        
        # Notional value of each trade
        notionals = quantities * prices * self.contract_size
        
        # We need equity at entry. Use cumulative_pnl to estimate equity
        equities = np.array([getattr(t, 'equity_at_entry', self.initial_capital + (t.cumulative_pnl - t.pnl)) for t in self.trades])
        
        leverages = np.where(equities > 0, notionals / equities, 0)
        
        return float(np.max(leverages)) if len(leverages) > 0 else 0.0