"""
Monte Carlo & Bootstrap Risk Evaluation Module

This module implements:
1. Monte Carlo Simulation - Stress testing through shuffling (Risk of Ruin, Worst Case)
2. Bootstrap Analysis - Reliability testing through resampling (Lucky vs Robust)

Core Objectives:
- Stress Test: Can the strategy survive worst-case scenarios?
- Reliability Test: Is the backtest result reliable or just lucky?

Author: NoraQuant Engine
Created: 2026-02-10
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo & Bootstrap simulations"""
    num_simulations: int = 1000
    initial_capital: float = 1000000.0
    confidence_level: float = 0.95
    seed: Optional[int] = None
    
    # Risk thresholds
    ruin_threshold: float = 0.5  # 50% drawdown = "ruin"
    var_percentile: float = 0.05  # 5% VaR
    
    # Analysis options
    calculate_var: bool = True
    calculate_cvar: bool = True
    calculate_risk_of_ruin: bool = True
    analyze_drawdown_dist: bool = True


class MonteCarloEngine:
    """
    Monte Carlo Engine - Stress Test through Shuffling
    
    Purpose: Test strategy robustness by randomizing trade order
    - Keeps same wins/losses, only changes order
    - Finds worst-case scenarios (max drawdown, risk of ruin)
    - Answers: "What if bad trades cluster together?"
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def shuffle_trades(self, pnl_list: np.ndarray) -> np.ndarray:
        """
        Shuffle trade order randomly
        
        Args:
            pnl_list: Array of profit/loss values
            
        Returns:
            Shuffled array
        """
        shuffled = pnl_list.copy()
        np.random.shuffle(shuffled)
        return shuffled
    
    def calculate_equity_curve(self, pnl_list: np.ndarray, initial_capital: float) -> np.ndarray:
        """
        Calculate equity curve from P&L list
        
        Args:
            pnl_list: Array of profit/loss percentages (e.g., [0.01, -0.005, 0.02])
            initial_capital: Starting capital
            
        Returns:
            Array of equity values
        """
        # Convert percentage returns to equity curve
        returns = 1 + pnl_list
        equity = initial_capital * np.cumprod(returns)
        return equity
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate max drawdown and drawdown series
        
        Args:
            equity_curve: Array of equity values
            
        Returns:
            Tuple of (max_drawdown_pct, drawdown_series)
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Max drawdown (negative value)
        max_dd = np.min(drawdown)
        
        return max_dd, drawdown
    
    def run_simulations(self, pnl_list: np.ndarray) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations
        
        Args:
            pnl_list: Array of P&L percentages
            
        Returns:
            Dictionary with simulation results
        """
        n_sims = self.config.num_simulations
        initial_capital = self.config.initial_capital
        
        # Storage for results
        final_equity = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        final_returns = np.zeros(n_sims)
        
        # Store some equity curves for visualization (storage-efficient)
        sample_indices = np.linspace(0, n_sims-1, min(100, n_sims), dtype=int)
        sample_curves = []
        
        for i in range(n_sims):
            # Shuffle trades
            shuffled_pnl = self.shuffle_trades(pnl_list)
            
            # Calculate equity curve
            equity_curve = self.calculate_equity_curve(shuffled_pnl, initial_capital)
            
            # Calculate metrics
            final_equity[i] = equity_curve[-1]
            final_returns[i] = (equity_curve[-1] - initial_capital) / initial_capital
            max_dd, _ = self.calculate_drawdown(equity_curve)
            max_drawdowns[i] = max_dd
            
            # Store sample curves for visualization
            if i in sample_indices:
                sample_curves.append(equity_curve.tolist())
        
        return {
            'final_equity': final_equity,
            'final_returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'sample_curves': sample_curves,
            'num_simulations': n_sims
        }


class BootstrapEngine:
    """
    Bootstrap Engine - Reliability Test through Resampling
    
    Purpose: Test if backtest results are reliable or lucky
    - Random sampling WITH replacement
    - Creates distribution of possible outcomes
    - Answers: "Is my profit typical or an outlier?"
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
    
    def bootstrap_sample(self, pnl_list: np.ndarray) -> np.ndarray:
        """
        Create bootstrap sample with replacement
        
        Args:
            pnl_list: Array of P&L values
            
        Returns:
            Bootstrap sample (same length as original)
        """
        n = len(pnl_list)
        indices = np.random.choice(n, size=n, replace=True)
        return pnl_list[indices]
    
    def run_bootstrap(self, pnl_list: np.ndarray) -> Dict[str, Any]:
        """
        Run bootstrap analysis
        
        Args:
            pnl_list: Array of P&L percentages
            
        Returns:
            Dictionary with bootstrap results
        """
        n_sims = self.config.num_simulations
        initial_capital = self.config.initial_capital
        
        # Storage
        final_returns = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        
        for i in range(n_sims):
            # Bootstrap sample
            bootstrapped_pnl = self.bootstrap_sample(pnl_list)
            
            # Calculate equity curve
            equity_curve = MonteCarloEngine.calculate_equity_curve(
                self, bootstrapped_pnl, initial_capital
            )
            
            # Calculate metrics
            final_returns[i] = (equity_curve[-1] - initial_capital) / initial_capital
            max_dd, _ = MonteCarloEngine.calculate_drawdown(self, equity_curve)
            max_drawdowns[i] = max_dd
        
        return {
            'final_returns': final_returns,
            'max_drawdowns': max_drawdowns,
            'num_simulations': n_sims
        }


class RiskAnalyzer:
    """
    Risk Analyzer - Calculate risk metrics from simulation results
    
    Calculates:
    - Risk of Ruin (RoR)
    - Value at Risk (VaR)
    - Conditional VaR (CVaR / Expected Shortfall)
    - Worst Case Scenarios
    - Confidence intervals
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def calculate_risk_of_ruin(
        self, 
        max_drawdowns: np.ndarray,
        threshold: Optional[float] = None
    ) -> float:
        """
        Calculate Risk of Ruin
        
        Percentage of simulations where drawdown exceeds threshold
        
        Args:
            max_drawdowns: Array of max drawdown values (negative)
            threshold: Ruin threshold (e.g., -0.5 for 50% loss)
            
        Returns:
            Risk of Ruin as percentage (0-100)
        """
        if threshold is None:
            threshold = -self.config.ruin_threshold
        
        ruin_count = np.sum(max_drawdowns <= threshold)
        ror = (ruin_count / len(max_drawdowns)) * 100
        
        return ror
    
    def calculate_var(
        self,
        returns: np.ndarray,
        percentile: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Array of return values
            percentile: Percentile for VaR (e.g., 0.05 for 5% VaR)
            
        Returns:
            VaR value (negative number representing loss)
        """
        if percentile is None:
            percentile = self.config.var_percentile
        
        var = np.percentile(returns, percentile * 100)
        return var
    
    def calculate_cvar(
        self,
        returns: np.ndarray,
        percentile: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        Average of returns worse than VaR
        
        Args:
            returns: Array of return values
            percentile: Percentile for CVaR
            
        Returns:
            CVaR value (expected loss in worst cases)
        """
        if percentile is None:
            percentile = self.config.var_percentile
        
        var = self.calculate_var(returns, percentile)
        cvar = np.mean(returns[returns <= var])
        
        return cvar
    
    def calculate_worst_case(
        self,
        max_drawdowns: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate worst case scenarios
        
        Args:
            max_drawdowns: Array of max drawdown values
            returns: Array of return values
            
        Returns:
            Dictionary with worst case metrics
        """
        return {
            'worst_drawdown': float(np.min(max_drawdowns)),
            'worst_return': float(np.min(returns)),
            'worst_final_equity': float(
                self.config.initial_capital * (1 + np.min(returns))
            )
        }
    
    def calculate_confidence_interval(
        self,
        values: np.ndarray,
        confidence: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval
        
        Args:
            values: Array of values
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if confidence is None:
            confidence = self.config.confidence_level
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(values, lower_percentile)
        upper = np.percentile(values, upper_percentile)
        
        return lower, upper
    
    def analyze_drawdown_distribution(
        self,
        max_drawdowns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze drawdown distribution
        
        Args:
            max_drawdowns: Array of max drawdown values
            
        Returns:
            Dictionary with distribution statistics
        """
        # Convert to positive percentages for easier reading
        dd_pct = -max_drawdowns * 100
        
        return {
            'mean': float(np.mean(dd_pct)),
            'median': float(np.median(dd_pct)),
            'std': float(np.std(dd_pct)),
            'min': float(np.min(dd_pct)),
            'max': float(np.max(dd_pct)),
            'percentile_25': float(np.percentile(dd_pct, 25)),
            'percentile_75': float(np.percentile(dd_pct, 75)),
            'percentile_95': float(np.percentile(dd_pct, 95))
        }
    
    def get_percentiles(
        self,
        values: np.ndarray,
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Dict[str, float]:
        """
        Get multiple percentiles for visualization
        
        Args:
            values: Array of values
            percentiles: List of percentile values (0-100)
            
        Returns:
            Dictionary mapping percentile to value
        """
        result = {}
        for p in percentiles:
            result[f'p{p}'] = float(np.percentile(values, p))
        return result


class VerdictEngine:
    """
    Verdict Engine - Generate final assessment
    
    Generates verdict based on comprehensive analysis:
    - ROBUST: Strategy is solid, low risk
    - LUCKY: Results may be due to randomness/overfitting
    - DANGEROUS: High risk, unstable performance
    - WARNING: Some concerns but not critical
    """
    
    @staticmethod
    def generate_verdict(
        original_return: float,
        original_mdd: float,
        mc_results: Dict[str, Any],
        bs_results: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate final verdict
        
        Args:
            original_return: Original backtest return
            original_mdd: Original backtest max drawdown
            mc_results: Monte Carlo simulation results
            bs_results: Bootstrap analysis results
            risk_metrics: Calculated risk metrics
            
        Returns:
            Dictionary with verdict and reasoning
        """
        # Extract key metrics
        ror = risk_metrics.get('risk_of_ruin', 0)
        worst_dd = risk_metrics.get('worst_case', {}).get('worst_drawdown', 0)
        
        # Bootstrap comparison
        bs_returns = bs_results['final_returns']
        bs_mean = np.mean(bs_returns)
        bs_std = np.std(bs_returns)
        
        # Check if original is outlier (more than 2 std above mean)
        is_outlier = original_return > (bs_mean + 2 * bs_std)
        
        # Worst case MDD comparison
        mdd_increase = abs(worst_dd / original_mdd) if original_mdd != 0 else 1
        
        # Decision logic
        issues = []
        warnings = []
        
        # Risk of Ruin check
        if ror > 5:
            issues.append(f"High Risk of Ruin: {ror:.1f}%")
        elif ror > 1:
            warnings.append(f"Moderate Risk of Ruin: {ror:.1f}%")
        
        # Worst case check
        if abs(worst_dd) > 0.6:  # 60% drawdown
            issues.append(f"Extreme worst-case drawdown: {abs(worst_dd)*100:.1f}%")
        elif mdd_increase > 2:
            warnings.append(f"Worst-case DD is {mdd_increase:.1f}x worse than original")
        
        # Outlier check (Lucky?)
        if is_outlier:
            warnings.append("Original profit is unusually high (potential overfit)")
        
        # Bootstrap reliability check
        if original_return < bs_mean - bs_std:
            warnings.append("Original performance below bootstrap average")
        
        # Determine status
        if len(issues) > 0:
            status = "DANGEROUS"
            badge = "❌"
            message = "High risk detected. " + " ".join(issues)
        elif is_outlier and len(warnings) >= 2:
            status = "LUCKY"
            badge = "⚠️"
            message = "Results may be due to luck/overfitting. " + " ".join(warnings)
        elif len(warnings) > 0:
            status = "WARNING"
            badge = "⚠️"
            message = "Some concerns detected. " + " ".join(warnings)
        else:
            status = "ROBUST"
            badge = "✅"
            message = f"Strategy is robust. Risk of Ruin < 1%. Consistent performance."
        
        return {
            'status': status,
            'badge': badge,
            'message': message,
            'issues': issues,
            'warnings': warnings,
            'metrics': {
                'risk_of_ruin': ror,
                'worst_case_mdd_pct': abs(worst_dd) * 100,
                'mdd_multiplier': mdd_increase,
                'is_outlier': is_outlier,
                'bootstrap_mean_return': bs_mean,
                'bootstrap_std_return': bs_std
            }
        }


class MonteCarloSimulator:
    """
    Main Monte Carlo & Bootstrap Simulator
    
    Orchestrates the entire risk evaluation process:
    1. Load trade data
    2. Run Monte Carlo simulations (Shuffle)
    3. Run Bootstrap analysis (Resample)
    4. Calculate risk metrics
    5. Generate verdict
    6. Prepare visualization data
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulator
        
        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self.config = config or SimulationConfig()
        self.mc_engine = MonteCarloEngine(self.config)
        self.bs_engine = BootstrapEngine(self.config)
        self.risk_analyzer = RiskAnalyzer(self.config)
    
    def run_full_analysis(
        self,
        pnl_list: List[float],
        original_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run complete Monte Carlo & Bootstrap analysis
        
        Args:
            pnl_list: List of P&L percentages (e.g., [0.01, -0.005, 0.02])
            original_metrics: Original backtest metrics (roi, mdd, etc.)
            
        Returns:
            Complete analysis results with verdict and visualizations
        """
        # Convert to numpy array
        pnl_array = np.array(pnl_list, dtype=np.float64)
        
        if len(pnl_array) == 0:
            raise ValueError("P&L list is empty")
        
        print(f"[Carlo] Starting analysis on {len(pnl_array)} trades...")
        print(f"[Carlo] Running {self.config.num_simulations} simulations...")
        
        # Calculate original metrics if not provided
        if original_metrics is None:
            original_metrics = self._calculate_original_metrics(pnl_array)
        
        # 1. Run Monte Carlo (Shuffle)
        print("[Carlo] Running Monte Carlo simulations (Shuffle)...")
        mc_results = self.mc_engine.run_simulations(pnl_array)
        
        # 2. Run Bootstrap (Resample)
        print("[Carlo] Running Bootstrap analysis (Resample)...")
        bs_results = self.bs_engine.run_bootstrap(pnl_array)
        
        # 3. Calculate risk metrics
        print("[Carlo] Calculating risk metrics...")
        risk_metrics = self._calculate_all_risk_metrics(mc_results, bs_results)
        
        # 4. Generate verdict
        print("[Carlo] Generating verdict...")
        verdict = VerdictEngine.generate_verdict(
            original_return=original_metrics.get('roi', 0) / 100,
            original_mdd=original_metrics.get('mdd', 0) / 100,
            mc_results=mc_results,
            bs_results=bs_results,
            risk_metrics=risk_metrics
        )
        
        # 5. Prepare chart data
        print("[Carlo] Preparing visualization data...")
        chart_data = self._prepare_chart_data(mc_results, bs_results, original_metrics)
        
        # 6. Calculate threshold statistics
        print("[Carlo] Calculating threshold statistics...")
        threshold_stats = self.calculate_threshold_statistics(
            mc_results,
            thresholds=[20, 15, 10, 5]
        )
        
        # 7. Compile final results
        results = {
            'summary': {
                'num_trades': len(pnl_array),
                'num_simulations': self.config.num_simulations,
                'confidence_level': self.config.confidence_level,
                'original_metrics': original_metrics
            },
            'monte_carlo': {
                'worst_case': risk_metrics['worst_case'],
                'drawdown_dist': risk_metrics['drawdown_distribution'],
                'return_percentiles': risk_metrics['mc_return_percentiles']
            },
            'bootstrap': {
                'mean_return': float(np.mean(bs_results['final_returns'])),
                'std_return': float(np.std(bs_results['final_returns'])),
                'confidence_interval': risk_metrics['bs_confidence_interval'],
                'return_percentiles': risk_metrics['bs_return_percentiles']
            },
            'risk_metrics': {
                'risk_of_ruin': risk_metrics['risk_of_ruin'],
                'var': risk_metrics.get('var', None),
                'cvar': risk_metrics.get('cvar', None),
                'probability_of_profit': risk_metrics['probability_of_profit']
            },
            'threshold_stats': threshold_stats,
            'verdict': verdict,
            'charts': chart_data
        }
        
        print(f"[Carlo] Analysis complete. Verdict: {verdict['status']}")
        return results
    
    def _calculate_original_metrics(self, pnl_array: np.ndarray) -> Dict[str, float]:
        """Calculate metrics from original P&L sequence"""
        equity = self.mc_engine.calculate_equity_curve(
            pnl_array, 
            self.config.initial_capital
        )
        
        final_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital
        max_dd, _ = self.mc_engine.calculate_drawdown(equity)
        
        return {
            'roi': final_return * 100,
            'mdd': max_dd * 100,
            'final_equity': float(equity[-1]),
            'profit': float(equity[-1] - self.config.initial_capital)
        }
    
    def _calculate_all_risk_metrics(
        self,
        mc_results: Dict[str, Any],
        bs_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate all risk metrics from simulation results"""
        metrics = {}
        
        # Risk of Ruin
        if self.config.calculate_risk_of_ruin:
            metrics['risk_of_ruin'] = self.risk_analyzer.calculate_risk_of_ruin(
                mc_results['max_drawdowns']
            )
        
        # VaR and CVaR
        if self.config.calculate_var:
            metrics['var'] = self.risk_analyzer.calculate_var(
                mc_results['final_returns']
            ) * 100  # Convert to percentage
        
        if self.config.calculate_cvar:
            metrics['cvar'] = self.risk_analyzer.calculate_cvar(
                mc_results['final_returns']
            ) * 100
        
        # Worst case
        metrics['worst_case'] = self.risk_analyzer.calculate_worst_case(
            mc_results['max_drawdowns'],
            mc_results['final_returns']
        )
        
        # Drawdown distribution
        if self.config.analyze_drawdown_dist:
            metrics['drawdown_distribution'] = self.risk_analyzer.analyze_drawdown_distribution(
                mc_results['max_drawdowns']
            )
        
        # Probability of profit
        profitable_count = np.sum(mc_results['final_returns'] > 0)
        metrics['probability_of_profit'] = (profitable_count / len(mc_results['final_returns'])) * 100
        
        # Confidence intervals
        metrics['mc_confidence_interval'] = self.risk_analyzer.calculate_confidence_interval(
            mc_results['final_returns']
        )
        metrics['bs_confidence_interval'] = self.risk_analyzer.calculate_confidence_interval(
            bs_results['final_returns']
        )
        
        # Percentiles for visualization
        metrics['mc_return_percentiles'] = self.risk_analyzer.get_percentiles(
            mc_results['final_returns'] * 100  # Convert to percentage
        )
        metrics['bs_return_percentiles'] = self.risk_analyzer.get_percentiles(
            bs_results['final_returns'] * 100
        )
        
        return metrics
    
    def _prepare_chart_data(
        self,
        mc_results: Dict[str, Any],
        bs_results: Dict[str, Any],
        original_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Prepare data for frontend charts
        
        Returns data for:
        - Cone chart (equity curves)
        - Return distribution histogram
        - Drawdown distribution histogram
        """
        # 1. Cone Chart Data (Monte Carlo equity curves)
        cone_data = {
            'sample_curves': mc_results['sample_curves'],
            'percentile_curves': self._calculate_percentile_curves(mc_results)
        }
        
        # 2. Return Distribution (Histogram data)
        mc_returns_pct = mc_results['final_returns'] * 100
        bs_returns_pct = bs_results['final_returns'] * 100
        
        return_hist = {
            'monte_carlo': self._create_histogram_data(mc_returns_pct, bins=50),
            'bootstrap': self._create_histogram_data(bs_returns_pct, bins=50),
            'original_return': original_metrics.get('roi', 0)
        }
        
        # 3. Drawdown Distribution
        dd_hist = self._create_histogram_data(
            -mc_results['max_drawdowns'] * 100,  # Convert to positive percentage
            bins=50
        )
        dd_hist['original_mdd'] = abs(original_metrics.get('mdd', 0))
        
        return {
            'cone': cone_data,
            'return_distribution': return_hist,
            'drawdown_distribution': dd_hist
        }
    
    def calculate_threshold_statistics(
        self,
        mc_results: Dict[str, Any],
        thresholds: List[float] = [20, 15, 10, 5]
    ) -> Dict[str, Any]:
        """
        Calculate pass statistics for different MDD thresholds
        
        Args:
            mc_results: Monte Carlo simulation results
            thresholds: List of MDD thresholds in percentage (e.g., [20, 15, 10, 5])
            
        Returns:
            Dictionary with pass rates and statistics for each threshold
        """
        max_drawdowns = mc_results['max_drawdowns']
        final_returns = mc_results['final_returns']
        num_sims = len(max_drawdowns)
        
        threshold_stats = []
        
        for threshold_pct in thresholds:
            threshold_decimal = -threshold_pct / 100  # Convert to negative decimal
            
            # Find simulations that pass this threshold (MDD better than threshold)
            passing_mask = max_drawdowns >= threshold_decimal
            num_passed = np.sum(passing_mask)
            pass_rate = (num_passed / num_sims) * 100
            
            # Calculate statistics for passing simulations
            if num_passed > 0:
                passing_returns = final_returns[passing_mask]
                passing_mdd = max_drawdowns[passing_mask]
                
                # Calculate annualized Sharpe ratio approximation
                # Sharpe = (mean return) / (std of returns) * sqrt(periods)
                # For Monte Carlo, we use return distribution's std as volatility measure
                mean_return = np.mean(passing_returns)
                std_return = np.std(passing_returns)
                
                # Sharpe approx: just return/risk ratio, no time scaling needed for comparison
                # Cap at reasonable values to avoid overflow
                if std_return > 0.0001:  # Avoid division by near-zero
                    sharpe_approx = mean_return / std_return
                    # Cap sharpe at reasonable range [-10, 10]
                    sharpe_approx = np.clip(sharpe_approx, -10, 10)
                else:
                    # If no variance, set to 0 (deterministic outcome)
                    sharpe_approx = 0.0
                
                # Calculate average metrics
                avg_roi_pct = float(np.mean(passing_returns) * 100)
                avg_mdd_pct = float(abs(np.mean(passing_mdd)) * 100)
                avg_sharpe = float(sharpe_approx)
                
                stats = {
                    'threshold_pct': threshold_pct,
                    'num_passed': int(num_passed),
                    'num_total': int(num_sims),
                    'pass_rate_pct': float(pass_rate),
                    'stats': {
                        'avg_mdd_pct': avg_mdd_pct,
                        'avg_roi_pct': avg_roi_pct,
                        'avg_sharpe': avg_sharpe
                    },
                    'metrics': {
                        'roi': {
                            'mean': avg_roi_pct,
                            'median': float(np.median(passing_returns) * 100),
                            'min': float(np.min(passing_returns) * 100),
                            'max': float(np.max(passing_returns) * 100),
                            'std': float(np.std(passing_returns) * 100)
                        },
                        'mdd': {
                            'mean': avg_mdd_pct,
                            'median': float(abs(np.median(passing_mdd)) * 100),
                            'min': float(abs(np.max(passing_mdd)) * 100),  # max because values are negative
                            'max': float(abs(np.min(passing_mdd)) * 100),  # min because values are negative
                            'std': float(np.std(passing_mdd) * 100)
                        },
                        'sharpe_approx': avg_sharpe
                    }
                }
            else:
                # No simulations passed this threshold
                stats = {
                    'threshold_pct': threshold_pct,
                    'num_passed': 0,
                    'num_total': int(num_sims),
                    'pass_rate_pct': 0.0,
                    'stats': None,
                    'metrics': None
                }
            
            threshold_stats.append(stats)
        
        # Calculate overall pass rate (at least one threshold passed)
        overall_passed = np.sum(max_drawdowns >= -thresholds[-1] / 100)  # Use strictest threshold
        overall_pass_rate = (overall_passed / num_sims) * 100
        
        return {
            'thresholds': threshold_stats,
            'overall_pass_rate_pct': float(overall_pass_rate),
            'num_simulations': int(num_sims)
        }
    
    def _calculate_percentile_curves(self, mc_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Calculate percentile curves for cone chart
        
        Returns 5%, 25%, 50%, 75%, 95% percentiles at each time step
        """
        sample_curves = np.array(mc_results['sample_curves'])
        
        if len(sample_curves) == 0:
            return {}
        
        # Calculate percentiles at each time step
        percentiles = {
            'p5': np.percentile(sample_curves, 5, axis=0).tolist(),
            'p25': np.percentile(sample_curves, 25, axis=0).tolist(),
            'p50': np.percentile(sample_curves, 50, axis=0).tolist(),
            'p75': np.percentile(sample_curves, 75, axis=0).tolist(),
            'p95': np.percentile(sample_curves, 95, axis=0).tolist()
        }
        
        return percentiles
    
    def _create_histogram_data(
        self,
        values: np.ndarray,
        bins: int = 50
    ) -> Dict[str, Any]:
        """Create histogram data for frontend"""
        # Auto-adjust bins if data range is too small
        data_range = np.max(values) - np.min(values)
        
        if data_range == 0:
            # All values are the same
            return {
                'counts': [len(values)],
                'bin_edges': [float(values[0]) - 0.5, float(values[0]) + 0.5],
                'bin_centers': [float(values[0])],
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': 0.0
            }
        
        # Limit bins to reasonable number based on data
        max_bins = min(bins, len(values) // 2, int(data_range / 0.01) + 1)
        max_bins = max(max_bins, 5)  # At least 5 bins
        
        counts, bin_edges = np.histogram(values, bins=max_bins)
        
        # Calculate bin centers for display
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return {
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': bin_centers.tolist(),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values))
        }
    
    def run_quick_analysis(
        self,
        pnl_list: List[float],
        num_simulations: int = 100
    ) -> Dict[str, Any]:
        """
        Run quick analysis with fewer simulations
        
        Useful for preview/testing
        
        Args:
            pnl_list: List of P&L percentages
            num_simulations: Number of simulations (default: 100)
            
        Returns:
            Simplified results
        """
        # Temporarily override config
        original_sims = self.config.num_simulations
        self.config.num_simulations = num_simulations
        
        try:
            results = self.run_full_analysis(pnl_list)
            return results
        finally:
            # Restore original config
            self.config.num_simulations = original_sims


# Export all classes
__all__ = [
    'SimulationConfig',
    'MonteCarloEngine',
    'BootstrapEngine',
    'RiskAnalyzer',
    'VerdictEngine',
    'MonteCarloSimulator'
]
