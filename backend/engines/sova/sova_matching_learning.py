"""
SOVA LEFT BRAIN: The Neuron Matrix (ML Core)
Role: Signal Processing, Reinforced Memory, Tactical Governance, Predictive Intelligence
Architecture: Multi-Layer Neural Perception + Synaptic Vortex Memory + Structural ML Judge
Intelligence Level: Senior AI Agent (QuantaAlpha/RD-Agent Grade)
"""

import os
import json
import re
import math
import logging
import hashlib
import importlib
import warnings
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from collections import defaultdict, deque
from pathlib import Path

from sova_paths import memory_path, get_vortex_root

warnings.filterwarnings('ignore')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    raise ImportError("NumPy is required for SOVA. Install: pip install numpy")

try:
    from scipy import stats as scipy_stats
    from scipy.fft import fft, ifft
    from scipy.signal import welch, find_peaks, butter, filtfilt
    from scipy.linalg import svd
    from scipy.spatial.distance import euclidean
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    lgb = importlib.import_module("lightgbm")
    HAS_LGB = True
except ImportError:
    lgb = None
    HAS_LGB = False

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SOVA.LeftBrain")

@dataclass
class BacktestMetrics:
    ID: str = ""
    IC: float = 0.0
    ICIR: float = 0.0
    RankIC: float = 0.0
    Sharpe: float = 0.0
    Sortino: float = 0.0
    Calmar: float = 0.0
    MDD: float = 0.0
    AnnRet: float = 0.0
    WinRate: float = 0.0
    ProfitFactor: float = 0.0
    Expectancy: float = 0.0
    Fitness: float = 0.0
    Complexity: int = 0
    Novelty: float = 1.0
    
    def compute_fitness(self) -> float:
        # TIES BREAKER: Use more decimal points and penalties
        ic_component = abs(self.IC) * 2.0 + abs(self.RankIC) * 1.5
        sharpe_component = max(0, self.Sharpe) * 1.0
        
        # Realism Penalty: can be disabled to avoid capping high-IC factors
        disable_overfit_penalty = str(os.environ.get("SOVA_DISABLE_OVERFIT_PENALTY", "1") or "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        overfit_penalty = 1.0
        if not disable_overfit_penalty:
            if abs(self.IC) > 0.1:
                # v5.2 Elite: Steep penalty for breaking the 0.1 barrier
                overfit_penalty = 0.1
            elif abs(self.IC) > 0.08:
                # Subtle penalty for entering the 'High Risk' zone
                overfit_penalty = 0.8
            
        risk_penalty = (1.0 + abs(self.MDD) * 3.0)
        # Complexity penalty: steeper for long expressions
        complexity_penalty = 1.0 + (self.Complexity / 50.0)**2
        
        self.Fitness = (
            (ic_component + sharpe_component) 
            * self.Novelty 
            * overfit_penalty
            / (risk_penalty * complexity_penalty)
        )
        # Add a tiny epsilon based on complexity to break remaining ties (prefer simpler)
        self.Fitness += (1.0 / (self.Complexity + 1)) * 1e-6
        
        return self.Fitness

@dataclass
class MemoryImpression:
    Regime: str
    Expression: str
    Metrics: BacktestMetrics
    Timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    Reinforcement: float = 1.0
    AccessCount: int = 0
    LastAccess: str = field(default_factory=lambda: datetime.now().isoformat())
    ASTSignature: str = ""
    SemanticHash: str = ""
    ParentExpressions: List[str] = field(default_factory=list)
    EvolutionGeneration: int = 0

@dataclass
class MarketDNA:
    regime: str
    volatility: float
    hurst: float
    fractal_dimension: float
    spectral_entropy: float
    trend: float
    volume_surge: float
    microstructure_efficiency: float
    momentum_persistence: float
    mean_reversion_strength: float
    regime_confidence: float
    regime_transition_probability: float
    adaptive_window_size: int
    regime_duration_estimate: int
    volatility_regime: str
    trend_regime: str
    liquidity_regime: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SignalDenoiser:
    @staticmethod
    def kalman_filter(data: np.ndarray, x0: float = 0, p0: float = 1, q: float = 0.001, r: float = 0.1) -> np.ndarray:
        n = len(data)
        xhat = np.zeros(n)
        p = np.zeros(n)
        xhat[0], p[0] = x0, p0
        for k in range(1, n):
            xhat_minus = xhat[k-1]
            p_minus = p[k-1] + q
            k_gain = p_minus / (p_minus + r)
            xhat[k] = xhat_minus + k_gain * (data[k] - xhat_minus)
            p[k] = (1 - k_gain) * p_minus
        return xhat
    
    @staticmethod
    def wavelet_denoise(data: np.ndarray, threshold_scale: float = 0.5) -> np.ndarray:
        if not HAS_SCIPY:
            return data
        coeffs = fft(data)
        threshold = threshold_scale * np.median(np.abs(coeffs))
        coeffs[np.abs(coeffs) < threshold] = 0
        return np.real(ifft(coeffs))
    
    @staticmethod
    def butterworth_filter(data: np.ndarray, cutoff: float = 0.1, order: int = 3) -> np.ndarray:
        if len(data) < 20 or not HAS_SCIPY:
            return data
        try:
            b, a = butter(order, cutoff, btype='low')
            return filtfilt(b, a, data)
        except:
            return data

class AdvancedSignalProcessor:
    @staticmethod
    def hurst_exponent(ts: np.ndarray) -> float:
        if len(ts) < 20:
            return 0.5
        lags = range(2, min(20, len(ts) // 3))
        tau = np.array([np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags])
        if len(tau) < 3 or np.all(tau == 0):
            return 0.5
        poly = np.polyfit(np.log(list(lags)), np.log(tau + 1e-10), 1)
        return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
    
    @staticmethod
    def fractal_dimension(ts: np.ndarray) -> float:
        if len(ts) < 20:
            return 1.0
        n = len(ts)
        x = np.arange(2, min(11, n // 10))
        lk = []
        for k in x:
            lm = []
            for m in range(k):
                segment = ts[m::k]
                if len(segment) > 1:
                    l_mk = np.sum(np.abs(np.diff(segment))) * (n - 1) / (len(segment) * k)
                    lm.append(l_mk)
            if lm:
                lk.append(np.mean(lm))
        if len(lk) < 2:
            return 1.0
        lk = np.array(lk)
        return float(np.clip(np.polyfit(np.log(1/x[:len(lk)]), np.log(lk + 1e-10), 1)[0], 0.5, 2.0))
    
    @staticmethod
    def spectral_entropy(ts: np.ndarray) -> float:
        if not HAS_SCIPY or len(ts) < 10:
            return 0.0
        try:
            f, psd = welch(ts, nperseg=min(len(ts)//2, 256))
            psd_norm = psd / (np.sum(psd) + 1e-12)
            return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        except:
            return 0.0
    
    @staticmethod
    def dominant_frequency(ts: np.ndarray) -> float:
        if not HAS_SCIPY or len(ts) < 10:
            return 0.0
        try:
            f, psd = welch(ts, nperseg=min(len(ts)//2, 256))
            return float(f[np.argmax(psd)])
        except:
            return 0.0
    
    @staticmethod
    def autocorrelation_structure(ts: np.ndarray, max_lag: int = 20) -> np.ndarray:
        if len(ts) < max_lag + 5:
            return np.zeros(max_lag)
        acf = np.correlate(ts - np.mean(ts), ts - np.mean(ts), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf[:max_lag+1] / (acf[0] + 1e-10)
        return acf[1:]
    
    @staticmethod
    def lyapunov_exponent(ts: np.ndarray, lag: int = 1) -> float:
        if len(ts) < 50:
            return 0.0
        n = len(ts) - lag
        divergences = np.abs(ts[lag:] - ts[:n])
        divergences = divergences[divergences > 1e-10]
        if len(divergences) < 10:
            return 0.0
        return float(np.mean(np.log(divergences + 1e-10)))

class NeuralPerceptionLayer:
    def __init__(self):
        self.denoiser = SignalDenoiser()
        self.processor = AdvancedSignalProcessor()
        self.perception_history: deque = deque(maxlen=50)
        
    def sense(self, price: np.ndarray, volume: Optional[np.ndarray] = None, 
              high: Optional[np.ndarray] = None, low: Optional[np.ndarray] = None,
              single_asset_mode: bool = False) -> MarketDNA:
        if not HAS_NUMPY or len(price) < 100:
            return self._default_dna()
        
        # ELITE UPGRADE: Support single-asset (Forex/XAU) high-precision sensing
        if single_asset_mode:
            logger.info("[PERCEPTION] Operating in Single-Asset (XAU/Forex) Mastery mode.")
        
        rets = np.diff(np.log(price + 1e-10))
        rets_clean = self.denoiser.kalman_filter(rets)
        
        vol_60 = np.std(rets[-60:]) * math.sqrt(252)
        vol_20 = np.std(rets[-20:]) * math.sqrt(252)
        vol_5 = np.std(rets[-5:]) * math.sqrt(252)
        
        hurst = self.processor.hurst_exponent(price[-100:])
        fd = self.processor.fractal_dimension(price[-100:])
        entropy = self.processor.spectral_entropy(rets[-60:])
        lyap = self.processor.lyapunov_exponent(rets[-60:])
        
        ma5 = np.mean(price[-5:])
        ma20 = np.mean(price[-20:])
        ma60 = np.mean(price[-60:])
        trend = (ma5 / ma60) - 1.0
        
        v_surge = 1.0
        if volume is not None and len(volume) >= 20:
            v_surge = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-9)
        
        microstructure_eff = 1.0
        if high is not None and low is not None and len(high) >= 20:
            true_range = np.maximum(high[-20:] - low[-20:], 
                                   np.abs(high[-20:] - np.roll(price[-20:], 1)))
            close_to_close_move = np.abs(np.diff(price[-21:]))
            microstructure_eff = np.mean(close_to_close_move) / (np.mean(true_range) + 1e-9)
        
        acf = self.processor.autocorrelation_structure(rets_clean[-60:], max_lag=10)
        momentum_persistence = float(np.mean(acf[:5]))
        mean_reversion_strength = float(-np.min(acf))
        
        regime, regime_conf = self._classify_regime(
            vol_60, hurst, fd, entropy, trend, v_surge, lyap, momentum_persistence,
            single_asset_mode=single_asset_mode
        )
        
        vol_regime = self._classify_volatility_regime(vol_5, vol_20, vol_60)
        trend_regime = self._classify_trend_regime(trend, ma5, ma20, ma60)
        liquidity_regime = self._classify_liquidity_regime(v_surge, microstructure_eff)
        
        transition_prob = self._estimate_transition_probability(regime, regime_conf)
        adaptive_window = self._compute_adaptive_window(vol_20, hurst)
        duration_est = self._estimate_regime_duration(regime, regime_conf)
        
        dna = MarketDNA(
            regime=regime,
            volatility=vol_60,
            hurst=hurst,
            fractal_dimension=fd,
            spectral_entropy=entropy,
            trend=trend,
            volume_surge=v_surge,
            microstructure_efficiency=microstructure_eff,
            momentum_persistence=momentum_persistence,
            mean_reversion_strength=mean_reversion_strength,
            regime_confidence=regime_conf,
            regime_transition_probability=transition_prob,
            adaptive_window_size=adaptive_window,
            regime_duration_estimate=duration_est,
            volatility_regime=vol_regime,
            trend_regime=trend_regime,
            liquidity_regime=liquidity_regime
        )
        
        self.perception_history.append(dna)
        return dna
    
    def _classify_regime(self, vol: float, hurst: float, fd: float, entropy: float,
                        trend: float, v_surge: float, lyap: float, momentum: float,
                        single_asset_mode: bool = False) -> Tuple[str, float]:
        regime_scores = defaultdict(float)
        
        # ELITE UPGRADE: Forex/XAU Specific Regimes
        if single_asset_mode:
            # News shock / High Volatility Spike
            if vol > 0.60 and abs(trend) > 0.10:
                regime_scores["NEWS_SHOCK_VOLATILITY"] = 0.95
            
            # Liquidity gaps / Weekend gaps (simulated via high vol/low surge)
            if v_surge < 0.5 and vol > 0.40:
                regime_scores["LIQUIDITY_GAP_RISK"] = 0.85

        # Standard / Hybrid Regimes
        if vol > 0.50 and trend < -0.08 and v_surge > 2.0:
            regime_scores["CAPITULATION_CRASH"] = 0.9
        if vol > 0.45 and lyap < -1.5 and trend < -0.05:
            regime_scores["CAPITULATION_CRASH"] += 0.7
        
        if hurst > 0.70 and trend > 0.06 and momentum > 0.3:
            regime_scores["EXPONENTIAL_BULL"] = 0.85
        if vol < 0.25 and hurst > 0.65 and trend > 0.04:
            regime_scores["EXPONENTIAL_BULL"] += 0.6
        
        if fd > 1.45 or entropy > 2.8:
            regime_scores["CHAOTIC_NOISE"] = 0.8
        if lyap > -0.5 and entropy > 2.5:
            regime_scores["CHAOTIC_NOISE"] += 0.5
        
        if hurst < 0.38 and abs(trend) < 0.03:
            regime_scores["MEAN_REVERSION_ZONE"] = 0.75
        if momentum < 0 and entropy < 1.5:
            regime_scores["MEAN_REVERSION_ZONE"] += 0.6
        
        if v_surge > 1.8 and abs(trend) < 0.025 and vol > 0.30:
            regime_scores["DISTRIBUTION_ANOMALY"] = 0.7
        
        if 0.45 < hurst < 0.60 and 0.02 < trend < 0.08 and vol < 0.35:
            regime_scores["STABLE_ACCUMULATION"] = 0.65
        
        if 0.45 < hurst < 0.60 and -0.08 < trend < -0.02 and vol < 0.35:
            regime_scores["STABLE_EROSION"] = 0.65
        
        if not regime_scores:
            if trend > 0:
                regime_scores["STABLE_ACCUMULATION"] = 0.3
            else:
                regime_scores["STABLE_EROSION"] = 0.3
        
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        return best_regime[0], min(best_regime[1], 1.0)
    
    def _classify_volatility_regime(self, vol5: float, vol20: float, vol60: float) -> str:
        vol_expanding = vol5 > vol20 > vol60
        vol_contracting = vol5 < vol20 < vol60
        
        if vol60 > 0.60:
            return "HIGH_VOLATILITY_CRISIS" if vol_expanding else "HIGH_VOLATILITY_STABLE"
        elif vol60 > 0.35:
            return "MEDIUM_VOLATILITY_EXPANDING" if vol_expanding else "MEDIUM_VOLATILITY"
        else:
            return "LOW_VOLATILITY_COMPRESSION" if vol_contracting else "LOW_VOLATILITY"
    
    def _classify_trend_regime(self, trend: float, ma5: float, ma20: float, ma60: float) -> str:
        if ma5 > ma20 > ma60 and trend > 0.05:
            return "STRONG_UPTREND"
        elif ma5 > ma20 and trend > 0.02:
            return "EMERGING_UPTREND"
        elif ma5 < ma20 < ma60 and trend < -0.05:
            return "STRONG_DOWNTREND"
        elif ma5 < ma20 and trend < -0.02:
            return "EMERGING_DOWNTREND"
        else:
            return "SIDEWAYS_CONSOLIDATION"
    
    def _classify_liquidity_regime(self, v_surge: float, microstructure: float) -> str:
        if v_surge > 2.5:
            return "PANIC_LIQUIDITY_SURGE"
        elif v_surge > 1.5:
            return "HIGH_LIQUIDITY"
        elif v_surge < 0.6 and microstructure < 0.7:
            return "THIN_LIQUIDITY_CRISIS"
        elif v_surge < 0.8:
            return "LOW_LIQUIDITY"
        else:
            return "NORMAL_LIQUIDITY"
    
    def _estimate_transition_probability(self, regime: str, confidence: float) -> float:
        if confidence > 0.80:
            return 0.10
        elif confidence > 0.60:
            return 0.25
        elif confidence > 0.40:
            return 0.50
        else:
            return 0.75
    
    def _compute_adaptive_window(self, vol: float, hurst: float) -> int:
        base_window = 20
        vol_factor = 1.0 + (vol / 0.30)
        hurst_factor = 1.0 if hurst > 0.5 else 1.5
        window = int(base_window * vol_factor * hurst_factor)
        return np.clip(window, 10, 60)

    def interpret_regime_with_llm(self, dna: "MarketDNA") -> str:
        """
        Ask the real LLM to interpret the market regime and explain its implications
        for alpha factor design. This replaces the hard-coded STRATEGIC_AXIOMS strings
        with actual LLM reasoning based on the computed market metrics.

        Called AFTER if-else classification (which stays for speed), to add
        genuine understanding on top of the label.

        Returns:
            A string of LLM-generated regime context (logged and attached to the
            hypothesis for downstream use). Falls back to empty string on failure.
        """
        try:
            from sova_cloud_brain import QwenFreeClient
        except ImportError:
            return ""

        system = (
            "You are a senior quantitative analyst. You interpret market microstructure metrics "
            "and explain their implications for factor alpha research. Be concise and precise. "
            "Use mathematical reasoning. Respond in ENGLISH only."
        )

        prompt = f"""Analyze the following market state and provide a brief regime interpretation.

## Computed Market Metrics
- Regime Label: {dna.regime}
- Regime Confidence: {dna.regime_confidence:.2f}
- Volatility (annualized): {dna.volatility:.3f} | Regime: {dna.volatility_regime}
- Hurst Exponent: {dna.hurst:.3f}  (>0.5=trend-persistent, <0.5=mean-reverting, =0.5=random)
- Fractal Dimension: {dna.fractal_dimension:.3f}
- Spectral Entropy: {dna.spectral_entropy:.3f}  (higher = more stochastic/chaotic)
- Trend Slope: {dna.trend:.4f} | Trend Regime: {dna.trend_regime}
- Volume Surge: {dna.volume_surge:.2f}x  (>1.5 = elevated institutional activity)
- Momentum Persistence: {dna.momentum_persistence:.3f}
- Mean Reversion Strength: {dna.mean_reversion_strength:.3f}
- Regime Transition Probability: {dna.regime_transition_probability:.2f}

## Answer these 3 questions in 2–3 sentences total:
1. WHY are these specific metrics producing the "{dna.regime}" label?
2. What type of alpha signal (momentum, mean-reversion, volatility, volume) is MOST exploitable now?
3. What is the primary risk factor for any alpha operating in this regime?
"""
        try:
            client = QwenFreeClient()
            response = client.chat(prompt, system)
            if response:
                logger.info(f"[LLM-Regime] {dna.regime}:\n{response[:400]}")
            return response or ""
        except Exception as e:
            logger.warning(f"[LLM-Regime] Failed: {e}")
            return ""

    
    def _estimate_regime_duration(self, regime: str, confidence: float) -> int:
        duration_map = {
            "CAPITULATION_CRASH": 5,
            "EXPONENTIAL_BULL": 30,
            "CHAOTIC_NOISE": 10,
            "MEAN_REVERSION_ZONE": 15,
            "DISTRIBUTION_ANOMALY": 8,
            "STABLE_ACCUMULATION": 25,
            "STABLE_EROSION": 20
        }
        base_duration = duration_map.get(regime, 15)
        return int(base_duration * confidence)
    
    def _default_dna(self) -> MarketDNA:
        return MarketDNA(
            regime="NEUTRAL",
            volatility=0.30,
            hurst=0.50,
            fractal_dimension=1.20,
            spectral_entropy=1.50,
            trend=0.0,
            volume_surge=1.0,
            microstructure_efficiency=1.0,
            momentum_persistence=0.0,
            mean_reversion_strength=0.0,
            regime_confidence=0.30,
            regime_transition_probability=0.50,
            adaptive_window_size=20,
            regime_duration_estimate=15,
            volatility_regime="MEDIUM_VOLATILITY",
            trend_regime="SIDEWAYS_CONSOLIDATION",
            liquidity_regime="NORMAL_LIQUIDITY"
        )

class EnsembleJudge:
    """
    ELITE UPGRADE: multi-model ensemble for high-precision Alpha quality prediction.
    Uses an ensemble of LightGBM models with different loss functions.
    """
    def __init__(self):
        self.models: List[Any] = []
        # Each item: (features, target_ic, sample_weight)
        self.training_data: List[Tuple[List[float], float, float]] = []
        self.persist_training: bool = os.environ.get("SOVA_JUDGE_PERSIST", "1").strip().lower() not in {"0", "false", "no"}
        self.training_data_path = Path(
            os.environ.get("SOVA_JUDGE_TRAINING_DATA", str(memory_path("judge_training.jsonl")))
        )
        self.feature_importance: Dict[str, float] = {}
        self.last_update = 0
        self.failure_memory_path = memory_path("failure_learning.json")
        self.failure_cases: List[Dict[str, Any]] = []
        self.strategy_outcomes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success": 1.0, "failure": 1.0, "avg_improvement": 0.0, "count": 0.0}
        )
        self._load_failure_memory()
        self._load_training_data()

    def _load_training_data(self) -> None:
        if not self.persist_training:
            return
        try:
            if not self.training_data_path.exists():
                return
            loaded: List[Tuple[List[float], float, float]] = []
            # Load last ~4000 lines defensively; keep most recent 2000 samples.
            with self.training_data_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-4000:]
            for line in lines:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                feats = rec.get("features")
                ic = rec.get("ic")
                w = rec.get("weight", 1.0)
                if not isinstance(feats, list) or not feats:
                    continue
                try:
                    feats_f = [float(x) for x in feats]
                    ic_f = float(ic)
                    w_f = float(w)
                except Exception:
                    continue
                loaded.append((feats_f, ic_f, float(np.clip(w_f, 0.0, 10.0))))

            if loaded:
                self.training_data = loaded[-2000:]
                # Warm-start: train immediately if enough samples.
                if HAS_LGB and len(self.training_data) >= 120:
                    self._retrain_ensemble()
                    logger.info(f"[JUDGE] Loaded {len(self.training_data)} persisted samples from {self.training_data_path}.")
        except Exception as e:
            logger.warning(f"[JUDGE] Failed to load training data: {e}")

    def _append_training_example(
        self,
        expr: str,
        regime: str,
        features: List[float],
        ic: float,
        weight: float,
        dna: Optional[MarketDNA] = None,
    ) -> None:
        if not self.persist_training:
            return
        try:
            self.training_data_path.parent.mkdir(parents=True, exist_ok=True)
            rec: Dict[str, Any] = {
                "ts": datetime.now().isoformat(),
                "expr": str(expr or ""),
                "regime": str(regime or ""),
                "ic": float(ic),
                "weight": float(weight),
                "features": [float(x) for x in (features or [])],
            }
            if dna is not None:
                rec["dna"] = {
                    "volatility": float(getattr(dna, "volatility", 0.0) or 0.0),
                    "hurst": float(getattr(dna, "hurst", 0.0) or 0.0),
                    "fractal_dimension": float(getattr(dna, "fractal_dimension", 0.0) or 0.0),
                    "spectral_entropy": float(getattr(dna, "spectral_entropy", 0.0) or 0.0),
                    "trend": float(getattr(dna, "trend", 0.0) or 0.0),
                    "volume_surge": float(getattr(dna, "volume_surge", 0.0) or 0.0),
                    "momentum_persistence": float(getattr(dna, "momentum_persistence", 0.0) or 0.0),
                    "mean_reversion_strength": float(getattr(dna, "mean_reversion_strength", 0.0) or 0.0),
                    "regime_confidence": float(getattr(dna, "regime_confidence", 0.0) or 0.0),
                }

            with self.training_data_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[JUDGE] Failed to persist training example: {e}")
        
    def _extract_motifs(self, expr: str, regime: str, dna: Optional[MarketDNA] = None) -> List[float]:
        tokens = re.findall(r'[A-Z_]+\(|\$\w+|[\d.]+', expr)
        structural_features = [
            float(len(tokens)),
            float(expr.count("RANK")),
            float(expr.count("TS_STD")),
            float(expr.count("TS_CORR")),
            float(expr.count("DELTA")),
            float(expr.count("TS_MEAN")),
            float(expr.count("Ref")),
            float(expr.count("*")),
            float(expr.count("/")),
            float(len(set(re.findall(r'\$\w+', expr)))),
            float(len(set(re.findall(r'[A-Z_]+', expr)))),
            float(expr.count("(")),
        ]
        
        regimes = [
            "CAPITULATION_CRASH", "EXPONENTIAL_BULL", "CHAOTIC_NOISE", 
            "MEAN_REVERSION_ZONE", "DISTRIBUTION_ANOMALY", 
            "STABLE_ACCUMULATION", "STABLE_EROSION"
        ]
        regime_features = [1.0 if r == regime else 0.0 for r in regimes]
        
        market_features = [0.0] * 9
        if dna:
            market_features = [
                dna.volatility, dna.hurst, dna.fractal_dimension,
                dna.spectral_entropy, dna.trend, dna.volume_surge,
                dna.momentum_persistence, dna.mean_reversion_strength,
                dna.regime_confidence
            ]
        
        return structural_features + regime_features + market_features
    
    def update(
        self,
        expr: str,
        regime: str,
        ic: float,
        dna: Optional[MarketDNA] = None,
        sample_weight: float = 1.0,
    ):
        features = self._extract_motifs(expr, regime, dna)
        try:
            w = float(sample_weight)
        except Exception:
            w = 1.0
        w = float(np.clip(w, 0.0, 10.0))
        self.training_data.append((features, float(ic), w))
        self._append_training_example(expr, regime, features, float(ic), w, dna)
        
        # Keep window of 2000 samples
        if len(self.training_data) > 2000:
            self.training_data = self.training_data[-2000:]
        
        # Retrain every 20 samples once we reach 100
        if len(self.training_data) >= 100 and HAS_LGB and (len(self.training_data) % 20 == 0):
            self._retrain_ensemble()
    
    def _retrain_ensemble(self):
        X = np.array([d[0] for d in self.training_data], dtype=float)
        y = np.array([d[1] for d in self.training_data], dtype=float)
        w = np.array([d[2] for d in self.training_data], dtype=float)

        # Time-ordered holdout (no shuffle) to avoid overfitting on the latest distribution.
        n = len(y)
        if n < 120:
            return
        n_valid = max(50, int(n * 0.2))
        split = max(1, n - n_valid)
        X_train, X_valid = X[:split], X[split:]
        y_train, y_valid = y[:split], y[split:]
        w_train, w_valid = w[:split], w[split:]
        
        self.models = []
        objectives = ["huber", "fair", "regression"]

        base_seed = int(os.environ.get("SOVA_JUDGE_SEED", "42") or "42")
        num_round = int(os.environ.get("SOVA_JUDGE_NUM_BOOST_ROUND", "600") or "600")
        es_rounds = int(os.environ.get("SOVA_JUDGE_EARLY_STOP", "60") or "60")

        for obj in objectives:
            params = {
                "objective": obj,
                "metric": "rmse",
                "verbosity": -1,
                "learning_rate": float(os.environ.get("SOVA_JUDGE_LR", "0.05") or "0.05"),
                "num_leaves": int(os.environ.get("SOVA_JUDGE_NUM_LEAVES", "31") or "31"),
                "max_depth": int(os.environ.get("SOVA_JUDGE_MAX_DEPTH", "6") or "6"),
                "min_data_in_leaf": int(os.environ.get("SOVA_JUDGE_MIN_DATA_IN_LEAF", "40") or "40"),
                "feature_fraction": float(os.environ.get("SOVA_JUDGE_FEATURE_FRACTION", "0.80") or "0.80"),
                "bagging_fraction": float(os.environ.get("SOVA_JUDGE_BAGGING_FRACTION", "0.80") or "0.80"),
                "bagging_freq": int(os.environ.get("SOVA_JUDGE_BAGGING_FREQ", "1") or "1"),
                "lambda_l1": float(os.environ.get("SOVA_JUDGE_L1", "0.0") or "0.0"),
                "lambda_l2": float(os.environ.get("SOVA_JUDGE_L2", "2.0") or "2.0"),
                "seed": base_seed + len(self.models),
                "feature_fraction_seed": base_seed + 11 + len(self.models),
                "bagging_seed": base_seed + 23 + len(self.models),
            }

            try:
                ds_train = lgb.Dataset(X_train, label=y_train, weight=w_train)
                ds_valid = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, reference=ds_train)
                model = lgb.train(
                    params,
                    ds_train,
                    num_boost_round=num_round,
                    valid_sets=[ds_valid],
                    callbacks=[lgb.early_stopping(stopping_rounds=es_rounds, verbose=False)],
                )
                self.models.append(model)
            except Exception as e:
                logger.error(f"[ENSEMBLE] Model {obj} failed: {e}")
        
        logger.info(
            f"[JUDGE] Ensemble trained on {len(self.training_data)} samples (train={len(y_train)}, valid={len(y_valid)}) "
            f"with {len(self.models)} models."
        )

    # Regime-calibrated base IC (v5.2 Elite: Targeting 0.05-0.08 range)
    _REGIME_BASE_IC = {
        "EXPONENTIAL_BULL":    0.045, "STABLE_ACCUMULATION": 0.040,
        "MEAN_REVERSION_ZONE": 0.035, "DISTRIBUTION_ANOMALY": 0.030,
        "STABLE_EROSION":      0.028, "NEUTRAL":             0.030,
        "CAPITULATION_CRASH":  0.025, "CHAOTIC_NOISE":        0.015,
        "TREND_FOLLOWING":     0.042, "MOMENTUM_FLOW":        0.040,
    }

    def _heuristic_forecast(self, features: list, regime: str, dna: Optional[Any]) -> float:
        """Domain-knowledge heuristic - CHÂN THỰC đánh giá (không chia đều điểm).
        
        Philosophy: Cố gắng estimate IC chính xác, KHÔNG cố tình balance distribution.
        Low-quality formulas sẽ có low scores (phải chịu), high-quality có high scores.
        """
        n_tokens  = features[0]
        n_rank    = features[1]
        n_ts_std  = features[2]
        n_ts_corr = features[3]
        n_delta   = features[4]
        n_ts_mean = features[5]
        n_depth   = features[11]  # open-paren count ≈ nesting depth

        # Base IC: Thực tế từ Qlib benchmarks (KHÔNG inflate để chia đều)
        score = self._REGIME_BASE_IC.get(regime, 0.013)

        # Operator quality weights - AGGRESSIVE scoring
        # Good operators get BIG bonus, bad ones get PENALTY
        score += min(n_rank, 1)  * 0.006   # RANK caps at 1x (more = redundant noise)
        score += n_ts_corr       * 0.012   # TS_CORR is structurally informative (BOOST)
        score += n_delta         * 0.008   # captures momentum (BOOST)
        score += n_ts_std        * 0.005   # variance normalisation
        score += min(n_ts_mean, 2) * 0.003   # TS_MEAN caps at 2x (more = over-smoothing)

        # Complexity penalties - EXTREMELY STRICT (bad formulas bị penalty cực nặng)
        if n_tokens > 18:
            score -= (n_tokens - 18) * 0.004  # 4x penalty for complexity
        if n_rank > 1:
            # Nested RANK is FATAL - exponential penalty
            score -= (n_rank - 1) ** 2 * 0.015  # Quadratic penalty: 2nd RANK = -0.015, 3rd = -0.060
        if n_depth > 8:
            score -= (n_depth - 8) ** 1.5 * 0.005  # Super-linear penalty for deep nesting
        if n_ts_mean > 2:
            # Excessive smoothing penalty (4x TS_MEAN is useless)
            score -= (n_ts_mean - 2) ** 2 * 0.012

        # Market-state adaptive bonus - AGGRESSIVE rewards for regime-fit
        if dna:
            if dna.hurst < 0.45 and n_ts_mean > 0:
                score += 0.010  # Mean-reversion regime → TS_MEAN bonus (BOOST)
            if dna.hurst > 0.55 and n_delta > 0:
                score += 0.009  # Trending regime → DELTA bonus (BOOST)
            if dna.volatility > 0.02 and n_ts_std > 0:
                score += 0.006  # High vol → volatility operators bonus
            
            # SEVERE penalty for regime mismatch (bad formulas phải chịu)
            if dna.hurst > 0.55 and n_ts_mean > n_delta:  # Trending but using mean-reversion ops
                score -= 0.015  # SEVERE penalty
            if dna.hurst < 0.45 and n_delta > n_ts_mean:  # Mean-reversion but using momentum ops
                score -= 0.015  # SEVERE penalty

        # Final clip: allow optional IC cap via env (disabled by default)
        disable_ic_cap = str(os.environ.get("SOVA_DISABLE_IC_CAP", "1") or "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        if not disable_ic_cap:
            # v5.2 Elite Target: [0.05 - 0.08].
            # Anything > 0.1 is suspect and capped to discourage overfitting.
            if score > 0.10:
                logger.debug(f"[JUDGE] Heuristic forecast {score:.4f} capped at 0.1 (Overfit Risk).")
                return 0.10
            return float(np.clip(score, 0.001, 0.10))
        return float(np.clip(score, 0.001, 1.0))

    def forecast_fitness(self, expr: str, regime: str, dna: Optional[MarketDNA] = None) -> float:
        features = self._extract_motifs(expr, regime, dna)
        if not self.models or not HAS_LGB:
            return self._heuristic_forecast(features, regime, dna)

        try:
            feat_arr = np.array(features).reshape(1, -1)
            predictions = [m.predict(feat_arr)[0] for m in self.models]
            avg_pred = np.mean(predictions)
            return float(np.clip(avg_pred, -0.5, 0.5))
        except Exception:
            return self._heuristic_forecast(features, regime, dna)

    def diagnose_failure(self, expr: str, regime: str, ic: float,
                         icir: float = 0.0, arr: float = 0.0, mdd: float = 0.0,
                         dna: Optional[MarketDNA] = None) -> Dict[str, Any]:
        """Learn from failed backtests and explain why the factor failed.

        Returns a structured diagnosis with root causes and ranked refinement strategies.
        """
        features = self._extract_motifs(expr, regime, dna)
        n_tokens = int(features[0])
        n_rank = int(features[1])
        n_ts_std = int(features[2])
        n_ts_corr = int(features[3])
        n_delta = int(features[4])
        n_ts_mean = int(features[5])
        n_mul = int(features[7])
        n_div = int(features[8])
        n_vars = int(features[9])
        n_depth = int(features[11])

        causes: List[Tuple[str, float, str]] = []
        preferred: List[str] = []

        if n_tokens > 22 or n_depth > 6 or n_mul + n_div > 3:
            severity = min(0.95, 0.45 + (n_tokens - 18) * 0.02 + max(0, n_depth - 6) * 0.05)
            causes.append((
                "complexity_overload",
                severity,
                "Formula is too long/deep, so signal quality is diluted and generalization collapses."
            ))
            preferred.extend(["simplify_to_core", "ast_prune_deepest", "collapse_redundant", "strip_outer_layer"])

        if n_rank > 1:
            causes.append((
                "redundant_normalization",
                0.78,
                "Nested ranking layers suppress economically meaningful magnitude information."
            ))
            preferred.extend(["strip_outer_layer", "collapse_redundant"])

        if dna and dna.hurst > 0.55 and n_ts_mean > n_delta:
            causes.append((
                "regime_mismatch_trend",
                0.82,
                "Market is trend-dominant but expression leans too heavily on smoothing/mean-reversion structure."
            ))
            preferred.extend(["switch_operator_family", "add_cross_variable", "change_primary_variable"])

        if dna and dna.hurst < 0.45 and n_delta > n_ts_mean:
            causes.append((
                "regime_mismatch_reversion",
                0.82,
                "Market is mean-reverting but expression is momentum-heavy and reacts in the wrong direction."
            ))
            preferred.extend(["switch_operator_family", "widen_windows", "change_primary_variable"])

        if n_vars <= 1 and ic < 0.02:
            causes.append((
                "single_channel_signal",
                0.66,
                "Expression depends on only one information channel, so it misses confirmation from flow, volatility, or cross-variable structure."
            ))
            preferred.extend(["add_cross_variable", "change_primary_variable"])

        if n_ts_std == 0 and abs(mdd) > 0.15:
            causes.append((
                "missing_risk_normalization",
                0.74,
                "Factor lacks volatility normalization and therefore scales exposure poorly in stressed regimes."
            ))
            preferred.extend(["add_volatility_norm", "add_rank_wrapper"])

        if ic >= 0.02 and icir < 0.30:
            causes.append((
                "phase_specific_alpha",
                0.80,
                "Signal works in isolated phases, but stability across windows/regimes is weak."
            ))
            preferred.extend(["widen_windows", "add_volatility_norm", "add_rank_wrapper"])

        if abs(ic) <= 0.005:
            causes.append((
                "pure_noise",
                0.88,
                "Observed IC is indistinguishable from noise; current operator family is not extracting a real edge."
            ))
            preferred.extend(["rebuild_from_hypothesis", "switch_operator_family", "flip_direction"])

        if not causes:
            causes.append((
                "weak_signal",
                0.60,
                "Signal is directionally plausible but lacks enough predictive strength to clear deployment thresholds."
            ))
            preferred.extend(["change_primary_variable", "add_cross_variable", "shorten_windows"])

        causes.sort(key=lambda x: x[1], reverse=True)
        preferred_ranked = self._rank_strategies(preferred, regime, causes)

        diagnosis = {
            "primary_cause": causes[0][0],
            "confidence": round(causes[0][1], 3),
            "root_causes": [
                {"name": name, "severity": round(severity, 3), "explanation": explanation}
                for name, severity, explanation in causes[:4]
            ],
            "preferred_strategies": preferred_ranked,
            "recommended_actions": self._build_recommended_actions(causes, preferred_ranked, expr, dna),
            "failure_signature": {
                "tokens": n_tokens,
                "depth": n_depth,
                "variables": n_vars,
                "rank_layers": n_rank,
                "momentum_ops": n_delta,
                "mean_ops": n_ts_mean,
                "risk_ops": n_ts_std,
                "cross_ops": n_ts_corr,
            },
        }

        self.failure_cases.append({
            "expr": expr[:180],
            "regime": regime,
            "ic": ic,
            "icir": icir,
            "arr": arr,
            "mdd": mdd,
            "diagnosis": diagnosis,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.failure_cases) > 500:
            self.failure_cases = self.failure_cases[-400:]
        self._persist_failure_memory()
        return diagnosis

    def record_refinement_result(self, diagnosis: str, strategy: str,
                                 old_ic: float, new_ic: float, regime: str):
        """Learn whether a refinement strategy actually improved the factor."""
        key = f"{regime}:{diagnosis}:{strategy}"
        stats = self.strategy_outcomes[key]
        improvement = new_ic - old_ic
        stats["count"] += 1.0
        stats["avg_improvement"] = (
            (stats["avg_improvement"] * (stats["count"] - 1.0)) + improvement
        ) / stats["count"]
        if improvement > 0.002:
            stats["success"] += 1.0
        else:
            stats["failure"] += 1.0
        self._persist_failure_memory()

    def _rank_strategies(self, strategies: List[str], regime: str,
                         causes: List[Tuple[str, float, str]]) -> List[str]:
        unique_strategies = []
        for strategy in strategies:
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)
        scored: List[Tuple[str, float]] = []
        primary = causes[0][0] if causes else "weak_signal"
        for strategy in unique_strategies:
            key = f"{regime}:{primary}:{strategy}"
            stats = self.strategy_outcomes[key]
            posterior_mean = stats["success"] / (stats["success"] + stats["failure"])
            score = posterior_mean + stats["avg_improvement"] * 8.0
            scored.append((strategy, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [strategy for strategy, _ in scored[:5]]

    def _build_recommended_actions(self, causes: List[Tuple[str, float, str]],
                                   strategies: List[str], expr: str,
                                   dna: Optional[MarketDNA]) -> List[str]:
        actions: List[str] = []
        primary = causes[0][0] if causes else "weak_signal"
        if primary == "complexity_overload":
            actions.append("Reduce formula depth to <= 4 and keep only the strongest two operator layers.")
        if primary.startswith("regime_mismatch") and dna:
            if dna.hurst > 0.55:
                actions.append("Shift toward momentum/trend operators and reduce excess smoothing in this trending regime.")
            else:
                actions.append("Shift toward mean-reversion / range-position operators for this reverting regime.")
        if primary == "missing_risk_normalization":
            actions.append("Normalize the core signal by TS_STD or another volatility proxy before ranking.")
        if primary == "single_channel_signal":
            actions.append("Add a second information channel, typically volume or return, to confirm the primary signal.")
        if primary == "pure_noise":
            actions.append("Do not over-tune this exact structure; change operator family or rebuild from hypothesis.")
        if not actions:
            actions.append("Refine the factor using the top-ranked strategy and retest on the same regime slice.")
        if strategies:
            actions.append(f"Strategy priority: {', '.join(strategies[:3])}.")
        return actions[:4]

    def _load_failure_memory(self):
        if not self.failure_memory_path.exists():
            return
        try:
            with open(self.failure_memory_path, "r") as src:
                data = json.load(src)
            self.failure_cases = data.get("failure_cases", [])[-400:]
            raw_stats = data.get("strategy_outcomes", {})
            for key, value in raw_stats.items():
                self.strategy_outcomes[key] = {
                    "success": float(value.get("success", 1.0)),
                    "failure": float(value.get("failure", 1.0)),
                    "avg_improvement": float(value.get("avg_improvement", 0.0)),
                    "count": float(value.get("count", 0.0)),
                }
        except Exception as e:
            logger.warning(f"[JUDGE] Failed to load failure memory: {e}")

    def _persist_failure_memory(self):
        try:
            self.failure_memory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failure_memory_path, "w") as dst:
                json.dump(
                    {
                        "failure_cases": self.failure_cases[-400:],
                        "strategy_outcomes": dict(self.strategy_outcomes),
                    },
                    dst,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"[JUDGE] Failed to persist failure memory: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  GPT-5.1 AXIS 3: FAILURE DNA
#  Tracks structural hashes of failed formula patterns to prevent re-generation.
#  "Elite ML never repeats its mistakes."
# ═══════════════════════════════════════════════════════════════════════════════

class FailureDNA:
    """
    Maintains a persistent registry of failed formula structural signatures.
    When a formula fails (IC < 0.01), its structural skeleton is hashed and stored.
    Future HypothesisGenerator calls check against this DNA to avoid regenerating
    the same structure.
    """

    SAVE_PATH_NAME = "failure_dna.json"

    def __init__(self, vortex_path: Optional["Path"] = None):
        from pathlib import Path as _Path
        if vortex_path:
            self._file = _Path(vortex_path) / self.SAVE_PATH_NAME
        else:
            try:
                self._file = get_vortex_root() / self.SAVE_PATH_NAME
            except Exception:
                import tempfile
                self._file = _Path(tempfile.gettempdir()) / self.SAVE_PATH_NAME
        self._dna: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        try:
            if self._file.exists():
                with open(self._file, "r") as f:
                    self._dna = json.load(f)
        except Exception:
            self._dna = {}

    def _save(self):
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._file, "w") as f:
                json.dump(self._dna, f, indent=2)
        except Exception:
            pass

    def _structural_hash(self, expression: str) -> str:
        """
        Extract the structural skeleton of a formula (operators + nesting depth),
        ignoring specific variable names and window values.
        E.g., RANK(TS_CORR($close, $volume, 10)) → RANK_TS_CORR_2depth
        """
        import re as _re
        # Remove all variable names and numbers → keep only operators
        skeleton = _re.sub(r'\$\w+', 'VAR', expression)
        skeleton = _re.sub(r'\b\d+(?:\.\d+)?\b', 'N', skeleton)
        skeleton = _re.sub(r'[\s,]', '', skeleton)
        # Compute nesting depth
        depth = max(skeleton.count('('), 1)
        key = f"{skeleton[:80]}_d{depth}"
        return key

    def record_failure(self, expression: str, ic: float, regime: str = ""):
        """Record a failed formula's structural DNA."""
        h = self._structural_hash(expression)
        self._dna[h] = {
            "ic": round(ic, 5),
            "regime": regime,
            "expression_sample": expression[:120],
            "count": self._dna.get(h, {}).get("count", 0) + 1,
        }
        self._save()
        logger.debug(f"[FailureDNA] Recorded failure: {h[:50]} IC={ic:.4f}")

    def is_failed_structure(self, expression: str, ic_threshold: float = 0.01) -> bool:
        """Return True if this expression matches a known failing structural pattern."""
        h = self._structural_hash(expression)
        if h in self._dna:
            prev_ic = self._dna[h].get("ic", 0.0)
            if prev_ic < ic_threshold:
                logger.debug(f"[FailureDNA] Blocked duplicate structure: {h[:50]}")
                return True
        return False

    def failure_count(self) -> int:
        return len(self._dna)


# ═══════════════════════════════════════════════════════════════════════════════
#  GPT-5.1 AXIS 3: ENSEMBLE MEMORY BANK
#  Learns generalization patterns from the 10 most recent alphas.
#  Unlike individual memory, this synthesizes cross-alpha knowledge.
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleMemoryBank:
    """
    Meta-learning layer that synthesizes knowledge from the N most recent alphas.
    Produces:
    - operator_success_rates: which operators succeed most often
    - window_success_rates: which windows work best
    - regime_insights: per-regime success patterns
    - avoid_operators: operators with persistent failure rates
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._recent_alphas: List[Dict[str, Any]] = []  # Each entry: {expression, ic, mdd, regime, ...}

    def record(self, expression: str, ic: float, mdd: float = 0.0,
               regime: str = "", icir: float = 0.0):
        """Record a completed alpha evaluation."""
        entry = {
            "expression": expression,
            "ic": ic,
            "mdd": mdd,
            "regime": regime,
            "icir": icir,
            "success": ic >= 0.05,  # v5.2 Elite: Institutional success starts at 0.05
        }
        self._recent_alphas.append(entry)
        # Keep only the last N
        if len(self._recent_alphas) > self.max_history:
            self._recent_alphas.pop(0)

    def synthesize(self) -> Dict[str, Any]:
        """
        Synthesize learning from all recorded alphas.
        Returns a context dict usable as feedback hints.
        """
        if not self._recent_alphas:
            return {}

        import re as _re
        op_success: Dict[str, List[float]] = {}
        op_failure_count: Dict[str, int] = {}

        for alpha in self._recent_alphas:
            # Extract operators used in expression
            ops = _re.findall(r'\b([A-Z][A-Z_0-9]+)\b', alpha["expression"])
            ops = [o for o in ops if o not in ("VAR", "N", "RANK") or True]
            for op in set(ops):
                op_success.setdefault(op, [])
                op_success[op].append(alpha["ic"])
                if not alpha["success"]:
                    op_failure_count[op] = op_failure_count.get(op, 0) + 1

        # Compute average IC per operator
        op_avg_ic = {op: sum(ics) / len(ics) for op, ics in op_success.items() if ics}

        # Rank operators by avg IC
        best_ops = sorted(op_avg_ic, key=op_avg_ic.get, reverse=True)[:5]
        worst_ops = [op for op, cnt in op_failure_count.items() if cnt >= 2]

        # Recent trend: are we improving or degrading?
        recent_ics = [a["ic"] for a in self._recent_alphas[-5:]]
        trend = "improving" if len(recent_ics) >= 2 and recent_ics[-1] > recent_ics[0] else "stagnating"

        return {
            "best_operators": best_ops,
            "avoid_operators": worst_ops,
            "average_ic": sum(a["ic"] for a in self._recent_alphas) / len(self._recent_alphas),
            "success_rate": sum(1 for a in self._recent_alphas if a["success"]) / len(self._recent_alphas),
            "trend": trend,
            "n_alphas_evaluated": len(self._recent_alphas),
        }

    def get_operator_hints(self) -> Tuple[List[str], List[str]]:
        """Return (preferred_operators, avoid_operators) based on ensemble learning."""
        synthesis = self.synthesize()
        return synthesis.get("best_operators", []), synthesis.get("avoid_operators", [])


class SynapticVortexMemory:
    def __init__(self, path: Optional[str] = None):
        self.path = get_vortex_root() if not path else Path(path)
        self.path.mkdir(exist_ok=True)
        self.vortex: Dict[str, List[MemoryImpression]] = {}
        self.global_hashes: Set[str] = set()
        self.regime_statistics: Dict[str, Dict[str, Any]] = {}
        # GPT-5.1 Axis 3: Failure DNA prevents structural repetition
        self.failure_dna = FailureDNA(vortex_path=self.path)
        # GPT-5.1 Axis 3: Ensemble memory synthesizes cross-alpha knowledge
        self.ensemble_bank = EnsembleMemoryBank(max_history=10)
        self._hydrate()

    
    def _hydrate(self):
        for f in self.path.glob("*.json"):
            regime = f.stem.upper()
            try:
                with open(f, "r") as src:
                    data = json.load(src)
                    self.vortex[regime] = [
                        MemoryImpression(
                            Regime=e.get('Regime', regime),
                            Expression=e['Expression'],
                            Metrics=BacktestMetrics(**e['Metrics']) if isinstance(e['Metrics'], dict) else e['Metrics'],
                            Timestamp=e.get('Timestamp', datetime.now().isoformat()),
                            Reinforcement=e.get('Reinforcement', 1.0),
                            AccessCount=e.get('AccessCount', 0),
                            LastAccess=e.get('LastAccess', datetime.now().isoformat()),
                            ASTSignature=e.get('ASTSignature', ''),
                            SemanticHash=e.get('SemanticHash', ''),
                            ParentExpressions=e.get('ParentExpressions', []),
                            EvolutionGeneration=e.get('EvolutionGeneration', 0)
                        )
                        for e in data
                    ]
                    for imp in self.vortex[regime]:
                        self.global_hashes.add(imp.SemanticHash)
                logger.info(f"[VORTEX] Loaded {len(self.vortex[regime])} from {regime}")
            except Exception as e:
                logger.error(f"[VORTEX] Failed to load {f}: {e}")
    
    def imprint(self, impression: MemoryImpression):
        r = impression.Regime.upper()
        if r not in self.vortex:
            self.vortex[r] = []
        
        impression.SemanticHash = self._compute_semantic_hash(impression.Expression)
        impression.ASTSignature = self._extract_ast_signature(impression.Expression)
        
        norm_expr = re.sub(r'\s+', '', impression.Expression.lower())
        found = False
        
        for synapse in self.vortex[r]:
            synapse_norm = re.sub(r'\s+', '', synapse.Expression.lower())
            if synapse_norm == norm_expr:
                synapse.Reinforcement += 0.25
                synapse.Metrics.IC = (synapse.Metrics.IC * 0.65) + (impression.Metrics.IC * 0.35)
                synapse.Metrics.Sharpe = (synapse.Metrics.Sharpe * 0.65) + (impression.Metrics.Sharpe * 0.35)
                synapse.Metrics.compute_fitness()
                synapse.Timestamp = impression.Timestamp
                synapse.AccessCount += 1
                synapse.LastAccess = impression.Timestamp
                found = True
                logger.info(f"[VORTEX] Reinforced in {r}: R={synapse.Reinforcement:.2f}")
                break
        
        if not found:
            impression.Metrics.compute_fitness()
            self.vortex[r].append(impression)
            self.global_hashes.add(impression.SemanticHash)
            logger.info(f"[VORTEX] New memory in {r}: Fitness={impression.Metrics.Fitness:.4f}")
        
        if len(self.vortex[r]) > 200:
            self.vortex[r].sort(key=lambda x: (x.Metrics.Fitness * x.Reinforcement), reverse=True)
            removed = self.vortex[r][150:]
            for imp in removed:
                if imp.SemanticHash in self.global_hashes:
                    self.global_hashes.remove(imp.SemanticHash)
            self.vortex[r] = self.vortex[r][:150]
            logger.info(f"[VORTEX] Pruned {r}: kept 150")
        
        self._persist(r)
    
    def recall(self, regime: str, top_n: int = 10, diversity_boost: float = 0.2) -> List[MemoryImpression]:
        r_upper = regime.upper()
        candidates = list(self.vortex.get(r_upper, []))
        
        if len(candidates) < 5:
            analogies = self._get_analogous_regimes(r_upper)
            for analog in analogies:
                analog_candidates = self.vortex.get(analog, [])
                candidates.extend([c for c in analog_candidates])
        
        for c in candidates:
            c.AccessCount += 1
            c.LastAccess = datetime.now().isoformat()
        
        scored_candidates = []
        for c in candidates:
            fitness_score = c.Metrics.Fitness * c.Reinforcement
            try:
                recency_bonus = 1.0 / (1.0 + (datetime.now() - datetime.fromisoformat(c.Timestamp)).days / 30.0)
            except:
                recency_bonus = 1.0
            diversity_penalty = 1.0 - (diversity_boost * (candidates.count(c) / len(candidates)))
            
            total_score = fitness_score * (1.0 + recency_bonus * 0.1) * diversity_penalty
            scored_candidates.append((c, total_score))
        
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [c for c, _ in scored_candidates[:top_n]]
        
        logger.info(f"[VORTEX] Recalled {len(selected)} from {r_upper}")
        return selected
    
    def _get_analogous_regimes(self, regime: str) -> List[str]:
        similar_map = {
            "CAPITULATION_CRASH": ["DISTRIBUTION_ANOMALY", "STABLE_EROSION"],
            "EXPONENTIAL_BULL": ["STABLE_ACCUMULATION"],
            "CHAOTIC_NOISE": ["MEAN_REVERSION_ZONE", "DISTRIBUTION_ANOMALY"],
            "MEAN_REVERSION_ZONE": ["CHAOTIC_NOISE", "STABLE_ACCUMULATION"],
            "DISTRIBUTION_ANOMALY": ["EXPONENTIAL_BULL", "CAPITULATION_CRASH"],
            "STABLE_ACCUMULATION": ["MEAN_REVERSION_ZONE", "EXPONENTIAL_BULL"],
            "STABLE_EROSION": ["CAPITULATION_CRASH", "DISTRIBUTION_ANOMALY"]
        }
        return similar_map.get(regime, [])
    
    def _compute_semantic_hash(self, expression: str) -> str:
        normalized = re.sub(r'\s+', '', expression.lower())
        normalized = re.sub(r'\d+', 'N', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _extract_ast_signature(self, expression: str) -> str:
        operators = re.findall(r'[A-Z_]+', expression)
        return "_".join(sorted(set(operators)))
    
    def _persist(self, regime: str):
        filepath = self.path / f"{regime.lower()}.json"
        try:
            with open(filepath, "w") as dst:
                data = [
                    {
                        **asdict(imp),
                        'Metrics': asdict(imp.Metrics)
                    }
                    for imp in self.vortex[regime]
                ]
                json.dump(data, dst, indent=2)
        except Exception as e:
            logger.error(f"[VORTEX] Failed to persist {regime}: {e}")
    
    def get_statistics(self, regime: str) -> Dict[str, Any]:
        r = regime.upper()
        if r not in self.vortex:
            return {}
        
        memories = self.vortex[r]
        ics = [m.Metrics.IC for m in memories]
        sharpes = [m.Metrics.Sharpe for m in memories]
        fitnesses = [m.Metrics.Fitness for m in memories]
        
        return {
            "total_memories": len(memories),
            "avg_ic": np.mean(ics) if ics else 0.0,
            "max_ic": np.max(ics) if ics else 0.0,
            "avg_sharpe": np.mean(sharpes) if sharpes else 0.0,
            "max_sharpe": np.max(sharpes) if sharpes else 0.0,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "max_fitness": np.max(fitnesses) if fitnesses else 0.0
        }

# Backward-compatibility alias for tests that used the old class name
StructuralMLJudge = EnsembleJudge

class GovernanceOracle:
    def __init__(self):
        self.fingerprints: Set[str] = set()
        self.signal_hashes: Set[str] = set() # NEW: Semantic Deduplication
        self.ast_cache: Dict[str, Any] = {}
        
    def validate(self, expression: str, factor_values: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        if not expression or not expression.strip():
            return False, "Empty expression"
        
        # 1. Syntax Check
        tokens = re.findall(r'[A-Z_]+\(|\$\w+|[\d.]+', expression)
        if not tokens:
            return False, "No valid tokens found"
        if len(tokens) > 80:
            return False, "Entropy Exceeded (>80 tokens)"
        if expression.count("(") != expression.count(")"):
            return False, "Parentheses mismatch"
        if expression.count("RANK(RANK(") > 0:
            return False, "Redundant nested RANK"
        
        # 2. Syntax Duplicate Check
        fp = hashlib.md5(re.sub(r'\s+', '', expression.lower()).encode()).hexdigest()
        if fp in self.fingerprints:
            return False, "Duplicate pattern (Syntax)"
        
        # 3. SEMANTIC DEDUPLICATION (NEW)
        # Check if the generated signal is identical to a previous one
        if factor_values is not None and len(factor_values) > 0:
            # Sample 200 points to create a representative signal hash
            indices = np.linspace(0, len(factor_values)-1, min(200, len(factor_values)), dtype=int)
            sample = factor_values[indices]
            # Use a fuzzy hash by rounding values to 4 decimals to catch marginal differences
            sig_hash = hashlib.md5(np.round(sample, 4).tobytes()).hexdigest()
            if sig_hash in self.signal_hashes:
                return False, "Duplicate signal (Semantic)"
            self.signal_hashes.add(sig_hash)
        
        # 4. Look-ahead Check
        if re.search(r'Ref\(.*,\s*-\d+\)', expression):
            return False, "Look-ahead bias detected (Negative Ref shift)"
        
        self.fingerprints.add(fp)
        
        # Cleanup
        if len(self.fingerprints) > 10000:
            self.fingerprints = set(list(self.fingerprints)[5000:])
        if len(self.signal_hashes) > 10000:
            self.signal_hashes = set(list(self.signal_hashes)[5000:])
            
        return True, "Valid"


class ICCalculator:
    @staticmethod
    def spearman_rank(x: np.ndarray, y: np.ndarray) -> float:
        if not HAS_SCIPY:
            return ICCalculator.pearson(x, y)
        
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if np.sum(mask) < 20:
            return 0.0
        
        try:
            rho, _ = scipy_stats.spearmanr(x[mask], y[mask])
            return float(rho) if not math.isnan(rho) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def pearson(x: np.ndarray, y: np.ndarray) -> float:
        mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if np.sum(mask) < 20:
            return 0.0
        
        x_clean, y_clean = x[mask], y[mask]
        if np.std(x_clean) == 0 or np.std(y_clean) == 0:
            return 0.0
        
        return float(np.corrcoef(x_clean, y_clean)[0, 1])

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("SOVA LEFT BRAIN Test")
    logger.info("="*80)
    
    perception = NeuralPerceptionLayer()
    vortex = SynapticVortexMemory()
    oracle = GovernanceOracle()
    judge = StructuralMLJudge()
    
    price = np.random.randn(300).cumsum() + 10000
    volume = np.random.rand(300) * 1000000
    high = price + np.random.rand(300) * 50
    low = price - np.random.rand(300) * 50
    
    dna = perception.sense(price, volume, high, low)
    logger.info(f"[TEST] DNA: {dna.regime} (conf={dna.regime_confidence:.2f})")
    
    test_alpha = "RANK(DELTA($close, 5))"
    ok, msg = oracle.validate(test_alpha)
    logger.info(f"[TEST] Validation: {ok} - {msg}")
    
    metrics = BacktestMetrics(IC=0.045, Sharpe=2.1, MDD=0.12)
    metrics.compute_fitness()
    
    impression = MemoryImpression(Regime=dna.regime, Expression=test_alpha, Metrics=metrics)
    vortex.imprint(impression)
    
    recalled = vortex.recall(dna.regime, top_n=5)
    logger.info(f"[TEST] Recalled {len(recalled)} memories")
    
    logger.info("="*80)
