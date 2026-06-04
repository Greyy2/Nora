
import os
import sys
import numpy as np
import logging

# Add paths
sys.path.append("/home/vmc01/vinh/noraquantengine/QuantaAlpha")
sys.path.append("/home/vmc01/vinh/noraquantengine/Sova")

from quantaalpha.miner.market_intelligence_layer import MarketIntelligenceLayer
from sova_matching_learning import StructuralMLJudge, MarketDNA, BacktestMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyV5.2")

def test_mil_capping():
    logger.info("--- Testing MIL IC Capping ---")
    mil = MarketIntelligenceLayer()
    import pandas as pd
    
    # Create fake "perfectly overfit" data (e.g., linear trend)
    close = np.linspace(100, 200, 300)
    volume = np.ones(300)
    df = pd.DataFrame({"close": close, "volume": volume})
    
    result = mil.analyze(df, symbol="TEST_OVERFIT")
    logger.info(f"MIL IC Scores: {result['ic_scores']}")
    logger.info(f"Overfit Risk Families: {result['overfit_risk']}")
    
    # Assert No score exceeds 0.1 (absolute)
    for f, score in result['ic_scores'].items():
        if abs(score) > 0.1001:
             raise ValueError(f"MIL Failed: {f} score {score} > 0.1")
    logger.info("✅ MIL IC Capping Verified.")

def test_judge_overfit_penalty():
    logger.info("--- Testing ML Judge Overfit Penalty ---")
    judge = StructuralMLJudge()
    
    # Case 1: Elite IC (0.07)
    metrics_elite = BacktestMetrics(IC=0.07, Sharpe=2.5, MDD=-0.1)
    fitness_elite = metrics_elite.compute_fitness()
    
    # Case 2: Overfit IC (0.12)
    metrics_overfit = BacktestMetrics(IC=0.12, Sharpe=4.0, MDD=-0.05)
    fitness_overfit = metrics_overfit.compute_fitness()
    
    logger.info(f"Elite Fitness (IC=0.07): {fitness_elite:.4f}")
    logger.info(f"Overfit Fitness (IC=0.12): {fitness_overfit:.4f}")
    
    if fitness_overfit > fitness_elite:
        raise ValueError("ML Judge Failed: Overfit formula (IC=0.12) has higher fitness than Elite formula (IC=0.07)")
    
    logger.info("✅ ML Judge Overfit Penalty Verified (0.1 IC Wall enforced).")

def test_heuristic_forecast_cap():
    logger.info("--- Testing Heuristic Forecast Cap ---")
    judge = StructuralMLJudge()
    # Mock some "too good" features
    # structural_features: [n_tokens, n_rank, n_ts_std, n_ts_corr, n_delta, n_ts_mean, ...]
    # Give it lots of TS_CORR and DELTA to push score high
    features = [10.0, 1.0, 1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0]
    # Add regime and market features (zeros for simplicity)
    features += [0.0] * 7 + [0.0] * 9
    
    score = judge._heuristic_forecast(features, "EXPONENTIAL_BULL", None)
    logger.info(f"Heuristic Forecast Score: {score:.4f}")
    if score > 0.1001:
        raise ValueError(f"Heuristic Forecast Failed: Score {score} > 0.1")
    logger.info("✅ Heuristic Forecast Capping Verified.")

if __name__ == "__main__":
    try:
        test_mil_capping()
        test_judge_overfit_penalty()
        test_heuristic_forecast_cap()
        logger.info("\n🎉 V5.2 ELITE IC CONVERGENCE VERIFIED.")
    except Exception as e:
        logger.error(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
