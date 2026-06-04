import sys
import logging
import json
from pathlib import Path

# Setup paths
parent = Path(__file__).resolve().parent
sys.path.append(str(parent))
sys.path.append(str(parent.parent / "Sova"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("V6_Gates")

from quantaalpha.llm import sova_adapter

def test_audit_gates():
    logger.info("--- Testing V6 Audit Gates (Seed & Evolution Superiority) ---")
    import os
    os.environ["USE_SOVA"] = "1"
    os.environ["SOVA_REQUIRE_REAL_BACKTEST"] = "0"
    os.environ["SOVA_VERIFY_BEAM"] = "2"
    
    # We run 2 rounds to allow evolution to happen
    result_raw = sova_adapter.run_generation_training_backtest_cycle(
        system_prompt="You are a pure quant engine.",
        user_prompt="Find a momentum factor.",
        max_rounds=2,
    )
    result = json.loads(result_raw)
    
    audit = result.get("audit_gates", {})
    logger.info(f"Audit Gates Output: {json.dumps(audit, indent=2)}")
    
    if "seed_ic_verified" not in audit:
        raise ValueError("Missing Seed IC tracking")
    if "evolved_superiority_passed" not in audit:
        raise ValueError("Missing Evolution tracking")
        
    logger.info("✅ V6 Audit Gates Verified.")

if __name__ == "__main__":
    test_audit_gates()
