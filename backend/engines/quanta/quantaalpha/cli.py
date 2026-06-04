"""
QuantaAlpha CLI entry.

Commands:
  quantaalpha mine       - run factor mining
  quantaalpha backtest   - run backtest
  quantaalpha ui         - start log Web UI
  quantaalpha health_check - environment health check
"""

from pathlib import Path
from dotenv import load_dotenv

# Load .env (prefer project root, fallback to cwd)
_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(".env")

from quantaalpha.utils.fire_compat import fire
from quantaalpha.pipeline.factor_mining import main as mine
from quantaalpha.pipeline.factor_backtest import main as backtest
from quantaalpha.pipeline.sova_dual_market import main as sova_dual
from quantaalpha.app.utils.health_check import health_check
from quantaalpha.app.utils.info import collect_info


def app():
    fire.Fire(
        {
            "mine": mine,
            "backtest": backtest,
            "sova_dual": sova_dual,
            "health_check": health_check,
            "collect_info": collect_info,
        }
    )


if __name__ == "__main__":
    app()
