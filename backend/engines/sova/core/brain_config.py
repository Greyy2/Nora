from pydantic_settings import BaseSettings
from pathlib import Path
import os

class SovaSettings(BaseSettings):
    # Identity
    BRAIN_NAME: str = "SOVA Sovereign Intelligence"
    VERSION: str = "4.0.0"
    
    # Paths
    SOVA_ROOT: Path = Path(__file__).resolve().parents[1]
    WORKSPACE_ROOT: Path = SOVA_ROOT.parent
    MEMORY_DIR: Path = WORKSPACE_ROOT / "data" / "sova_memories"
    
    # Model Settings
    PRIMARY_MODEL: str = os.getenv("SOVA_MODEL", "gpt-4-turbo-preview")
    TEMPERATURE: float = 0.3 # Low temp for precise math
    
    # Optimization Logic
    USE_THOMPSON_SAMPLING: bool = True
    USE_VECTOR_MEMORY: bool = True
    MAX_REFINEMENT_ROUNDS: int = 3
    
    class Config:
        env_file = ".env"
        env_prefix = "SOVA_"

settings = SovaSettings()
settings.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
