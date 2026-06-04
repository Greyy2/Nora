from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Any, List
import os
from urllib.parse import urlparse, urlunparse


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
ENV_FILE = PROJECT_ROOT / ".env"

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on", "dev", "debug", "development"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off", "prod", "production", "release"}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def _is_docker_backend() -> bool:
    # Use an explicit flag; generic container detection is noisy in dev/CI shells.
    return _coerce_bool(os.getenv("DOCKER_BACKEND"), False)


def _normalize_mongo_uri(uri: str) -> str:
    """Keep docker hostnames inside Docker, map them to local exposed port outside."""
    if _is_docker_backend() or not uri:
        return uri

    parsed = urlparse(uri)
    if parsed.scheme not in {"mongodb", "mongodb+srv"}:
        return uri
    if parsed.hostname not in {"mongodb", "grey-mongodb"}:
        return uri

    auth_prefix = ""
    if "@" in parsed.netloc:
        auth_prefix = parsed.netloc.rsplit("@", 1)[0] + "@"

    return urlunparse(parsed._replace(netloc=f"{auth_prefix}localhost:27020"))

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        case_sensitive=True,
        extra="ignore",
    )

    # Project Identity
    APP_NAME: str = "Grey Backend API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    DOCKER_BACKEND: bool = False

    # Infrastructure Paths
    # Grey/backend/core/config.py -> parents[2] is Grey/
    BACKEND_ROOT: Path = BACKEND_ROOT
    PROJECT_ROOT: Path = PROJECT_ROOT
    
    # In Docker: /app/backend/core/config.py -> PROJECT_ROOT = /app
    # WORKSPACE_ROOT would be / which is wrong for Docker
    # Use environment variable or fallback to PROJECT_ROOT for Docker compatibility
    WORKSPACE_ROOT: Path = Path(os.getenv("WORKSPACE_ROOT", str(PROJECT_ROOT.parent if PROJECT_ROOT.parent != Path("/") else PROJECT_ROOT)))

    # Standardized Data & Result Paths (defaults to Grey/data in this unified layout)
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    RESULT_DIR: Path = DATA_DIR / "results"
    
    # Legacy/Local paths for Grey specifically
    GREY_TMP_DIR: Path = PROJECT_ROOT / "tmp"
    GREY_RESULTS_DIR: Path = PROJECT_ROOT / "results"

    # AI & External Engines
    QUANTA_ALPHA_ROOT: Path = Path(
        os.getenv("QUANTA_ALPHA_ROOT", str(PROJECT_ROOT / "backend" / "engines" / "quanta"))
    )
    SOVA_ROOT: Path = Path(
        os.getenv("SOVA_ROOT", str(PROJECT_ROOT / "backend" / "engines" / "sova"))
    )
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Security & Artifacts
    # Mount point for the frontend to access results
    ARTIFACT_MOUNT_PATH: str = "/vinh" 
    
    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27020/vinh"
    MONGO_DB: str = "vinh"

    # Resource Management
    MAX_WORKERS: int = int(os.getenv("GREY_MAX_WORKERS", "4"))

    @field_validator("DEBUG", "DOCKER_BACKEND", mode="before")
    @classmethod
    def parse_boolish_env(cls, value: Any) -> bool:
        return _coerce_bool(value, False)

    @field_validator("MONGO_URI", mode="after")
    @classmethod
    def normalize_local_mongo_uri(cls, value: str) -> str:
        return _normalize_mongo_uri(value)

settings = Settings()

# Ensure critical directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
settings.GREY_TMP_DIR.mkdir(parents=True, exist_ok=True)
settings.GREY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
