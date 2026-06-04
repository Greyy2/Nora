from __future__ import annotations

import os
from pathlib import Path


_SOVA_HOME = Path(__file__).resolve().parent
_QUANTA_ALPHA_HOME = _SOVA_HOME.parent / "QuantaAlpha"

# V3: Centralized data storage under QuantaAlpha/data/sova_memory
_DEFAULT_MEMORY_ROOT = _QUANTA_ALPHA_HOME / "data" / "sova_memory"
_DEFAULT_VORTEX_ROOT = _QUANTA_ALPHA_HOME / "data" / "sova_memory" / "vortex"


def get_memory_root() -> Path:
    raw = os.environ.get("SOVA_MEMORY_ROOT", "").strip()
    root = Path(raw).expanduser() if raw else _DEFAULT_MEMORY_ROOT
    return root.resolve()


def get_vortex_root() -> Path:
    raw = os.environ.get("SOVA_VORTEX_ROOT", "").strip()
    root = Path(raw).expanduser() if raw else _DEFAULT_VORTEX_ROOT
    return root.resolve()


def memory_path(*parts: str) -> Path:
    p = get_memory_root()
    for part in parts:
        p = p / part
    return p


def vortex_path(*parts: str) -> Path:
    p = get_vortex_root()
    for part in parts:
        p = p / part
    return p
