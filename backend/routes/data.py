"""Data metadata routes for Backtest/WFA selectors."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter

from services.cache_service import cache_get, cache_set

router = APIRouter(prefix="/api", tags=["data"])

PROJECT_ROOT = Path(__file__).parent.parent.parent
BASE_DATA_DIR = PROJECT_ROOT / "data"
_ASSET_TIMEFRAME_SUFFIX = re.compile(r"_(\d+[mhdw])$", re.IGNORECASE)


def _resolve_first_existing(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_dir():
                return candidate
        except Exception:
            continue
    return None


def _category_paths() -> Dict[str, Path]:
    """Return available category -> directory mapping.

    Priority:
    - Explicit env aliases
    - Well-known project folders
    - Auto-discovered folders inside Grey/data
    """
    mapping: Dict[str, Path] = {}

    default_okx = BASE_DATA_DIR / "OKX"
    default_forex = BASE_DATA_DIR / "forex"

    grey_alias = Path(os.getenv("GREY_DATA_ALIAS_GREY", str(default_okx))).expanduser()
    vinh_alias_env = os.getenv("GREY_DATA_ALIAS_VINH", "").strip()
    vinh_candidates = []
    if vinh_alias_env:
        vinh_candidates.append(Path(vinh_alias_env).expanduser())
    vinh_candidates.extend([
        BASE_DATA_DIR / "vinh",
        PROJECT_ROOT.parent / "data" / "vinh",
        PROJECT_ROOT.parent / "Gone" / "vinh" / "kema" / "data" / "OKX_split",
    ])
    vinh_alias = _resolve_first_existing(vinh_candidates)

    if grey_alias.exists() and grey_alias.is_dir():
        mapping["grey"] = grey_alias
    if default_okx.exists() and default_okx.is_dir():
        mapping["OKX"] = default_okx
    if default_forex.exists() and default_forex.is_dir():
        mapping["forex"] = default_forex
    if vinh_alias is not None:
        mapping["vinh"] = vinh_alias

    # Keep compatibility with any existing folders in Grey/data.
    try:
        for entry in BASE_DATA_DIR.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                mapping.setdefault(entry.name, entry)
    except Exception:
        pass

    return mapping


def _ordered_categories(keys: List[str]) -> List[str]:
    priority = ["vinh", "grey", "OKX", "forex"]
    present = set(keys)
    ordered = [name for name in priority if name in present]
    ordered.extend(sorted([name for name in keys if name not in set(priority)], key=str.lower))
    return ordered


def _extract_asset_name(stem: str) -> str:
    # Normalize BTCUSDT_4h -> BTCUSDT to avoid duplicate symbols per timeframe file.
    return _ASSET_TIMEFRAME_SUFFIX.sub("", stem)


@router.get("/categories")
async def get_categories() -> List[str]:
    """Return available categories, preferring vinh/grey first."""
    cache_key = "categories:v2"
    cached = cache_get("data", cache_key)
    if isinstance(cached, list):
        return cached

    paths = _category_paths()
    categories = _ordered_categories(list(paths.keys()))
    if not categories:
        categories = ["grey", "OKX", "forex"]

    cache_set("data", cache_key, categories, ttl_seconds=60)
    return categories


@router.get("/assets")
async def get_assets(category: str = "grey") -> List[str]:
    """Return unique assets for a category using cached directory scanning."""
    paths = _category_paths()
    if not paths:
        return []

    # Accept case-insensitive category input.
    normalized_map = {key.lower(): key for key in paths.keys()}
    selected_key = normalized_map.get((category or "").lower(), "grey" if "grey" in paths else "OKX")
    if selected_key not in paths:
        selected_key = next(iter(paths.keys()))

    target_dir = paths[selected_key]
    try:
        dir_mtime = int(target_dir.stat().st_mtime)
    except Exception:
        dir_mtime = 0

    cache_key = f"assets:v3:{selected_key}:{dir_mtime}"
    cached = cache_get("data", cache_key)
    if isinstance(cached, list):
        return cached

    try:
        asset_map: Dict[str, str] = {}
        with os.scandir(target_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.lower().endswith(".pkl"):
                    continue
                stem = _extract_asset_name(Path(entry.name).stem)
                if not stem:
                    continue
                lower = stem.lower()
                prev = asset_map.get(lower)
                if prev is None:
                    asset_map[lower] = stem
                    continue

                # Prefer higher-case variant (BTCUSDT over btcusdt).
                prev_upper = sum(1 for ch in prev if ch.isupper())
                curr_upper = sum(1 for ch in stem if ch.isupper())
                if curr_upper > prev_upper:
                    asset_map[lower] = stem

        assets = sorted(asset_map.values())
        cache_set("data", cache_key, assets, ttl_seconds=120)
        return assets
    except Exception as exc:
        print(f"Error scanning assets for category={selected_key}: {exc}")
        return []

@router.get("/timeframes")
async def get_timeframes() -> List[str]:
    """
    Get list of available timeframes (1h-23h, 1d)
    """
    timeframes = [f"{i}h" for i in range(1, 24)] # 1h to 23h
    timeframes.append("1d")
    return timeframes