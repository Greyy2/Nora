"""
Sova AI Chat Routes — /api/sova/*

Exposes the SovaAnalysisOrchestrator (3-role AI pipeline) to the Grey
frontend via a single POST endpoint: POST /api/sova/chat
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("Grey.SovaRoutes")

# ── Path bootstrap ────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parents[1]          # Grey/backend
_ENGINES_DIR = _BACKEND_DIR / "engines" / "quanta"
_FB_DIR      = _ENGINES_DIR / "frontend_backend"

for _p in [str(_ENGINES_DIR), str(_FB_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Lazy-load orchestrator ────────────────────────────────────────
_orch      = None
_orch_lock = threading.Lock()
_orch_err  = None


def _get_orch():
    global _orch, _orch_err
    if _orch is None:
        with _orch_lock:
            if _orch is None:
                try:
                    from sova_orchestrator import SovaAnalysisOrchestrator
                    # project_root = Grey/ (two levels above backend/)
                    project_root = _BACKEND_DIR.parent
                    _orch = SovaAnalysisOrchestrator(project_root=project_root)
                    logger.info("[SovaRoutes] Orchestrator loaded ✓  root=%s", project_root)
                except Exception as exc:
                    _orch_err = str(exc)
                    logger.error("[SovaRoutes] Failed to load orchestrator: %s", exc)
    return _orch


# ── Pydantic models ───────────────────────────────────────────────

class SovaChatContext(BaseModel):
    asset:            Optional[str]                  = None
    timeframe:        Optional[str]                  = None
    regime:           Optional[Dict[str, Any]]        = None
    backtest_summary: Optional[Dict[str, Any]]        = None
    candle_snapshot:  Optional[List[Dict[str, Any]]]  = None
    history:          Optional[List[Dict[str, Any]]]  = None  # last 5–10 messages


class SovaChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: Optional[SovaChatContext] = None


# ── Router ────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/sova", tags=["sova-ai"])


@router.post("/chat")
async def sova_chat(req: SovaChatRequest):
    """
    POST /api/sova/chat
    Accepts a user message + trading context.
    Returns a SovaAnalysis object with summary text and chart overlay schema.
    """
    orch = _get_orch()
    if orch is None:
        raise HTTPException(
            status_code=503,
            detail=f"Sova orchestrator not available: {_orch_err or 'unknown error'}",
        )

    ctx: Dict[str, Any] = {}
    if req.context:
        ctx = req.context.model_dump(exclude_none=True)

    try:
        result = await asyncio.to_thread(orch.analyze, req.message, ctx)
        return {
            "success": True,
            "data": result.to_dict(),
        }
    except Exception as exc:
        logger.exception("[SovaRoutes] analyze() error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
async def sova_health():
    """Quick liveness probe for Sova AI."""
    orch = _get_orch()
    return {
        "status": "ok" if orch else "degraded",
        "orchestrator": "loaded" if orch else f"error: {_orch_err}",
    }
