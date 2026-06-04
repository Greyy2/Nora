"""Quanta/SOVA integration routes.

These endpoints allow the Grey frontend (or any client) to trigger the
QuantaAlpha dual-flow smoketest and poll for status/logs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import settings
from services.task_manager import task_manager

router = APIRouter(prefix="/api/ai/quanta", tags=["quanta-sova"])

class QuantaRunRequest(BaseModel):
    """Run QuantaAlpha dual-flow smoke test (stock+forex) in background."""
    max_rounds: int = Field(2, ge=1, le=20)
    stock_attempts: int = Field(3, ge=1, le=20)
    verify_beam: Optional[int] = Field(None, ge=1, le=64)
    forex_mdd_cap: Optional[float] = Field(None, ge=0.01, le=0.50)


def _get_task_or_404(task_id: str) -> dict:
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.post("/run")
async def run_quanta(request: QuantaRunRequest, background_tasks: BackgroundTasks):
    task_id = task_manager.create_task("quanta_smoketest")
    
    # Standardized output directory using core.config
    output_dir = settings.RESULT_DIR / f"sova_dual_flow_api_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "SOVA_REQUIRE_REAL_BACKTEST": "1",
        "SOVA_REFLECTION": "1",
        "SOVA_GENERALIZATION_GATES": "1",
        "SOVA_GENERALIZATION_REQUIRE_ON_READY": "1"
    }

    if request.verify_beam is not None:
        env["SOVA_VERIFY_BEAM"] = str(int(request.verify_beam))
    if request.forex_mdd_cap is not None:
        env["SOVA_FOREX_MDD_CAP"] = str(float(request.forex_mdd_cap))

    cmd = [
        sys.executable,
        "-u",
        str((settings.QUANTA_ALPHA_ROOT / "run_dual_flow_smoketest.py").resolve()),
        "--max-rounds",
        str(int(request.max_rounds)),
        "--stock-attempts",
        str(int(request.stock_attempts)),
        "--output-dir",
        str(output_dir),
    ]

    task_manager.update_task(task_id, cmd=cmd, output_dir=str(output_dir))
    
    # Launch in background
    background_tasks.add_task(
        task_manager.run_subprocess, 
        task_id=task_id, 
        cmd=cmd, 
        env=env
    )

    return {
        "success": True,
        "job_id": task_id,
        "task_id": task_id,
        "status_url": f"/api/ai/quanta/progress/{task_id}",
        "logs_url": f"/api/ai/quanta/logs/{task_id}",
        "result_url": f"/api/ai/quanta/result/{task_id}",
    }

@router.get("/progress/{task_id}")
async def get_quanta_progress(task_id: str):
    task = _get_task_or_404(task_id)
    return {"success": True, "data": task}


@router.get("/logs/{task_id}")
async def get_quanta_logs(task_id: str, tail: int = Query(250, ge=1, le=5000)):
    _get_task_or_404(task_id)
    logs = task_manager.get_task_logs(task_id, last_lines=tail)
    return {"success": True, "job_id": task_id, "tail": tail, "log": logs}


@router.get("/result/{task_id}")
async def get_quanta_result(task_id: str):
    task = _get_task_or_404(task_id)
    output_dir = (task.get("metadata") or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="output_dir not available")

    summary_path = Path(output_dir) / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="summary.json not ready")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse summary.json: {e}")

    return {
        "success": True,
        "job_id": task_id,
        "output_dir": output_dir,
        "summary": summary,
    }


@router.post("/stop/{task_id}")
async def stop_quanta(task_id: str):
    _get_task_or_404(task_id)
    task_manager.cancel_task(task_id)
    return {"success": True, "job_id": task_id, "message": "Stop requested"}

@router.delete("/cancel/{task_id}")
async def cancel_quanta(task_id: str):
    _get_task_or_404(task_id)
    task_manager.cancel_task(task_id)
    return {"success": True, "message": "Task cancellation requested"}
