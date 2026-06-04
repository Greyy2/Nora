"""
Grey Backend - Backtest & Optimization Routes
Standardized for Institutional Grade Performance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import settings
from database.mongo_service import MongoService
from optimize.history import create_batch, update_status
from services.optimization_service import generate_configs_logic, run_optimization_logic
from services.task_manager import TaskStatus, task_manager

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

# --- Models ---

class RangeConfig(BaseModel):
    start: float
    end: float
    step: float

class TimeframeConfig(BaseModel):
    start: str
    end: str
    step: str
    unit: str = 'h'

class IndicatorConfig(BaseModel):
    ema: RangeConfig
    atr: RangeConfig
    high_vf: RangeConfig
    low_vf: RangeConfig
    multiple: int = 1
    side: str = 'both'

class PSConfig(BaseModel):
    ir: RangeConfig
    er: RangeConfig
    or_: RangeConfig = Field(alias='or')

class BrokerConfig(BaseModel):
    max_slot: int = 1
    slippage_pct: float = 0.3
    initial_capital: float = 1000000
    commission_pct: float = 0.1

class OptimizeRequest(BaseModel):
    asset: str
    timeframes: Union[TimeframeConfig, List[str]]
    indicator: IndicatorConfig
    ps: PSConfig
    broker: BrokerConfig
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    batch_id: Optional[str] = None
    collection_type: Optional[str] = "backtest"
    data_type: str = "OKX"

# --- Helper Functions ---

async def _background_optimization_flow(task_id: str, batch_id: str, config: Dict[str, Any], collection_type: str):
    """Orchestrates generation and optimization in sequence."""
    try:
        # 1. Generation Phase
        task_manager.update_task(task_id, status=TaskStatus.GENERATING, message="📝 Đang khởi tạo chiến thuật...")
        total_configs = await task_manager.run_cpu_bound(task_id, generate_configs_logic, task_id, batch_id, config, collection_type)
        
        # 2. Optimization Phase
        task_manager.update_task(task_id, status=TaskStatus.RUNNING, total=total_configs, message="🚀 Đang bắt đầu tối ưu hóa...")
        await task_manager.run_cpu_bound(task_id, run_optimization_logic, task_id, batch_id, config, collection_type)
        
    except Exception as e:
        task_manager.update_task(task_id, status=TaskStatus.FAILED, error=str(e))

# --- Routes ---

@router.post("/run")
async def run_backtest(request: OptimizeRequest, background_tasks: BackgroundTasks):
    """Combined Generate + Run optimization endpoint."""
    mongo = MongoService()
    try:
        batch_id = request.batch_id or f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:12]}"
        task_id = task_manager.create_task("optimization", {"batch_id": batch_id})

        config = request.model_dump(by_alias=True)
        broker = config.pop('broker', {})
        config.update(broker)

        create_batch(batch_id, config, mongo, collection_type=request.collection_type)
        
        background_tasks.add_task(
            _background_optimization_flow,
            task_id=task_id,
            batch_id=batch_id,
            config=config,
            collection_type=request.collection_type
        )

        return {
            "success": True,
            "batch_id": batch_id,
            "task_id": task_id,
            "message": "Optimization task queued"
        }
    finally:
        mongo.close()

@router.get("/progress/{batch_id}")
async def get_progress(batch_id: str, task_id: Optional[str] = None):
    """
    Get optimization progress with high-precision memory polling.
    If task_id is provided, it checks TaskManager memory first for sub-second updates.
    """
    # 1. Check memory for real-time smoothness
    if task_id:
        task = task_manager.get_task(task_id)
        if task and task['status'] in [TaskStatus.RUNNING, TaskStatus.GENERATING]:
            return {
                "batch_id": batch_id,
                "task_id": task_id,
                "status": task['status'],
                "completed": task['completed'],
                "total": task['total'],
                "percentage": task['progress'],
                "speed": task['speed'],
                "message": task['message'],
                "source": "memory"
            }

    # 2. Fallback to MongoDB (Legacy or Completed tasks)
    mongo = MongoService()
    try:
        history = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        if not history:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        progress = history.get('progress', {})
        return {
            "batch_id": batch_id,
            "status": history.get('status'),
            "completed": progress.get('completed', 0),
            "total": progress.get('total', 0),
            "percentage": progress.get('percentage', 0),
            "speed": progress.get('speed', 0),
            "message": progress.get('message', ""),
            "source": "database"
        }
    finally:
        mongo.close()

@router.get("/logs/{task_id}")
async def get_task_logs(task_id: str, last_lines: int = Query(50, ge=1, le=500)):
    """
    Log Diary Endpoint: Get the latest execution logs for a task.
    This fulfills the requirement for a QuantaAlpha-style log view.
    """
    logs = task_manager.get_task_logs(task_id, last_lines)
    return {
        "task_id": task_id,
        "logs": logs,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/results/{batch_id}")
async def get_results(
    batch_id: str, 
    page: int = Query(1, ge=1), 
    limit: int = Query(100, ge=1, le=1000),
    sort_by: str = "metrics.sharpe",
    sort_order: int = -1,
    collection_type: str = "backtest"
):
    """
    Paginated Results Endpoint: Handle millions of configs without crashing.
    Fulfills the 'smoothly show data' requirement for large datasets.
    """
    mongo = MongoService()
    try:
        # Determine collection
        collection = mongo.backtest_result if collection_type == "backtest" else mongo.wfo_result
        
        skip = (page - 1) * limit
        cursor = collection.find({"batch_id": batch_id}).sort(sort_by, sort_order).skip(skip).limit(limit)
        
        results = []
        for doc in cursor:
            doc['_id'] = str(doc['_id'])
            results.append(doc)
            
        total_count = collection.count_documents({"batch_id": batch_id})
        
        return {
            "batch_id": batch_id,
            "page": page,
            "limit": limit,
            "total": total_count,
            "total_pages": (total_count + limit - 1) // limit,
            "data": results
        }
    finally:
        mongo.close()

@router.post("/stop/{batch_id}")
async def stop_backtest(batch_id: str, task_id: Optional[str] = None):
    """Stop a running optimization."""
    if task_id:
        task_manager.cancel_task(task_id)
    
    mongo = MongoService()
    try:
        update_status(batch_id, 'cancelled', mongo)
        return {"success": True, "message": "Optimization cancelled"}
    finally:
        mongo.close()
