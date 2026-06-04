"""Grey WFA analysis routes.

These endpoints back the frontend WFA tab which runs IOS/WFA analysis on top of an
existing backtest batch (source_batch_id).

Endpoints:
- POST /api/wfa/run
- GET /api/wfa/progress/{batch_id}
- GET /api/wfa/runs
- POST /api/wfa/stop/{batch_id}
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import os
import logging

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

from database.mongo_service import MongoService
from optimize.wfa.ios_engine import IOSEngine
from optimize.wfa.wfa_engine import WFAEngine

router = APIRouter(prefix="/api/wfa", tags=["wfa"])

logger = logging.getLogger(__name__)


CONTROL_DIR = os.getenv('GREY_WFA_CONTROL_DIR', '/tmp/wfa_control')
os.makedirs(CONTROL_DIR, exist_ok=True)


class StopRequested(Exception):
    """Raised when a stop signal was requested for a running analysis."""


def _make_bson_safe(value: Any) -> Any:
    """Best-effort conversion to Mongo/BSON-safe primitives.

    We store a *compact* analysis result in `wfa-analysis`. Full engine outputs
    can contain numpy arrays, pandas objects, or custom classes that PyMongo
    cannot encode.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value

    if np is not None:
        try:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _make_bson_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_make_bson_safe(v) for v in value]

    # Fallback: stringify unknown types
    try:
        return str(value)
    except Exception:
        return None


def _compact_engine_result(result: Any) -> Dict[str, Any]:
    """Store only small, stable fields from engine results."""

    if not isinstance(result, dict):
        return {'status': 'unknown', 'raw': _make_bson_safe(result)}

    # Drop huge/non-serializable fields (but keep limited strategies)
    drop_keys = {
        'equity_curve',
        'trades',
        'df',
        'data_frames',
        'raw_results',
    }

    compact: Dict[str, Any] = {}
    for k, v in result.items():
        if k in drop_keys:
            continue
        
        # Keep 'results' but limit to top 5000 strategies to avoid MongoDB size issues
        if k == 'results' and isinstance(v, list):
            # Take top N strategies (already sorted by engine)
            limited_results = v[:5000] if len(v) > 5000 else v
            compact[k] = _make_bson_safe(limited_results)
        elif k == 'strategies' and isinstance(v, list):
            # Same for 'strategies' key (some engines use this)
            limited_strategies = v[:5000] if len(v) > 5000 else v
            compact[k] = _make_bson_safe(limited_strategies)
        else:
            compact[k] = _make_bson_safe(v)

    # Ensure there's always a status
    if 'status' not in compact:
        compact['status'] = 'unknown'

    return compact


def _stop_file_path(batch_id: str) -> str:
    return os.path.join(CONTROL_DIR, f"{batch_id}.stop")


def _is_stop_requested(batch_id: str) -> bool:
    return os.path.exists(_stop_file_path(batch_id))


class InSamplePeriod(BaseModel):
    start: str
    end: str


class OutSamplePeriod(BaseModel):
    start: str
    end: str


class WFAConfig(BaseModel):
    start_date: str
    end_date: str
    is_val: int  # In-sample months
    oos_val: int  # Out-sample months
    step_val: int  # Step months
    step_type: str = 'monthly'


class SourceFilter(BaseModel):
    expression: str


class WFARunRequest(BaseModel):
    mode: str  # 'ios' or 'wfa'
    source_batch_id: str
    in_sample_period: Optional[InSamplePeriod] = None
    out_sample_period: Optional[OutSamplePeriod] = None
    wfa: Optional[WFAConfig] = None
    top_n: Optional[int] = 20
    source_filter: Optional[SourceFilter] = None
    expression: Optional[str] = None          # IS filter expression
    oos_expression: Optional[str] = None      # OOS filter expression
    correlation_threshold: Optional[float] = 0.8
    # WFA from IOS: pass IOS-selected config IDs to narrow down the search space
    ios_batch_id: Optional[str] = None
    selected_config_ids: Optional[List[str]] = None


class WFARunResponse(BaseModel):
    job_id: str
    batch_id: str
    mode: str
    status: str
    message: str


class WFARunSummary(BaseModel):
    batch_id: str
    job_id: str
    mode: str
    status: str
    source_batch_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    current_stage: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class WFAStopResponse(BaseModel):
    batch_id: str
    status: str
    message: str


def _progress_updater(batch_id: str, mongo: MongoService, stage: str):
    """Create a progress callback that persists progress into wfa-analysis.

    Engines call this periodically; we also use it as a stop-check hook.
    """

    def _cb(data: Dict[str, Any]):
        if _is_stop_requested(batch_id):
            raise StopRequested()

        completed = int(data.get('completed', 0) or 0)
        total = int(data.get('total', 0) or 0)
        message = data.get('message')
        if not message and total:
            message = f"{completed}/{total}"

        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'running',
                    'current_stage': stage,
                    'progress': {
                        'completed': completed,
                        'total': total,
                        'configs_per_sec': data.get('configs_per_sec'),
                        'eta_seconds': data.get('eta_seconds'),
                        'elapsed': data.get('elapsed'),
                        'stage': stage,
                        'message': message,
                    },
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )

    return _cb


def _normalize_progress_payload(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize progress fields so frontend consumers can rely on stable keys."""
    progress = analysis.get('progress')
    if not isinstance(progress, dict):
        progress = {}

    completed = int(progress.get('completed', 0) or 0)
    total = int(progress.get('total', 0) or 0)
    stage = analysis.get('current_stage') or progress.get('stage')
    message = analysis.get('message') or progress.get('message')

    raw_speed = progress.get('configs_per_sec')
    try:
        configs_per_sec = float(raw_speed) if raw_speed is not None else None
    except Exception:
        configs_per_sec = None

    raw_eta = progress.get('eta_seconds')
    try:
        eta_seconds = float(raw_eta) if raw_eta is not None else None
    except Exception:
        eta_seconds = None

    if eta_seconds is None and configs_per_sec and configs_per_sec > 0 and total > 0:
        eta_seconds = max(0.0, float(total - completed) / configs_per_sec)

    normalized_progress = dict(progress)
    normalized_progress.update(
        {
            'completed': completed,
            'total': total,
            'stage': stage,
            'message': message,
            'configs_per_sec': configs_per_sec,
            'eta_seconds': eta_seconds,
        }
    )

    return {
        'stage': stage,
        'message': message,
        'configs_per_sec': configs_per_sec,
        'eta_seconds': eta_seconds,
        'progress': normalized_progress,
    }


def run_ios_background(batch_id: str, config: Dict[str, Any]):
    """Background task for IOS analysis"""
    mongo = MongoService()
    try:
        engine = IOSEngine(
            batch_id,
            config,
            mongo=mongo,
            progress_callback=_progress_updater(batch_id, mongo, stage='ios'),
        )
        result = engine.run()

        stored_result = _compact_engine_result(result)

        final_status = 'stopped' if _is_stop_requested(batch_id) or result.get('status') == 'stopped' else 'completed'
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': final_status,
                    'result': stored_result,
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    except StopRequested:
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'stopped',
                    'message': 'Stopped by user',
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    except Exception as e:
        logger.exception('IOS background failed for %s', batch_id)
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'failed',
                    'error': str(e),
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    finally:
        try:
            mongo.close()
        except Exception:
            pass


def run_wfa_background(batch_id: str, config: Dict[str, Any]):
    """Background task for WFA analysis"""
    mongo = MongoService()
    try:
        engine = WFAEngine(
            batch_id,
            config,
            mongo=mongo,
            progress_callback=_progress_updater(batch_id, mongo, stage='wfa'),
        )
        result = engine.run()

        stored_result = _compact_engine_result(result)

        final_status = 'stopped' if _is_stop_requested(batch_id) or result.get('status') == 'stopped' else 'completed'
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': final_status,
                    'result': stored_result,
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    except StopRequested:
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'stopped',
                    'message': 'Stopped by user',
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    except Exception as e:
        logger.exception('WFA background failed for %s', batch_id)
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'failed',
                    'error': str(e),
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=True,
        )
    finally:
        try:
            mongo.close()
        except Exception:
            pass


@router.post("/run", response_model=WFARunResponse)
async def run_wfa_analysis(
    request: WFARunRequest,
    background_tasks: BackgroundTasks
):
    """
    Run IOS or WFA analysis
    
    - **mode**: 'ios' or 'wfa'
    - **source_batch_id**: Source backtest batch ID
    - **in_sample_period**: IS period (for IOS mode)
    - **out_sample_period**: OS period (for IOS mode)
    - **wfa**: WFA config (for WFA mode)
    """
    mongo = MongoService()
    
    # Generate batch ID
    batch_id = f"{request.mode}_{int(datetime.utcnow().timestamp())}"
    job_id = str(uuid.uuid4())
    
    # Check if source has results (any status - just verify data exists)
    result_count = mongo.backtest_result.count_documents({
        'batch_id': request.source_batch_id
    })
    if result_count == 0:
        raise HTTPException(status_code=400, detail=f"No results found in batch {request.source_batch_id}")

    # Source metadata can come from multiple places depending on legacy/production flows.
    # Prefer backtest-batch (if present), otherwise infer from campaign/config/results.
    source_batch = mongo.db['backtest-batch'].find_one({'batch_id': request.source_batch_id})
    inferred_asset = None
    inferred_initial_capital = None
    inferred_commission_pct = None
    inferred_slippage_pct = None
    inferred_indicator = None

    if not source_batch:
        # Try reconstruct from config collections (best effort)
        try:
            from routes.campaigns import reconstruct_config_from_db  # local import to avoid heavy imports at module load
            reconstructed = reconstruct_config_from_db(mongo, request.source_batch_id, collection_type='backtest') or {}
        except Exception:
            reconstructed = {}

        if reconstructed:
            inferred_asset = reconstructed.get('asset')
            inferred_indicator = reconstructed.get('indicator')
            broker = reconstructed.get('broker', {}) or {}
            inferred_initial_capital = broker.get('capital') or broker.get('initial_capital')
            inferred_commission_pct = broker.get('commission') or broker.get('commission_pct')
            inferred_slippage_pct = broker.get('skid') or broker.get('slippage_pct')

        # Always be able to infer at least symbol from earliest successful result
        if not inferred_asset:
            earliest = mongo.backtest_result.find_one(
                {'batch_id': request.source_batch_id, 'status': 'success'},
                sort=[('created_at', 1)],
            )
            if earliest:
                inferred_asset = earliest.get('symbol') or earliest.get('asset')

        # If still nothing, treat as not found
        if not inferred_asset:
            raise HTTPException(status_code=404, detail=f"Source batch {request.source_batch_id} not found")

    # Normalize source fields with safe defaults
    asset = source_batch.get('asset') if source_batch else (inferred_asset or 'BTCUSDT')
    initial_capital = source_batch.get('initial_capital') if source_batch else (inferred_initial_capital or 10000)
    commission_pct = source_batch.get('commission_pct') if source_batch else (inferred_commission_pct if inferred_commission_pct is not None else 0.1)
    slippage_pct = source_batch.get('slippage_pct') if source_batch else (inferred_slippage_pct if inferred_slippage_pct is not None else 0.0)
    indicator = source_batch.get('indicator') if source_batch else (inferred_indicator or {})
    
    try:
        if request.mode == 'ios':
            # IOS Mode
            if not request.in_sample_period or not request.out_sample_period:
                raise HTTPException(status_code=400, detail="in_sample_period and out_sample_period required for IOS mode")

            config = {
                'batch_id': batch_id,
                'mode': 'ios',
                'source_batch_id': request.source_batch_id,
                'asset': asset,
                'initial_capital': float(initial_capital),
                'commission_pct': float(commission_pct),
                'slippage_pct': float(slippage_pct),
                'indicator': indicator,
                'source_filter': {'expression': request.source_filter.expression} if request.source_filter else None,
                'ios': {
                    'in_sample_period': {
                        'start': request.in_sample_period.start,
                        'end': request.in_sample_period.end,
                    },
                    'out_sample_period': {
                        'start': request.out_sample_period.start,
                        'end': request.out_sample_period.end,
                    },
                    'expression': request.expression,
                    'oos_expression': request.oos_expression,
                    'correlation_threshold': request.correlation_threshold,
                    'top_n': request.top_n,
                },
            }
            
            # Create analysis record
            mongo.db['wfa-analysis'].insert_one({
                'job_id': job_id,
                'batch_id': batch_id,
                'mode': 'ios',
                'source_batch_id': request.source_batch_id,
                'config': config,
                'status': 'running',
                'current_stage': 'ios',
                'progress': {'completed': 0, 'total': 0, 'stage': 'ios', 'message': 'Starting...'},
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            
            # Start background task
            background_tasks.add_task(run_ios_background, batch_id, config)
            
        elif request.mode == 'wfa':
            # WFA Mode
            if not request.wfa:
                raise HTTPException(status_code=400, detail="wfa config required for WFA mode")

            config = {
                'batch_id': batch_id,
                'mode': 'wfa',
                'source_batch_id': request.source_batch_id,
                'asset': asset,
                'start_date': request.wfa.start_date,
                'end_date': request.wfa.end_date,
                'initial_capital': float(initial_capital),
                'commission_pct': float(commission_pct),
                'slippage_pct': float(slippage_pct),
                'indicator': indicator,
                'expression': request.expression,
                'oos_expression': request.oos_expression,
                'correlation_threshold': request.correlation_threshold,
                'top_n': request.top_n,
                'source_filter': {'expression': request.source_filter.expression} if request.source_filter else None,
                'ios_batch_id': request.ios_batch_id,
                'selected_config_ids': request.selected_config_ids,
                'wfa': {
                    'start_date': request.wfa.start_date,
                    'end_date': request.wfa.end_date,
                    'is_val': request.wfa.is_val,
                    'oos_val': request.wfa.oos_val,
                    'step_val': request.wfa.step_val,
                    'step_type': request.wfa.step_type,
                },
            }
            
            # Create analysis record
            mongo.db['wfa-analysis'].insert_one({
                'job_id': job_id,
                'batch_id': batch_id,
                'mode': 'wfa',
                'source_batch_id': request.source_batch_id,
                'config': config,
                'status': 'running',
                'current_stage': 'wfa',
                'progress': {'completed': 0, 'total': 0, 'stage': 'wfa', 'message': 'Starting...'},
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            })
            
            # Start background task
            background_tasks.add_task(run_wfa_background, batch_id, config)
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}. Must be 'ios' or 'wfa'")
        
        return WFARunResponse(
            job_id=job_id,
            batch_id=batch_id,
            mode=request.mode,
            status='running',
            message=f'{request.mode.upper()} analysis started successfully'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")
    finally:
        mongo.close()


@router.get("/progress/{batch_id}")
async def get_wfa_progress(batch_id: str):
    """
    Get WFA/IOS analysis progress
    """
    mongo = MongoService()
    try:
        analysis = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if not analysis:
            raise HTTPException(status_code=404, detail=f"Analysis {batch_id} not found")

        normalized = _normalize_progress_payload(analysis)

        return {
            'batch_id': batch_id,
            'job_id': analysis.get('job_id'),
            'mode': analysis.get('mode'),
            'status': analysis.get('status'),
            'current_stage': analysis.get('current_stage'),
            'stage': normalized['stage'],
            'progress': normalized['progress'],
            'message': normalized['message'],
            'configs_per_sec': normalized['configs_per_sec'],
            'eta_seconds': normalized['eta_seconds'],
            'result': analysis.get('result'),
            'error': analysis.get('error'),
            'created_at': analysis.get('created_at'),
            'updated_at': analysis.get('updated_at'),
        }
    finally:
        mongo.close()


@router.get("/runs", response_model=List[WFARunSummary])
async def list_wfa_runs(
    status: Optional[str] = Query(None, description="Filter by status (running|completed|failed|stopped|stopping)"),
    limit: int = Query(50, ge=1, le=200),
):
    """List WFA/IOS analyses from wfa-analysis collection."""
    mongo = MongoService()
    try:
        query: Dict[str, Any] = {}
        if status:
            query['status'] = status
        docs = list(mongo.db['wfa-analysis'].find(query).sort('created_at', -1).limit(limit))
        out: List[WFARunSummary] = []
        for d in docs:
            out.append(
                WFARunSummary(
                    batch_id=d.get('batch_id'),
                    job_id=d.get('job_id'),
                    mode=d.get('mode'),
                    status=d.get('status'),
                    source_batch_id=d.get('source_batch_id'),
                    created_at=d.get('created_at'),
                    updated_at=d.get('updated_at'),
                    current_stage=d.get('current_stage'),
                    progress=d.get('progress'),
                    message=d.get('message'),
                )
            )
        return out
    finally:
        mongo.close()


@router.post("/stop/{batch_id}", response_model=WFAStopResponse)
async def stop_wfa_run(batch_id: str):
    """Request stop for a running analysis (best-effort)."""
    # Create stop file
    try:
        with open(_stop_file_path(batch_id), 'w') as f:
            f.write(datetime.utcnow().isoformat())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create stop signal: {e}")

    mongo = MongoService()
    try:
        mongo.db['wfa-analysis'].update_one(
            {'batch_id': batch_id},
            {
                '$set': {
                    'status': 'stopping',
                    'message': 'Stop requested',
                    'updated_at': datetime.utcnow(),
                }
            },
            upsert=False,
        )
    finally:
        mongo.close()

    return WFAStopResponse(batch_id=batch_id, status='stopping', message='Stop requested')


@router.get("/campaign/{batch_id}")
async def get_wfa_campaign(batch_id: str):
    """Get WFA campaign metadata (compatible with /api/campaigns/{id})."""
    mongo = MongoService()
    try:
        doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"WFA campaign {batch_id} not found")
        
        # Convert to campaign-like format
        return {
            'id': doc.get('batch_id'),
            'friendly_name': f"WFA_{doc.get('mode', 'wfa').upper()}_{doc.get('source_batch_id', '')[:8]}",
            'status': doc.get('status', 'UNKNOWN').upper(),
            'collection_type': 'wfa',
            'type': 'wfa',
            'mode': doc.get('mode'),
            'config': doc.get('config', {}),
            'source_batch_id': doc.get('source_batch_id'),
            'created_at': doc.get('created_at'),
            'updated_at': doc.get('updated_at'),
            'current_stage': doc.get('current_stage'),
            'progress': doc.get('progress'),
            'result': doc.get('result')
        }
    finally:
        mongo.close()


@router.get("/campaign/{batch_id}/results")
async def get_wfa_results(
    batch_id: str,
    limit: int = Query(200000, ge=1),
    sort_by: str = Query('roi', description="Sort field"),
    sort_order: int = Query(-1, description="1 for ascending, -1 for descending")
):
    """Get WFA results/strategies (compatible with /api/campaigns/{id}/results)."""
    mongo = MongoService()
    try:
        doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"WFA campaign {batch_id} not found")
        
        # Extract results from WFA doc
        result = doc.get('result', {})
        strategies = result.get('strategies', []) if isinstance(result, dict) else []
        
        return {
            'results': strategies[:limit] if strategies else [],
            'total': len(strategies),
            'limit': limit
        }
    finally:
        mongo.close()


@router.get("/campaign/{batch_id}/top-strategies")
async def get_wfa_top_strategies(
    batch_id: str,
    limit: int = Query(20, ge=1, le=100)
):
    """Get top WFA strategies (compatible with /api/campaigns/{id}/top-strategies)."""
    mongo = MongoService()
    try:
        doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"WFA campaign {batch_id} not found")
        
        result = doc.get('result', {})
        strategies = result.get('strategies', []) if isinstance(result, dict) else []
        
        return strategies[:limit] if strategies else []
    finally:
        mongo.close()


@router.get("/campaign/{batch_id}/chart-data")
async def get_wfa_chart_data(
    batch_id: str,
    points: int = Query(1000, ge=10, le=10000)
):
    """Get WFA chart data (compatible with /api/campaigns/{id}/chart-data)."""
    mongo = MongoService()
    try:
        doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if not doc:
            raise HTTPException(status_code=404, detail=f"WFA campaign {batch_id} not found")
        
        # Return empty chart data for now - can enhance later
        return {
            'chart_data': [],
            'points': 0
        }
    finally:
        mongo.close()
