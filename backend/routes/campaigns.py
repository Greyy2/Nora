"""
Campaign Routes - API endpoints for campaign management (Backtest & WFA)

Endpoints:
- GET /api/campaigns?type={backtest|wfa} - List all campaigns
- GET /api/campaigns/{batch_id} - Get campaign details
- GET /api/campaigns/{batch_id}/results - Get campaign results
- DELETE /api/campaigns/{batch_id} - Delete campaign
- GET /api/campaigns/stats?type={backtest|wfa} - Get dashboard stats
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from database.mongo_service import MongoService
from services.cache_service import cache_get, cache_set, cache_invalidate_prefix
from utils.error_handler import handle_mongo_error, MongoConnectionError
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = {
    'campaign_list': 90,
    'campaign_detail': 15,
    'campaign_stats': 30,
    'pipeline_summary': 30,
    'campaign_results': 15,
    'top_strategies': 20,
    'chart_data': 20,
}
_ALLOWED_SORT_FIELDS = {
    'roi', 'winRate', 'sharpe', 'mdd', 'cagr', 'profit', 'totalTrades',
    'ema', 'atr', 'ir', 'er', 'or',
}
_ALLOWED_FILTER_METRICS = {
    'profit', 'winRate', 'cagr', 'mdd', 'roi', 'sharpe', 'totalTrades',
}
_ALLOWED_FILTER_OPERATORS = {'>', '>=', '<', '<=', '=='}
_COMPLETED_STATUSES = {'success', 'completed', 'done', 'finished'}


def _is_completed_status(value: Any) -> bool:
    return str(value or '').strip().lower() in _COMPLETED_STATUSES


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value:
        return str(value)
    return datetime.utcnow().isoformat()


def _cache_get(cache_key: str):
    return cache_get('campaigns', cache_key)


def _cache_set(cache_key: str, value: Any, ttl_seconds: int):
    cache_set('campaigns', cache_key, value, ttl_seconds)


def _cache_invalidate(prefix: Optional[str] = None):
    cache_invalidate_prefix('campaigns', prefix or '')


def _make_cache_key(scope: str, *parts: Any) -> str:
    normalized_parts = []
    for part in parts:
        if isinstance(part, (dict, list, tuple)):
            normalized_parts.append(repr(part))
        else:
            normalized_parts.append(str(part))
    return f"{scope}:{'|'.join(normalized_parts)}"


def _stringify_timeframe(tf: Any) -> str:
    """Best-effort timeframe normalization for reporting.

    Notes:
    - DB may contain legacy numeric timeframes (e.g. 8) or strings ('8', '8h').
    - We do NOT force-append units here because that can hide upstream data issues.
    """
    if tf is None:
        return ''
    try:
        return str(tf)
    except Exception:
        return ''


def _build_timeframes_from_cfg(timeframes_cfg: Any) -> List[str]:
    """Build list of timeframes from config.

    Supports:
    - dict: {start,end,step,unit}
    - list: ['1h','2h',...]
    - scalar: '4h'
    """
    if isinstance(timeframes_cfg, dict):
        def parse_int(x, default: int) -> int:
            if x is None:
                return default
            s = str(x).strip().lower()
            # drop suffix if present
            for suffix in ['h', 'm', 'd', 'w']:
                if s.endswith(suffix):
                    s = s[:-1]
                    break
            try:
                return int(float(s))
            except Exception:
                return default

        start = parse_int(timeframes_cfg.get('start'), 1)
        end = parse_int(timeframes_cfg.get('end'), start)
        step = max(1, parse_int(timeframes_cfg.get('step'), 1))
        unit = str(timeframes_cfg.get('unit') or 'h').strip().lower()[:1]
        if end < start:
            start, end = end, start
        return [f"{i}{unit}" for i in range(start, end + 1, step)]

    if isinstance(timeframes_cfg, list):
        return [_stringify_timeframe(x) for x in timeframes_cfg if _stringify_timeframe(x)]

    s = _stringify_timeframe(timeframes_cfg)
    return [s] if s else []


def _normalize_timeframes_to_dict(timeframes_cfg: Any) -> Dict[str, Any]:
    """
    Normalize timeframes config to dict format {start, end, step, unit}.
    
    Handles:
    - dict: Return as-is
    - list: Calculate start/end/step from array (e.g., ["1h", "2h", "3h"] -> {start: "1h", end: "3h", step: "1h"})
    - scalar: Convert to single-value dict
    
    Returns:
        Dict with keys: start, end, step, unit
    """
    # Already dict format
    if isinstance(timeframes_cfg, dict):
        return timeframes_cfg
    
    # List format - calculate start/end/step
    if isinstance(timeframes_cfg, list) and len(timeframes_cfg) > 0:
        def parse_tf(tf_str):
            """Parse timeframe string like '1h' -> (1, 'h')"""
            s = str(tf_str).strip().lower()
            unit = 'h'  # default
            for suffix in ['h', 'm', 'd', 'w']:
                if s.endswith(suffix):
                    unit = suffix
                    s = s[:-1]
                    break
            try:
                value = int(float(s))
                return value, unit
            except:
                return None, unit
        
        # Parse first and last timeframes
        first_val, unit = parse_tf(timeframes_cfg[0])
        last_val, _ = parse_tf(timeframes_cfg[-1])
        
        if first_val is None or last_val is None:
            # Fallback if parsing fails
            return {
                'start': str(timeframes_cfg[0]),
                'end': str(timeframes_cfg[-1]),
                'step': '1h',
                'unit': 'h'
            }
        
        # Calculate step from first two elements
        step_val = 1
        if len(timeframes_cfg) >= 2:
            second_val, _ = parse_tf(timeframes_cfg[1])
            if second_val is not None:
                step_val = second_val - first_val
        
        return {
            'start': f"{first_val}{unit}",
            'end': f"{last_val}{unit}",
            'step': f"{step_val}{unit}",
            'unit': unit
        }
    
    # Scalar format - single value
    s = _stringify_timeframe(timeframes_cfg)
    if s:
        return {
            'start': s,
            'end': s,
            'step': '1h',
            'unit': 'h'
        }
    
    # Fallback
    return {
        'start': '1h',
        'end': '1h',
        'step': '1h',
        'unit': 'h'
    }


def _wfa_extract_strategies(wfa_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of strategy-like dicts for WFA/IOS analysis documents.

    Preference order:
    - result['strategies']
    - result['results']
    - synthesize from result['analysis']['windows'][*]['best_config']
    """
    if not isinstance(wfa_doc, dict):
        return []
    res = wfa_doc.get('result') or {}
    if not isinstance(res, dict):
        return []

    for key in ('strategies', 'results'):
        val = res.get(key)
        if isinstance(val, list) and len(val) > 0:
            return val

    analysis = res.get('analysis')
    if not isinstance(analysis, dict):
        return []
    windows = analysis.get('windows')
    if not isinstance(windows, dict):
        return []

    synthesized: List[Dict[str, Any]] = []
    for wid, w in windows.items():
        if not isinstance(w, dict):
            continue
        best_cfg = w.get('best_config')
        if best_cfg is None:
            continue
        oos_roi = w.get('best_oos_roi')
        try:
            roi_val = float(oos_roi) if oos_roi is not None else 0.0
        except Exception:
            roi_val = 0.0

        synthesized.append({
            'window_id': str(wid),
            'config': best_cfg,
            # Provide a metrics-ish shape for frontend fallback table
            'roi': roi_val,
            'sharpe': 0,
            'mdd': 0,
            'win_rate': 0,
            'total_trades': 0,
            'oos_roi': oos_roi,
            'is_roi': w.get('best_is_roi'),
            'score': oos_roi,
        })

    synthesized.sort(key=lambda x: (x.get('roi') or 0), reverse=True)
    return synthesized


def _reconstruct_config_from_params(params: Dict[str, Any], metadata: Dict[str, Any], source_batch_id: Optional[str] = None) -> Dict[str, Any]:
    ps = params.get('strategy', {}).get('ps', {})
    bse = params.get('strategy', {}).get('bse', {})
    is_on_going_val = bse.get('is_on_going', 0.8)
    if isinstance(is_on_going_val, bool):
        is_on_going_val = 0.8 if is_on_going_val else 0.0

    config = {
        'asset': metadata.get('asset', params.get('asset', 'BTCUSDT')),
        'indicator': {
            'ema': {'start': params.get('length_ema'), 'end': params.get('length_ema'), 'step': 1},
            'atr': {'start': params.get('length_atr'), 'end': params.get('length_atr'), 'step': 1},
            'high_vf': {'start': params.get('long_vol_factor'), 'end': params.get('long_vol_factor'), 'step': 0.1},
            'low_vf': {'start': params.get('short_vol_factor'), 'end': params.get('short_vol_factor'), 'step': 0.1},
            'multiple': params.get('multiple', 1)
        },
        'ps': {
            'ir': {'start': ps.get('ir'), 'end': ps.get('ir'), 'step': 0.01},
            'er': {'start': ps.get('er'), 'end': ps.get('er'), 'step': 0.1},
            'or': {'start': ps.get('or'), 'end': ps.get('or'), 'step': 0.05}
        },
        'timeframes': {
            'start': metadata.get('timeframe', params.get('timeframe', '1h')),
            'end': metadata.get('timeframe', params.get('timeframe', '1h')),
            'step': '1h',
            'unit': 'h'
        },
        'broker': {
            'max_slot': params.get('max_slot', 1),
            'skid': metadata.get('slippage_pct', params.get('skid', 0.3)),
            'capital': params.get('capital', 1000000),
            'commission': metadata.get('commission_pct', params.get('commission', 0.1)),
            'is_on_going': is_on_going_val
        }
    }

    if source_batch_id:
        config['source_batch_id'] = source_batch_id

    if 'timeframes' in config:
        config['timeframes'] = _normalize_timeframes_to_dict(config['timeframes'])
    return config


# Import progress_tracker to check in-memory status (avoid race condition)
try:
    from routes.backtest import progress_tracker
except ImportError:
    progress_tracker = {}

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])


def reconstruct_config_from_db(mongo: MongoService, batch_id: str, collection_type: str = 'backtest') -> Dict[str, Any]:
    """
    Reconstruct campaign configuration from database collections.
    
    This function attempts to rebuild the config object from:
    1. backtest-config or wfa-config collections
    2. Fallback to earliest result document if config collection is empty
    
    Args:
        mongo: MongoService instance
        batch_id: Campaign batch ID
        collection_type: 'backtest' or 'wfa'
    
    Returns:
        Dict containing reconstructed config, or empty dict if reconstruction fails
    """
    config_coll = mongo.wfa_config if collection_type == 'wfa' else mongo.backtest_config
    result_coll = mongo.wfa_result if collection_type == 'wfa' else mongo.backtest_result
    
    # Try to get config from config collection
    cfg_doc = config_coll.find_one({'batch_id': batch_id})
    
    if not cfg_doc or 'params' not in cfg_doc:
        # No config found, return empty
        return {}
    
    p = cfg_doc['params']
    ps = p.get('strategy', {}).get('ps', {})
    bse = p.get('strategy', {}).get('bse', {})
    metadata = cfg_doc.get('metadata', {})
    
    # Get earliest result for fallback data
    earliest = result_coll.find_one({'batch_id': batch_id}, sort=[('created_at', 1)])
    
    # Handle is_on_going: true -> 0.8, false -> 0.0
    is_on_going_val = bse.get('is_on_going', 0.8)
    if isinstance(is_on_going_val, bool):
        is_on_going_val = 0.8 if is_on_going_val else 0.0
    
    return {
        'asset': metadata.get('asset', p.get('asset', earliest.get('symbol', 'BTCUSDT') if earliest else 'BTCUSDT')),
        'indicator': {
            'ema': {'start': p.get('length_ema'), 'end': p.get('length_ema'), 'step': 1},
            'atr': {'start': p.get('length_atr'), 'end': p.get('length_atr'), 'step': 1},
            'high_vf': {'start': p.get('long_vol_factor'), 'end': p.get('long_vol_factor'), 'step': 0.1},
            'low_vf': {'start': p.get('short_vol_factor'), 'end': p.get('short_vol_factor'), 'step': 0.1},
            'multiple': p.get('multiple', 1)
        },
        'ps': {
            'ir': {'start': ps.get('ir'), 'end': ps.get('ir'), 'step': 0.01},
            'er': {'start': ps.get('er'), 'end': ps.get('er'), 'step': 0.1},
            'or': {'start': ps.get('or'), 'end': ps.get('or'), 'step': 0.05}
        },
        'timeframes': {
            'start': metadata.get('timeframe', p.get('timeframe', '1h')),
            'end': metadata.get('timeframe', p.get('timeframe', '1h')),
            'step': '1h',
            'unit': 'h'
        },
        'broker': {
            'max_slot': p.get('max_slot', 1),
            'skid': metadata.get('slippage_pct', p.get('skid', 0.3)),
            'capital': p.get('capital', 1000000),
            'commission': metadata.get('commission_pct', p.get('commission', 0.1)),
            'is_on_going': is_on_going_val
        }
    }

class CampaignResponse(BaseModel):
    """Campaign summary response"""
    id: str
    friendly_name: str
    createdAt: str
    type: str  # 'backtest' or 'wfa'
    status: str  # 'SUCCESS', 'RUNNING', 'FAILED'
    config: Dict[str, Any]
    stats: Dict[str, Any]
    filters: Optional[List[Dict[str, Any]]] = []


class DashboardStatsResponse(BaseModel):
    """Dashboard statistics matching frontend expectations"""
    totalCampaigns: int
    completionRate: float
    completedCount: int
    avgWinRate: float
    avgMaxDrawdown: float
    avgROI: float
    topROI: float


@router.get("/{batch_id}/timeframe-stats")
async def get_campaign_timeframe_stats(batch_id: str):
    """Debug helper: show how many configs/results per timeframe.

    This answers questions like:
    - 'I ran 10 timeframes, why do I see mostly 8h in results?'

    The most common reason is: many timeframes failed (missing data / invalid params),
    and the UI only shows successful results.
    """
    mongo = MongoService()
    try:
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id}, {'_id': 0, 'config': 1, 'collection_type': 1})
        campaign_type = (campaign or {}).get('collection_type') or ('wfa' if batch_id.startswith('WFA_') else 'backtest')
        
        if campaign_type == 'wfa':
            config_collection = mongo.wfa_config
            result_collection = mongo.wfa_result
        else:
            config_collection = mongo.backtest_config
            result_collection = mongo.backtest_result

        cfg = (campaign or {}).get('config') or {}
        expected_cfg = cfg.get('timeframes')
        expected_list = _build_timeframes_from_cfg(expected_cfg) if expected_cfg is not None else []

        # Group configs by params.timeframe
        cfg_pipe = [
            {'$match': {'batch_id': batch_id}},
            {'$group': {'_id': '$params.timeframe', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        cfg_groups = list(config_collection.aggregate(cfg_pipe))
        configs_by_tf = [{'timeframe': _stringify_timeframe(g.get('_id')), 'count': int(g.get('count', 0))} for g in cfg_groups]

        # Group results by timeframe for success/failed
        success_pipe = [
            {'$match': {'batch_id': batch_id, 'status': 'success'}},
            {'$group': {'_id': '$result.all.timeframe', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        failed_pipe = [
            {'$match': {'batch_id': batch_id, 'status': {'$ne': 'success'}}},
            {'$group': {'_id': '$result.all.timeframe', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]

        success_groups = list(result_collection.aggregate(success_pipe))
        failed_groups = list(result_collection.aggregate(failed_pipe))

        results_success_by_tf = [{'timeframe': _stringify_timeframe(g.get('_id')), 'count': int(g.get('count', 0))} for g in success_groups]
        results_failed_by_tf = [{'timeframe': _stringify_timeframe(g.get('_id')), 'count': int(g.get('count', 0))} for g in failed_groups]

        total_configs = int(config_collection.count_documents({'batch_id': batch_id}))
        total_success = int(result_collection.count_documents({'batch_id': batch_id, 'status': 'success'}))
        total_failed = int(result_collection.count_documents({'batch_id': batch_id, 'status': {'$ne': 'success'}}))

        return {
            'batch_id': batch_id,
            'type': campaign_type,
            'expected_timeframes': {
                'config': expected_cfg,
                'list': expected_list,
                'count': len(expected_list)
            },
            'configs': {
                'total': total_configs,
                'by_timeframe': configs_by_tf
            },
            'results': {
                'success_total': total_success,
                'failed_total': total_failed,
                'success_by_timeframe': results_success_by_tf,
                'failed_by_timeframe': results_failed_by_tf
            }
        }
    finally:
        mongo.close()


@router.get("/stats")
async def get_dashboard_stats(
    type: str = Query("backtest", description="Campaign type: backtest or wfa")
):
    """Get dashboard statistics for campaigns"""
    cache_key = _make_cache_key('stats', type)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = MongoService()
    try:
        # Get all campaigns from history
        campaigns_from_history = list(mongo.db['optimize_history'].find({
            'collection_type': type
        }).sort('created_at', -1))
        
        # Always get complete list from results
        result_coll = mongo.wfa_result if type == 'wfa' else mongo.backtest_result
        batch_ids = result_coll.distinct('batch_id')
        
        # Get batch_ids that are already in history
        existing_batch_ids = {c.get('batch_id') for c in campaigns_from_history}
        
        # Count inferred campaigns
        inferred_count = 0
        for bid in batch_ids:
            if not bid or bid in existing_batch_ids:
                continue
            is_wfa = bid.startswith('WFA_')
            # Skip if type mismatch
            if type == 'wfa' and not is_wfa:
                continue
            if type == 'backtest' and is_wfa:
                continue
            inferred_count += 1
        
        total_campaigns = len(campaigns_from_history) + inferred_count
        
        # If no campaigns at all, aggregate from results
        if total_campaigns == 0:
            result_coll = mongo.wfa_result if type == 'wfa' else mongo.backtest_result
            # Match batch IDs by prefix to ensure they belong to the requested type
            match_pattern = r'^WFA_.*' if type == 'wfa' else r'^(?!WFA_).*'
            pipeline = [
                {'$match': {'batch_id': {'$regex': match_pattern}}},
                {'$group': {
                    '_id': '$batch_id',
                    'count': {'$sum': 1},
                    'avgROI': {'$avg': '$result.all.roi'},
                    'maxROI': {'$max': '$result.all.roi'},
                    'avgWinRate': {'$avg': '$result.all.winRate'},
                    'avgMDD': {'$avg': '$result.all.mdd'}
                }}
            ]
            agg_results = list(result_coll.aggregate(pipeline))
            
            if not agg_results:
                response = DashboardStatsResponse(
                    totalCampaigns=0,
                    completedCount=0,
                    completionRate=0.0,
                    avgWinRate=0.0,
                    avgMaxDrawdown=0.0,
                    avgROI=0.0,
                    topROI=0.0
                )
                _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_stats'])
                return response
            
            total = len(agg_results)
            avg_roi = sum(r['avgROI'] for r in agg_results if r['avgROI']) / total if total > 0 else 0
            top_roi = max(r['maxROI'] for r in agg_results if r['maxROI']) if total > 0 else 0
            avg_win_rate = sum(r['avgWinRate'] for r in agg_results if r['avgWinRate']) / total if total > 0 else 0
            avg_mdd = sum(r['avgMDD'] for r in agg_results if r['avgMDD']) / total if total > 0 else 0
            
            response = DashboardStatsResponse(
                totalCampaigns=total,
                completionRate=100.0,
                completedCount=total,
                avgWinRate=round(avg_win_rate, 2),
                avgMaxDrawdown=round(avg_mdd, 2),
                avgROI=round(avg_roi, 2),
                topROI=round(top_roi, 2)
            )
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_stats'])
            return response
        
        # Calculate stats from all campaigns
        successful = sum(1 for c in campaigns_from_history if c.get('status') == 'completed')
        # Inferred campaigns are considered completed
        successful += inferred_count
        
        completion_rate = (successful / total_campaigns * 100) if total_campaigns > 0 else 0.0
        
        # Aggregate metrics from ALL results in database (not just campaigns)
        result_coll = mongo.wfa_result if type == 'wfa' else mongo.backtest_result
        
        # Match batch IDs by type
        is_wfa_type = type == 'wfa'
        pipeline = [
            {
                '$match': {
                    'batch_id': {'$regex': '^WFA_' if is_wfa_type else '^(?!WFA_)'}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'avgWinRate': {'$avg': '$result.all.winRate'},
                    'avgMDD': {'$avg': '$result.all.mdd'},
                    'avgROI': {'$avg': '$result.all.roi'},
                    'maxROI': {'$max': '$result.all.roi'}
                }
            }
        ]
        
        agg_results = list(result_coll.aggregate(pipeline))
        
        if agg_results and agg_results[0]['_id'] is None:
            stats = agg_results[0]
            avg_win_rate = stats.get('avgWinRate', 0) or 0
            avg_mdd = stats.get('avgMDD', 0) or 0
            avg_roi = stats.get('avgROI', 0) or 0
            top_roi = stats.get('maxROI', 0) or 0
        else:
            avg_win_rate = avg_mdd = avg_roi = top_roi = 0.0
        
        response = DashboardStatsResponse(
            totalCampaigns=total_campaigns,
            completionRate=round(completion_rate, 1),
            completedCount=successful,
            avgWinRate=round(avg_win_rate, 2),
            avgMaxDrawdown=round(avg_mdd, 2),
            avgROI=round(avg_roi, 2),
            topROI=round(top_roi, 2)
        )
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_stats'])
        return response
        
    finally:
        mongo.close()


@router.get("")
async def list_campaigns(
    type: str = Query("backtest", description="Campaign type: backtest or wfa"),
    limit: int = Query(100, description="Max results to return"),
    summary: bool = Query(True, description="Fast summary mode for dashboard lists"),
    include_config: bool = Query(False, description="Include full config payload in list response")
):
    """List all campaigns (backtest or WFA)"""
    # Validate campaign type
    if type not in ('backtest', 'wfa'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid campaign type '{type}'. Must be 'backtest' or 'wfa'"
        )
    
    cache_key = _make_cache_key('list', type, limit, summary, include_config)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = None
    try:
        mongo = MongoService()
        # Include WFA analysis jobs (new pipeline stored in wfa-analysis) in WFA dashboard list
        wfa_analysis_campaigns: List[Dict[str, Any]] = []
        if type == 'wfa':
            try:
                docs = list(mongo.db['wfa-analysis'].find({
                    'mode': {'$in': ['wfa', 'WFA']}
                }).sort('created_at', -1).limit(limit))

                source_batch_ids = list({
                    doc.get('source_batch_id')
                    for doc in docs
                    if doc.get('source_batch_id')
                })
                ios_analysis_by_source: Dict[str, Any] = {}
                if source_batch_ids:
                    ios_pipeline = [
                        {'$match': {
                            'source_batch_id': {'$in': source_batch_ids},
                            'mode': {'$in': ['ios', 'IOS']}
                        }},
                        {'$sort': {'created_at': -1}},
                        {'$group': {
                            '_id': '$source_batch_id',
                            'doc': {'$first': '$$ROOT'}
                        }}
                    ]
                    for grouped in mongo.db['wfa-analysis'].aggregate(ios_pipeline):
                        ios_doc = grouped.get('doc') or {}
                        ios_res = ios_doc.get('result') or {}
                        if isinstance(ios_res, dict):
                            ios_analysis_by_source[grouped.get('_id')] = ios_res.get('analysis') or ios_res

                for doc in docs:
                    res = doc.get('result') or {}
                    analysis = res.get('analysis') if isinstance(res, dict) else {}
                    windows_src = (analysis.get('windows') or {}) if isinstance(analysis, dict) else {}

                    passed_configs = 0
                    if isinstance(windows_src, dict):
                        for _wid, w in windows_src.items():
                            if isinstance(w, dict):
                                passed_configs += int(w.get('configs_count', 0) or 0)

                    progress = doc.get('progress') or {}
                    total_configs = int(progress.get('total', 0) or 0)
                    passed_pct = (passed_configs / (total_configs or 1)) * 100.0

                    # Attach matching IOS analysis (so dashboard can show IOS pass stats)
                    source_batch_id = doc.get('source_batch_id')
                    ios_analysis = ios_analysis_by_source.get(source_batch_id)

                    wfa_analysis_campaigns.append({
                        'id': doc.get('batch_id'),
                        'friendly_name': f"WFA_{(doc.get('mode', 'wfa') or 'wfa').upper()}_{(doc.get('source_batch_id') or '')[:8]}",
                        'createdAt': doc.get('created_at').isoformat() if doc.get('created_at') else datetime.utcnow().isoformat(),
                        'type': 'wfa',
                        'collection_type': 'wfa',
                        'status': (doc.get('status') or 'UNKNOWN').upper(),
                        'config': doc.get('config', {}),
                        'stats': {
                            'total': int(progress.get('total', 0) or 0),
                            'completed': int(progress.get('completed', 0) or 0),
                            'pending': max(0, int(progress.get('total', 0) or 0) - int(progress.get('completed', 0) or 0)),
                        },
                        'source_batch_id': doc.get('source_batch_id'),
                        'analysis': ios_analysis,
                        'wfa_analysis': {
                            'aggregate': {
                                'completed_windows': int((res.get('completed_windows') if isinstance(res, dict) else 0) or 0),
                                'total_windows': int((res.get('windows') if isinstance(res, dict) else 0) or 0),
                                'avg_oos_roi': float((res.get('avg_oos_roi') if isinstance(res, dict) else 0) or 0),
                                'total_oos_roi': float((res.get('total_oos_roi') if isinstance(res, dict) else 0) or 0),
                                'passed_configs': int(passed_configs),
                                'passed_pct': float(passed_pct),
                            }
                        },
                        'filters': [],
                    })
            except Exception:
                # best-effort; keep history-based list even if this fails
                pass
        # Get campaigns from optimize_history
        # NOTE: dashboard summary mode does not need full config payload unless explicitly requested.
        history_projection = {
            '_id': 0,
            'batch_id': 1,
            'friendly_name': 1,
            'created_at': 1,
            'collection_type': 1,
            'status': 1,
            'filters': 1,
            'summary': 1,
            'stats': 1,
            'progress': 1,
            'generation': 1,
            'source_batch_id': 1,
        }
        if include_config:
            history_projection['config'] = 1

        campaigns_from_history = list(
            mongo.db['optimize_history']
            .find({'collection_type': type}, history_projection)
            .sort('created_at', -1)
            .limit(limit)
        )

        if summary:
            all_campaigns: List[Dict[str, Any]] = []

            for campaign in campaigns_from_history:
                bid = campaign.get('batch_id')
                if not bid:
                    continue

                db_status = campaign.get('status', 'pending')
                if bid in progress_tracker:
                    tracker_status = progress_tracker[bid].get('status', '')
                    if tracker_status == 'completed':
                        db_status = 'completed'
                    elif tracker_status == 'running':
                        db_status = 'running'
                    elif tracker_status == 'processing':
                        db_status = 'completed'
                    elif tracker_status == 'failed':
                        db_status = 'failed'

                status_map = {
                    'pending': 'PENDING',
                    'generated': 'GENERATED',
                    'running': 'RUNNING',
                    'completed': 'SUCCESS',
                    'success': 'SUCCESS',
                    'failed': 'FAILED',
                    'paused': 'PAUSED',
                    'stopped': 'STOPPED',
                    'cancelled': 'CANCELLED',
                }
                campaign_status = status_map.get(db_status, 'PENDING')

                progress = campaign.get('progress', {}) or {}
                generation = campaign.get('generation', {}) or {}
                summary_stats = campaign.get('summary', {}) or {}

                total = int(
                    progress.get('total', 0)
                    or generation.get('total_inserted', 0)
                    or summary_stats.get('total', 0)
                    or 0
                )
                completed = int(
                    progress.get('completed', 0)
                    or summary_stats.get('completed', 0)
                    or 0
                )

                config = campaign.get('config', {}) if include_config else {}
                if include_config and 'timeframes' in config:
                    config['timeframes'] = _normalize_timeframes_to_dict(config['timeframes'])

                raw_filters = campaign.get('filters', []) or []
                compact_filters = []
                for f in raw_filters[:5]:
                    if not isinstance(f, dict):
                        continue
                    compact_filters.append({
                        'id': f.get('id'),
                        'expression': f.get('expression', ''),
                        'matched': int(f.get('matched', 0) or 0),
                        'total': int(f.get('total', 0) or 0),
                        'percentage': float(f.get('percentage', 0) or 0),
                        'createdAt': f.get('createdAt'),
                    })

                all_campaigns.append({
                    'id': bid,
                    'friendly_name': campaign.get('friendly_name', bid),
                    'createdAt': campaign.get('created_at', datetime.now()).isoformat() if isinstance(campaign.get('created_at'), datetime) else str(campaign.get('created_at', '')),
                    'type': type,
                    'status': campaign_status,
                    'config': config,
                    'stats': {
                        'total': total,
                        'completed': completed,
                        'pending': max(0, total - completed),
                    },
                    'filters': compact_filters,
                    'source_batch_id': campaign.get('source_batch_id')
                })

            if wfa_analysis_campaigns:
                existing_ids = {c.get('id') for c in all_campaigns}
                for c in wfa_analysis_campaigns:
                    if c.get('id') and c.get('id') not in existing_ids:
                        if not include_config:
                            c = {**c, 'config': {}}
                        all_campaigns.append(c)

            all_campaigns.sort(key=lambda x: x['createdAt'], reverse=True)
            response = all_campaigns[:limit]
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_list'])
            return response

        # Non-summary mode keeps the older compatibility behavior below.
        
        # Get batch_ids already in history
        existing_batch_ids = {c.get('batch_id') for c in campaigns_from_history}
        
        # FAST PATH: If we have enough campaigns from history, skip expensive aggregation
        result_coll = mongo.wfa_result if type == 'wfa' else mongo.backtest_result
        config_coll = mongo.wfa_config if type == 'wfa' else mongo.backtest_config
        
        inferred_campaigns = []
        
        # Only run aggregation if we need more campaigns
        if len(campaigns_from_history) < limit:
            # Use faster distinct on indexed field, then filter
            all_batch_ids = result_coll.distinct('batch_id')
            
            # Filter to only batch_ids not in history and matching type
            batch_ids_to_infer = [
                bid for bid in all_batch_ids 
                if bid and bid not in existing_batch_ids
                and ((type == 'wfa' and bid.startswith('WFA_')) or (type == 'backtest' and not bid.startswith('WFA_')))
            ][:limit - len(campaigns_from_history)]  # Only take what we need
            
            if batch_ids_to_infer:
                # Batch fetch stats using aggregation on subset
                pipeline = [
                    {'$match': {'batch_id': {'$in': batch_ids_to_infer}}},
                    {'$group': {
                        '_id': '$batch_id',
                        'count': {'$sum': 1},
                        'earliest': {'$min': '$created_at'}
                    }}
                ]
                batch_stats = {doc['_id']: doc for doc in result_coll.aggregate(pipeline)}
                
                # Batch fetch history (small) and ONE config per batch using $group
                history_map = {doc.get('batch_id'): doc for doc in mongo.db['optimize_history'].find({'batch_id': {'$in': batch_ids_to_infer}})}
                
                # Get ONE config per batch_id using aggregation (instead of fetching all)
                config_pipeline = [
                    {'$match': {'batch_id': {'$in': batch_ids_to_infer}}},
                    {'$group': {
                        '_id': '$batch_id',
                        'doc': {'$first': '$$ROOT'}
                    }}
                ]
                config_map = {doc['_id']: doc['doc'] for doc in config_coll.aggregate(config_pipeline)}
                
                for bid in batch_ids_to_infer:
                    stats = batch_stats.get(bid, {})
                    earliest_date = stats.get('earliest')
                    count = stats.get('count', 0)
                    
                    # Use pre-fetched config
                    cfg_doc = config_map.get(bid)
                    if cfg_doc and 'params' in cfg_doc:
                        p = cfg_doc['params']
                        ps = p.get('strategy', {}).get('ps', {})
                        bse = p.get('strategy', {}).get('bse', {})
                        metadata = cfg_doc.get('metadata', {})
                        is_on_going_val = bse.get('is_on_going', 0.8)
                        if isinstance(is_on_going_val, bool):
                            is_on_going_val = 0.8 if is_on_going_val else 0.0
                        config = {
                            'asset': metadata.get('asset', p.get('asset', 'BTCUSDT')),
                            'indicator': {
                                'ema': {'start': p.get('length_ema'), 'end': p.get('length_ema'), 'step': 1},
                                'atr': {'start': p.get('length_atr'), 'end': p.get('length_atr'), 'step': 1},
                                'high_vf': {'start': p.get('long_vol_factor'), 'end': p.get('long_vol_factor'), 'step': 0.1},
                                'low_vf': {'start': p.get('short_vol_factor'), 'end': p.get('short_vol_factor'), 'step': 0.1},
                                'multiple': p.get('multiple', 1)
                            },
                            'ps': {
                                'ir': {'start': ps.get('ir'), 'end': ps.get('ir'), 'step': 0.01},
                                'er': {'start': ps.get('er'), 'end': ps.get('er'), 'step': 0.1},
                                'or': {'start': ps.get('or'), 'end': ps.get('or'), 'step': 0.05}
                            },
                            'timeframes': {
                                'start': metadata.get('timeframe', p.get('timeframe', '1h')),
                                'end': metadata.get('timeframe', p.get('timeframe', '1h')),
                                'step': '1h', 'unit': 'h'
                            },
                            'broker': {
                                'max_slot': p.get('max_slot', 1),
                                'skid': metadata.get('slippage_pct', p.get('skid', 0.3)),
                                'capital': p.get('capital', 1000000),
                                'commission': metadata.get('commission_pct', p.get('commission', 0.1)),
                                'is_on_going': is_on_going_val
                            }
                        }
                    else:
                        config = {}
                    
                    history_entry = history_map.get(bid)
                    filters = history_entry.get('filters', []) if history_entry else []
                    
                    if 'timeframes' in config:
                        config['timeframes'] = _normalize_timeframes_to_dict(config['timeframes'])
                    
                    try:
                        ema_start = config.get('indicator', {}).get('ema', {}).get('start', 5)
                        atr_start = config.get('indicator', {}).get('atr', {}).get('start', 10)
                        lowvf_start = config.get('indicator', {}).get('low_vf', {}).get('start', 0.5)
                        prefix = "wfa" if type == "wfa" else "kema"
                        friendly_name = f"{prefix}_{ema_start}_{atr_start}_{lowvf_start}"
                    except:
                        friendly_name = bid
                    
                    inferred_campaigns.append({
                        'id': bid,
                        'friendly_name': friendly_name,
                        'createdAt': earliest_date.isoformat() if earliest_date and isinstance(earliest_date, datetime) else datetime.now().isoformat(),
                        'type': type,
                        'status': 'SUCCESS',
                        'config': config,
                        'stats': {'total': count, 'completed': count, 'pending': 0},
                        'filters': filters,
                        'source_batch_id': config.get('source_batch_id')
                    })
        
        # Combine history campaigns and inferred campaigns
        all_campaigns = []
        
        # Pre-fetch result counts for all history batch_ids to avoid N+1 queries
        history_batch_ids = [c.get('batch_id') for c in campaigns_from_history if c.get('batch_id')]
        result_counts_pipeline = [
            {'$match': {'batch_id': {'$in': history_batch_ids}}},
            {'$group': {'_id': '$batch_id', 'count': {'$sum': 1}}}
        ]
        result_counts = {doc['_id']: doc['count'] for doc in result_coll.aggregate(result_counts_pipeline)}
        
        # Pre-fetch config existence for pending campaigns
        pending_bids = [c.get('batch_id') for c in campaigns_from_history if c.get('status') in ['pending', 'generating', 'generated']]
        config_exists = set()
        if pending_bids:
            for doc in config_coll.find({'batch_id': {'$in': pending_bids}}, {'batch_id': 1}):
                config_exists.add(doc.get('batch_id'))
        
        # Format campaigns from history
        for campaign in campaigns_from_history:
            bid = campaign.get('batch_id')
            
            # 🔧 FIX DUPLICATE: Skip campaigns that have no configs and no results
            db_status = campaign.get('status', 'pending')
            if db_status in ['pending', 'generating', 'generated']:
                has_configs = bid in config_exists
                has_results = result_counts.get(bid, 0) > 0
                
                # Skip if no configs and no results (orphaned/replaced campaign)
                if not has_configs and not has_results:
                    print(f"⏩ Skipping orphaned campaign {bid} (no configs, no results)")
                    continue
            
            # 🔧 FIX RACE CONDITION: Check progress_tracker (in-memory) first
            # If batch is in progress_tracker, use its status (more up-to-date than MongoDB)
            if bid in progress_tracker:
                tracker_status = progress_tracker[bid].get('status', '')
                # Map tracker status to campaign status
                if tracker_status == 'completed':
                    db_status = 'completed'
                elif tracker_status == 'running':
                    db_status = 'running'
                elif tracker_status == 'processing':
                    db_status = 'completed'  # Processing = post-optimization, treat as completed
                elif tracker_status == 'failed':
                    db_status = 'failed'
                print(f"🔍 [Campaign List] {bid}: DB status='{campaign.get('status')}' -> Using tracker status='{tracker_status}' -> Final='{db_status}'")
            
            # Determine status based on campaign status field
            # Get progress stats (prefer history totals; never shrink total)
            progress = campaign.get('progress', {}) or {}
            generation = campaign.get('generation', {}) or {}
            hist_total = progress.get('total', 0) or generation.get('total_inserted', 0) or 0
            total = int(hist_total or 0)
            completed = int(progress.get('completed', 0) or 0)

            # If campaign is marked stopped but results are already complete, normalize to completed.
            if db_status == 'stopped':
                results_count = result_counts.get(bid, 0)  # Use pre-fetched count
                # Treat as completed if results cover the expected total.
                if results_count > 0 and (total == 0 or results_count >= total):
                    db_status = 'completed'
                    completed = results_count
                    total = int(max(total, completed))
                    try:
                        mongo.db['optimize_history'].update_one(
                            {'batch_id': bid},
                            {
                                '$set': {
                                    'status': 'completed',
                                    'progress.completed': int(completed),
                                    'progress.total': int(total),
                                    'progress.percentage': 100.0,
                                    'progress.eta_seconds': 0.0,
                                    'progress.eta_at': datetime.utcnow(),
                                    'progress.message': 'Hoàn tất!',
                                    'updated_at': datetime.utcnow(),
                                }
                            }
                        )
                    except Exception:
                        pass

            status_map = {
                'pending': 'PENDING',
                'generated': 'GENERATED',
                'running': 'RUNNING',
                'completed': 'SUCCESS',
                'success': 'SUCCESS',
                'failed': 'FAILED',
                'stopped': 'STOPPED'
            }
            campaign_status = status_map.get(db_status, 'PENDING')
            
            # If status is generated and total is unknown, avoid heavy counts in list view.
            # Runner/progress endpoint will correct totals shortly.
            if campaign_status == 'GENERATED' and total == 0:
                completed = 0
            
            # Fetch completed count from results if completed; keep total as campaign total (configs)
            if campaign_status == 'SUCCESS':
                result_coll = mongo.wfa_result if type == 'wfa' else mongo.backtest_result
                count = result_coll.count_documents({'batch_id': bid})
                if count > 0:
                    completed = int(count)
                    total = int(max(total, completed))
            
            stats = {
                'total': total,
                'completed': completed,
                'pending': total - completed
            }
            
            # Get config if not present or incomplete
            config = campaign.get('config', {})
            if not config or not isinstance(config, dict) or not config.get('asset'):
                # Use centralized reconstruction
                config = reconstruct_config_from_db(mongo, bid, type)
            
            # Normalize timeframes: Handle list format from optimize_history
            if 'timeframes' in config:
                config['timeframes'] = _normalize_timeframes_to_dict(config['timeframes'])

            all_campaigns.append({
                'id': bid,
                'friendly_name': campaign.get('friendly_name', bid),
                'createdAt': campaign.get('created_at', datetime.now()).isoformat() if isinstance(campaign.get('created_at'), datetime) else str(campaign.get('created_at', '')),
                'type': type,
                'status': campaign_status,
                'config': config,
                'stats': stats,
                'filters': campaign.get('filters', []),
                'source_batch_id': campaign.get('source_batch_id') or config.get('source_batch_id')  # For WFA campaigns
            })
        
        # Add inferred campaigns
        all_campaigns.extend(inferred_campaigns)

        # Add new WFA analysis campaigns (dedupe by id)
        if wfa_analysis_campaigns:
            existing_ids = {c.get('id') for c in all_campaigns}
            for c in wfa_analysis_campaigns:
                if c.get('id') and c.get('id') not in existing_ids:
                    all_campaigns.append(c)
        
        # Sort by createdAt desc
        all_campaigns.sort(key=lambda x: x['createdAt'], reverse=True)
        response = all_campaigns[:limit]
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_list'])
        return response
    
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection error in list_campaigns: {str(e)}")
        raise MongoConnectionError(
            detail="Database connection failed. Please ensure MongoDB is running and accessible."
        )
    except Exception as e:
        logger.error(f"Error in list_campaigns: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list campaigns: {e.__class__.__name__}"
        )
    finally:
        if mongo:
            mongo.close()


@router.get("/pipeline-summary")
async def get_pipeline_summary(
    limit: int = Query(200, ge=1, le=500, description="Maximum recent docs to inspect per stage")
):
    """Lightweight pipeline status for unlocking the Result tab."""
    cache_key = _make_cache_key('pipeline-summary', limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = None
    try:
        mongo = MongoService()

        history_projection = {
            '_id': 0,
            'batch_id': 1,
            'friendly_name': 1,
            'collection_type': 1,
            'status': 1,
            'created_at': 1,
            'source_batch_id': 1,
        }

        backtests = []
        for doc in mongo.history.find(
            {'collection_type': 'backtest'},
            history_projection,
        ).sort('created_at', -1).limit(limit):
            if _is_completed_status(doc.get('status')):
                backtests.append({
                    'id': doc.get('batch_id'),
                    'friendly_name': doc.get('friendly_name') or doc.get('batch_id'),
                    'createdAt': _to_iso(doc.get('created_at')),
                })

        wfa_docs: List[Dict[str, Any]] = []
        for doc in mongo.history.find(
            {'collection_type': 'wfa'},
            history_projection,
        ).sort('created_at', -1).limit(limit):
            if _is_completed_status(doc.get('status')) and doc.get('source_batch_id'):
                wfa_docs.append({
                    'id': doc.get('batch_id'),
                    'friendly_name': doc.get('friendly_name') or doc.get('batch_id'),
                    'source_batch_id': doc.get('source_batch_id'),
                    'createdAt': _to_iso(doc.get('created_at')),
                })

        analysis_projection = {
            '_id': 0,
            'batch_id': 1,
            'status': 1,
            'mode': 1,
            'source_batch_id': 1,
            'created_at': 1,
            'result': 1,
        }
        for doc in mongo.wfa_analysis.find(
            {'mode': {'$in': ['wfa', 'WFA']}},
            analysis_projection,
        ).sort('created_at', -1).limit(limit):
            status_value = doc.get('status') or ('completed' if doc.get('result') else '')
            if _is_completed_status(status_value) and doc.get('source_batch_id'):
                wfa_docs.append({
                    'id': doc.get('batch_id'),
                    'friendly_name': doc.get('batch_id'),
                    'source_batch_id': doc.get('source_batch_id'),
                    'createdAt': _to_iso(doc.get('created_at')),
                })

        wfas_by_id: Dict[str, Dict[str, Any]] = {}
        for doc in sorted(wfa_docs, key=lambda item: item.get('createdAt') or '', reverse=True):
            if doc.get('id') and doc.get('id') not in wfas_by_id:
                wfas_by_id[doc['id']] = doc
        wfas = list(wfas_by_id.values())

        carlo_results = list(
            mongo.carlo_result.find(
                {},
                {'_id': 0, 'batch_id': 1, 'config_hash': 1, 'created_at': 1, 'status': 1},
            ).sort('created_at', -1).limit(limit)
        )
        config_hashes = list({
            doc.get('config_hash')
            for doc in carlo_results
            if doc.get('config_hash')
        })
        carlo_config_by_hash: Dict[str, Dict[str, Any]] = {}
        if config_hashes:
            for doc in mongo.carlo_config.find(
                {'config_hash': {'$in': config_hashes}},
                {'_id': 0, 'config_hash': 1, 'config': 1},
            ):
                carlo_config_by_hash[doc.get('config_hash')] = doc.get('config') or {}

        carlos = []
        for doc in carlo_results:
            cfg = carlo_config_by_hash.get(doc.get('config_hash'), {})
            carlos.append({
                'id': doc.get('batch_id'),
                'source_campaign_id': cfg.get('source_campaign_id'),
                'source_type': cfg.get('source_type'),
                'createdAt': _to_iso(doc.get('created_at')),
            })

        wfas_by_source: Dict[str, List[Dict[str, Any]]] = {}
        for wfa in wfas:
            wfas_by_source.setdefault(wfa.get('source_batch_id'), []).append(wfa)

        carlos_by_source: Dict[str, Dict[str, Any]] = {}
        for carlo in sorted(carlos, key=lambda item: item.get('createdAt') or '', reverse=True):
            source_id = carlo.get('source_campaign_id')
            if source_id and source_id not in carlos_by_source:
                carlos_by_source[source_id] = carlo

        chains = []
        for backtest in backtests:
            bt_id = backtest.get('id')
            if not bt_id:
                continue
            for wfa in wfas_by_source.get(bt_id, []):
                carlo = carlos_by_source.get(wfa.get('id')) or carlos_by_source.get(bt_id)
                if not carlo:
                    continue
                chains.append({
                    'backtest_id': bt_id,
                    'wfa_id': wfa.get('id'),
                    'carlo_id': carlo.get('id'),
                    'createdAt': max(backtest.get('createdAt'), wfa.get('createdAt'), carlo.get('createdAt')),
                })

        chains.sort(key=lambda item: item.get('createdAt') or '', reverse=True)
        response = {
            'enabled': len(chains) > 0,
            'completedTriples': len(chains),
            'counts': {
                'backtests': len(backtests),
                'wfas': len(wfas),
                'carlos': len(carlos),
            },
            'latestChain': chains[0] if chains else None,
            'generatedAt': datetime.utcnow().isoformat(),
        }
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['pipeline_summary'])
        return response

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection error in get_pipeline_summary: {str(e)}")
        raise MongoConnectionError(
            detail="Database connection failed. Please ensure MongoDB is running and accessible."
        )
    except Exception as e:
        logger.error(f"Error in get_pipeline_summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load pipeline summary: {e.__class__.__name__}"
        )
    finally:
        if mongo:
            mongo.close()


@router.get("/{batch_id}")
async def get_campaign(batch_id: str):
    """Get campaign details by batch_id"""
    if not batch_id or not isinstance(batch_id, str) or len(batch_id) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid batch_id"
        )
    
    cache_key = _make_cache_key('detail', batch_id)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = None
    try:
        mongo = MongoService()
        # Check if this is a WFA/IOS analysis campaign (stored in wfa-analysis collection)
        wfa_doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if wfa_doc:
            # Normalize analysis result into fields expected by frontend views
            res = wfa_doc.get('result') or {}
            mode = (wfa_doc.get('mode') or 'wfa').lower()

            wfa_analysis = None
            ios_analysis = None

            if isinstance(res, dict) and mode == 'wfa':
                analysis = res.get('analysis') or {}
                windows_src = (analysis.get('windows') or {}) if isinstance(analysis, dict) else {}

                windows_norm = {}
                if isinstance(windows_src, dict):
                    for wid, w in windows_src.items():
                        if not isinstance(w, dict):
                            continue
                        period = w.get('period') or {}
                        # Use avg metrics when available
                        is_roi = w.get('avg_is_roi', 0) or 0
                        oos_roi = w.get('avg_oos_roi', 0) or 0
                        windows_norm[str(wid)] = {
                            'status': 'completed',
                            'period': period,
                            'is_metrics': {
                                'roi': float(is_roi) if isinstance(is_roi, (int, float)) else 0,
                                'sharpe': 0,
                                'mdd': 0,
                            },
                            'oos_metrics': {
                                'roi': float(oos_roi) if isinstance(oos_roi, (int, float)) else 0,
                                'sharpe': 0,
                                'mdd': 0,
                            },
                            'total_metrics': {},
                            'passed_filters': int(w.get('configs_count', 0) or 0),
                            'params': {
                                'best_config': w.get('best_config'),
                                'best_is_roi': w.get('best_is_roi'),
                                'best_oos_roi': w.get('best_oos_roi'),
                            },
                        }

                # Aggregate
                completed_windows = len(windows_norm)
                total_windows = int(res.get('windows', completed_windows) or completed_windows)
                oos_rois = [v.get('oos_metrics', {}).get('roi', 0) or 0 for v in windows_norm.values()]
                avg_oos_roi = (sum(oos_rois) / len(oos_rois)) if oos_rois else 0
                total_oos_roi = sum(oos_rois) if oos_rois else 0

                passed_configs = 0
                if isinstance(windows_src, dict):
                    for _wid, w in windows_src.items():
                        if isinstance(w, dict):
                            passed_configs += int(w.get('configs_count', 0) or 0)
                prog = wfa_doc.get('progress') or {}
                total_configs = int(prog.get('total', 0) or 0)
                passed_pct = (passed_configs / (total_configs or 1)) * 100.0

                wfa_analysis = {
                    'windows': windows_norm,
                    'aggregate': {
                        'completed_windows': completed_windows,
                        'total_windows': total_windows,
                        'avg_oos_roi': avg_oos_roi,
                        'total_oos_roi': total_oos_roi,
                        'passed_configs': int(passed_configs),
                        'passed_pct': float(passed_pct),
                    },
                }

            if isinstance(res, dict) and mode == 'ios':
                # IOS engine commonly returns selection stats under result['analysis'] or top-level keys
                ios_analysis = res.get('analysis') or res

            # If this is a WFA campaign (mode=wfa), try to attach the corresponding IOS analysis
            # so the UI can split results into IOS vs WFA tabs.
            if mode == 'wfa' and ios_analysis is None:
                ios_batch_id = wfa_doc.get('ios_batch_id') or (wfa_doc.get('config') or {}).get('ios_batch_id')
                ios_doc = None
                if ios_batch_id:
                    ios_doc = mongo.db['wfa-analysis'].find_one({'batch_id': ios_batch_id})

                if ios_doc is None:
                    source_batch_id = wfa_doc.get('source_batch_id')
                    if source_batch_id:
                        # Prefer an IOS doc created before/at this WFA run (closest prior).
                        wfa_created = wfa_doc.get('created_at')
                        query = {
                            'source_batch_id': source_batch_id,
                            'mode': {'$in': ['ios', 'IOS']},
                            'status': {'$in': ['completed', 'COMPLETED', 'success', 'SUCCESS']}
                        }
                        if isinstance(wfa_created, datetime):
                            query['created_at'] = {'$lte': wfa_created}

                        ios_doc = mongo.db['wfa-analysis'].find_one(query, sort=[('created_at', -1)])

                        # Fallback: latest IOS regardless of time filter
                        if ios_doc is None:
                            ios_doc = mongo.db['wfa-analysis'].find_one({
                                'source_batch_id': source_batch_id,
                                'mode': {'$in': ['ios', 'IOS']}
                            }, sort=[('created_at', -1)])

                if isinstance(ios_doc, dict):
                    ios_res = ios_doc.get('result') or {}
                    if isinstance(ios_res, dict):
                        ios_analysis = ios_res.get('analysis') or ios_res

            # Return WFA campaign in compatible format
            response = {
                'id': wfa_doc.get('batch_id'),
                'friendly_name': f"WFA_{(wfa_doc.get('mode', 'wfa') or 'wfa').upper()}_{wfa_doc.get('source_batch_id', '')[:8]}",
                'status': wfa_doc.get('status', 'UNKNOWN').upper(),
                'collection_type': 'wfa',
                'type': 'wfa',
                'mode': wfa_doc.get('mode'),
                'config': wfa_doc.get('config', {}),
                'source_batch_id': wfa_doc.get('source_batch_id'),
                'ios_batch_id': wfa_doc.get('ios_batch_id') or (wfa_doc.get('config') or {}).get('ios_batch_id'),
                'createdAt': wfa_doc.get('created_at').isoformat() if wfa_doc.get('created_at') else None,
                'updatedAt': wfa_doc.get('updated_at').isoformat() if wfa_doc.get('updated_at') else None,
                'current_stage': wfa_doc.get('current_stage'),
                'progress': wfa_doc.get('progress'),
                # Fields consumed by WFAResultsView (WFA) and IOS tab
                'wfa_analysis': wfa_analysis,
                'analysis': ios_analysis,
                'stats': {
                    'total': wfa_doc.get('progress', {}).get('total', 0),
                    'completed': wfa_doc.get('progress', {}).get('completed', 0),
                    'pending': max(0, wfa_doc.get('progress', {}).get('total', 0) - wfa_doc.get('progress', {}).get('completed', 0))
                },
                'filters': []
            }
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_detail'])
            return response
        
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        
        if not campaign:
            # Try to infer from results
            is_wfa = batch_id.startswith('WFA_')
            collection_type = 'wfa' if is_wfa else 'backtest'
            result_coll = mongo.wfa_result if is_wfa else mongo.backtest_result
            
            exists = result_coll.find_one({'batch_id': batch_id})
            if not exists:
                # Provide detailed 404 message for debugging
                checked_collections = ['wfa-analysis', 'optimize_history', 'wfa_result' if is_wfa else 'backtest_result']
                raise HTTPException(
                    status_code=404, 
                    detail=f"Campaign {batch_id} not found in any collection ({', '.join(checked_collections)}). It may have been deleted or never created."
                )
            
            # Create inferred campaign doc
            count = result_coll.count_documents({'batch_id': batch_id})
            
            # Use centralized reconstruction
            config = reconstruct_config_from_db(mongo, batch_id, collection_type)

            response = {
                'id': batch_id,
                'friendly_name': batch_id,
                'createdAt': exists.get('created_at', datetime.now()).isoformat() if isinstance(exists.get('created_at'), datetime) else str(exists.get('created_at', '')),
                'type': collection_type,
                'status': 'SUCCESS',
                'config': config,
                'stats': {'total': count, 'completed': count, 'pending': 0},
                'filters': []
            }
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_detail'])
            return response
        
        # Campaign exists in history
        campaign_type = campaign.get('collection_type', 'backtest')
        
        # Check if config is missing or incomplete
        config = campaign.get('config', {})
        if not config or not isinstance(config, dict) or not config.get('asset'):
            # Reconstruct from database
            config = reconstruct_config_from_db(mongo, batch_id, campaign_type)
        
        # Determine status (normalize STOPPED -> SUCCESS if results are complete)
        raw_status = campaign.get('status', 'pending')
        if raw_status == 'stopped':
            progress = campaign.get('progress', {}) or {}
            generation = campaign.get('generation', {}) or {}
            total_hint = int(progress.get('total', 0) or generation.get('total_inserted', 0) or 0)
            
            if campaign_type == 'wfa':
                result_coll = mongo.wfa_result
            else:
                result_coll = mongo.backtest_result
                
            results_count = int(result_coll.count_documents({'batch_id': batch_id}) or 0)
            if results_count > 0 and (total_hint == 0 or results_count >= total_hint):
                raw_status = 'completed'
                try:
                    mongo.db['optimize_history'].update_one(
                        {'batch_id': batch_id},
                        {
                            '$set': {
                                'status': 'completed',
                                'progress.completed': int(results_count),
                                'progress.total': int(max(total_hint, results_count)),
                                'progress.percentage': 100.0,
                                'progress.eta_seconds': 0.0,
                                'progress.eta_at': datetime.utcnow(),
                                'progress.message': 'Hoàn tất!',
                                'updated_at': datetime.utcnow(),
                            }
                        }
                    )
                except Exception:
                    pass

        status_map = {
            'pending': 'PENDING',
            'generated': 'GENERATED',
            'running': 'RUNNING',
            'completed': 'SUCCESS',
            'success': 'SUCCESS',
            'failed': 'FAILED',
            'paused': 'PAUSED',
            'stopped': 'STOPPED',
            'cancelled': 'CANCELLED'
        }
        status = status_map.get(raw_status, 'PENDING')
        
        result = {
            'id': campaign.get('batch_id'),
            'friendly_name': campaign.get('friendly_name', batch_id),
            'createdAt': campaign.get('created_at', datetime.now()).isoformat() if isinstance(campaign.get('created_at'), datetime) else str(campaign.get('created_at', '')),
            'type': campaign_type,
            'status': status,
            'config': config,
            'stats': campaign.get('summary', campaign.get('stats', {})),
            'filters': campaign.get('filters', [])
        }
        
        # Include analysis and wfa_analysis if available
        if 'analysis' in campaign:
            result['analysis'] = campaign['analysis']
        if 'wfa_analysis' in campaign:
            result['wfa_analysis'] = campaign['wfa_analysis']
        
        _cache_set(cache_key, result, _CACHE_TTL_SECONDS['campaign_detail'])
        return result
    
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection error in get_campaign({batch_id}): {str(e)}")
        raise MongoConnectionError(
            detail="Database connection failed. Please ensure MongoDB is running."
        )
    except Exception as e:
        logger.error(f"Error in get_campaign({batch_id}): {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get campaign: {type(e).__name__}"
        )
    finally:
        if mongo:
            mongo.close()


@router.get("/{batch_id}/results")
async def get_campaign_results(
    batch_id: str,
    limit: int = Query(200, description="Max results to return"),
    skip: int = Query(0, description="Number of results to skip"),
    sort_by: str = Query("roi", description="Field to sort by"),
    sort_order: int = Query(-1, description="Sort order: 1=asc, -1=desc"),
    filter_id: Optional[str] = Query(None, description="Filter ID to apply"),
    include_stats: bool = Query(True, description="Include full-batch aggregate stats")
):
    """Get results for a campaign, optionally filtered"""
    normalized_sort_by = sort_by if sort_by in _ALLOWED_SORT_FIELDS else 'roi'
    normalized_sort_order = -1 if sort_order != 1 else 1
    cache_key = _make_cache_key(
        'results',
        batch_id,
        limit,
        skip,
        normalized_sort_by,
        normalized_sort_order,
        filter_id or '',
        include_stats,
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = MongoService()
    try:
        # Check if this is a WFA analysis campaign
        wfa_doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if wfa_doc:
            # Return WFA results
            strategies = _wfa_extract_strategies(wfa_doc)
            response = {
                'results': strategies[skip:skip+limit] if strategies else [],
                'total': len(strategies),
                'limit': limit
            }
            if include_stats:
                response['stats'] = {'count': len(strategies)}
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_results'])
            return response
        
        # First get campaign to determine type
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        
        # Determine which collection to use
        if campaign:
            campaign_type = campaign.get('collection_type', 'backtest')
        else:
            if batch_id.startswith('WFA_'):
                campaign_type = 'wfa'
            else:
                campaign_type = 'backtest'
            
        if campaign_type == 'wfa':
            result_collection = mongo.wfa_result
        else:
            result_collection = mongo.backtest_result
        
        # Build query
        query = {
            'batch_id': batch_id,
            'status': {'$ne': 'failed'} 
        }

        # Apply filter rules if filter_id is provided
        if filter_id and campaign and 'filters' in campaign:
            # Find the filter metadata
            filter_meta = next((f for f in campaign['filters'] if str(f.get('id')) == filter_id), None)
            if filter_meta and 'rules' in filter_meta:
                rules = filter_meta['rules']
                for rule in rules:
                    metric = rule.get('metric')
                    op = rule.get('operator')
                    val = rule.get('value')
                    
                    if metric in _ALLOWED_FILTER_METRICS and op in _ALLOWED_FILTER_OPERATORS and val is not None:
                        mongo_op = {
                            '>': '$gt', '>=': '$gte', '<': '$lt', '<=': '$lte', '==': '$eq'
                        }.get(op, '$eq')
                        try:
                            numeric_value = float(val)
                        except (TypeError, ValueError):
                            continue
                        query[f"result.all.{metric}"] = {mongo_op: numeric_value}

        projection = {
            '_id': 0,
            'config_hash': 1,
            'result.all': 1,
            'result.roi': 1,
            'result.winRate': 1,
            'result.sharpe': 1,
            'result.mdd': 1,
            'result.cagr': 1,
            'params': 1,
        }
        sort_field = f"result.all.{normalized_sort_by}"

        # Get results
        results = list(result_collection.find(query, projection).sort(sort_field, normalized_sort_order).skip(skip).limit(limit))

        config_hashes = [r.get('config_hash') for r in results if r.get('config_hash')]
        config_map: Dict[str, Dict[str, Any]] = {}
        if config_hashes:
            config_coll = mongo.wfa_config if campaign_type == 'wfa' else mongo.backtest_config
            config_docs = config_coll.find(
                {'config_hash': {'$in': list(set(config_hashes))}},
                {'_id': 0, 'config_hash': 1, 'params': 1}
            )
            config_map = {
                doc.get('config_hash'): (doc.get('params') or {})
                for doc in config_docs
                if doc.get('config_hash')
            }
        
        # Format results
        formatted_results = []
        for result in results:
            # FIX Structure: stats are in result.all, params are in results (saved in post_process)
            # However, looking at the actual data, params might be missing in result doc if not saved correctly
            # Let's check where params are. In post_process.py, it's NOT saved in result doc.
            # BUT in generation.py, it IS saved in config doc.
            
            metrics = result.get('result', {}).get('all', {})
            
            # If params missing in result doc, try to find in config doc
            params = result.get('params')
            if not params:
                params = config_map.get(result.get('config_hash'))
            
            if not params:
                params = {}
            
            # FIX: Lấy timeframe từ params (chính xác), không lấy từ metrics (có thể sai)
            timeframe_value = params.get('timeframe') or metrics.get('timeframe', 'N/A')
            
            # 🔧 FIX: Use 'is not None' check for params that can be 0
            metrics_or = metrics.get('or')
            params_or = params.get('strategy', {}).get('ps', {}).get('or')
            or_value = metrics_or if metrics_or is not None else params_or
            
            metrics_ir = metrics.get('ir')
            params_ir = params.get('strategy', {}).get('ps', {}).get('ir')
            ir_value = metrics_ir if metrics_ir is not None else params_ir
            
            metrics_er = metrics.get('er')
            params_er = params.get('strategy', {}).get('ps', {}).get('er')
            er_value = metrics_er if metrics_er is not None else params_er
            
            formatted_results.append({
                'config_hash': result.get('config_hash'),
                'asset': params.get('asset', 'N/A'),
                'timeframe': timeframe_value,
                'ema': metrics.get('ema') or params.get('length_ema'),
                'atr': metrics.get('atr') or params.get('length_atr'),
                'highSF': metrics.get('highVf') or params.get('long_vol_factor'),
                'lowSF': metrics.get('lowVf') or params.get('short_vol_factor'),
                'ir': ir_value,
                'er': er_value,
                'or': or_value,
                'skid': metrics.get('skid') or params.get('slippage_pct') or 0.4,
                'commission': metrics.get('commissionPct') or params.get('commission_pct') or 0.1,
                'roi': metrics.get('roi', 0),
                'winRate': metrics.get('winRate', 0),
                'totalTrades': metrics.get('totalTrades', 0),
                'profit': metrics.get('profit', 0),
                'finalEquity': metrics.get('finalEquity', 0),
                'mdd': metrics.get('mdd', 0),
                'sharpe': metrics.get('sharpe', 0),
                'sortino': metrics.get('sortino', 0),
                'cagr': metrics.get('cagr', 0),
                'maxLeverage': metrics.get('maxLeverage', 0),
                'maxConsecutiveLosses': metrics.get('maxConsecutiveLosses', 0),
                'maxDrawdownDuration': metrics.get('maxDrawdownDuration', 0),
                'dateRange': metrics.get('dateRange', ''),
                'status': 'success'
            })
        
        # Get total count
        total_count = result_collection.count_documents(query)
        
        # Calculate Aggregation Stats (for full data summary)
        stats_pipeline = [
            {'$match': query},
            {
                '$group': {
                    '_id': None,
                    'avgRoi': {'$avg': '$result.all.roi'},
                    'avgWinRate': {'$avg': '$result.all.winRate'},
                    'avgMDD': {'$avg': '$result.all.mdd'},
                    'avgSharpe': {'$avg': '$result.all.sharpe'},
                    'avgCagr': {'$avg': '$result.all.cagr'},
                    'count': {'$sum': 1}
                }
            }
        ]
        
        agg_stats = None
        if include_stats:
            agg_stats = {}
            try:
                agg_res = list(result_collection.aggregate(stats_pipeline))
                if agg_res:
                    s = agg_res[0]
                    agg_stats = {
                        'count': s.get('count', 0),
                        'avgRoi': s.get('avgRoi', 0),
                        'avgWinRate': s.get('avgWinRate', 0),
                        'avgMDD': s.get('avgMDD', 0),
                        'avgSharpe': s.get('avgSharpe', 0),
                        'avgCagr': s.get('avgCagr', 0)
                    }
            except Exception as e:
                print(f"Error calculating stats: {e}")

        response = {
            'results': formatted_results,
            'total': total_count,
            'limit': limit,
            'skip': skip
        }
        if include_stats:
            response['stats'] = agg_stats
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['campaign_results'])
        return response
        
    finally:
        mongo.close()


@router.delete("/{batch_id}")
async def delete_campaign(batch_id: str):
    """Delete a campaign and all its results"""
    mongo = MongoService()
    try:
        # Support deleting WFA/IOS analysis jobs stored in `wfa-analysis`.
        analysis_deleted = 0

        try:
            analysis_deleted = mongo.db['wfa-analysis'].delete_one({'batch_id': batch_id}).deleted_count
        except Exception:
            analysis_deleted = 0

        # First get campaign to determine type
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        
        # Determine campaign type
        if campaign:
            campaign_type = campaign.get('collection_type', 'backtest')
        else:
            if batch_id.startswith('WFA_'):
                campaign_type = 'wfa'
            else:
                campaign_type = 'backtest'
        
        if campaign_type == 'wfa':
            config_collection = mongo.wfa_config
            result_collection = mongo.wfa_result
        else:
            config_collection = mongo.backtest_config
            result_collection = mongo.backtest_result
        
        # Delete all related data
        config_deleted = config_collection.delete_many({'batch_id': batch_id})
        result_deleted = result_collection.delete_many({'batch_id': batch_id})
        history_deleted = mongo.db['optimize_history'].delete_one({'batch_id': batch_id})
        
        print(f"DEBUG DELETE {batch_id}: type={campaign_type}, config={config_deleted.deleted_count}, result={result_deleted.deleted_count}, history={history_deleted.deleted_count}")
        
        # Always return success if something was found in either place
        if config_deleted.deleted_count == 0 and result_deleted.deleted_count == 0 and history_deleted.deleted_count == 0 and analysis_deleted == 0:
             raise HTTPException(status_code=404, detail=f"Campaign {batch_id} not found in results or history")

        _cache_invalidate(_make_cache_key('detail', batch_id))
        _cache_invalidate(_make_cache_key('results', batch_id))
        _cache_invalidate(_make_cache_key('top', batch_id))
        _cache_invalidate(_make_cache_key('chart_data', batch_id))
        _cache_invalidate('list:')
        _cache_invalidate('stats:')
        
        return {
            'success': True,
            'deleted': {
                'configs': config_deleted.deleted_count,
                'results': result_deleted.deleted_count,
                'history': history_deleted.deleted_count,
                'analysis': analysis_deleted,
            }
        }
        
    finally:
        mongo.close()


@router.post("/{batch_id}/filter")
async def save_campaign_filter(batch_id: str, filter_data: Dict[str, Any]):
    """Save a filter result to campaign history"""
    mongo = MongoService()
    try:
        # Check if campaign exists in history
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        
        print(f"[FILTER] Saving filter for batch_id: {batch_id}")
        print(f"[FILTER] Campaign exists: {campaign is not None}")
        print(f"[FILTER] Filter data: {filter_data}")
        
        filter_entry = {
            'id': filter_data.get('id', int(datetime.utcnow().timestamp() * 1000)),
            'expression': filter_data.get('expression'),
            'rules': filter_data.get('rules'),
            'stats': filter_data.get('stats'),
            'matched': filter_data.get('matched', 0),
            'total': filter_data.get('total', 0),
            'percentage': filter_data.get('percentage', 0),
            'createdAt': datetime.utcnow().isoformat()
        }
        
        if campaign:
            # Append to filters list
            result = mongo.db['optimize_history'].update_one(
                {'batch_id': batch_id},
                {'$push': {'filters': filter_entry}}
            )
            print(f"[FILTER] Updated existing campaign: {result.modified_count} modified")
        else:
            # Create a basic history entry if it doesn't exist (inferred campaign)
            is_wfa = batch_id.startswith('WFA_')
            new_history = {
                'batch_id': batch_id,
                'collection_type': 'wfa' if is_wfa else 'backtest',
                'status': 'completed',
                'created_at': datetime.utcnow(),
                'filters': [filter_entry]
            }
            result = mongo.db['optimize_history'].insert_one(new_history)
            print(f"[FILTER] Created new campaign: {result.inserted_id}")

        _cache_invalidate(_make_cache_key('detail', batch_id))
        _cache_invalidate(_make_cache_key('results', batch_id))
        _cache_invalidate('list:')
            
        return {"success": True, "filter": filter_entry}
        
    finally:
        mongo.close()


@router.post("/{batch_id}/execute-filter")
async def execute_campaign_filter(batch_id: str, filter_data: Dict[str, Any]):
    """Execute filter logic on server-side and save result"""
    mongo = MongoService()
    try:
        rules = filter_data.get('rules', [])
        filter_id = filter_data.get('id', int(datetime.utcnow().timestamp() * 1000))
        
        # Determine campaign type
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        if campaign:
            campaign_type = campaign.get('collection_type', 'backtest')
        else:
            if batch_id.startswith('WFA_'):
                campaign_type = 'wfa'
            else:
                campaign_type = 'backtest'
            
        if campaign_type == 'wfa':
            result_collection = mongo.wfa_result
        else:
            result_collection = mongo.backtest_result
        
        # Build aggregation pipeline
        match_stage = {'batch_id': batch_id, 'status': {'$ne': 'failed'}}
        
        for rule in rules:
            metric = rule.get('metric')
            op = rule.get('operator')
            val = rule.get('value')
            
            if metric in _ALLOWED_FILTER_METRICS and op in _ALLOWED_FILTER_OPERATORS and val is not None:
                mongo_op = {
                    '>': '$gt', '>=': '$gte', '<': '$lt', '<=': '$lte', '==': '$eq'
                }.get(op, '$eq')
                try:
                    numeric_value = float(val)
                except (TypeError, ValueError):
                    continue
                match_stage[f"result.all.{metric}"] = {mongo_op: numeric_value}
        
        pipeline = [
            {'$match': match_stage},
            {'$group': {
                '_id': None,
                'count': {'$sum': 1},
                'avgRoi': {'$avg': '$result.all.roi'},
                'avgWinRate': {'$avg': '$result.all.winRate'},
                'avgSharpe': {'$avg': '$result.all.sharpe'},
                'avgMdd': {'$avg': '$result.all.mdd'},
                'avgCagr': {'$avg': '$result.all.cagr'}
            }}
        ]
        
        agg_results = list(result_collection.aggregate(pipeline))
        
        if not agg_results:
            stats = {'count': 0, 'avgRoi': 0, 'avgWinRate': 0, 'avgSharpe': 0, 'avgMdd': 0, 'avgCagr': 0}
        else:
            res = agg_results[0]
            stats = {
                'count': res.get('count', 0),
                'avgRoi': round(res.get('avgRoi', 0) or 0, 2),
                'avgWinRate': round(res.get('avgWinRate', 0) or 0, 2),
                'avgSharpe': round(res.get('avgSharpe', 0) or 0, 3),
                'avgMdd': round(res.get('avgMdd', 0) or 0, 2),
                'avgCagr': round(res.get('avgCagr', 0) or 0, 2)
            }
            
        # Get GLOBAL total for percentage (not just currently finished results)
        global_total = (
            campaign.get('summary', {}).get('total') or 
            campaign.get('progress', {}).get('total') or 
            0
        ) if campaign else 0
        
        # If progress.total not found, fallback to finished results count
        if global_total <= 0:
            global_total = result_collection.count_documents({'batch_id': batch_id, 'status': {'$ne': 'failed'}})
            
        matched_count = stats['count']
        percentage = (matched_count / global_total * 100) if global_total > 0 else 0
        
        # Helper to build expression string (same as frontend)
        # Note: In a real app, we might want to share this logic
        metric_map = {
            'profit': 'Profit ($)', 'winRate': 'Win Rate (%)', 'cagr': 'CAGR (%)',
            'mdd': 'MDD (%)', 'roi': 'ROI (%)', 'sharpe': 'Sharpe'
        }
        
        expression_parts = []
        for rule in rules:
            label = metric_map.get(rule['metric'], rule['metric'])
            expression_parts.append(f"{label} {rule['operator']} {rule['value']}")
        expression = " AND ".join(expression_parts)
        
        filter_entry = {
            'id': filter_id,
            'expression': expression,
            'rules': rules,
            'stats': stats,
            'matched': matched_count,
            'total': global_total,
            'percentage': round(percentage, 2),
            'createdAt': datetime.utcnow().isoformat()
        }
        
        # Save to optimize_history
        if campaign:
            mongo.db['optimize_history'].update_one(
                {'batch_id': batch_id},
                {'$push': {'filters': filter_entry}}
            )
        else:
            # Create inferred campaign if needed
            new_history = {
                'batch_id': batch_id,
                'collection_type': campaign_type,
                'status': 'completed',
                'created_at': datetime.utcnow(),
                'filters': [filter_entry]
            }
            mongo.db['optimize_history'].insert_one(new_history)

        _cache_invalidate(_make_cache_key('detail', batch_id))
        _cache_invalidate(_make_cache_key('results', batch_id))
        _cache_invalidate('list:')
            
        return {"success": True, "filter": filter_entry}
        
    finally:
        mongo.close()


@router.delete("/{batch_id}/filters/{filter_id}")
async def delete_campaign_filter(batch_id: str, filter_id: int):
    """Delete a specific filter from campaign history"""
    mongo = MongoService()
    try:
        # Remove filter from optimize_history
        result = mongo.db['optimize_history'].update_one(
            {'batch_id': batch_id},
            {'$pull': {'filters': {'id': filter_id}}}
        )
        
        if result.modified_count > 0:
            _cache_invalidate(_make_cache_key('detail', batch_id))
            _cache_invalidate(_make_cache_key('results', batch_id))
            _cache_invalidate('list:')
            return {"success": True, "message": "Filter deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Filter not found")
            
    finally:
        mongo.close()


@router.get("/{batch_id}/top-strategies")
async def get_top_strategies(
    batch_id: str,
    limit: int = Query(20, description="Number of top strategies to return")
):
    """
    Lấy top strategies thực sự từ toàn bộ database (không giới hạn 200k)
    Sắp xếp theo ROI giảm dần
    """
    cache_key = _make_cache_key('top', batch_id, limit)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = MongoService()
    try:
        # Check if this is a WFA analysis campaign
        wfa_doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id})
        if wfa_doc:
            strategies = _wfa_extract_strategies(wfa_doc)
            response = {
                'top_strategies': strategies[:limit] if strategies else [],
                'total': len(strategies),
                'limit': limit
            }
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['top_strategies'])
            return response
        
        # Xác định campaign type
        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id})
        if not campaign:
            raise HTTPException(status_code=404, detail=f"Campaign {batch_id} not found")
        
        campaign_type = campaign.get('collection_type', 'backtest')
        
        if campaign_type == 'wfa':
            result_collection = mongo.wfa_result
            config_collection = mongo.wfa_config
        else:
            result_collection = mongo.backtest_result
            config_collection = mongo.backtest_config
        
        # Query top strategies từ database
        # NOTE: Projection cannot include both "result" and "result.all" due to MongoDB path collision.
        success_query = {
            'batch_id': batch_id,
            'status': {'$in': ['success', 'SUCCESS']}
        }
        projection = {'_id': 0, 'config_hash': 1, 'result': 1}

        # Prefer nested ROI; fallback to flat ROI for legacy documents.
        raw_results = list(
            result_collection
            .find(success_query, projection)
            .sort('result.all.roi', -1)
            .limit(limit)
        )

        if not raw_results:
            raw_results = list(
                result_collection
                .find(success_query, projection)
                .sort('result.roi', -1)
                .limit(limit)
            )

        config_hashes = [r.get('config_hash') for r in raw_results if r.get('config_hash')]
        config_map: Dict[str, Dict[str, Any]] = {}
        if config_hashes:
            config_docs = config_collection.find(
                {'config_hash': {'$in': list(set(config_hashes))}},
                {'_id': 0, 'config_hash': 1, 'params': 1}
            )
            config_map = {
                doc.get('config_hash'): (doc.get('params') if isinstance(doc.get('params'), dict) else {})
                for doc in config_docs
                if doc.get('config_hash')
            }
        
        formatted_results = []
        
        # Enrich với config params và flatten metrics
        for result in raw_results:
            config_hash = result.get('config_hash')
            
            # Linh hoạt lấy metrics từ result.all (Structure B) hoặc trực tiếp từ result (Structure A)
            res_obj = result.get('result', {})
            if isinstance(res_obj, dict) and 'all' in res_obj:
                metrics = res_obj.get('all', {})
            else:
                metrics = res_obj if isinstance(res_obj, dict) else {}
            
            # Khởi tạo row với các metrics cơ bản và các keys fallback
            row = {
                'config_hash': config_hash,
                'roi': metrics.get('roi') or metrics.get('total_roi') or 0,
                'winRate': metrics.get('winRate') or metrics.get('win_rate') or metrics.get('positive_ratio', 0),
                'totalTrades': metrics.get('totalTrades') or metrics.get('total_trades', 0),
                'profit': metrics.get('profit') or metrics.get('total_profit') or metrics.get('net_profit', 0),
                'finalEquity': metrics.get('finalEquity') or metrics.get('final_equity', 0),
                'mdd': metrics.get('mdd') or metrics.get('max_drawdown') or metrics.get('max_drawdown_pct', 0),
                'sharpe': metrics.get('sharpe') or metrics.get('sharpe_ratio', 0),
                'sortino': metrics.get('sortino', metrics.get('sortino_ratio')),
                'cagr': metrics.get('cagr') or metrics.get('annualized_return', 0),
                'maxLeverage': metrics.get('maxLeverage') or metrics.get('max_leverage', 0),
                'maxConsecutiveLosses': metrics.get('maxConsecutiveLosses') or metrics.get('max_consecutive_losses', 0),
                'maxDrawdownDuration': metrics.get('maxDrawdownDuration') or metrics.get('max_drawdown_duration', 0),
                'status': 'success'
            }
            
            if config_hash:
                params = config_map.get(config_hash)
                if isinstance(params, dict) and params:
                    indicator = params.get('strategy', {}).get('indicator', {})
                    ps = params.get('strategy', {}).get('ps', {})
                    broker = params.get('broker', {})
                    
                    # Thêm params vào row với fallback cho các keys cũ
                    row['ema'] = indicator.get('ema') or params.get('length_ema')
                    row['atr'] = indicator.get('atr') or params.get('length_atr')
                    row['highSF'] = indicator.get('high_vf') or indicator.get('highVf') or params.get('long_vol_factor')
                    row['lowSF'] = indicator.get('low_vf') or indicator.get('lowVf') or params.get('short_vol_factor')
                    row['timeframe'] = params.get('timeframe')
                    row['commission'] = broker.get('commission') or params.get('commission_pct')
                    row['ir'] = ps.get('ir')
                    row['er'] = ps.get('er')
                    row['or'] = ps.get('or')
                    row['skid'] = broker.get('skid') or params.get('slippage_pct')
            
            formatted_results.append(row)
        
        # Đếm tổng số configs
        total_count = result_collection.count_documents(success_query)
        
        response = {
            'top_strategies': formatted_results,
            'total': total_count,
            'limit': limit
        }
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['top_strategies'])
        return response

    except Exception as e:
        logger.error(f"Error in get_top_strategies({batch_id}): {str(e)}", exc_info=True)
        # Degrade gracefully: keep UI responsive even if aggregation fails.
        return {
            'top_strategies': [],
            'total': 0,
            'limit': limit,
        }
        
    finally:
        mongo.close()


@router.get("/{batch_id}/chart-data")
async def get_chart_data(
    batch_id: str,
    points: int = Query(1000, description="Number of data points for chart")
):
    """
    Lấy dữ liệu cho biểu đồ từ toàn bộ database
    Sử dụng uniform sampling để giảm số lượng điểm
    """
    safe_points = max(100, min(points, 3000))
    cache_key = _make_cache_key('chart_data', batch_id, safe_points)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    mongo = MongoService()
    try:
        wfa_doc = mongo.db['wfa-analysis'].find_one({'batch_id': batch_id}, {'_id': 1})
        if wfa_doc:
            response = {
                'chart_data': [],
                'total_configs': 0,
                'sampling_method': 'none',
                'points_returned': 0,
            }
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['chart_data'])
            return response

        campaign = mongo.db['optimize_history'].find_one({'batch_id': batch_id}, {'_id': 0, 'collection_type': 1})
        if not campaign:
            raise HTTPException(status_code=404, detail=f"Campaign {batch_id} not found")

        campaign_type = campaign.get('collection_type', 'backtest')
        result_collection = mongo.wfa_result if campaign_type == 'wfa' else mongo.backtest_result
        base_query = {'batch_id': batch_id, 'status': 'success'}

        total_count = result_collection.count_documents(base_query)
        if total_count == 0:
            response = {
                'chart_data': [],
                'total_configs': 0,
                'sampling_method': 'none',
                'points_returned': 0,
            }
            _cache_set(cache_key, response, _CACHE_TTL_SECONDS['chart_data'])
            return response

        projection = {
            '_id': 0,
            'result.all': 1,
            'result.roi': 1,
            'result.winRate': 1,
            'result.sharpe': 1,
            'result.mdd': 1,
            'result.cagr': 1,
        }

        if total_count <= safe_points:
            raw_docs = list(result_collection.find(base_query, projection).sort('result.all.roi', -1))
            sampling_method = 'all'
        else:
            raw_docs = list(result_collection.find(base_query, projection).sort('result.all.roi', -1).limit(safe_points))
            sampling_method = 'top_roi_limit'

        chart_data = []
        for item in raw_docs:
            res_obj = item.get('result', {})
            metrics = res_obj.get('all', {}) if isinstance(res_obj, dict) and 'all' in res_obj else (res_obj if isinstance(res_obj, dict) else {})
            chart_data.append({
                'roi': metrics.get('roi') or metrics.get('total_roi') or 0,
                'winRate': metrics.get('winRate') or metrics.get('win_rate') or metrics.get('positive_ratio', 0),
                'sharpe': metrics.get('sharpe') or metrics.get('sharpe_ratio', 0),
                'mdd': metrics.get('mdd') or metrics.get('max_drawdown') or metrics.get('max_drawdown_pct', 0),
                'cagr': metrics.get('cagr') or metrics.get('annualized_return', 0),
            })

        response = {
            'chart_data': chart_data,
            'total_configs': total_count,
            'sampling_method': sampling_method,
            'points_returned': len(chart_data),
        }
        _cache_set(cache_key, response, _CACHE_TTL_SECONDS['chart_data'])
        return response
    finally:
        mongo.close()
