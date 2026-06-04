"""
Grey AI API Routes - FastAPI endpoints for AI advisor
"""

from __future__ import annotations

import asyncio
from typing import Optional, List, Dict
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import settings

# Attempt imports with fallback
_AI_IMPORT_ERROR: Optional[str] = None
try:
    # bot is at the root of noraquantengine
    import sys
    sys.path.append(str(settings.WORKSPACE_ROOT / "bot"))
    
    from ai_engine.advisor import GreyAdvisor
    from core.indicator import calculate_keltner_bands
except Exception as e:
    GreyAdvisor = None
    calculate_keltner_bands = None
    _AI_IMPORT_ERROR = f"{type(e).__name__}: {e}"

router = APIRouter(prefix="/api/ai", tags=["grey-ai"])

class AIAnalysisRequest(BaseModel):
    """Request model for AI analysis"""
    asset: str
    timeframe: str
    mode: str  # 'long' | 'short' | 'sideway'
    ema_length: int = 50
    atr_length: int = 14
    long_vol_factor: float = 2.0
    short_vol_factor: float = 2.0
    chart_data: Optional[List[Dict]] = None

def _perform_analysis_sync(request: AIAnalysisRequest):
    """Synchronous CPU-bound analysis logic."""
    df = pd.DataFrame(request.chart_data)
    if 'time' in df.columns:
        if pd.api.types.is_numeric_dtype(df['time']):
            tmax = float(df['time'].max()) if len(df) else 0.0
            unit = 'ms' if tmax > 1e12 else 's'
            df['time'] = pd.to_datetime(df['time'], unit=unit, utc=True)
        else:
            df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
    
    # Calculate indicators
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    
    ema, atr, upper_band, lower_band = calculate_keltner_bands(
        close=close,
        high=high,
        low=low,
        ema_length=request.ema_length,
        atr_length=request.atr_length,
        multiplier=request.long_vol_factor
    )
    
    # Recalculate bands with separate multipliers
    upper_band = ema + (atr * request.long_vol_factor)
    lower_band = ema - (atr * request.short_vol_factor)
    
    # Initialize and run advisor
    advisor = GreyAdvisor()
    return advisor.analyze_and_advise(
        mode=request.mode,
        df=df,
        ema=ema,
        upper_band=upper_band,
        lower_band=lower_band,
        atr=atr,
        ema_length=request.ema_length,
        atr_length=request.atr_length,
        current_trades=None
    )

@router.post("/analyze")
async def analyze_market(request: AIAnalysisRequest):
    """
    Grey AI Analysis Endpoint (Non-blocking)
    """
    if GreyAdvisor is None or calculate_keltner_bands is None:
        raise HTTPException(
            status_code=503,
            detail=f"Grey AI engine is not available ({_AI_IMPORT_ERROR})",
        )

    if request.mode not in ['long', 'short', 'sideway']:
        raise HTTPException(status_code=400, detail="Mode must be 'long', 'short', or 'sideway'")
    
    if not request.chart_data:
        raise HTTPException(status_code=400, detail="chart_data is required")
    
    if len(request.chart_data) < max(request.ema_length, request.atr_length) + 10:
        raise HTTPException(status_code=400, detail="Insufficient data for analysis")

    try:
        # Offload CPU-bound work to a separate thread to keep API responsive
        result = await asyncio.to_thread(_perform_analysis_sync, request)
        return {
            'success': True,
            'data': result
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@router.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'service': 'Grey AI Advisor',
        'version': settings.VERSION,
        'available': GreyAdvisor is not None,
    }

@router.get("/modes")
async def get_available_modes():
    return {
        'success': True,
        'modes': [
            {'id': 'long', 'name': 'Long Trading', 'icon': '📈'},
            {'id': 'short', 'name': 'Short Trading', 'icon': '📉'},
            {'id': 'sideway', 'name': 'Sideway/Range', 'icon': '↔️'}
        ]
    }
