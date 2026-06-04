"""
Chart Routes - FastAPI endpoints for chart data

Endpoints:
- GET /api/chart-data - Get OHLCV + indicators for chart rendering
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from services.chart_service import get_chart_data

router = APIRouter(prefix="/api", tags=["chart"])


class ChartDataResponse(BaseModel):
    """Response model for chart data"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


@router.get("/chart-data", response_model=ChartDataResponse)
async def fetch_chart_data(
    asset: str = Query(..., description="Asset symbol (e.g., BTCUSDT)"),
    timeframe: str = Query(..., description="Timeframe (1h, 4h, 1d)"),
    ema_length: int = Query(50, description="EMA period", ge=1, le=200),
    atr_length: int = Query(14, description="ATR period", ge=1, le=100),
    long_vol_factor: float = Query(2.0, description="Long volatility multiplier", ge=0.1, le=10.0),
    short_vol_factor: float = Query(2.0, description="Short volatility multiplier", ge=0.1, le=10.0),
    rsi_ema_len: int = Query(9, description="RSI EMA period", ge=1, le=100),
    rsi_wma_len: int = Query(45, description="RSI WMA period", ge=1, le=200),
    rsi_period: int = Query(14, description="RSI period", ge=2, le=100),
    rsi_signal_period: int = Query(14, description="RSI Signal period", ge=2, le=100),
    wma_smoothing: int = Query(3, description="WMA Delta smoothing", ge=1, le=20),
    max_bars: int = Query(2200, description="Maximum bars returned to frontend", ge=300, le=10000),
    include_sova: bool = Query(False, description="Enable optional Sova advisory call"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    data_type: str = Query("OKX", description="Data category (OKX, forex, XAU)")
):
    """
    Get chart data (OHLCV + indicators) for frontend rendering
    
    Returns:
        {
            "success": true,
            "data": {
                "candlestick": [
                    {"time": "2024-01-01T00:00:00Z", "open": 42000, "high": 42500, "low": 41800, "close": 42300, "volume": 1234}
                ],
                "indicators": {
                    "ema": [{"time": "2024-01-01T00:00:00Z", "value": 42100}],
                    "upper_band": [...],
                    "lower_band": [...],
                    "atr": [...]
                }
            }
        }
    """
    try:
        result = await get_chart_data(
            asset=asset,
            timeframe=timeframe,
            ema_length=ema_length,
            atr_length=atr_length,
            long_vol_factor=long_vol_factor,
            short_vol_factor=short_vol_factor,
            rsi_ema_len=rsi_ema_len,
            rsi_wma_len=rsi_wma_len,
            rsi_period=rsi_period,
            rsi_signal_period=rsi_signal_period,
            wma_smoothing=wma_smoothing,
            max_bars=max_bars,
            include_sova=include_sova,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type
        )
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Failed to get chart data'))
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
