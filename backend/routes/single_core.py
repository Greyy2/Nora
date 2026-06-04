"""
Single Core Routes - FastAPI endpoints for single strategy backtest

Endpoints:
- POST /api/single-core - Run single backtest (detailed result)
- GET /api/single-core/{id} - Get backtest result (future: from DB)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from services.backtest_service import run_backtest 
from database.mongo_service import MongoService

router = APIRouter(prefix="/api", tags=["single-core"])

class BacktestRequest(BaseModel):
    """Request model for backtest"""
    asset: str = Field(..., description="Asset (BTCUSDT, ETHUSDT, etc)")
    timeframe: str = Field(..., description="Timeframe (1h, 4h, 1d)")
    initial_capital: float = Field(1000000.0, description="Initial capital")
    commission_pct: float = Field(0.1, description="Commission %")
    risk_per_trade_pct: float = Field(0.02, description="Risk per trade %")
    max_risk_equity_pct: float = Field(0.50, description="Max risk equity %")
    ema_length: int = Field(50, description="EMA length")
    atr_length: int = Field(14, description="ATR length")
    multiplier: float = Field(2.0, description="Keltner Multiplier")
    long_vol_factor: Optional[float] = Field(2.0, description="Long Vol Matrix")
    short_vol_factor: Optional[float] = Field(2.0, description="Short Vol Matrix")
    trade_option: Optional[str] = Field('Both', description="Both, Long Only, Short Only")
    start_date: Optional[str] = Field(None, description="Start date(YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date(YYYY-MM-DD)")
    skid_pct: Optional[float] = Field(0.0, description="Slippage %")
    is_on_going: Optional[bool] = Field(True, description="Enable On-going risk")
    on_going_risk: Optional[float] = Field(0.95, description="On-going risk limit")
    use_delta: Optional[bool] = Field(None, description="Use delta-based position sizing (must match optimizer)")
    data_type: Optional[str] = Field("OKX", description="Data category (OKX, forex, XAU)")

class BacktestResponse(BaseModel):
    """Response model for backtest"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class ReplayRequest(BaseModel):
    """Replay a stored optimizer config exactly (no frontend mapping)."""
    batch_id: str = Field(..., description="Optimization batch_id")
    config_hash: str = Field(..., description="Config hash in backtest-config")

@router.post("/single-core", response_model=BacktestResponse)
async def create_single_backtest(request: BacktestRequest):
    """
    Run a new backtest

    Returns:
        {
            "success": bool,
            "data": {
                "chart": {...},
                "trades": {...},
                "metrics": {...},
                "equity_curve": {...}
            }
        }
    """
    try:
        print(f"🔍 [Single-Core] Request: asset={request.asset}, timeframe={request.timeframe}, side={request.trade_option}, type={request.data_type}")
        
        result = run_backtest(
            asset=request.asset,
            timeframe=request.timeframe,
            initial_capital=request.initial_capital,
            commission=request.commission_pct,
            slippage_pct=request.skid_pct,
            risk_per_trade_pct=request.risk_per_trade_pct,
            max_risk_equity_pct=request.max_risk_equity_pct,
            ema_length=request.ema_length,
            atr_length=request.atr_length,
            multiplier=request.multiplier,
            long_vol_factor=request.long_vol_factor,
            short_vol_factor=request.short_vol_factor,
            trade_option=request.trade_option,
            is_on_going=request.is_on_going,
            on_going_risk=request.on_going_risk,
            use_delta=request.use_delta,
            start_date=request.start_date,
            end_date=request.end_date,
            data_type=request.data_type
        )

        if not result['success']:
            print(f"❌ [Single-Core] Backtest failed: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get('error', 'Backtest failed'))

        print(f"✅ [Single-Core] Backtest completed successfully")
        return result

    except ValueError as e:
        print(f"❌ [Single-Core] ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        print(f"❌ [Single-Core] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/single-core/replay", response_model=BacktestResponse)
async def replay_single_backtest(request: ReplayRequest):
    """Replay a specific strategy from DB to ensure 100% parity with optimization."""
    mongo = MongoService()
    try:
        cfg = mongo.backtest_config.find_one(
            {'batch_id': request.batch_id, 'config_hash': request.config_hash},
            {'_id': 0, 'params': 1, 'metadata': 1}
        )
        if not cfg or not cfg.get('params'):
            raise HTTPException(status_code=404, detail='Config not found')

        params = cfg['params']
        metadata = cfg.get('metadata') or {}

        hist = mongo.db['optimize_history'].find_one(
            {'batch_id': request.batch_id},
            {'_id': 0, 'config': 1}
        )
        hist_cfg = (hist or {}).get('config') or {}
        broker_cfg = hist_cfg.get('broker') or {}

        asset = metadata.get('asset') or hist_cfg.get('asset') or 'BTCUSDT'
        timeframe = params.get('timeframe') or params.get('frequency') or metadata.get('timeframe') or '1h'

        initial_capital = broker_cfg.get('initial_capital') or hist_cfg.get('initial_capital') or 1000000.0
        commission_pct = broker_cfg.get('commission_pct') or hist_cfg.get('commission_pct') or metadata.get('commission_pct') or 0.1
        slippage_pct = broker_cfg.get('slippage_pct') or hist_cfg.get('slippage_pct') or metadata.get('slippage_pct') or 0.0
        start_date = hist_cfg.get('start_date')
        end_date = hist_cfg.get('end_date')

        strategy = params.get('strategy') or {}
        ps = (strategy.get('ps') or {})
        bse = (strategy.get('bse') or {})

        # Enforce that we run with explicitly provided/stored params (no silent defaults).
        missing_fields = []
        for key in ['ir', 'er', 'or']:
            if key not in ps:
                missing_fields.append(f"strategy.ps.{key}")
        if 'is_on_going' not in bse:
            missing_fields.append('strategy.bse.is_on_going')
        if 'use_delta' not in params:
            missing_fields.append('use_delta')
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required params in stored config: {', '.join(missing_fields)}"
            )

        result = run_backtest(
            asset=asset,
            timeframe=timeframe,
            initial_capital=float(initial_capital),
            commission=float(commission_pct),
            slippage_pct=float(slippage_pct),
            risk_per_trade_pct=float(ps.get('ir')),
            max_risk_equity_pct=float(ps.get('er')),
            ema_length=int(params.get('length_ema', 50)),
            atr_length=int(params.get('length_atr', 14)),
            multiplier=float(params.get('multiple', 1)),
            long_vol_factor=float(params.get('long_vol_factor', 2.0)),
            short_vol_factor=float(params.get('short_vol_factor', 2.0)),
            trade_option='Both',
            is_on_going=bool(bse.get('is_on_going')),
            on_going_risk=float(ps.get('or')),
            use_delta=bool(params.get('use_delta')),
            start_date=start_date,
            end_date=end_date,
            data_type=hist_cfg.get('data_type', 'OKX')
        )

        if not result.get('success'):
            raise HTTPException(status_code=400, detail=result.get('error', 'Backtest failed'))

        # Add a small hint for debugging payload parity (non-breaking)
        try:
            result.setdefault('data', {})
            result['data'].setdefault('replay', {})
            result['data']['replay'].update({'batch_id': request.batch_id, 'config_hash': request.config_hash})
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


class RegimeRequest(BaseModel):
    """Request for regime analysis on chart data"""
    asset: str
    timeframe: str
    data_type: Optional[str] = "OKX"
    ema_length: int = 50
    atr_length: int = 14
    long_vol_factor: float = 2.0
    short_vol_factor: float = 1.3
    account_equity: Optional[float] = None
    max_risk_pct: float = 1.0


@router.post("/single-core/regime")
async def analyze_market_regime(request: RegimeRequest):
    """
    Analyze current market regime and get trading blueprint
    
    Returns regime analysis with:
        - Multi-timeframe view (1h, 4h, 1d)
        - Market status classification
        - Trading permissions (allow_long/short)
        - Model selection (4-3-3 vs 2-2-2)
        - Risk-adjusted position sizing
        - Next 5 candles prediction
    """
    try:
        from regime import analyze_regime_v3, format_regime_output
        from core.load_data import DataLoader
        from core.indicator import IndicatorCalculator
        import pandas as pd
        import numpy as np
        
        loader = DataLoader(data_dir=request.data_type or 'OKX')
        df = loader.load_data(request.asset, request.timeframe)
        
        if df is None or len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        indicator_calc = IndicatorCalculator()
        ema = indicator_calc.calculate_ema(df['close'], request.ema_length)
        atr = indicator_calc.calculate_atr(df['high'], df['low'], df['close'], request.atr_length)
        
        upper_band = ema + (request.long_vol_factor * atr)
        lower_band = ema - (request.short_vol_factor * atr)
        
        volume = df['volume'].to_numpy() if 'volume' in df.columns else None
        
        regime_output = analyze_regime_v3(
            df=df,
            ema=ema,
            atr=atr,
            upper_band=upper_band,
            lower_band=lower_band,
            volume=volume
        )
        
        regime_data = format_regime_output(regime_output)
        
        response = {
            'success': True,
            'data': {
                'regime': regime_data,
                'asset': request.asset,
                'timeframe': request.timeframe,
            }
        }
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Regime analysis failed: {str(e)}")