from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, List, Optional
import tempfile
import re
from pathlib import Path

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False

from database.mongo_service import MongoService

from services.sheets_apikey_service import (
    export_to_google_sheet,
    upload_excel_to_google_sheet,
)


router = APIRouter(prefix="/api/sheets", tags=["sheets"])


def _kema_sheet_title(*, asset: str, filter_id: Optional[int] = None, suffix: str | None = None) -> str:
    a = (asset or "").strip() or "ASSET"
    title = f"kema_{a}_ema_atr_vf"
    if filter_id is not None:
        title = f"{title}_filter_{int(filter_id)}"
    if suffix:
        title = f"{title}_{suffix}"
    return title


def _resolve_asset_for_batch(mongo: MongoService, *, batch_id: str, campaign_type: str) -> str:
    # Try optimize_history first
    campaign = mongo.db["optimize_history"].find_one({"batch_id": batch_id})
    for key in ("asset", "symbol", "pair"):
        v = (campaign or {}).get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Fallback to any config doc under this batch
    config_coll = mongo.wfo_config if campaign_type == "wfo" else mongo.backtest_config
    config_doc = config_coll.find_one({"batch_id": batch_id})
    params = (config_doc or {}).get("params", {}) if isinstance(config_doc, dict) else {}
    v = params.get("asset")
    return (v or "").strip() if isinstance(v, str) else ""


class ExportRequest(BaseModel):
    title: str
    headers: List[str]
    data: List[List[Any]]
    share_email: Optional[str] = None


class BacktestResultsExportRequest(BaseModel):
    """Request model for exporting backtest results directly from MongoDB"""
    batch_id: str
    filter_id: Optional[int] = None
    sort_by: str = "roi"
    sort_order: int = -1
    share_email: Optional[str] = None


class StrategyTradesExportRequest(BaseModel):
    """Request model for exporting strategy trades from backend"""
    batch_id: str
    config_hash: str
    title: Optional[str] = None
    share_email: Optional[str] = None



@router.post("/export")
def export_sheet(req: ExportRequest):
    """Export data to Google Sheets using system-wide authorization."""
    try:
        url = export_to_google_sheet(
            title=req.title,
            headers=req.headers,
            data=req.data,
            share_email=req.share_email,
        )
        return {"success": True, "sheet_url": url, "rows_exported": len(req.data)}
    except ValueError as e:
        # Not authorized / missing token.json
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


def _format_number(val, decimals=2):
    """Format number with fixed decimals, matching frontend display."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return f"{val:.{decimals}f}"
    return str(val)


def _format_currency(val):
    """Format currency with thousand separators, matching frontend toLocaleString."""
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        return f"{val:,.2f}"
    return str(val)


@router.post("/export/backtest-results")
def export_backtest_results(req: BacktestResultsExportRequest):
    """Export full backtest optimization results to Google Sheets.
    
    Backend queries full dataset from Mongo and formats exactly like frontend table display.
    Frontend only needs to provide batch_id - no data fetching/formatting on client side.
    """

    mongo = MongoService()
    try:
        # Determine campaign type and collections
        campaign = mongo.db["optimize_history"].find_one({"batch_id": req.batch_id})
        campaign_type = (campaign or {}).get("collection_type") or (
            "wfo" if req.batch_id.startswith("WFO_") else "backtest"
        )
        result_collection = mongo.wfo_result if campaign_type == "wfo" else mongo.backtest_result

        query: dict[str, Any] = {
            "batch_id": req.batch_id,
            "status": {"$ne": "failed"},
        }

        # Apply saved filter rules from optimize_history if requested
        if req.filter_id is not None and campaign and "filters" in campaign:
            filter_meta = next(
                (
                    f
                    for f in campaign.get("filters", [])
                    if int(f.get("id", -1)) == int(req.filter_id)
                ),
                None,
            )
            if not filter_meta:
                raise HTTPException(status_code=404, detail=f"Filter {req.filter_id} not found")

            rules = filter_meta.get("rules", [])
            for rule in rules:
                metric = rule.get("metric")
                op = rule.get("operator")
                val = rule.get("value")
                if not metric or not op or val is None:
                    continue
                mongo_op = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "==": "$eq"}.get(
                    op
                )
                if not mongo_op:
                    continue
                try:
                    query[f"result.all.{metric}"] = {mongo_op: float(val)}
                except Exception:
                    continue

        sort_field = f"result.all.{req.sort_by}" if req.sort_by else "result.all.roi"

        # Headers matching frontend table columns exactly
        headers = [
            "STT",
            "EMA", "ATR", "HIGH SF", "LOW SF",
            "TIMEFRAME",
            "COMMISSION", "IR", "ER", "OR", "SKID",
            "TOTAL TRADES", "FINAL EQUITY",
            "WIN RATE", "MDD", "CAGR", "PROFIT", "SHARPE", "ROI", "SORTINO",
            "MAX LEV", "STREAK", "DD DAYS"
        ]

        data: List[List[Any]] = []
        idx = 0

        cursor = result_collection.find(query).sort(sort_field, req.sort_order)
        for doc in cursor:
            metrics = doc.get("result", {}).get("all", {})

            # Get params from result doc or fallback to config collection
            params = doc.get("params")
            if not params:
                config_coll = mongo.wfo_config if campaign_type == "wfo" else mongo.backtest_config
                config_doc = config_coll.find_one({"config_hash": doc.get("config_hash")})
                if config_doc:
                    params = config_doc.get("params", {})
            if not params:
                params = {}

            # Extract values with same logic as frontend
            ema = metrics.get("ema") or params.get("length_ema") or ""
            atr = metrics.get("atr") or params.get("length_atr") or ""
            
            highSF = metrics.get("highVf") or params.get("long_vol_factor")
            highSF_str = _format_number(highSF, 2) if highSF is not None else "-"
            
            lowSF = metrics.get("lowVf") or params.get("short_vol_factor")
            lowSF_str = _format_number(lowSF, 2) if lowSF is not None else "-"
            
            timeframe = params.get("timeframe") or metrics.get("timeframe", "")
            
            commission = metrics.get("commissionPct") or params.get("commission_pct")
            commission_str = _format_number(commission, 2) if commission is not None else ""
            
            # IR with fallback to 'is' (old field name)
            ir = metrics.get("ir")
            if ir is None:
                ir = metrics.get("is")
            if ir is None:
                ir = params.get("strategy", {}).get("ps", {}).get("ir")
            ir_str = _format_number(ir, 2) if ir is not None else ""
            
            # ER
            er = metrics.get("er")
            if er is None:
                er = params.get("strategy", {}).get("ps", {}).get("er")
            er_str = _format_number(er, 2) if er is not None else ""
            
            # OR with default 0.95
            or_val = metrics.get("or")
            if or_val is None:
                or_val = params.get("strategy", {}).get("ps", {}).get("or")
            or_str = _format_number(or_val, 2) if or_val is not None else "0.95"
            
            # Skid with default 0.30
            skid = metrics.get("skid") or params.get("slippage_pct")
            skid_str = _format_number(skid, 2) if skid is not None else "0.30"
            
            totalTrades = metrics.get("totalTrades", 0)
            
            finalEquity = metrics.get("finalEquity")
            finalEquity_str = _format_currency(finalEquity) if finalEquity is not None else ""
            
            winRate = metrics.get("winRate")
            winRate_str = f"{_format_number(winRate, 2)}%" if winRate is not None else ""
            
            mdd = metrics.get("mdd")
            mdd_str = f"{_format_number(mdd, 2)}%" if mdd is not None else ""
            
            cagr = metrics.get("cagr")
            cagr_str = f"{_format_number(cagr, 2)}%" if cagr is not None else ""
            
            profit = metrics.get("profit")
            profit_str = _format_currency(profit) if profit is not None else ""
            
            sharpe = metrics.get("sharpe")
            sharpe_str = _format_number(sharpe, 2) if sharpe is not None else ""
            
            roi = metrics.get("roi")
            roi_str = f"{_format_number(roi, 2)}%" if roi is not None else ""
            
            sortino = metrics.get("sortino")
            sortino_str = _format_number(sortino, 2) if sortino is not None else "-"
            
            maxLeverage = metrics.get("maxLeverage")
            maxLev_str = f"{_format_number(maxLeverage, 2)}x" if maxLeverage is not None else ""
            
            maxConsecutiveLosses = metrics.get("maxConsecutiveLosses", "")
            maxDrawdownDuration = metrics.get("maxDrawdownDuration", "")

            idx += 1
            data.append(
                [
                    idx,
                    ema,
                    atr,
                    highSF_str,
                    lowSF_str,
                    timeframe,
                    commission_str,
                    ir_str,
                    er_str,
                    or_str,
                    skid_str,
                    totalTrades,
                    finalEquity_str,
                    winRate_str,
                    mdd_str,
                    cagr_str,
                    profit_str,
                    sharpe_str,
                    roi_str,
                    sortino_str,
                    maxLev_str,
                    maxConsecutiveLosses,
                    maxDrawdownDuration,
                ]
            )

        if not data:
            raise HTTPException(status_code=404, detail="No results found")

        asset = _resolve_asset_for_batch(mongo, batch_id=req.batch_id, campaign_type=campaign_type)
        title = _kema_sheet_title(asset=asset, filter_id=req.filter_id)

        try:
            url = export_to_google_sheet(
                title=title,
                headers=headers,
                data=data,
                share_email=req.share_email,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

        return {"success": True, "sheet_url": url, "rows_exported": len(data)}
    finally:
        mongo.close()


@router.post("/export/strategy-trades")
def export_strategy_trades(req: StrategyTradesExportRequest):
    """Export full trade list for a specific strategy configuration.
    
    Re-runs backtest for the config to get complete trade details.
    """
    from services.backtest_service import run_single_backtest
    
    mongo = MongoService()
    try:
        # Get config from MongoDB
        campaign = mongo.db["optimize_history"].find_one({"batch_id": req.batch_id})
        campaign_type = (campaign or {}).get("collection_type") or (
            "wfo" if req.batch_id.startswith("WFO_") else "backtest"
        )
        
        config_coll = mongo.wfo_config if campaign_type == "wfo" else mongo.backtest_config
        config_doc = config_coll.find_one({"config_hash": req.config_hash})
        
        if not config_doc:
            raise HTTPException(status_code=404, detail=f"Config {req.config_hash} not found")
        
        params = config_doc.get("params", {})
        
        # Re-run backtest to get full trades
        backtest_result = run_single_backtest(
            asset=params.get("asset", "BTCUSDT"),
            timeframe=params.get("timeframe", "1h"),
            start_date="",
            end_date="",
            ema_length=params.get("length_ema", 15),
            atr_length=params.get("length_atr", 14),
            long_vol_factor=params.get("long_vol_factor", 2.0),
            short_vol_factor=params.get("short_vol_factor", 2.0),
            multiplier=params.get("multiple", 1),
            initial_capital=params.get("capital", 1000000),
            commission_pct=params.get("commission_pct", 0.1),
            skid_pct=params.get("slippage_pct", 0.4),
            risk_per_trade_pct=params.get("strategy", {}).get("ps", {}).get("ir", 0.02),
            max_risk_equity_pct=params.get("strategy", {}).get("ps", {}).get("er", 0.5),
            is_on_going=params.get("strategy", {}).get("bse", {}).get("is_on_going", True),
            on_going_risk=params.get("strategy", {}).get("ps", {}).get("or", 0.95),
            trade_option="Both",
            data_type=params.get("data_type", "OKX")
        )
        
        if not backtest_result.get("success"):
            raise HTTPException(status_code=500, detail=backtest_result.get("error", "Backtest failed"))
        
        data_obj = backtest_result.get("data", {})
        trades = data_obj.get("trades") or data_obj.get("list_of_trades") or []
        
        if not trades:
            raise HTTPException(status_code=404, detail="No trades found")
        
        # Format trades for sheet export
        headers = [
            "#", "Type", "Entry Time", "Entry Price", "Exit Time", "Exit Price",
            "Quantity", "Position Size", "Gross PnL", "Commission", "Net PnL", "PnL%",
            "MFE", "MAE", "MFE%", "MAE%", "Bars", "Exit Reason",
            "OR Risk Before", "OR Risk After", "OR Contracts Before", "OR Contracts After",
            "OR Unrealized PnL", "OR Equity", "Equity After Exit"
        ]
        
        data = []
        for i, trade in enumerate(trades, 1):
            data.append([
                i,
                trade.get("direction", "LONG"),
                trade.get("entry_time", "-"),
                _format_number(trade.get("entry_price"), 4),
                trade.get("exit_time", "-"),
                _format_number(trade.get("exit_price"), 4),
                _format_number(trade.get("quantity"), 4),
                _format_currency(trade.get("pos_size")),
                _format_currency(trade.get("pnl")),
                _format_currency(trade.get("commission")),
                _format_currency(trade.get("net_pnl")),
                f"{_format_number(trade.get('pnl_pct'), 2)}%" if trade.get("pnl_pct") is not None else "-",
                _format_currency(trade.get("mfe")),
                _format_currency(trade.get("mae")),
                f"{_format_number(trade.get('mfe_pct'), 2)}%" if trade.get("mfe_pct") is not None else "-",
                f"{_format_number(trade.get('mae_pct'), 2)}%" if trade.get("mae_pct") is not None else "-",
                trade.get("bars", 0),
                trade.get("exit_reason", "-"),
                _format_number(trade.get("or_risk_before"), 4),
                _format_number(trade.get("or_risk_after"), 4),
                _format_number(trade.get("or_contracts_before"), 2),
                _format_number(trade.get("or_contracts_after"), 2),
                _format_currency(trade.get("or_unrealized_pnl")),
                _format_currency(trade.get("or_on_going_equity")),
                _format_currency(trade.get("equity_after_exit"))
            ])
        
        asset = params.get("asset", "") if isinstance(params, dict) else ""
        title = _kema_sheet_title(asset=str(asset or "").strip(), suffix="trades")
        
        url = export_to_google_sheet(
            title=title,
            headers=headers,
            data=data,
            share_email=req.share_email,
        )
        
        return {"success": True, "sheet_url": url, "rows_exported": len(data)}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        mongo.close()


@router.post("/export/excel-upload")
def export_excel_upload(req: BacktestResultsExportRequest):
    """Export backtest results via Excel file upload (preserves Excel formatting).
    
    Workflow:
    1. Query data from MongoDB
    2. Generate formatted Excel file (like stress_test results)
    3. Upload Excel to Google Sheets (auto-converts with formatting intact)
    4. Return Google Sheets URL
    
    This method is faster and preserves Excel formatting better than cell-by-cell upload.
    """
    mongo = MongoService()
    temp_excel_path = None
    
    try:
        # Determine campaign type and collections
        campaign = mongo.db["optimize_history"].find_one({"batch_id": req.batch_id})
        if not campaign:
            raise HTTPException(status_code=404, detail=f"Campaign {req.batch_id} not found")
            
        campaign_type = campaign.get("collection_type") or (
            "wfo" if req.batch_id.startswith("WFO_") else "backtest"
        )
        result_collection = mongo.wfo_result if campaign_type == "wfo" else mongo.backtest_result
        config_collection = mongo.wfo_config if campaign_type == "wfo" else mongo.backtest_config

        query: dict[str, Any] = {
            "batch_id": req.batch_id,
            "status": {"$ne": "failed"},
        }

        # Apply saved filter rules if requested
        if req.filter_id is not None and "filters" in campaign:
            filter_meta = next(
                (f for f in campaign.get("filters", []) if int(f.get("id", -1)) == int(req.filter_id)),
                None,
            )
            if not filter_meta:
                raise HTTPException(status_code=404, detail=f"Filter {req.filter_id} not found")

            rules = filter_meta.get("rules", [])
            for rule in rules:
                metric = rule.get("metric")
                op = rule.get("operator")
                val = rule.get("value")
                if not metric or not op or val is None:
                    continue
                mongo_op = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "==": "$eq"}.get(op)
                if not mongo_op:
                    continue
                try:
                    query[f"result.all.{metric}"] = {mongo_op: float(val)}
                except Exception:
                    continue

        sort_field = f"result.all.{req.sort_by}" if req.sort_by else "result.all.roi"

        # Query results
        cursor = result_collection.find(query).sort(sort_field, req.sort_order)
        results = list(cursor)
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")

        print(f"[ExcelUpload] Processing {len(results)} results for {req.batch_id}")

        # Get config for metadata
        config_doc = config_collection.find_one({"batch_id": req.batch_id})
        if not config_doc:
            raise HTTPException(status_code=404, detail="Campaign config not found")
        
        config = config_doc.get("config", {})
        
        # ============================================================
        # GENERATE EXCEL FILE (Format matching stress_test)
        # ============================================================
        
        if not XLSXWRITER_AVAILABLE:
            raise HTTPException(status_code=500, detail="xlsxwriter not installed")
        
        # Create temp Excel file
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xlsx', delete=False) as tmp_file:
            temp_excel_path = Path(tmp_file.name)
        
        print(f"[ExcelUpload] Creating Excel file: {temp_excel_path}")
        
        # Process data
        data_rows = []
        for res in results:
            metrics = res.get("result", {}).get("all", {})
            params = res.get("params", {})
            ps = params.get("strategy", {}).get("ps", {})
            
            tf = metrics.get("timeframe") or params.get("timeframe", "N/A")
            
            # Calculate skid from base_slippage / sqrt(tf)
            base_slippage = config.get("slippage_pct", 0.5)
            tf_num = int(re.search(r'(\d+)', tf).group(1)) if re.search(r'(\d+)', tf) else 1
            import math
            skid = base_slippage / math.sqrt(tf_num)
            
            data_rows.append([
                tf,
                params.get("length_ema"),
                round(params.get("long_vol_factor", 0), 2),
                round(params.get("short_vol_factor", 0), 2),
                round(ps.get("ir", 0), 2),
                round(ps.get("er", 0), 2),
                round(ps.get("or", 0), 2),
                round(skid, 2),
                round(metrics.get("roi", 0), 2),
                round(metrics.get("winRate", 0), 2),
                metrics.get("totalTrades", 0),
                round(metrics.get("profit", 0), 2),
                round(metrics.get("finalEquity", 10000.0), 2),
                round(metrics.get("mdd", 0), 2),
                round(metrics.get("sharpe", 0), 2),
                round(metrics.get("cagr", 0), 2)
            ])
        
        # Sort by tf → ema → high_vf → low_vf → ir → er → or
        def extract_tf_num(tf_str):
            match = re.search(r'(\d+)', str(tf_str))
            return int(match.group(1)) if match else 999
        
        data_rows.sort(key=lambda row: (
            extract_tf_num(row[0]),
            row[1] or 0,
            row[2] or 0,
            row[3] or 0,
            row[4] or 0,
            row[5] or 0,
            row[6] or 0
        ))
        
        print(f"[ExcelUpload] Sorted {len(data_rows)} rows")
        
        # Create workbook
        workbook = xlsxwriter.Workbook(str(temp_excel_path), {'constant_memory': True})
        
        # Formats (matching stress_test exactly)
        header_format = workbook.add_format({
            'bold': True,
            'font_color': '#FFFFFF',
            'bg_color': '#4472C4',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'font_size': 11
        })
        
        tf_format = workbook.add_format({
            'bg_color': '#CCE5FF',
            'font_color': '#000000',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        param_format = workbook.add_format({
            'bg_color': '#FFF4CC',
            'font_color': '#000000',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        metric_format = workbook.add_format({
            'bg_color': '#E6F7E6',
            'font_color': '#000000',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        metric_positive = workbook.add_format({
            'bg_color': '#E6F7E6',
            'font_color': '#008000',
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        metric_negative = workbook.add_format({
            'bg_color': '#E6F7E6',
            'font_color': '#C00000',
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        # Results sheet
        ws = workbook.add_worksheet('Results')
        headers = ['Timeframe', 'EMA', 'VF_Long', 'VF_Short', 'IR', 'ER', 'OR', 'Skid',
                   'ROI%', 'WinRate%', 'Trades', 'Profit', 'FinalCapital', 'MDD%', 'Sharpe', 'CAGR%']
        
        for col_idx, header in enumerate(headers):
            ws.write(0, col_idx, header, header_format)
        
        ws.set_column(0, 0, 12)
        ws.set_column(1, 7, 10)
        ws.set_column(8, 15, 12)
        
        for row_idx, row_data in enumerate(data_rows, start=1):
            ws.write(row_idx, 0, row_data[0], tf_format)
            
            for col_idx in range(1, 8):
                ws.write(row_idx, col_idx, row_data[col_idx], param_format)
            
            for col_idx in range(8, 16):
                value = row_data[col_idx]
                if value > 0:
                    ws.write(row_idx, col_idx, value, metric_positive)
                elif value < 0:
                    ws.write(row_idx, col_idx, value, metric_negative)
                else:
                    ws.write(row_idx, col_idx, value, metric_format)
        
        # Config sheet
        ws_config = workbook.add_worksheet('Config')
        config_header_format = workbook.add_format({
            'bold': True,
            'font_color': '#FFFFFF',
            'bg_color': '#4472C4',
            'border': 1
        })
        
        config_label_format = workbook.add_format({
            'bold': True,
            'bg_color': '#E6E6FA',
            'border': 1,
            'align': 'left',
            'valign': 'vcenter'
        })
        
        config_value_format = workbook.add_format({
            'bg_color': '#FFFFFF',
            'border': 1,
            'align': 'left',
            'valign': 'vcenter'
        })
        
        ws_config.set_column(0, 0, 25)
        ws_config.set_column(1, 1, 40)
        
        config_data = [
            ['Parameter', 'Value'],
            ['Batch ID', req.batch_id],
            ['Asset', config.get('asset')],
            ['Date Range', f"{config.get('start_date')} to {config.get('end_date')}"],
            ['Initial Capital', config.get('initial_capital')],
            ['Commission %', config.get('commission_pct')],
            ['Slippage %', config.get('slippage_pct')],
            ['Total Results', len(data_rows)],
        ]
        
        for row_idx, (label, value) in enumerate(config_data):
            if row_idx == 0:
                ws_config.write(row_idx, 0, label, config_header_format)
                ws_config.write(row_idx, 1, value, config_header_format)
            else:
                ws_config.write(row_idx, 0, label, config_label_format)
                ws_config.write(row_idx, 1, value, config_value_format)
        
        workbook.close()
        
        print(f"[ExcelUpload] Excel file created: {temp_excel_path.stat().st_size / 1024:.1f} KB")
        
        # ============================================================
        # UPLOAD TO GOOGLE SHEETS
        # ============================================================
        
        asset = _resolve_asset_for_batch(mongo, batch_id=req.batch_id, campaign_type=campaign_type)
        title = _kema_sheet_title(asset=asset, filter_id=req.filter_id)
        
        try:
            url = upload_excel_to_google_sheet(
                excel_path=temp_excel_path,
                title=title,
                share_email=req.share_email,
            )
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
        
        print(f"[ExcelUpload] ✓ Successfully uploaded to Google Sheets")
        print(f"[ExcelUpload] URL: {url}")
        
        return {"success": True, "sheet_url": url, "rows_exported": len(data_rows)}
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_excel_path and temp_excel_path.exists():
            try:
                temp_excel_path.unlink()
                print(f"[ExcelUpload] Cleaned up temp file")
            except Exception as e:
                print(f"[ExcelUpload] Warning: Could not delete temp file: {e}")
        
        mongo.close()
