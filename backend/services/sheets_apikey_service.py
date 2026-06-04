from __future__ import annotations

"""
Google Sheets export via Service Account (no OAuth required).
Uses nora-service-sheet.json in the backend/ directory.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

PARENT_FOLDER_ID = '1_6c7kHPK6chnF4m3zdhTQMt6aLGL8MjQ'


# ============================================================================
# SERVICE ACCOUNT CREDENTIALS
# ============================================================================

def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _service_account_path() -> Path:
    """Find nora-service-sheet.json (service account key)."""
    for p in [
        _backend_root() / "nora-service-sheet.json",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Service account key not found. Expected: backend/nora-service-sheet.json"
    )


def _load_credentials() -> Credentials:
    """Load Service Account credentials."""
    path = _service_account_path()
    return Credentials.from_service_account_file(str(path), scopes=SCOPES)


# ============================================================================
# FORMATTING HELPERS (same as OAuth version)
# ============================================================================

def _col_letter(col_index: int) -> str:
    result = ""
    col_index += 1
    while col_index > 0:
        col_index -= 1
        result = chr(col_index % 26 + ord('A')) + result
        col_index //= 26
    return result


def _build_format_requests(headers: List[str], data_row_count: int) -> List[Dict[str, Any]]:
    num_cols = len(headers)
    total_rows = 1 + data_row_count

    BLUE_LIGHT   = {"red": 204/255, "green": 229/255, "blue": 1.0}
    YELLOW_LIGHT = {"red": 1.0, "green": 244/255, "blue": 204/255}
    GREEN_LIGHT  = {"red": 230/255, "green": 247/255, "blue": 230/255}
    PINK_LIGHT   = {"red": 1.0, "green": 230/255, "blue": 230/255}
    WHITE        = {"red": 1.0, "green": 1.0, "blue": 1.0}
    GREEN_TEXT   = {"red": 0, "green": 128/255, "blue": 0}
    RED_TEXT     = {"red": 192/255, "green": 0, "blue": 0}

    requests = []

    # Freeze header row
    requests.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": 0,
                "gridProperties": {"frozenRowCount": 1}
            },
            "fields": "gridProperties.frozenRowCount"
        }
    })

    # Header formatting: blue bg, white bold text, centered
    requests.append({
        "repeatCell": {
            "range": {"sheetId": 0, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": num_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 68/255, "green": 114/255, "blue": 196/255},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                "horizontalAlignment": "CENTER",
                "verticalAlignment": "MIDDLE"
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"
        }
    })

    # Auto-resize columns
    requests.append({
        "autoResizeDimensions": {
            "dimensions": {"sheetId": 0, "dimension": "COLUMNS",
                           "startIndex": 0, "endIndex": num_cols}
        }
    })

    headers_upper = [h.upper() for h in headers]
    is_trades_sheet = any(h in ["ENTRY TIME", "EXIT TIME", "ENTRY PRICE", "EXIT PRICE"] for h in headers_upper)

    def _col_range(col_idx):
        return {"sheetId": 0, "startRowIndex": 1, "endRowIndex": total_rows,
                "startColumnIndex": col_idx, "endColumnIndex": col_idx + 1}

    def _bg_format(col_idx, bg_color, align="CENTER"):
        return {"repeatCell": {
            "range": _col_range(col_idx),
            "cell": {"userEnteredFormat": {
                "backgroundColor": bg_color,
                "horizontalAlignment": align,
                "verticalAlignment": "MIDDLE"
            }},
            "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
        }}

    def _cond_text(col_idx, condition_type, value, color):
        return {"addConditionalFormatRule": {
            "rule": {
                "ranges": [_col_range(col_idx)],
                "booleanRule": {
                    "condition": {"type": condition_type, "values": [{"userEnteredValue": value}]},
                    "format": {"textFormat": {"foregroundColor": color}}
                }
            },
            "index": 0
        }}

    if is_trades_sheet:
        for col_idx, h in enumerate(headers_upper):
            if h in ["TYPE", "DIRECTION"]:
                requests.append(_bg_format(col_idx, WHITE))
                requests.append(_cond_text(col_idx, "TEXT_CONTAINS", "LONG", {**GREEN_TEXT, "bold": True}))
                requests.append(_cond_text(col_idx, "TEXT_CONTAINS", "SHORT", {**RED_TEXT, "bold": True}))
            elif "ENTRY" in h:
                requests.append(_bg_format(col_idx, YELLOW_LIGHT))
            elif "EXIT" in h:
                requests.append(_bg_format(col_idx, BLUE_LIGHT))
            elif "POSITION SIZE" in h or "QTY" in h or "QUANTITY" in h:
                requests.append(_bg_format(col_idx, PINK_LIGHT))
            elif any(x in h for x in ["PNL", "P&L", "PROFIT", "LOSS"]):
                requests.append(_bg_format(col_idx, WHITE, align="RIGHT"))
                requests.append(_cond_text(col_idx, "NUMBER_GREATER", "0", GREEN_TEXT))
                requests.append(_cond_text(col_idx, "NUMBER_LESS", "0", RED_TEXT))
    else:
        timeframe_cols, param_cols, metric_cols = [], [], []
        for col_idx, h in enumerate(headers_upper):
            if "TIMEFRAME" in h or h in ["STT", "#"]:
                timeframe_cols.append(col_idx)
            elif any(x in h for x in ["EMA", "ATR", "VF", "SF", "HIGH", "LOW", "IR", "ER", "OR", "SKID", "COMMISSION"]):
                param_cols.append(col_idx)
            elif any(x in h for x in ["ROI", "CAGR", "PROFIT", "SHARPE", "SORTINO", "WIN", "MDD", "TRADES", "EQUITY", "FINAL", "MAX"]):
                metric_cols.append(col_idx)

        for col_idx in timeframe_cols:
            requests.append(_bg_format(col_idx, BLUE_LIGHT))
        for col_idx in param_cols:
            requests.append(_bg_format(col_idx, YELLOW_LIGHT))
        for col_idx in metric_cols:
            requests.append(_bg_format(col_idx, GREEN_LIGHT))
            h = headers_upper[col_idx]
            if any(x in h for x in ["ROI", "CAGR", "PROFIT", "SHARPE", "SORTINO", "WIN"]):
                requests.append(_cond_text(col_idx, "NUMBER_GREATER", "0", {**GREEN_TEXT, "bold": True}))
                requests.append(_cond_text(col_idx, "NUMBER_LESS", "0", {**RED_TEXT, "bold": True}))
            elif "MDD" in h:
                requests.append(_cond_text(col_idx, "NUMBER_LESS", "0", {**RED_TEXT, "bold": True}))

    return requests


# ============================================================================
# PUBLIC EXPORT FUNCTIONS
# ============================================================================

def export_to_google_sheet(
    *,
    title: str,
    headers: List[str],
    data: List[List[Any]],
    share_email: Optional[str] = None,
) -> str:
    """Export data to Google Sheets using Service Account (no OAuth required)."""
    creds = _load_credentials()
    gc = gspread.authorize(creds)

    print(f"[Sheets] Creating spreadsheet: {title} ({len(data)} rows)")

    if PARENT_FOLDER_ID:
        try:
            spreadsheet = gc.create(title, folder_id=PARENT_FOLDER_ID)
        except (TypeError, AttributeError):
            spreadsheet = gc.create(title)
    else:
        spreadsheet = gc.create(title)

    sheet = spreadsheet.sheet1
    sheet_id = sheet.id
    spreadsheet_id = spreadsheet.id

    # Resize if needed
    required_rows = 1 + len(data)
    required_cols = len(headers)
    if required_rows > sheet.row_count or required_cols > sheet.col_count:
        sheet.resize(rows=max(required_rows, sheet.row_count),
                     cols=max(required_cols, sheet.col_count, 26))

    sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    all_requests = []

    # Write data
    all_values = [headers] + data
    rows_data = []
    for row_values in all_values:
        cells = []
        for val in row_values:
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                cells.append({"userEnteredValue": {"numberValue": float(val)}})
            elif isinstance(val, bool):
                cells.append({"userEnteredValue": {"boolValue": val}})
            elif val is None or val == "":
                cells.append({"userEnteredValue": {"stringValue": ""}})
            else:
                cells.append({"userEnteredValue": {"stringValue": str(val)}})
        rows_data.append({"values": cells})

    all_requests.append({
        "updateCells": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0,
                "endRowIndex": len(all_values),
                "startColumnIndex": 0,
                "endColumnIndex": len(headers)
            },
            "rows": rows_data,
            "fields": "userEnteredValue"
        }
    })

    all_requests.extend(_build_format_requests(headers, len(data)))

    try:
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": all_requests}
        ).execute()
    except Exception as e:
        print(f"[Sheets] Batch update failed: {e}, trying simple write...")
        try:
            sheet.update(range_name="A1", values=all_values, value_input_option='RAW')
        except Exception as e2:
            raise RuntimeError(f"Export failed: {e2}") from e2

    # Share publicly (anyone with link can edit)
    try:
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        drive.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "anyone", "role": "writer", "allowFileDiscovery": False},
            fields="id",
        ).execute()
        print(f"[Sheets] ✓ Shared publicly")
    except Exception as e:
        print(f"[Sheets] Warning: Public sharing failed: {e}")

    if share_email:
        try:
            spreadsheet.share(share_email, perm_type="user", role="writer")
        except Exception:
            pass

    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?usp=sharing"
    print(f"[Sheets] ✓ Done: {url}")
    return url


def upload_excel_to_google_sheet(
    *,
    excel_path,
    title: Optional[str] = None,
    share_email: Optional[str] = None,
) -> str:
    """Upload Excel file to Google Sheets via Service Account."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if title is None:
        title = excel_path.stem

    creds = _load_credentials()
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    file_metadata = {
        'name': title,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
    }
    if PARENT_FOLDER_ID:
        file_metadata['parents'] = [PARENT_FOLDER_ID]

    media = MediaFileUpload(
        str(excel_path),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        resumable=True
    )

    print(f"[Sheets] Uploading Excel: {excel_path.name} ({excel_path.stat().st_size / 1024:.1f} KB)")

    file = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name'
    ).execute()

    spreadsheet_id = file.get('id')
    print(f"[Sheets] ✓ Uploaded: {file.get('name')}")

    try:
        drive.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "anyone", "role": "writer", "allowFileDiscovery": False},
            fields="id",
        ).execute()
        print(f"[Sheets] ✓ Shared publicly")
    except Exception as e:
        print(f"[Sheets] Warning: Public sharing failed: {e}")

    if share_email:
        try:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={"type": "user", "role": "writer", "emailAddress": share_email},
                fields="id",
            ).execute()
        except Exception:
            pass

    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?usp=sharing"
