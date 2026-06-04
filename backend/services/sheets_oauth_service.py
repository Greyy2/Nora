from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import os

import gspread
from google.auth.credentials import Credentials as GoogleAuthCredentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Optional folder ID to create sheets inside
PARENT_FOLDER_ID = '1_6c7kHPK6chnF4m3zdhTQMt6aLGL8MjQ'


# ============================================================================
# PATH & FILE HELPERS
# ============================================================================

def _grey_root() -> Path:
    """Root of Grey project (2 levels up from services/)."""
    return Path(__file__).resolve().parents[2]


def _sheet_dir() -> Path:
    """Directory for OAuth tokens and state files."""
    d = _grey_root() / "sheet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _creds_file() -> Path:
    """Token file for system-wide OAuth credentials."""
    return _sheet_dir() / "token.json"


def _state_file() -> Path:
    """Temporary state file during OAuth flow."""
    return _sheet_dir() / "state.json"


def _client_secrets_path() -> Path:
    """Path to OAuth client JSON file (fallback if env not set)."""
    preferred = _sheet_dir() / "nora-sheet.json"
    if preferred.exists():
        return preferred
    return _grey_root() / "nora-sheet.json"


def _read_json(path: Path) -> Dict[str, Any]:
    """Read and parse JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON to file."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# OAUTH CLIENT CONFIG
# ============================================================================

def _env_oauth_client() -> Optional[Dict[str, Any]]:
    """Build OAuth client config from env vars (GOOGLE_OAUTH_CLIENT_ID/SECRET).
    
    This avoids needing to download client JSON from Google Cloud Console.
    """
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    
    if not client_id or not client_secret:
        return None
    
    return {
        "web": {
            "client_id": client_id,
            "project_id": os.environ.get("GOOGLE_OAUTH_PROJECT_ID", "").strip() or None,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": client_secret,
        }
    }


def get_oauth_client_id() -> str:
    """Return currently active OAuth client_id (from env or JSON)."""
    env_cfg = _env_oauth_client()
    if env_cfg:
        return env_cfg.get("web", {}).get("client_id", "")
    
    try:
        secrets_path = _client_secrets_path()
        if secrets_path.exists():
            raw = _read_json(secrets_path)
            if "web" in raw:
                return raw["web"].get("client_id", "")
            if "installed" in raw:
                return raw["installed"].get("client_id", "")
    except Exception:
        pass
    
    return ""


def _flow(*, redirect_uri: str) -> Flow:
    """Create OAuth flow for authorization."""
    env_cfg = _env_oauth_client()
    if env_cfg:
        return Flow.from_client_config(env_cfg, scopes=SCOPES, redirect_uri=redirect_uri)
    
    secrets_path = _client_secrets_path()
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"OAuth not configured. Set GOOGLE_OAUTH_CLIENT_ID/SECRET env vars or create {secrets_path}"
        )
    
    return Flow.from_client_secrets_file(str(secrets_path), scopes=SCOPES, redirect_uri=redirect_uri)


# ============================================================================
# OAUTH AUTHORIZATION FLOW
# ============================================================================

def is_authorized() -> bool:
    """Check if system has valid OAuth token."""
    path = _creds_file()
    if not path.exists():
        return False
    
    try:
        info = _read_json(path)
        return bool(info.get("refresh_token"))
    except Exception:
        return False


def create_auth_url(*, redirect_uri: str) -> str:
    """Create authorization URL for system-wide OAuth consent."""
    nonce = secrets.token_urlsafe(24)
    state = f"system:{nonce}"
    
    _write_json(_state_file(), {
        "nonce": nonce,
        "created_at": int(time.time()),
        "redirect_uri": redirect_uri,
    })
    
    flow = _flow(redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    
    return auth_url


def exchange_code_and_store(state: str, code: str) -> None:
    """Exchange authorization code for credentials and save to token.json."""
    if not state or ":" not in state:
        raise ValueError("Invalid OAuth state")
    
    _, nonce = state.split(":", 1)
    
    sf = _state_file()
    if not sf.exists():
        raise ValueError("OAuth state not found. Start authorization again at /api/sheets/oauth/start")
    
    state_info = _read_json(sf)
    if state_info.get("nonce") != nonce:
        raise ValueError("OAuth state mismatch. Start authorization again.")
    
    redirect_uri = state_info.get("redirect_uri")
    flow = _flow(redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    creds = flow.credentials
    
    # Preserve existing refresh_token if Google doesn't send a new one
    if not creds.refresh_token:
        existing_path = _creds_file()
        if existing_path.exists():
            existing = _read_json(existing_path)
            creds.refresh_token = existing.get("refresh_token")
    
    _save_credentials(creds)
    
    # Cleanup state file
    try:
        sf.unlink(missing_ok=True)
    except Exception:
        pass


def _save_credentials(creds: Credentials) -> None:
    """Save credentials to token.json."""
    payload = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or SCOPES),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    _write_json(_creds_file(), payload)


def _load_credentials() -> Credentials:
    """Load OAuth credentials from token.json."""
    if not is_authorized():
        raise ValueError("Not authorized. Authorize at /api/sheets/oauth/start")
    
    info = _read_json(_creds_file())
    return Credentials(
        token=info.get("token"),
        refresh_token=info.get("refresh_token"),
        token_uri=info.get("token_uri"),
        client_id=info.get("client_id"),
        client_secret=info.get("client_secret"),
        scopes=info.get("scopes", SCOPES),
    )


# ============================================================================
# SHEET EXPORT WITH FORMATTING
# ============================================================================

def _col_letter(col_index: int) -> str:
    """Convert 0-based index to Excel column letter (A, B, ..., Z, AA, AB, ...)."""
    result = ""
    col_index += 1  # Make 1-based
    while col_index > 0:
        col_index -= 1
        result = chr(col_index % 26 + ord('A')) + result
        col_index //= 26
    return result


def _build_format_requests(headers: List[str], data_row_count: int) -> List[Dict[str, Any]]:
    """Build batch formatting requests matching Excel export style.
    
    Applies column background colors and conditional text colors in ONE batch:
    - Timeframe: Light blue bg
    - Params: Light yellow bg  
    - Metrics: Light green bg with conditional text colors
    - Trade-specific: Entry (yellow), Exit (blue), Size (pink), PnL (conditional)
    """
    num_cols = len(headers)
    total_rows = 1 + data_row_count
    
    # Color palette from Excel stress_test
    BLUE_LIGHT = {"red": 204/255, "green": 229/255, "blue": 1.0}        # #CCE5FF
    YELLOW_LIGHT = {"red": 1.0, "green": 244/255, "blue": 204/255}      # #FFF4CC
    GREEN_LIGHT = {"red": 230/255, "green": 247/255, "blue": 230/255}   # #E6F7E6
    PINK_LIGHT = {"red": 1.0, "green": 230/255, "blue": 230/255}        # #FFE6E6
    WHITE = {"red": 1.0, "green": 1.0, "blue": 1.0}
    
    # Text colors
    GREEN_TEXT = {"red": 0, "green": 128/255, "blue": 0}       # #008000
    RED_TEXT = {"red": 192/255, "green": 0, "blue": 0}         # #C00000
    BLACK_TEXT = {"red": 0, "green": 0, "blue": 0}
    
    requests = []
    
    # 1. Freeze header row
    requests.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": 0,
                "gridProperties": {"frozenRowCount": 1}
            },
            "fields": "gridProperties.frozenRowCount"
        }
    })
    
    # 2. Header row formatting (Blue bg, white text, bold, centered)
    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": 0,
                "startRowIndex": 0,
                "endRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": num_cols
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": {"red": 68/255, "green": 114/255, "blue": 196/255},  # #4472C4
                    "textFormat": {
                        "bold": True,
                        "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}
                    },
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment": "MIDDLE"
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"
        }
    })
    
    # 3. Auto-resize all columns
    requests.append({
        "autoResizeDimensions": {
            "dimensions": {
                "sheetId": 0,
                "dimension": "COLUMNS",
                "startIndex": 0,
                "endIndex": num_cols
            }
        }
    })
    
    # 4. Detect sheet type and apply column-specific formatting
    headers_upper = [h.upper() for h in headers]
    
    # Check if this is a trades sheet or results sheet
    is_trades_sheet = any(h in ["ENTRY TIME", "EXIT TIME", "ENTRY PRICE", "EXIT PRICE"] for h in headers_upper)
    
    if is_trades_sheet:
        # TRADE LIST FORMATTING
        # Trade direction column (LONG/SHORT in column "TYPE" or "DIRECTION")
        for col_idx, h in enumerate(headers_upper):
            if h in ["TYPE", "DIRECTION"]:
                # White background for trade direction column
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": total_rows,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": WHITE,
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                    }
                })
                # Conditional text color: LONG = green, SHORT = red
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "TEXT_CONTAINS",
                                    "values": [{"userEnteredValue": "LONG"}]
                                },
                                "format": {"textFormat": {"foregroundColor": GREEN_TEXT, "bold": True}}
                            }
                        },
                        "index": 0
                    }
                })
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "TEXT_CONTAINS",
                                    "values": [{"userEnteredValue": "SHORT"}]
                                },
                                "format": {"textFormat": {"foregroundColor": RED_TEXT, "bold": True}}
                            }
                        },
                        "index": 0
                    }
                })
            
            # Entry columns: yellow background
            elif "ENTRY" in h:
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": total_rows,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": YELLOW_LIGHT,
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                    }
                })
            
            # Exit columns: light blue background
            elif "EXIT" in h:
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": total_rows,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": BLUE_LIGHT,
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                    }
                })
            
            # Position size: light pink background
            elif "POSITION SIZE" in h or "QTY" in h or "QUANTITY" in h:
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": total_rows,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": PINK_LIGHT,
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                    }
                })
            
            # PnL columns: white bg with conditional text colors
            elif any(x in h for x in ["PNL", "P&L", "PROFIT", "LOSS"]):
                requests.append({
                    "repeatCell": {
                        "range": {
                            "sheetId": 0,
                            "startRowIndex": 1,
                            "endRowIndex": total_rows,
                            "startColumnIndex": col_idx,
                            "endColumnIndex": col_idx + 1
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": WHITE,
                                "horizontalAlignment": "RIGHT",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                    }
                })
                # Green if positive, red if negative
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "NUMBER_GREATER",
                                    "values": [{"userEnteredValue": "0"}]
                                },
                                "format": {"textFormat": {"foregroundColor": GREEN_TEXT}}
                            }
                        },
                        "index": 0
                    }
                })
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "NUMBER_LESS",
                                    "values": [{"userEnteredValue": "0"}]
                                },
                                "format": {"textFormat": {"foregroundColor": RED_TEXT}}
                            }
                        },
                        "index": 0
                    }
                })
    
    else:
        # BACKTEST RESULTS FORMATTING (matching Excel stress test)
        # Identify column groups
        timeframe_cols = []
        param_cols = []
        metric_cols = []
        
        for col_idx, h in enumerate(headers_upper):
            if "TIMEFRAME" in h or h in ["STT", "#"]:
                timeframe_cols.append(col_idx)
            elif any(x in h for x in ["EMA", "ATR", "VF", "SF", "HIGH", "LOW", "IR", "ER", "OR", "SKID", "COMMISSION"]):
                param_cols.append(col_idx)
            elif any(x in h for x in ["ROI", "CAGR", "PROFIT", "SHARPE", "SORTINO", "WIN", "MDD", "TRADES", "EQUITY", "FINAL", "MAX"]):
                metric_cols.append(col_idx)
        
        # Apply column background colors
        # Timeframe columns: Light blue
        for col_idx in timeframe_cols:
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 1,
                        "endRowIndex": total_rows,
                        "startColumnIndex": col_idx,
                        "endColumnIndex": col_idx + 1
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": BLUE_LIGHT,
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                }
            })
        
        # Params columns: Light yellow
        for col_idx in param_cols:
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 1,
                        "endRowIndex": total_rows,
                        "startColumnIndex": col_idx,
                        "endColumnIndex": col_idx + 1
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": YELLOW_LIGHT,
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                }
            })
        
        # Metrics columns: Light green with conditional text colors
        for col_idx in metric_cols:
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": 0,
                        "startRowIndex": 1,
                        "endRowIndex": total_rows,
                        "startColumnIndex": col_idx,
                        "endColumnIndex": col_idx + 1
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": GREEN_LIGHT,
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,horizontalAlignment,verticalAlignment)"
                }
            })
            
            # Add conditional text coloring for metrics (green positive, red negative)
            h = headers_upper[col_idx]
            if any(x in h for x in ["ROI", "CAGR", "PROFIT", "SHARPE", "SORTINO", "WIN"]):
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "NUMBER_GREATER",
                                    "values": [{"userEnteredValue": "0"}]
                                },
                                "format": {"textFormat": {"foregroundColor": GREEN_TEXT, "bold": True}}
                            }
                        },
                        "index": 0
                    }
                })
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "NUMBER_LESS",
                                    "values": [{"userEnteredValue": "0"}]
                                },
                                "format": {"textFormat": {"foregroundColor": RED_TEXT, "bold": True}}
                            }
                        },
                        "index": 0
                    }
                })
            elif "MDD" in h:  # MDD is special: negative is red (bad)
                requests.append({
                    "addConditionalFormatRule": {
                        "rule": {
                            "ranges": [{
                                "sheetId": 0,
                                "startRowIndex": 1,
                                "endRowIndex": total_rows,
                                "startColumnIndex": col_idx,
                                "endColumnIndex": col_idx + 1
                            }],
                            "booleanRule": {
                                "condition": {
                                    "type": "NUMBER_LESS",
                                    "values": [{"userEnteredValue": "0"}]
                                },
                                "format": {"textFormat": {"foregroundColor": RED_TEXT, "bold": True}}
                            }
                        },
                        "index": 0
                    }
                })
    
    return requests


def upload_excel_to_google_sheet(
    *,
    excel_path: str | Path,
    title: Optional[str] = None,
    share_email: Optional[str] = None,
) -> str:
    """Upload Excel file to Google Sheets (preserves formatting).
    
    This method uploads the Excel file directly and converts it to Google Sheets,
    which is faster and preserves Excel formatting better than manual cell-by-cell upload.
    
    Args:
        excel_path: Path to Excel file (.xlsx)
        title: Optional custom title (defaults to filename without extension)
        share_email: Optional email to share with
    
    Returns:
        Google Sheets URL
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    if title is None:
        title = excel_path.stem
    
    creds = _load_credentials()
    
    # Refresh token if expired
    if not creds.valid:
        creds.refresh(Request())
        _save_credentials(creds)
    
    # Build Drive service
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    
    # File metadata
    file_metadata = {
        'name': title,
        'mimeType': 'application/vnd.google-apps.spreadsheet',  # Convert to Google Sheets
    }
    
    # Add to folder if specified
    if PARENT_FOLDER_ID:
        file_metadata['parents'] = [PARENT_FOLDER_ID]
    
    # Upload Excel file with conversion to Sheets
    media = MediaFileUpload(
        str(excel_path),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        resumable=True
    )
    
    print(f"[Sheets] Uploading Excel file: {excel_path.name} ({excel_path.stat().st_size / 1024:.1f} KB)")
    
    file = drive.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, mimeType'
    ).execute()
    
    spreadsheet_id = file.get('id')
    print(f"[Sheets] ✓ Uploaded and converted to Google Sheet: {file.get('name')}")
    
    # Share with "anyone with link" can edit
    try:
        drive.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "anyone", "role": "writer", "allowFileDiscovery": False},
            fields="id",
        ).execute()
        print(f"[Sheets] ✓ Shared publicly (anyone with link can edit)")
    except Exception as e:
        print(f"[Sheets] Warning: Public sharing failed: {e}")
    
    # Optional: share with specific email
    if share_email:
        try:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={"type": "user", "role": "writer", "emailAddress": share_email},
                fields="id",
            ).execute()
            print(f"[Sheets] ✓ Shared with {share_email}")
        except Exception as e:
            print(f"[Sheets] Warning: Email sharing failed: {e}")
    
    url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?usp=sharing"
    return url


def export_to_google_sheet(
    *,
    title: str,
    headers: List[str],
    data: List[List[Any]],
    share_email: Optional[str] = None,
) -> str:
    """Export data to Google Sheets with frontend-matching formatting.
    
    ULTRA-OPTIMIZED for large datasets (inspired by xlsxwriter approach):
    - Single batch API call for BOTH data write + formatting (parallel processing)
    - Reduces 2 API calls to 1 call (like writing data+format simultaneously in xlsxwriter)
    - Handles 10K+ rows efficiently
    """
    creds = _load_credentials()
    
    # Refresh token if expired
    if not creds.valid:
        creds.refresh(Request())
        _save_credentials(creds)
    
    gc = gspread.authorize(creds)
    
    # Create spreadsheet
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
    
    # Resize sheet if needed
    required_rows = 1 + len(data)
    required_cols = len(headers)
    if required_rows > sheet.row_count or required_cols > sheet.col_count:
        new_rows = max(required_rows, sheet.row_count)
        new_cols = max(required_cols, sheet.col_count, 26)
        sheet.resize(rows=new_rows, cols=new_cols)
    
    # ============================================================
    # OPTIMIZATION: Combine data write + formatting in 1 API call
    # Like xlsxwriter: ws.write(row, col, value, format) - data+format together
    # Google Sheets: batchUpdate with updateCells + format requests together
    # Result: 2 API calls → 1 API call (much faster for large data)
    # ============================================================
    
    sheets_service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    all_requests = []
    
    # 1. Data Write Request (updateCells for all data at once)
    all_values = [headers] + data
    rows_data = []
    for row_values in all_values:
        cells = []
        for val in row_values:
            # Convert value to appropriate type
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
    
    # 2. Formatting Requests (all column colors, conditional formatting, etc.)
    format_requests = _build_format_requests(headers, len(data))
    if format_requests:
        all_requests.extend(format_requests)
    
    # Execute ALL requests in 1 batch API call (data + formatting in parallel)
    try:
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": all_requests}
        ).execute()
    except Exception as e:
        print(f"[Sheets] Error: Batch update (data+format) failed: {e}")
        # Fallback: try simple write without formatting
        try:
            sheet.update(range_name="A1", values=all_values, value_input_option='RAW')
            print(f"[Sheets] Fallback: Data written without formatting")
        except Exception as e2:
            print(f"[Sheets] Fallback also failed: {e2}")
            raise
    
    # Share with "anyone with link" can edit
    try:
        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        drive.permissions().create(
            fileId=spreadsheet_id,
            body={"type": "anyone", "role": "writer", "allowFileDiscovery": False},
            fields="id",
        ).execute()
    except Exception as e:
        print(f"[Sheets] Warning: Public sharing failed: {e}")
    
    # Optional: share with specific email
    if share_email:
        try:
            spreadsheet.share(share_email, perm_type="user", role="writer")
        except Exception:
            pass
    
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit?usp=sharing"
