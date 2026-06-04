import pandas as pd
import numpy as np
import pickle
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# Required columns for OHLCV data
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']


class DataLoader:
    """Load market data từ file - Enterprise grade"""
    
    # Global cache for dataframes
    _cache: Dict[Tuple[str, int, str], pd.DataFrame] = {}
    _raw_cache: Dict[Tuple[str, int], Any] = {}
    _file_index_cache: Dict[str, Dict[str, Any]] = {}
    _INDEX_TTL_SECONDS = 120
    _CACHE_LIMIT = 64
    _RAW_CACHE_LIMIT = 16

    @classmethod
    def _resolve_alias_dir(cls, alias: str) -> Optional[Path]:
        current = Path(__file__).parent.parent.parent
        key = (alias or '').strip().lower()

        if key == 'grey':
            candidate = Path(os.getenv('GREY_DATA_ALIAS_GREY', str(current / 'data' / 'OKX'))).expanduser()
            return candidate if candidate.exists() and candidate.is_dir() else None

        if key == 'vinh':
            env_candidate = os.getenv('GREY_DATA_ALIAS_VINH', '').strip()
            candidates = []
            if env_candidate:
                candidates.append(Path(env_candidate).expanduser())
            candidates.extend([
                current / 'data' / 'vinh',
                current.parent / 'data' / 'vinh',
                current.parent / 'Gone' / 'vinh' / 'kema' / 'data' / 'OKX_split',
            ])
            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    return candidate
            return None

        return None

    @classmethod
    def _get_filename_index(cls, data_dir: Path) -> Dict[str, Path]:
        """Build and cache a case-insensitive filename index for a directory."""
        try:
            key = str(data_dir.resolve())
        except Exception:
            key = str(data_dir)

        now = time.time()
        try:
            dir_mtime = int(data_dir.stat().st_mtime)
        except Exception:
            dir_mtime = 0

        cached = cls._file_index_cache.get(key)
        if cached:
            if cached.get('dir_mtime') == dir_mtime and (now - float(cached.get('built_at', 0))) < cls._INDEX_TTL_SECONDS:
                return cached.get('index', {})

        index: Dict[str, Path] = {}
        try:
            with os.scandir(data_dir) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    lowered = entry.name.lower()
                    if not (lowered.endswith('.pkl') or lowered.endswith('.csv')):
                        continue
                    index[lowered] = Path(entry.path)
        except Exception:
            index = {}

        cls._file_index_cache[key] = {
            'built_at': now,
            'dir_mtime': dir_mtime,
            'index': index,
        }
        return index

    @classmethod
    def _load_raw_file(cls, path: Path):
        """Load raw serialized data with mtime-aware cache."""
        path = Path(path)
        mtime_ns = int(path.stat().st_mtime_ns)
        raw_key = (str(path), mtime_ns)
        cached = cls._raw_cache.get(raw_key)
        if cached is not None:
            return cached, mtime_ns

        if path.suffix == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif path.suffix == '.csv':
            data = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        if len(cls._raw_cache) >= cls._RAW_CACHE_LIMIT:
            first = next(iter(cls._raw_cache))
            cls._raw_cache.pop(first, None)
        cls._raw_cache[raw_key] = data
        return data, mtime_ns
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Khởi tạo DataLoader
        
        Args:
            data_dir: Thư mục chứa data files. Mặc định: Grey/data/OKX
                     Có thể dùng: 'OKX', 'forex', hoặc absolute path
        """
        current = Path(__file__).parent.parent.parent
        if data_dir is None:
            self.data_dir = current / 'data' / 'OKX'
            return

        raw = str(data_dir).strip()
        alias_dir = self._resolve_alias_dir(raw)
        if alias_dir is not None:
            self.data_dir = alias_dir
            return

        if raw in ['OKX', 'forex', 'OKX_backup', 'XAU']:
            self.data_dir = current / 'data' / raw
            return

        self.data_dir = Path(raw)
    
    def load(self, asset: str, timeframe: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dữ liệu OHLCV
        
        Args:
            asset: Tên asset (ví dụ: BTCUSDT)
            timeframe: Khung thời gian (ví dụ: 4h)
            file_path: Đường dẫn file trực tiếp (override auto-detect)
        
        Returns:
            DataFrame với columns: [open, high, low, close, volume]
            index: timestamp (datetime, UTC-aware)
        """
        def _candidate_assets(sym: str) -> List[str]:
            sym = (sym or '').strip()
            if not sym:
                return []
            cands = [sym, sym.upper(), sym.lower()]

            upper = sym.upper()
            # Common alias: BTCUSD -> BTCUSDT (and similar)
            if upper.endswith('USD') and not upper.endswith('USDT'):
                usdt = upper + 'T'
                cands.extend([usdt, usdt.lower()])

            # De-dup while preserving order
            seen = set()
            out: List[str] = []
            for c in cands:
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)
            return out

        def _find_existing_path(candidates: List[Path]) -> Optional[Path]:
            for p in candidates:
                if p.exists():
                    return p
            return None

        filename_index = self._get_filename_index(self.data_dir)

        def _find_case_insensitive(target_name: str) -> Optional[Path]:
            # Linux FS is case-sensitive; use prebuilt lowercase index for O(1) lookup.
            return filename_index.get(target_name.lower())

        # Determine file path
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        else:
            # Auto-detect: try {asset}_{timeframe}.pkl, then {asset}.pkl
            path = None
            for sym in _candidate_assets(asset):
                candidates = [
                    self.data_dir / f"{sym}_{timeframe}.pkl",
                    self.data_dir / f"{sym.lower()}_{timeframe}.pkl",
                    self.data_dir / f"{sym}.pkl",
                    self.data_dir / f"{sym.lower()}.pkl",
                ]
                found = _find_existing_path(candidates)
                if found is not None:
                    path = found
                    break

                # Case-insensitive fallback for exact filenames
                for name in [
                    f"{sym}_{timeframe}.pkl",
                    f"{sym}.pkl",
                ]:
                    ci = _find_case_insensitive(name)
                    if ci is not None:
                        path = ci
                        break
                if path is not None:
                    break

            if path is None:
                # Last resort: if caller passed something like BTCUSD but only BTCUSDT.pkl exists,
                # try to find any file that starts with the base symbol.
                base = (asset or '').strip().upper()
                if base.endswith('USD') and not base.endswith('USDT'):
                    base = base + 'T'
                try:
                    for lowered_name, indexed_path in filename_index.items():
                        if not lowered_name.endswith('.pkl'):
                            continue
                        stem = Path(lowered_name).stem.upper()
                        if stem == base or stem.startswith(base):
                            p = indexed_path
                            path = p
                            break
                except Exception:
                    path = None
        
        if path is None or not path.exists():
            raise FileNotFoundError(f"File not found for asset={asset}, timeframe={timeframe} in {self.data_dir}")
        
        # Check cache (path + file mtime + timeframe)
        path_mtime_ns = int(path.stat().st_mtime_ns)
        cache_key = (str(path), path_mtime_ns, timeframe)
        if cache_key in self._cache:
            # print(f"   🚀 [Cache Hit] {asset} - {timeframe}")
            return self._cache[cache_key]

        # 1. Load raw data (mtime-aware cached)
        data, path_mtime_ns = self._load_raw_file(path)

        # 2. Extract specific timeframe or Resample
        if isinstance(data, dict):
            # A. Try exact string match ('1h', '4h')
            if timeframe in data:
                df = data[timeframe]
            # B. Try numeric string match ('1', '4')
            elif timeframe.rstrip('mhd') in data:
                df = data[timeframe.rstrip('mhd')]
            # C. Try integer match (1, 4)
            elif timeframe.rstrip('mhd').isdigit() and int(timeframe.rstrip('mhd')) in data:
                df = data[int(timeframe.rstrip('mhd'))]
            # D. Resample from '1h' if exists
            elif '1h' in data:
                print(f"   ℹ️  Timeframe {timeframe} not found in {path.name}. Resampling from 1h...")
                df = self._resample_ohlcv(data['1h'], timeframe)
            else:
                # Fallback: first available
                first_key = list(data.keys())[0]
                print(f"   ⚠️  Timeframe {timeframe} not found. Using {first_key}.")
                df = data[first_key]
        else:
            # It's a single DataFrame
            df = data
            # If requested timeframe is not 1h, try to resample
            if timeframe and timeframe != '1h' and len(df) > 0:
                # Need to check if it's actually 1h data before resampling
                # But for now, assume single file .pkl in OKX is 1h base
                print(f"   ℹ️  Resampling {asset} from base data to {timeframe}...")
                df = self._resample_ohlcv(df, timeframe)
        
        # 3. Standardize and validate
        df = self._standardize_columns(df)
        self._validate_schema(df)
        
        # Store in cache
        if len(self._cache) > self._CACHE_LIMIT: # simple limit
            # Remove first key
            it = iter(self._cache)
            first = next(it)
            del self._cache[first]
            
        self._cache[cache_key] = df

        return df

    def _resample_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample OHLCV data based on timeframe string (e.g. '1h', '4h')"""
        import re
        match = re.match(r'(\d+)([mhd])', timeframe)
        if not match:
            return df
            
        val, unit = match.groups()
        # rule = '1h', '4h', etc.
        rule = f"{val}{unit.lower()}"
        
        # Standardize first to get DatetimeIndex
        df = self._standardize_columns(df)
        
        # Aggregation logic
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add bid/ask if exist
        if 'open_bid' in df.columns:
            agg_dict['open_bid'] = 'first'
        if 'open_ask' in df.columns:
            agg_dict['open_ask'] = 'first'
        
        # Resample
        resampled = df.resample(rule).agg(agg_dict).dropna()
        
        return resampled

    def _load_pkl(self, path: Path, timeframe: str) -> pd.DataFrame:
        # Legacy method - no longer used by main load() but kept for compatibility
        return self.load(None, timeframe, file_path=str(path))
    
    def _load_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa columns names và set index
        
        Steps:
        1. Copy để tránh side-effects
        2. Lowercase tất cả columns
        3. Rename 'time' -> 'timestamp' nếu cần
        4. Convert timestamp to datetime (UTC-aware)
        5. Set timestamp làm index
        6. Sort theo timestamp
        
        Returns:
            DataFrame đã chuẩn hóa
        """
        # Copy để tránh side-effects
        df = df.copy()
        
        # Lowercase columns
        df.columns = [col.lower() for col in df.columns]
        
        # Rename time -> timestamp
        if 'time' in df.columns and 'timestamp' not in df.columns:
            df.rename(columns={'time': 'timestamp'}, inplace=True)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                # Already datetime, just set as index
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            else:
                # Robust dtype detection (not just int64)
                if np.issubdtype(df['timestamp'].dtype, np.integer):
                    # Unix timestamp (milliseconds or seconds)
                    # Use dropna() để tránh crash nếu row đầu là NaN
                    first_ts = df['timestamp'].dropna().iloc[0]
                    
                    if first_ts > 1e12:
                        # Milliseconds
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                    else:
                        # Seconds
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                else:
                    # String or other format
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                
                # Set làm index
                df.set_index('timestamp', inplace=True)
                
                # Sort theo timestamp (bắt buộc cho backtest)
                df.sort_index(inplace=True)
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame schema
        
        Checks:
        1. Required columns exist
        2. Columns are numeric
        3. Index is datetime
        
        Raises:
            ValueError: Schema không hợp lệ
        """
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {REQUIRED_COLUMNS}, Got: {list(df.columns)}"
            )
        
        # Check numeric dtype
        for col in REQUIRED_COLUMNS:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(
                    f"Column '{col}' must be numeric, got {df[col].dtype}"
                )
        
        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Index must be DatetimeIndex, got {type(df.index)}"
            )
        
        # Check timezone awareness (optional but recommended)
        if df.index.tz is None:
            # Warning: not an error, but log it
            # In production, you might want to enforce UTC
            pass