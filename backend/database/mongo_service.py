import os
import hashlib 
import json 
import threading
from datetime import datetime 
from typing import Dict, Any, List, Optional
import numpy as np
from core.config import settings

_MONGO_IMPORT_ERROR: Optional[str] = None
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
    from pymongo.collection import Collection
    from pymongo.errors import BulkWriteError, ConnectionFailure
except Exception as e:
    MongoClient = None  # type: ignore[assignment]
    ASCENDING = DESCENDING = UpdateOne = None  # type: ignore[assignment]
    Collection = object  # type: ignore[assignment]
    BulkWriteError = ConnectionFailure = Exception  # type: ignore[assignment]
    _MONGO_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# Global singleton client
_MONGO_CLIENT: Optional[Any] = None

def _get_client(uri: Optional[str] = None):
    """Get MongoDB client with connection pool"""
    if MongoClient is None:
        raise RuntimeError(
            "MongoDB client is not available in this environment "
            f"({_MONGO_IMPORT_ERROR})."
        )
    global _MONGO_CLIENT 
    if _MONGO_CLIENT is None:
        if not uri:
            uri = settings.MONGO_URI

        # CRITICAL: connect=False for fork-safety (multiprocessing)
        _MONGO_CLIENT = MongoClient(
            uri, 
            maxPoolSize=100,
            serverSelectionTimeoutMS=5000, 
            connectTimeoutMS=5000,
            connect=False
        )
    return _MONGO_CLIENT

class MongoService:
    """
    MongoDB Service with singleton pattern and optimized operations (v4.0)
    """

    _instance = None
    _indexes_initialized = False
    _indexes_initializing = False
    _index_lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MongoService, cls).__new__(cls)
            cls._instance._initialized = False 
        return cls._instance 
    
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        """Initialize MongoDB service using settings"""
        if getattr(self, '_initialized', False):
            return 
        
        self.client = _get_client(uri)
        # Use provided db_name or default from settings
        target_db = db_name or settings.MONGO_DB
        self.db = self.client[target_db]

        # Collections
        self.backtest_config: Collection = self.db['backtest-config']
        self.backtest_result: Collection = self.db['backtest-result']
        self.wfo_config: Collection = self.db['wfo-config']
        self.wfo_result: Collection = self.db['wfo-result']
        self.wfa_config: Collection = self.db['wfa-config']
        self.wfa_result: Collection = self.db['wfa-result']
        self.carlo_config: Collection = self.db['carlo-config']
        self.carlo_result: Collection = self.db['carlo-result']
        self.history: Collection = self.db['optimize_history']
        self.wfa_analysis: Collection = self.db['wfa-analysis']

        self._initialized = True
        # Ensure indexes are initialized once without blocking API request path
        self._ensure_indexes_async()

    def _ensure_indexes_async(self):
        """Initialize indexes in the background once per process."""
        if MongoService._indexes_initialized:
            return

        with MongoService._index_lock:
            if MongoService._indexes_initialized or MongoService._indexes_initializing:
                return
            MongoService._indexes_initializing = True

        def _runner():
            try:
                self.initialize_indexes()
                MongoService._indexes_initialized = True
            finally:
                MongoService._indexes_initializing = False

        threading.Thread(target=_runner, daemon=True, name="mongo-index-init").start()
    
    def initialize_indexes(self):
        """Create indexes for fast queries - Essential for Big Data"""
        def _safe_create_index(collection: Collection, keys, **kwargs):
            try:
                collection.create_index(keys, **kwargs)
            except Exception as exc:
                # Existing deployments may already have equivalent indexes with
                # different names/options. Never fail request path because of this.
                print(f"Index initialization warning ({collection.name}): {exc}")

        try:
            # Backtest-config - Non-unique vì database cũ có thể có duplicates
            _safe_create_index(self.backtest_config, [('config_hash', ASCENDING)], unique=False)
            _safe_create_index(self.backtest_config, [('batch_id', ASCENDING)])
            _safe_create_index(self.backtest_config, [('batch_id', ASCENDING), ('config_hash', ASCENDING)])
            
            # Backtest-result (CRITICAL FOR PERFORMANCE)
            _safe_create_index(self.backtest_result, [('batch_id', ASCENDING)])
            _safe_create_index(self.backtest_result, [('config_hash', ASCENDING)], unique=False)
            _safe_create_index(self.backtest_result, [('result.all.roi', DESCENDING)])
            _safe_create_index(self.backtest_result, [('result.all.sharpe', DESCENDING)])
            _safe_create_index(self.backtest_result, [('result.all.sharpe_ratio', DESCENDING)])
            _safe_create_index(self.backtest_result, [('batch_id', ASCENDING), ('status', ASCENDING), ('result.all.roi', DESCENDING)])
            _safe_create_index(self.backtest_result, [('batch_id', ASCENDING), ('status', ASCENDING), ('created_at', DESCENDING)])
            
            # History
            _safe_create_index(self.history, [('batch_id', ASCENDING)], unique=False)
            _safe_create_index(self.history, [('status', ASCENDING)])
            _safe_create_index(self.history, [('collection_type', ASCENDING), ('created_at', DESCENDING)])
            _safe_create_index(self.history, [('collection_type', ASCENDING), ('status', ASCENDING), ('created_at', DESCENDING)])
            _safe_create_index(self.history, [('batch_id', ASCENDING), ('created_at', DESCENDING)])
        except Exception as e:
            # Ignore index errors khi kết nối database có sẵn data
            print(f"Index initialization warning: {e}")
        
        # Others
        for coll in [self.wfo_result, self.wfa_result, self.carlo_result]:
            _safe_create_index(coll, [('batch_id', ASCENDING)])
            # Keep non-unique to avoid startup failures on legacy datasets
            # where duplicate config_hash values already exist.
            _safe_create_index(coll, [('config_hash', ASCENDING)], unique=False)
            _safe_create_index(coll, [('batch_id', ASCENDING), ('status', ASCENDING), ('result.all.roi', DESCENDING)])
            _safe_create_index(coll, [('created_at', DESCENDING)])

        try:
            _safe_create_index(self.carlo_config, [('config_hash', ASCENDING)], unique=False)
            _safe_create_index(self.carlo_config, [('batch_id', ASCENDING)])
            _safe_create_index(self.carlo_config, [('config.source_campaign_id', ASCENDING), ('created_at', DESCENDING)])
        except Exception as e:
            print(f"Carlo config index initialization warning: {e}")

        try:
            _safe_create_index(self.wfa_analysis, [('batch_id', ASCENDING)])
            _safe_create_index(self.wfa_analysis, [('status', ASCENDING), ('created_at', DESCENDING)])
            _safe_create_index(self.wfa_analysis, [('mode', ASCENDING), ('created_at', DESCENDING)])
            _safe_create_index(self.wfa_analysis, [('source_batch_id', ASCENDING), ('mode', ASCENDING), ('created_at', DESCENDING)])
        except Exception as e:
            print(f"WFA index initialization warning: {e}")

    def ping(self) -> bool:
        """Return True when the Mongo server is reachable."""
        try:
            self.client.admin.command('ping')
            self._last_ping_error = None
            return True
        except Exception as exc:
            self._last_ping_error = f"{type(exc).__name__}: {exc}"
            return False

    @staticmethod
    def sanitize_data(data):
        if isinstance(data, dict):
            return {k: MongoService.sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [MongoService.sanitize_data(i) for i in data]
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64)):
            val = float(data)
            return None if np.isnan(val) or np.isinf(val) else val
        elif isinstance(data, (np.bool_, bool)):
            return bool(data)
        elif data is None or isinstance(data, (str, int, float, bool)):
            return data
        return str(data)

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        s = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(s.encode('utf-8')).hexdigest()

    def get_strategies_collection(self, collection_type: str = 'backtest') -> Collection:
        """Select result collection by logical type."""
        if collection_type == 'wfo':
            return self.wfo_result
        if collection_type == 'wfa':
            return self.wfa_result
        if collection_type == 'carlo':
            return self.carlo_result
        return self.backtest_result

    # ============================================================
    # CARLO (MONTE CARLO) METHODS
    # ============================================================

    def save_carlo_config(self, batch_id: str, config: Dict[str, Any]) -> str:
        """Save Monte Carlo simulation config and return config_hash."""
        config_hash = self._generate_hash(config)
        document = {
            'config_hash': config_hash,
            'batch_id': batch_id,
            'config': self.sanitize_data(config),
            'created_at': datetime.utcnow(),
        }
        self.carlo_config.update_one(
            {'config_hash': config_hash},
            {'$set': document},
            upsert=True,
        )
        return config_hash

    def save_carlo_result(
        self,
        batch_id: str,
        config_hash: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save Monte Carlo simulation results."""
        document = {
            'config_hash': config_hash,
            'batch_id': batch_id,
            'results': self.sanitize_data(results),
            'metadata': metadata or {},
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }
        self.carlo_result.update_one(
            {'config_hash': config_hash},
            {'$set': document},
            upsert=True,
        )
        return config_hash

    def get_carlo_result(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get Monte Carlo result by batch_id."""
        return self.carlo_result.find_one({'batch_id': batch_id})

    def list_carlo_campaigns(self, limit: int = 50, skip: int = 0, sort_by: str = 'created_at') -> List[Dict[str, Any]]:
        """List Carlo campaigns from result collection."""
        cursor = self.carlo_result.find().sort(sort_by, DESCENDING).skip(skip).limit(limit)
        return list(cursor)

    def get_carlo_config(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """Get Carlo config by config_hash."""
        return self.carlo_config.find_one({'config_hash': config_hash})

    def delete_carlo_campaign(self, batch_id: str) -> int:
        """Delete Carlo campaign across config/result collections."""
        deleted_config = self.carlo_config.delete_many({'batch_id': batch_id}).deleted_count
        deleted_result = self.carlo_result.delete_many({'batch_id': batch_id}).deleted_count
        return deleted_config + deleted_result

    def save_carlo_stats_for_strategy(
        self,
        source_batch_id: str,
        source_collection: str,
        strategy_hash: str,
        carlo_stats: Dict[str, Any],
    ) -> bool:
        """Attach Carlo analysis to a source strategy document."""
        collection = self.get_strategies_collection(
            'backtest' if source_collection == 'backtest-result' else
            'wfa' if source_collection == 'wfa-result' else
            'wfo' if source_collection == 'wfo-result' else
            'carlo'
        )

        result = collection.update_one(
            {'batch_id': source_batch_id, 'config_hash': strategy_hash},
            {
                '$set': {
                    'carlo_analysis': self.sanitize_data(carlo_stats),
                    'carlo_updated_at': datetime.utcnow(),
                }
            },
        )
        return result.modified_count > 0

    def get_carlo_stats_for_strategy(
        self,
        source_batch_id: str,
        source_collection: str,
        strategy_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Read Carlo analysis attached to a source strategy."""
        collection = self.get_strategies_collection(
            'backtest' if source_collection == 'backtest-result' else
            'wfa' if source_collection == 'wfa-result' else
            'wfo' if source_collection == 'wfo-result' else
            'carlo'
        )

        doc = collection.find_one(
            {'batch_id': source_batch_id, 'config_hash': strategy_hash},
            {'_id': 0, 'carlo_analysis': 1},
        )
        if not doc:
            return None
        return doc.get('carlo_analysis')

    def list_backtest_results(self, batch_id: str, skip: int = 0, limit: int = 100, sort_by: str = 'result.all.roi'):
        """Paginated results for smooth UI"""
        query = {'batch_id': batch_id}
        cursor = self.backtest_result.find(query).sort(sort_by, DESCENDING).skip(skip).limit(limit)
        return list(cursor)

    def close(self):
        pass
