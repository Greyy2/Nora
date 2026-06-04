"""
History Module
Manages optimization batch history and metadata
"""

from typing import Dict, Any, Optional
from datetime import datetime

from database.mongo_service import MongoService

def create_batch(batch_id: str, config: Dict[str, Any], mongo: MongoService, collection_type: str = 'backtest') -> Dict[str, Any]:
    """
    Create new batch in optimize_history
    
    Args:
        batch_id: Batch ID
        config: Configuration dict
        mongo: MongoService instance
        collection_type: 'backtest' or 'wfo'
        
    Returns:
        Created batch document
    """
    # Sanitize config to convert NumPy types to Python native types
    clean_config = MongoService.sanitize_data(config)
    
    doc = {
        'batch_id': batch_id,
        'config': clean_config,
        'status': 'pending',
        'collection_type': collection_type,
        'created_at': datetime.utcnow(),
        'progress': {
            'completed': 0,
            'total': 0,
            'percentage': 0
        }
    }

    mongo.db['optimize_history'].insert_one(doc)
    return doc

def update_status(batch_id: str, status: str, mongo: MongoService):
    """
    Update batch status
    
    Args:
        batch_id: Batch ID
        status: New status (pending/running/success/failed)
        mongo: MongoService instance
    """
    print(f"🔧 [UPDATE_STATUS] Updating {batch_id} → {status}")
    result = mongo.db['optimize_history'].update_one(
        {'batch_id': batch_id}, 
        {'$set': {'status': status, 'updated_at': datetime.utcnow()}}
    )
    print(f"✅ [UPDATE_STATUS] MongoDB update result: matched={result.matched_count}, modified={result.modified_count}")

def save_filter(batch_id: str, filter_config: Dict[str, Any], mongo: MongoService):
    """
    Save filter configuration to history
    
    Args:
        batch_id: Batch ID
        filter_config: Filter configuration
        mongo: MongoService instance
    """
    # Sanitize filter config to convert NumPy types to Python native types
    clean_filter = MongoService.sanitize_data(filter_config)
    mongo.db['optimize_history'].update_one({'batch_id': batch_id}, {'$set': {'filter': clean_filter, 'filter_updated_at': datetime.utcnow()}})

def get_batch_info(batch_id: str, mongo: MongoService) -> Optional[Dict[str, Any]]:
    """
    Get batch metadata
    
    Args:
        batch_id: Batch ID
        mongo: MongoService instance
        
    Returns:
        Batch document or None
    """
    return mongo.db['optimize_history'].find_one({'batch_id': batch_id})