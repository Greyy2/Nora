import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from core.config import settings
from database.mongo_service import MongoService
from optimize.generation import save_to_mongodb
from optimize.optimizer import run_optimization
from optimize.wfa.wfa import run_wfa_optimization
from optimize.history import create_batch, update_status
from services.task_manager import task_manager, TaskStatus

def run_optimization_logic(task_id: str, batch_id: str, config: Dict[str, Any], collection_type: str):
    """
    Main logic for optimization. Updates both TaskManager and MongoDB.
    This function runs in a separate process.
    """
    mongo = MongoService()
    try:
        # Progress callback to bridge Optimizer -> TaskManager (via DB and memory)
        def on_progress(data: Dict[str, Any]):
            completed = data.get('completed', 0)
            total = data.get('total', 0)
            speed = data.get('speed', 0)
            
            # Update MongoDB for frontend legacy polling
            percentage = (completed / total * 100) if total > 0 else 0
            mongo.db['optimize_history'].update_one(
                {'batch_id': batch_id},
                {
                    '$set': {
                        'progress.completed': completed,
                        'progress.total': total,
                        'progress.percentage': round(percentage, 2),
                        'progress.speed': speed,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            # Since this is a separate process, TaskManager in the main process 
            # will poll this information. (In a true distributed system, we'd use Redis)

        update_status(batch_id, 'running', mongo)
        
        if collection_type == 'wfo':
            result = run_wfa_optimization(
                batch_id=batch_id,
                max_workers=settings.MAX_WORKERS,
                config=config,
                progress_callback=on_progress
            )
        else:
            result = run_optimization(
                batch_id=batch_id,
                max_workers=settings.MAX_WORKERS,
                config=config,
                collection_type=collection_type,
                progress_callback=on_progress
            )

        if result:
            update_status(batch_id, 'completed', mongo)
            return result
        return None
    finally:
        mongo.close()

def generate_configs_logic(task_id: str, batch_id: str, config: Dict[str, Any], collection_type: str):
    """Logic for generating configurations with smooth progress reporting."""
    mongo = MongoService()
    try:
        def on_gen_progress(data):
            completed = data.get('completed', 0)
            total = data.get('total', 0)
            # Update MongoDB
            mongo.db['optimize_history'].update_one(
                {'batch_id': batch_id},
                {'$set': {'progress.completed': completed, 'progress.total': total, 'status': 'generating'}}
            )

        total_inserted = save_to_mongodb(
            batch_id=batch_id,
            config=config,
            mongo=mongo,
            batch_size=50000,
            collection_type=collection_type,
            progress_callback=on_gen_progress
        )
        return total_inserted
    finally:
        mongo.close()
