"""
Post-Process Module - Production-Grade DB Writer (NEW STRUCTURE)
Saves results to backtest-result collection
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from queue import Queue, Empty, Full
import threading
import time
import json
from pathlib import Path
from bson import ObjectId
from pymongo.operations import UpdateOne
from pymongo.write_concern import WriteConcern
from database.mongo_service import MongoService
import numpy as np


def save_results_batch(
    results: List[Dict[str, Any]], 
    mongo: MongoService,
    write_concern: Optional[WriteConcern] = None,
    collection_type: str = 'backtest'
) -> Dict[str, int]:
    """
    Save batch of results to MongoDB using fast insert_many()
    
    Args:
        results: List of result dicts with 'config_hash', 'metrics', 'batch_id'
        mongo: MongoService instance
        write_concern: Optional write concern (e.g., w="majority")
        collection_type: 'backtest' or 'wfo' (determines which collection to use)
        
    Returns:
        Dict with matched, modified, failed counts
    """
    if not results:
        return {'matched': 0, 'modified': 0, 'failed': 0}
    
    documents = []
    
    for result in results:
        try:
            config_hash = result.get('config_hash')
            if not config_hash:
                # Skip silently - this is handled upstream
                continue
            
            now = datetime.now(timezone.utc)
            
            if result.get('status') == 'success':
                # CRITICAL: Sanitize metrics to convert NumPy types to Python native
                raw_metrics = result.get('metrics', {})
                clean_metrics = mongo.sanitize_data(raw_metrics)  # Use MongoService method
                
                # 🔧 FIX: Also save params to result document for easy access
                raw_params = result.get('params', {})
                clean_params = mongo.sanitize_data(raw_params)
                
                # Prepare document for insert_many
                doc = {
                    'config_hash': config_hash,
                    'batch_id': result.get('batch_id'),
                    'params': clean_params,  # Add params to result doc
                    'result': {
                        'all': clean_metrics,
                        'filter': []  # Empty initially
                    },
                    'status': 'success',  # Explicit status
                    'created_at': now,
                    'updated_at': now
                }
                
                if 'metadata' in result:
                    doc['metadata'] = result['metadata']
            else:
                # Failed result
                doc = {
                    'config_hash': config_hash,
                    'batch_id': result.get('batch_id'),
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error'),
                    'created_at': now,
                    'updated_at': now
                }
            
            documents.append(doc)
            
        except Exception as e:
            print(f"⚠️ Error preparing document for config_hash {result.get('config_hash', 'unknown')}: {e}")
            continue
    
    if not documents:
        return {'matched': 0, 'modified': 0, 'failed': 0}
    
    # Select collection based on type
    result_collection = mongo.wfo_result if collection_type == 'wfo' else mongo.backtest_result
    
    # Fast insert_many (10-20x faster than UpdateOne operations)
    try:
        if write_concern:
            result = result_collection.with_options(write_concern=write_concern).insert_many(
                documents, ordered=False
            )
        else:
            result = result_collection.insert_many(documents, ordered=False)
        
        inserted = len(result.inserted_ids) if result.inserted_ids else 0
        
        return {
            'matched': 0,
            'modified': 0,
            'upserted': inserted,
            'failed': len(documents) - inserted
        }
        
    except Exception as e:
        from pymongo.errors import BulkWriteError
        
        # BulkWriteError means some succeeded, some failed (duplicate keys)
        if isinstance(e, BulkWriteError):
            details = e.details
            write_errors = details.get('writeErrors', [])
            inserted = details.get('nInserted', 0)
            
            # Log first few errors for debugging (usually duplicate key errors)
            if write_errors:
                duplicate_count = sum(1 for err in write_errors if err.get('code') == 11000)
                other_errors = len(write_errors) - duplicate_count
                if other_errors > 0:
                    print(f"⚠️ Insert partial failure: {inserted} inserted, {duplicate_count} duplicates, {other_errors} errors")
                    for err in write_errors[:3]:
                        if err.get('code') != 11000:  # Not duplicate key
                            print(f"   Error {err.get('index')}: {err.get('errmsg', 'Unknown')}")
            
            # Return partial success stats
            return {
                'matched': 0,
                'modified': 0,
                'upserted': inserted,
                'failed': len(write_errors)
            }
        
        # Network/connection errors - retry with backoff
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                time.sleep(retry_delay)
                print(f"⚠️ Retry {attempt+1}/{max_retries} after error: {type(e).__name__}")
                
                if write_concern:
                    result = result_collection.with_options(write_concern=write_concern).insert_many(
                        documents, ordered=False
                    )
                else:
                    result = result_collection.insert_many(documents, ordered=False)
                
                inserted = len(result.inserted_ids) if result.inserted_ids else 0
                return {
                    'matched': 0,
                    'modified': 0,
                    'upserted': inserted,
                    'failed': len(documents) - inserted
                }
                
            except Exception as retry_err:
                if attempt == max_retries - 1:
                    print(f"❌ Failed after {max_retries} retries: {retry_err}")
                    # Return all as failed
                    return {
                        'matched': 0,
                        'modified': 0,
                        'upserted': 0,
                        'failed': len(documents)
                    }
                retry_delay *= 2
    
    return {'matched': 0, 'modified': 0, 'failed': len(operations)}


class ProductionDBWriter:
    def __init__(
        self, 
        batch_size: int = 10000,  # 10k items per batch
        flush_interval: float = 10.0,  # Flush every 10s
        max_queue_size: int = 50000,  # Not used, kept for compatibility
        dlq_path: Optional[str] = None,
        enable_metrics: bool = True,
        write_concern: Optional[WriteConcern] = None,
        collection_type: str = 'backtest'
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # RAM buffer (no queue, no timeout, no DLQ)
        self.buffer = []  # Main RAM storage
        self.buffer_lock = threading.Lock()  # Thread-safe access
        
        self.running = False
        self.thread = None
        self.enable_metrics = enable_metrics
        self.write_concern = write_concern
        self.collection_type = collection_type
        
        # Stats (thread-safe with lock)
        self.stats_lock = threading.Lock()
        self.total_submitted = 0
        self.total_matched = 0
        self.total_modified = 0
        self.total_upserted = 0
        self.total_failed = 0
        
        # Metrics
        self.metrics = {
            'buffer_size': 0,
            'flush_count': 0,
            'avg_flush_latency': 0.0
        }
        
        self.mongo = None
        
    def start(self):
        """Start writer thread"""
        self.running = True
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
        print(f"💾 DB Writer started: {self.batch_size:,} items/{self.flush_interval}s")
        
    def stop(self):
        """Stop with guaranteed flush"""
        with self.buffer_lock:
            buffer_size = len(self.buffer)
        
        if buffer_size > 0:
            print(f"🛑 Stopping DB Writer ({buffer_size:,} items in buffer)...")
        
        self.running = False
        
        # Wait for writer thread to finish
        if self.thread:
            self.thread.join(timeout=300)  # 5 min timeout for large batches
            if self.thread.is_alive():
                print(f"⚠️  Writer thread still alive after 300s timeout!")
        
        self._print_stats()
        
    def submit(self, result: Dict[str, Any]):
        """Submit to RAM buffer (no timeout, no dropping)"""
        with self.buffer_lock:
            self.buffer.append(result)
        with self.stats_lock:
            self.total_submitted += 1
            
    def _writer_loop(self):
        """Writer loop: periodic flush from RAM buffer"""
        # Single DB connection for entire session
        self.mongo = MongoService()
        
        # Health check
        if not self.mongo.ping():
            print(f"❌ DB Writer: Cannot connect to MongoDB!")
            print(f"   Check MONGO_URI environment variable")
            return
        
        last_flush = time.monotonic()
        
        try:
            while self.running:
                time.sleep(1)  # Check every second
                
                # Update metrics
                with self.buffer_lock:
                    buffer_size = len(self.buffer)
                
                if self.enable_metrics:
                    self.metrics['buffer_size'] = buffer_size
                
                # Flush conditions: 10k items OR 10s elapsed
                elapsed = time.monotonic() - last_flush
                should_flush = buffer_size >= self.batch_size or elapsed >= self.flush_interval
                
                if should_flush and buffer_size > 0:
                    # Extract batch from buffer (thread-safe)
                    with self.buffer_lock:
                        batch = self.buffer[:self.batch_size]
                        self.buffer = self.buffer[self.batch_size:]
                    
                    # Flush to DB
                    self._flush_buffer(batch)
                    last_flush = time.monotonic()
            
            # Final flush after stop
            with self.buffer_lock:
                remaining = self.buffer[:]
                self.buffer = []
            
            if remaining:
                print(f"🔥 Final flush: {len(remaining):,} items")
                # Flush in batches of batch_size
                for i in range(0, len(remaining), self.batch_size):
                    batch = remaining[i:i+self.batch_size]
                    self._flush_buffer(batch)
                
        except Exception as e:
            print(f"❌ Writer error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.mongo:
                self.mongo.close()
                
    def _flush_buffer(self, batch: List[Dict]):
        """Flush batch to DB"""
        if not batch:
            return
        
        flush_start = time.monotonic()
        batch_size = len(batch)
        
        try:
            stats = save_results_batch(batch, self.mongo, self.write_concern, self.collection_type)
            
            # Update metrics
            if self.enable_metrics:
                flush_latency = time.monotonic() - flush_start
                self.metrics['flush_count'] += 1
                prev_avg = self.metrics['avg_flush_latency']
                count = self.metrics['flush_count']
                self.metrics['avg_flush_latency'] = (
                    (prev_avg * (count - 1) + flush_latency) / count
                )
            
            # Update totals
            with self.stats_lock:
                self.total_matched += stats.get('matched', 0)
                self.total_modified += stats.get('modified', 0)
                self.total_upserted += stats.get('upserted', 0)
                self.total_failed += stats.get('failed', 0)
            
            # Log progress
            from tqdm import tqdm
            saved = stats.get('upserted', 0) + stats.get('modified', 0) + stats.get('matched', 0)
            if saved > 0:
                tqdm.write(f"   💾 Work 1: Đã lưu {saved:,}/{batch_size:,} kết quả vào database ({flush_latency:.1f}s)")
            
        except Exception as e:
            print(f"   ❌ Flush error: {e}")
            with self.stats_lock:
                self.total_failed += batch_size
            import traceback
            traceback.print_exc()
            
    def get_buffer_size(self):
        """Get current buffer size (thread-safe)"""
        with self.buffer_lock:
            return len(self.buffer)
                
    def _print_stats(self):
        print(f"✅ DB Writer stopped")
        print(f"   - Submitted: {self.total_submitted:,}")
        print(f"   - Matched:   {self.total_matched:,}")
        print(f"   - Modified:  {self.total_modified:,}")
        print(f"   - Upserted:  {self.total_upserted:,}")
        print(f"   - Failed:    {self.total_failed:,}")
        
        if self.enable_metrics:
            print(f"   📊 Metrics:")
            print(f"      - Flushes: {self.metrics['flush_count']}")
            print(f"      - Avg latency: {self.metrics['avg_flush_latency']:.3f}s")


def replay_dlq_file(dlq_file: Path, writer: ProductionDBWriter):
    """
    Replay failed batches from DLQ back to writer
    
    Args:
        dlq_file: Path to DLQ JSON file
        writer: ProductionDBWriter instance
    """
    try:
        with open(dlq_file, 'r') as f:
            dlq_data = json.load(f)
        
        results = dlq_data.get('results', [])
        print(f"🔁 Replaying {len(results)} results from {dlq_file.name}")
        
        for result in results:
            writer.submit(result)
        
        print(f"✅ Replayed {len(results)} results")
        
        # Archive DLQ file
        archive_path = dlq_file.parent / 'replayed' / dlq_file.name
        archive_path.parent.mkdir(exist_ok=True)
        dlq_file.rename(archive_path)
        
        print(f"📦 Archived to {archive_path}")
        
    except Exception as e:
        print(f"❌ Replay error: {e}")


def aggregate_and_save_results(batch_id: str, mongo: MongoService, collection_type: str = 'backtest') -> Dict[str, Any]:
    """Aggregate results and save summary"""
    try:
        # Use appropriate collection based on type
        result_collection = mongo.wfo_result if collection_type == 'wfo' else mongo.backtest_result
        
        total = result_collection.count_documents({'batch_id': batch_id})
        success = result_collection.count_documents({'batch_id': batch_id, 'status': 'success'})
        failed = result_collection.count_documents({'batch_id': batch_id, 'status': 'failed'})
        
        summary = {
            'total': total,
            'success': success,
            'failed': failed,
            'success_rate': (success / total * 100) if total > 0 else 0,
            'completed_at': datetime.now(timezone.utc)
        }
        
        mongo.db['optimize_history'].update_one(
            {'batch_id': batch_id},
            {'$set': {'status': 'completed', 'summary': summary, 'completed_at': datetime.now(timezone.utc)}}
        )
        
        print(f"✅ Aggregated: {success}/{total} ({summary['success_rate']:.1f}%)")
        return summary
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'error': str(e)}