import asyncio
import uuid
import subprocess
import os
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from core.config import settings

logger = logging.getLogger(__name__)

class TaskStatus:
    PENDING = "pending"
    GENERATING = "generating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Task:
    def __init__(self, task_id: str, task_type: str, metadata: Optional[Dict] = None):
        self.id = task_id
        self.type = task_type
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.metadata = metadata or {}
        self.progress = 0.0
        self.completed_count = 0
        self.total_count = 0
        self.speed = 0.0
        self.message = "Task initialized"
        self.result = None
        self.error = None
        self.pid = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at.isoformat() + "Z",
            "updated_at": self.updated_at.isoformat() + "Z",
            "metadata": self.metadata,
            "progress": self.progress,
            "completed": self.completed_count,
            "total": self.total_count,
            "speed": self.speed,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "pid": self.pid
        }

class TaskManager:
    """
    Institutional Task Manager for Grey.
    Standardizes background execution across all modules.
    """
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._subprocesses: Dict[str, subprocess.Popen] = {}
        self._executor = ProcessPoolExecutor(max_workers=settings.MAX_WORKERS)
        self._loop = asyncio.get_event_loop()

    def create_task(self, task_type: str, metadata: Optional[Dict] = None, task_id: Optional[str] = None) -> str:
        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        task = Task(tid, task_type, metadata)
        self._tasks[tid] = task
        return tid

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return a serialized task payload used by API routes."""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def update_task(self, task_id: str, **kwargs):
        if task_id in self._tasks:
            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
                elif key in ["completed", "total"]: # Compatibility mapping
                    if key == "completed": task.completed_count = value
                    if key == "total": task.total_count = value
                else:
                    task.metadata[key] = value
            task.updated_at = datetime.utcnow()

    def get_task_logs(self, task_id: str, last_lines: int = 100) -> str:
        """Reads the tail of the log file for a specific task."""
        log_path = settings.GREY_TMP_DIR / "logs" / task_id / "run.log"
        if not log_path.exists():
            return "Log file not found."
        
        try:
            with open(log_path, "rb") as f:
                # Efficiently seek to end and read last N lines
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                buffer_size = 8192
                lines_found = 0
                data = []
                
                pos = file_size
                while pos > 0 and lines_found <= last_lines:
                    seek_pos = max(0, pos - buffer_size)
                    f.seek(seek_pos)
                    chunk = f.read(pos - seek_pos)
                    lines_found += chunk.count(b'\n')
                    data.insert(0, chunk)
                    pos = seek_pos
                
                content = b"".join(data).decode("utf-8", errors="replace")
                return "\n".join(content.splitlines()[-last_lines:])
        except Exception as e:
            return f"Error reading logs: {e}"

    def update_progress(self, task_id: str, completed: int, total: int):
        """Standardized progress update with throughput calculation."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            now = datetime.utcnow()
            
            # Calculate speed (configs per second)
            elapsed = (now - task.updated_at).total_seconds()
            if elapsed > 0:
                delta_completed = completed - task.completed_count
                current_speed = delta_completed / elapsed
                # Simple EMA for smooth speed reporting
                task.speed = (task.speed * 0.7) + (current_speed * 0.3) if task.speed > 0 else current_speed
            
            task.completed_count = completed
            task.total_count = total
            task.progress = (completed / total * 100) if total > 0 else 0
            task.updated_at = now
            
            # Update message with ETA
            if task.speed > 0 and total > completed:
                eta_seconds = (total - completed) / task.speed
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                task.message = f"Processing: {completed}/{total} | Speed: {task.speed:.1f} cfg/s | ETA: {eta_str}"

    async def run_subprocess(self, task_id: str, cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict] = None):
        """Run a CLI command in background."""
        if task_id not in self._tasks: return
        
        log_dir = settings.GREY_TMP_DIR / "logs" / task_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"
        
        self.update_task(task_id, status=TaskStatus.RUNNING, message="🚀 Starting subprocess...")

        try:
            with open(log_file, "ab") as f:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd or settings.WORKSPACE_ROOT),
                    env={**os.environ, **(env or {})},
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
                
                self._subprocesses[task_id] = proc
                self.update_task(task_id, pid=proc.pid)

                # Wait for completion without blocking
                rc = await self._loop.run_in_executor(None, proc.wait)

                if rc == 0:
                    self.update_task(task_id, status=TaskStatus.COMPLETED, message="✅ Completed successfully")
                else:
                    self.update_task(task_id, status=TaskStatus.FAILED, message=f"❌ Failed (Exit Code: {rc})")

        except Exception as e:
            logger.error(f"Subprocess error: {e}")
            self.update_task(task_id, status=TaskStatus.FAILED, error=str(e), message="❌ Subprocess crashed")
        finally:
            self._subprocesses.pop(task_id, None)

    async def run_cpu_bound(self, task_id: str, func: Callable, *args, **kwargs):
        """Run a heavy CPU task in ProcessPoolExecutor."""
        self.update_task(task_id, status=TaskStatus.RUNNING, message="⚙️ Running CPU-bound task...")
        
        try:
            result = await self._loop.run_in_executor(self._executor, func, *args, **kwargs)
            self.update_task(task_id, status=TaskStatus.COMPLETED, result=result, message="✅ CPU task finished")
            return result
        except Exception as e:
            logger.error(f"CPU task error: {e}")
            self.update_task(task_id, status=TaskStatus.FAILED, error=str(e), message="❌ CPU task failed")
            raise e

    def cancel_task(self, task_id: str):
        if task_id in self._subprocesses:
            proc = self._subprocesses[task_id]
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                self.update_task(task_id, status=TaskStatus.CANCELLED, message="⚠️ Cancelled by user")
            except Exception as e:
                logger.warning(f"Failed to kill process group: {e}")
        
        # Note: ProcessPoolExecutor tasks cannot be easily cancelled once started in standard Python.
        # Future improvement: Use a library like 'billiard' (used by Celery).

# Global Singleton
task_manager = TaskManager()
