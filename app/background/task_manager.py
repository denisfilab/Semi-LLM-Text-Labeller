from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional
import threading
from enum import Enum
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TaskInfo:
    id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class TaskManager:
    def __init__(self, max_workers: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    def submit_task(self, func, *args, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        
        def wrapped_task(*args, **kwargs):
            with self._lock:
                self.tasks[task_id].status = TaskStatus.RUNNING
                self.tasks[task_id].start_time = datetime.now()
            
            # Execute function directly without try-except
            result = func(*args, **kwargs)
            
            with self._lock:
                self.tasks[task_id].status = TaskStatus.COMPLETED
                self.tasks[task_id].result = result
                self.tasks[task_id].end_time = datetime.now()
            
            return result
        
        with self._lock:
            self.tasks[task_id] = TaskInfo(
                id=task_id,
                status=TaskStatus.PENDING
            )
        
        self.executor.submit(wrapped_task, *args, **kwargs)
        return task_id

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        return self.tasks.get(task_id)

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskInfo]:
        start_time = time.time()
        while True:
            task_info = self.get_task_status(task_id)
            if task_info and task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return task_info
            
            if timeout and (time.time() - start_time) > timeout:
                return None
                
            time.sleep(0.1)

# Create global task manager instance
task_manager = TaskManager()