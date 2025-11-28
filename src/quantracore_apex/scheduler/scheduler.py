"""
QuantraCore Apex Scheduler — Task Scheduling System

Provides deterministic, research-only scheduled task execution.
No live trading or heavy automated processes.

Version: 8.1
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Any
from enum import Enum


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """
    Represents a scheduled task in the Apex system.
    
    All tasks are research-mode only and deterministic.
    """
    task_id: str
    name: str
    callback: Callable[[], Any]
    interval_seconds: int
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout_seconds: int = 300
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    failure_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_run(self, now: datetime) -> bool:
        """Check if task should run based on schedule."""
        if not self.enabled:
            return False
        if self.next_run is None:
            return True
        return now >= self.next_run
    
    def update_schedule(self, now: datetime) -> None:
        """Update next run time after execution."""
        self.last_run = now
        self.next_run = now + timedelta(seconds=self.interval_seconds)
        self.run_count += 1


@dataclass
class TaskResult:
    """Result of a scheduled task execution."""
    task_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    duration_ms: float
    result: Optional[Any] = None
    error: Optional[str] = None


class ApexScheduler:
    """
    QuantraCore Apex Scheduler
    
    Manages scheduled tasks for research and analysis workflows.
    All scheduling is deterministic and research-only.
    
    Features:
    - Priority-based task queue
    - Configurable intervals
    - Failure tracking and retry logic
    - Task history logging
    
    Note: This scheduler is for research automation only.
    No live trading or real-money operations are supported.
    """
    
    def __init__(self, max_concurrent: int = 2):
        """
        Initialize the scheduler.
        
        Args:
            max_concurrent: Maximum concurrent task executions
        """
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_history: List[TaskResult] = []
        self.max_concurrent = max_concurrent
        self.running = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        
    def add_task(self, task: ScheduledTask) -> None:
        """
        Add a scheduled task.
        
        Args:
            task: The task to schedule
        """
        with self._lock:
            self.tasks[task.task_id] = task
            if task.next_run is None:
                task.next_run = datetime.utcnow()
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a scheduled task.
        
        Args:
            task_id: ID of task to remove
            
        Returns:
            True if task was removed, False if not found
        """
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                return True
            return False
    
    def enable_task(self, task_id: str) -> bool:
        """Enable a task by ID."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].enabled = True
                return True
            return False
    
    def disable_task(self, task_id: str) -> bool:
        """Disable a task by ID."""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id].enabled = False
                return True
            return False
    
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Get all tasks that are due to run, sorted by priority."""
        now = datetime.utcnow()
        with self._lock:
            pending = [t for t in self.tasks.values() if t.should_run(now)]
            return sorted(pending, key=lambda x: x.priority.value, reverse=True)
    
    def execute_task(self, task: ScheduledTask) -> TaskResult:
        """
        Execute a single task.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult with execution details
        """
        start_time = datetime.utcnow()
        task.status = TaskStatus.RUNNING
        
        try:
            result = task.callback()
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            task.status = TaskStatus.COMPLETED
            task.update_schedule(end_time)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                result=result,
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            task.failure_count += 1
            task.status = TaskStatus.FAILED
            task.update_schedule(end_time)
            
            return TaskResult(
                task_id=task.task_id,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                error=str(e),
            )
    
    def run_once(self) -> List[TaskResult]:
        """
        Run all pending tasks once.
        
        Returns:
            List of task results
        """
        results = []
        pending = self.get_pending_tasks()
        
        for task in pending[:self.max_concurrent]:
            result = self.execute_task(task)
            results.append(result)
            self.task_history.append(result)
        
        return results
    
    def start(self, poll_interval: float = 1.0) -> None:
        """
        Start the scheduler background thread.
        
        Args:
            poll_interval: Seconds between schedule checks
        """
        if self.running:
            return
            
        self.running = True
        
        def _scheduler_loop():
            while self.running:
                self.run_once()
                time.sleep(poll_interval)
        
        self._thread = threading.Thread(target=_scheduler_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status summary."""
        with self._lock:
            return {
                "running": self.running,
                "total_tasks": len(self.tasks),
                "enabled_tasks": sum(1 for t in self.tasks.values() if t.enabled),
                "total_executions": len(self.task_history),
                "recent_failures": sum(
                    1 for r in self.task_history[-100:] if not r.success
                ),
                "compliance_note": "Research scheduler only - not for live trading",
            }
    
    def get_task_history(self, task_id: Optional[str] = None, limit: int = 50) -> List[TaskResult]:
        """
        Get task execution history.
        
        Args:
            task_id: Optional filter by task ID
            limit: Maximum results to return
            
        Returns:
            List of task results
        """
        history = self.task_history
        if task_id:
            history = [r for r in history if r.task_id == task_id]
        return history[-limit:]


def create_scan_task(
    task_id: str,
    symbols: List[str],
    scan_fn: Callable,
    interval_hours: int = 1,
) -> ScheduledTask:
    """
    Factory function to create a universe scan task.
    
    Args:
        task_id: Unique task identifier
        symbols: List of symbols to scan
        scan_fn: Function to execute scan
        interval_hours: Hours between scans
        
    Returns:
        Configured ScheduledTask
    """
    def _run_scan():
        results = []
        for symbol in symbols:
            try:
                result = scan_fn(symbol)
                results.append({"symbol": symbol, "result": result})
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})
        return results
    
    return ScheduledTask(
        task_id=task_id,
        name=f"Universe Scan ({len(symbols)} symbols)",
        callback=_run_scan,
        interval_seconds=interval_hours * 3600,
        priority=TaskPriority.NORMAL,
        metadata={"symbols": symbols},
    )
