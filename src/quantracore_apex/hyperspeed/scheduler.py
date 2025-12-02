"""
Overnight Training Scheduler.

Coordinates intensive learning cycles during off-market hours:
- Market close (4 PM ET) to pre-market (4 AM ET)
- Runs historical replay, battle simulations, and model training
- Maximizes learning while markets are closed
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz

from .models import HyperspeedConfig, TrainingCycle, HyperspeedMode

logger = logging.getLogger(__name__)


class SchedulerState(str, Enum):
    IDLE = "idle"
    WAITING = "waiting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ScheduledTask:
    """A task scheduled for overnight execution."""
    task_id: str
    name: str
    callback: Callable
    priority: int = 0
    estimated_duration_minutes: int = 60
    enabled: bool = True
    last_run: Optional[datetime] = None
    last_result: Optional[str] = None


class OvernightScheduler:
    """
    Manages overnight intensive training sessions.
    
    Automatically detects market hours and runs intensive
    learning cycles during off-hours for maximum efficiency.
    """
    
    MARKET_TZ = pytz.timezone("US/Eastern")
    
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    OVERNIGHT_START_HOUR = 17
    OVERNIGHT_END_HOUR = 4
    
    def __init__(self, config: Optional[HyperspeedConfig] = None):
        self.config = config or HyperspeedConfig()
        
        self._state = SchedulerState.IDLE
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self._tasks: Dict[str, ScheduledTask] = {}
        self._cycles_completed: List[TrainingCycle] = []
        self._current_cycle: Optional[TrainingCycle] = None
        
        self._on_cycle_start: Optional[Callable] = None
        self._on_cycle_complete: Optional[Callable] = None
        self._on_task_complete: Optional[Callable] = None
        
        logger.info("[OvernightScheduler] Initialized")
    
    def is_market_hours(self) -> bool:
        """Check if currently within market hours."""
        now = datetime.now(self.MARKET_TZ)
        
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(
            hour=self.MARKET_OPEN_HOUR,
            minute=self.MARKET_OPEN_MINUTE,
            second=0,
            microsecond=0,
        )
        market_close = now.replace(
            hour=self.MARKET_CLOSE_HOUR,
            minute=self.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0,
        )
        
        return market_open <= now <= market_close
    
    def is_overnight_window(self) -> bool:
        """Check if currently in overnight training window."""
        now = datetime.now(self.MARKET_TZ)
        
        hour = now.hour
        
        if hour >= self.OVERNIGHT_START_HOUR or hour < self.OVERNIGHT_END_HOUR:
            return True
        
        if now.weekday() >= 5:
            return True
        
        return False
    
    def get_time_until_overnight(self) -> timedelta:
        """Get time until next overnight window starts."""
        now = datetime.now(self.MARKET_TZ)
        
        if self.is_overnight_window():
            return timedelta(seconds=0)
        
        overnight_start = now.replace(
            hour=self.OVERNIGHT_START_HOUR,
            minute=0,
            second=0,
            microsecond=0,
        )
        
        if now.hour >= self.OVERNIGHT_START_HOUR:
            overnight_start += timedelta(days=1)
        
        return overnight_start - now
    
    def get_overnight_remaining(self) -> timedelta:
        """Get time remaining in current overnight window."""
        now = datetime.now(self.MARKET_TZ)
        
        if not self.is_overnight_window():
            return timedelta(seconds=0)
        
        if now.hour >= self.OVERNIGHT_START_HOUR:
            overnight_end = (now + timedelta(days=1)).replace(
                hour=self.OVERNIGHT_END_HOUR,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            overnight_end = now.replace(
                hour=self.OVERNIGHT_END_HOUR,
                minute=0,
                second=0,
                microsecond=0,
            )
        
        return max(timedelta(seconds=0), overnight_end - now)
    
    def register_task(
        self,
        task_id: str,
        name: str,
        callback: Callable,
        priority: int = 0,
        estimated_minutes: int = 60,
    ):
        """Register a task for overnight execution."""
        self._tasks[task_id] = ScheduledTask(
            task_id=task_id,
            name=name,
            callback=callback,
            priority=priority,
            estimated_duration_minutes=estimated_minutes,
        )
        logger.info(f"[OvernightScheduler] Registered task: {name}")
    
    def unregister_task(self, task_id: str):
        """Remove a registered task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info(f"[OvernightScheduler] Unregistered task: {task_id}")
    
    def set_callbacks(
        self,
        on_cycle_start: Optional[Callable] = None,
        on_cycle_complete: Optional[Callable] = None,
        on_task_complete: Optional[Callable] = None,
    ):
        """Set callbacks for scheduler events."""
        self._on_cycle_start = on_cycle_start
        self._on_cycle_complete = on_cycle_complete
        self._on_task_complete = on_task_complete
    
    def start(self):
        """Start the overnight scheduler."""
        if self._state == SchedulerState.RUNNING:
            logger.warning("[OvernightScheduler] Already running")
            return
        
        self._stop_event.clear()
        self._state = SchedulerState.WAITING
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("[OvernightScheduler] Started")
    
    def stop(self):
        """Stop the overnight scheduler."""
        self._stop_event.set()
        self._state = SchedulerState.IDLE
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        logger.info("[OvernightScheduler] Stopped")
    
    def _run_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                if self.is_overnight_window() and self.config.overnight_training_enabled:
                    self._run_overnight_cycle()
                else:
                    self._state = SchedulerState.WAITING
                    time.sleep(60)
            except Exception as e:
                logger.error(f"[OvernightScheduler] Error in loop: {e}")
                self._state = SchedulerState.ERROR
                time.sleep(300)
    
    def _run_overnight_cycle(self):
        """Execute a complete overnight training cycle."""
        self._state = SchedulerState.RUNNING
        
        cycle = TrainingCycle(
            mode=HyperspeedMode.OVERNIGHT_INTENSIVE,
        )
        self._current_cycle = cycle
        
        if self._on_cycle_start:
            try:
                self._on_cycle_start(cycle)
            except Exception as e:
                logger.error(f"[OvernightScheduler] Cycle start callback error: {e}")
        
        logger.info(f"[OvernightScheduler] Starting overnight cycle {cycle.cycle_id}")
        
        sorted_tasks = sorted(
            [t for t in self._tasks.values() if t.enabled],
            key=lambda t: -t.priority,
        )
        
        for task in sorted_tasks:
            if self._stop_event.is_set():
                break
            
            if not self.is_overnight_window():
                logger.info("[OvernightScheduler] Overnight window ended, stopping cycle")
                break
            
            remaining = self.get_overnight_remaining()
            if remaining.total_seconds() < task.estimated_duration_minutes * 60:
                logger.info(f"[OvernightScheduler] Skipping {task.name}, not enough time")
                continue
            
            logger.info(f"[OvernightScheduler] Running task: {task.name}")
            
            try:
                start_time = time.time()
                result = task.callback()
                elapsed = time.time() - start_time
                
                task.last_run = datetime.utcnow()
                task.last_result = "success"
                
                logger.info(f"[OvernightScheduler] Task {task.name} completed in {elapsed:.1f}s")
                
                if self._on_task_complete:
                    try:
                        self._on_task_complete(task, result)
                    except Exception as e:
                        logger.error(f"[OvernightScheduler] Task callback error: {e}")
                
            except Exception as e:
                task.last_result = f"error: {str(e)}"
                logger.error(f"[OvernightScheduler] Task {task.name} failed: {e}")
                cycle.errors.append(f"{task.name}: {str(e)}")
        
        cycle.completed_at = datetime.utcnow()
        cycle.actual_duration_seconds = (cycle.completed_at - cycle.started_at).total_seconds()
        
        self._cycles_completed.append(cycle)
        self._current_cycle = None
        
        if self._on_cycle_complete:
            try:
                self._on_cycle_complete(cycle)
            except Exception as e:
                logger.error(f"[OvernightScheduler] Cycle complete callback error: {e}")
        
        logger.info(f"[OvernightScheduler] Cycle {cycle.cycle_id} complete")
        
        if self.is_overnight_window():
            time.sleep(300)
    
    def run_now(self) -> Optional[TrainingCycle]:
        """Run a training cycle immediately (manual trigger)."""
        logger.info("[OvernightScheduler] Manual cycle triggered")
        
        original_state = self._state
        self._run_overnight_cycle()
        self._state = original_state
        
        return self._cycles_completed[-1] if self._cycles_completed else None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current scheduler state."""
        return {
            "state": self._state.value,
            "is_market_hours": self.is_market_hours(),
            "is_overnight_window": self.is_overnight_window(),
            "time_until_overnight": str(self.get_time_until_overnight()),
            "overnight_remaining": str(self.get_overnight_remaining()),
            "registered_tasks": len(self._tasks),
            "cycles_completed": len(self._cycles_completed),
            "current_cycle": self._current_cycle.cycle_id if self._current_cycle else None,
        }
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get all registered tasks."""
        return [
            {
                "task_id": t.task_id,
                "name": t.name,
                "priority": t.priority,
                "estimated_minutes": t.estimated_duration_minutes,
                "enabled": t.enabled,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "last_result": t.last_result,
            }
            for t in self._tasks.values()
        ]
    
    def get_recent_cycles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent training cycles."""
        return [c.to_dict() for c in self._cycles_completed[-limit:]]
