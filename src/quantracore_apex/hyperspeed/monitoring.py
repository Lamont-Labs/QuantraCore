"""
Enhanced Monitoring for Hyperspeed Learning System.

Provides thread lifecycle tracking, health checks, and alerting
for the OvernightScheduler and other hyperspeed components.
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThreadState(str, Enum):
    """Thread lifecycle states."""
    NOT_STARTED = "not_started"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    STALLED = "stalled"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ThreadHealthMetrics:
    """Health metrics for a monitored thread."""
    thread_id: str
    name: str
    state: ThreadState = ThreadState.NOT_STARTED
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    heartbeat_interval_seconds: int = 60
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    is_alive: bool = False
    uptime_seconds: float = 0.0


@dataclass
class HealthAlert:
    """A health alert from the monitoring system."""
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class SchedulerMonitor:
    """
    Monitors OvernightScheduler thread lifecycle and health.
    
    Features:
    - Thread state tracking
    - Heartbeat monitoring
    - Stall detection
    - Alert generation
    - Recovery suggestions
    """
    
    STALL_THRESHOLD_SECONDS = 300
    HEARTBEAT_GRACE_PERIOD = 2.0
    
    def __init__(self, scheduler=None):
        self.scheduler = scheduler
        self._metrics: Dict[str, ThreadHealthMetrics] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_handlers: List[Callable[[HealthAlert], None]] = []
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._alert_counter = 0
        
        logger.info("[SchedulerMonitor] Initialized")
    
    def register_thread(
        self,
        thread_id: str,
        name: str,
        heartbeat_interval: int = 60,
    ):
        """Register a thread for monitoring."""
        with self._lock:
            self._metrics[thread_id] = ThreadHealthMetrics(
                thread_id=thread_id,
                name=name,
                heartbeat_interval_seconds=heartbeat_interval,
            )
        
        logger.info(f"[Monitor] Registered thread: {name} ({thread_id})")
    
    def record_heartbeat(self, thread_id: str):
        """Record a heartbeat from a thread."""
        with self._lock:
            if thread_id in self._metrics:
                metrics = self._metrics[thread_id]
                metrics.last_heartbeat = datetime.utcnow()
                metrics.is_alive = True
                
                if metrics.state == ThreadState.STALLED:
                    metrics.state = ThreadState.RUNNING
                    self._generate_alert(
                        AlertLevel.INFO,
                        thread_id,
                        f"Thread {metrics.name} recovered from stall",
                    )
    
    def record_cycle_complete(self, thread_id: str, success: bool, error: Optional[str] = None):
        """Record completion of a work cycle."""
        with self._lock:
            if thread_id in self._metrics:
                metrics = self._metrics[thread_id]
                metrics.total_cycles += 1
                
                if success:
                    metrics.successful_cycles += 1
                else:
                    metrics.failed_cycles += 1
                    metrics.last_error = error
                    metrics.last_error_at = datetime.utcnow()
                    
                    if metrics.failed_cycles > 3:
                        self._generate_alert(
                            AlertLevel.WARNING,
                            thread_id,
                            f"Thread {metrics.name} has {metrics.failed_cycles} consecutive failures",
                            {"last_error": error},
                        )
    
    def update_state(self, thread_id: str, state: ThreadState):
        """Update the state of a monitored thread."""
        with self._lock:
            if thread_id in self._metrics:
                old_state = self._metrics[thread_id].state
                self._metrics[thread_id].state = state
                
                if state == ThreadState.RUNNING and old_state != ThreadState.RUNNING:
                    self._metrics[thread_id].started_at = datetime.utcnow()
                
                if state == ThreadState.ERROR:
                    self._generate_alert(
                        AlertLevel.ERROR,
                        thread_id,
                        f"Thread {self._metrics[thread_id].name} entered error state",
                    )
                
                logger.debug(f"[Monitor] Thread {thread_id} state: {old_state} -> {state}")
    
    def get_thread_health(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get health metrics for a specific thread."""
        with self._lock:
            if thread_id not in self._metrics:
                return None
            
            metrics = self._metrics[thread_id]
            
            if metrics.started_at:
                metrics.uptime_seconds = (datetime.utcnow() - metrics.started_at).total_seconds()
            
            return {
                "thread_id": metrics.thread_id,
                "name": metrics.name,
                "state": metrics.state.value,
                "started_at": metrics.started_at.isoformat() if metrics.started_at else None,
                "last_heartbeat": metrics.last_heartbeat.isoformat() if metrics.last_heartbeat else None,
                "is_alive": metrics.is_alive,
                "uptime_seconds": metrics.uptime_seconds,
                "total_cycles": metrics.total_cycles,
                "successful_cycles": metrics.successful_cycles,
                "failed_cycles": metrics.failed_cycles,
                "success_rate": (
                    metrics.successful_cycles / max(metrics.total_cycles, 1) * 100
                ),
                "last_error": metrics.last_error,
                "last_error_at": metrics.last_error_at.isoformat() if metrics.last_error_at else None,
            }
    
    def get_all_health(self) -> Dict[str, Any]:
        """Get health metrics for all monitored threads."""
        with self._lock:
            threads = {}
            for thread_id in self._metrics:
                threads[thread_id] = self.get_thread_health(thread_id)
            
            active_count = sum(1 for m in self._metrics.values() if m.state == ThreadState.RUNNING)
            stalled_count = sum(1 for m in self._metrics.values() if m.state == ThreadState.STALLED)
            error_count = sum(1 for m in self._metrics.values() if m.state == ThreadState.ERROR)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_threads": len(self._metrics),
                    "active_threads": active_count,
                    "stalled_threads": stalled_count,
                    "error_threads": error_count,
                    "overall_health": "healthy" if (stalled_count == 0 and error_count == 0) else "degraded",
                },
                "threads": threads,
                "recent_alerts": [self._alert_to_dict(a) for a in self._alerts[-10:]],
            }
    
    def check_for_stalls(self):
        """Check all threads for stall conditions."""
        now = datetime.utcnow()
        
        with self._lock:
            for thread_id, metrics in self._metrics.items():
                if metrics.state != ThreadState.RUNNING:
                    continue
                
                if metrics.last_heartbeat is None:
                    continue
                
                elapsed = (now - metrics.last_heartbeat).total_seconds()
                threshold = metrics.heartbeat_interval_seconds * self.HEARTBEAT_GRACE_PERIOD
                
                if elapsed > threshold and elapsed < self.STALL_THRESHOLD_SECONDS:
                    logger.warning(f"[Monitor] Thread {metrics.name} heartbeat delayed: {elapsed:.1f}s")
                
                elif elapsed >= self.STALL_THRESHOLD_SECONDS:
                    if metrics.state != ThreadState.STALLED:
                        metrics.state = ThreadState.STALLED
                        self._generate_alert(
                            AlertLevel.CRITICAL,
                            thread_id,
                            f"Thread {metrics.name} appears stalled (no heartbeat for {elapsed:.0f}s)",
                            {
                                "last_heartbeat": metrics.last_heartbeat.isoformat(),
                                "elapsed_seconds": elapsed,
                                "threshold_seconds": self.STALL_THRESHOLD_SECONDS,
                            },
                        )
    
    def start_monitoring(self, check_interval: int = 30):
        """Start the background monitoring loop."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("[Monitor] Already running")
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True,
            name="SchedulerMonitor",
        )
        self._monitor_thread.start()
        
        logger.info(f"[Monitor] Started with {check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("[Monitor] Stopped")
    
    def _monitoring_loop(self, interval: int):
        """Background loop that checks thread health."""
        while not self._stop_event.is_set():
            try:
                self.check_for_stalls()
                
                if self.scheduler:
                    self._check_scheduler_state()
                
            except Exception as e:
                logger.error(f"[Monitor] Error in monitoring loop: {e}")
            
            self._stop_event.wait(interval)
    
    def _check_scheduler_state(self):
        """Check scheduler-specific health conditions."""
        if not self.scheduler:
            return
        
        state = self.scheduler._state.value if hasattr(self.scheduler, '_state') else "unknown"
        
        if state == "error":
            self._generate_alert(
                AlertLevel.ERROR,
                "scheduler",
                "OvernightScheduler is in error state",
            )
    
    def _generate_alert(
        self,
        level: AlertLevel,
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Generate and store an alert."""
        self._alert_counter += 1
        
        alert = HealthAlert(
            alert_id=f"alert_{self._alert_counter}",
            level=level,
            component=component,
            message=message,
            details=details or {},
        )
        
        self._alerts.append(alert)
        
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]
        
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"[Monitor] Alert handler error: {e}")
        
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(level, logging.INFO)
        
        logger.log(log_level, f"[ALERT] {level.value.upper()}: {message}")
    
    def _alert_to_dict(self, alert: HealthAlert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": alert.alert_id,
            "level": alert.level.value,
            "component": alert.component,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "acknowledged": alert.acknowledged,
            "details": alert.details,
        }
    
    def register_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """Register a handler for alerts."""
        self._alert_handlers.append(handler)
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        component: Optional[str] = None,
        unacknowledged_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get filtered list of alerts."""
        filtered = self._alerts
        
        if level:
            filtered = [a for a in filtered if a.level == level]
        
        if component:
            filtered = [a for a in filtered if a.component == component]
        
        if unacknowledged_only:
            filtered = [a for a in filtered if not a.acknowledged]
        
        return [self._alert_to_dict(a) for a in filtered]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_recovery_suggestions(self, thread_id: str) -> List[str]:
        """Get recovery suggestions for a troubled thread."""
        metrics = self._metrics.get(thread_id)
        if not metrics:
            return []
        
        suggestions = []
        
        if metrics.state == ThreadState.STALLED:
            suggestions.append("Restart the scheduler using /hyperspeed/overnight/stop then /start")
            suggestions.append("Check server logs for blocking operations")
            suggestions.append("Verify API rate limits haven't been exceeded")
        
        if metrics.state == ThreadState.ERROR:
            suggestions.append("Check the last error message in thread health metrics")
            suggestions.append("Verify all required environment variables are set")
            suggestions.append("Review recent code changes that may have introduced bugs")
        
        if metrics.failed_cycles > 5:
            suggestions.append("Consider reducing batch sizes to prevent timeouts")
            suggestions.append("Check for memory leaks or resource exhaustion")
        
        return suggestions
