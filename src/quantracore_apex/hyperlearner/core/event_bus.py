"""
HyperLearner Event Bus.

Central nervous system capturing ALL events across the platform
for hyper-velocity learning.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import threading
import queue
import logging

from ..models import (
    LearningEvent,
    EventCategory,
    EventType,
    LearningPriority,
)


logger = logging.getLogger(__name__)


class EventBus:
    """
    Universal event bus capturing every action and decision.
    
    Features:
    - Async event processing for zero latency impact
    - Priority queuing for critical events
    - Multi-subscriber support for parallel processing
    - Event correlation for linking related events
    """
    
    def __init__(self, max_queue_size: int = 100000):
        self._event_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._subscribers: Dict[EventCategory, List[Callable]] = defaultdict(list)
        self._global_subscribers: List[Callable] = []
        self._event_history: List[LearningEvent] = []
        self._max_history: int = 50000
        self._lock = threading.Lock()
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        self._events_processed = 0
        self._events_dropped = 0
        
        self._correlation_map: Dict[str, List[str]] = defaultdict(list)
        
    def start(self):
        """Start the event processing thread."""
        if self._running:
            return
        self._running = True
        self._processor_thread = threading.Thread(target=self._process_events, daemon=True)
        self._processor_thread.start()
        logger.info("[HyperLearner] Event bus started")
        
    def stop(self):
        """Stop the event processing thread."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        logger.info("[HyperLearner] Event bus stopped")
        
    def emit(
        self,
        category: EventCategory,
        event_type: EventType,
        source_component: str,
        context: Dict[str, Any],
        symbol: Optional[str] = None,
        decision_made: Optional[str] = None,
        confidence: float = 0.0,
        priority: LearningPriority = LearningPriority.MEDIUM,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Emit an event to be captured for learning.
        
        Returns the event_id for correlation.
        """
        timestamp = datetime.utcnow()
        event_id = LearningEvent.generate_id(timestamp, event_type, context)
        
        event = LearningEvent(
            event_id=event_id,
            timestamp=timestamp,
            category=category,
            event_type=event_type,
            source_component=source_component,
            symbol=symbol,
            context=context,
            decision_made=decision_made,
            confidence=confidence,
        )
        
        if correlation_id:
            with self._lock:
                self._correlation_map[correlation_id].append(event_id)
        
        try:
            priority_value = -priority.value
            self._event_queue.put_nowait((priority_value, timestamp, event))
        except queue.Full:
            self._events_dropped += 1
            logger.warning(f"[HyperLearner] Event queue full, dropped event: {event_type.value}")
            return event_id
            
        return event_id
    
    def subscribe(
        self,
        callback: Callable[[LearningEvent], None],
        category: Optional[EventCategory] = None,
    ):
        """Subscribe to events (optionally filtered by category)."""
        if category:
            self._subscribers[category].append(callback)
        else:
            self._global_subscribers.append(callback)
            
    def get_correlated_events(self, correlation_id: str) -> List[str]:
        """Get all event IDs linked by a correlation ID."""
        with self._lock:
            return self._correlation_map.get(correlation_id, [])
            
    def get_recent_events(
        self,
        limit: int = 100,
        category: Optional[EventCategory] = None,
        event_type: Optional[EventType] = None,
    ) -> List[LearningEvent]:
        """Get recent events with optional filtering."""
        with self._lock:
            events = list(self._event_history)
            
        if category:
            events = [e for e in events if e.category == category]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
            
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        with self._lock:
            history_count = len(self._event_history)
            
        return {
            "running": self._running,
            "queue_size": self._event_queue.qsize(),
            "events_processed": self._events_processed,
            "events_dropped": self._events_dropped,
            "history_size": history_count,
            "subscriber_count": sum(len(s) for s in self._subscribers.values()) + len(self._global_subscribers),
            "correlation_chains": len(self._correlation_map),
        }
        
    def _process_events(self):
        """Background thread processing events."""
        while self._running:
            try:
                priority, timestamp, event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            with self._lock:
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history = self._event_history[-self._max_history:]
                    
            for callback in self._global_subscribers:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"[HyperLearner] Subscriber error: {e}")
                    
            for callback in self._subscribers.get(event.category, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"[HyperLearner] Category subscriber error: {e}")
                    
            self._events_processed += 1


_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
        _global_event_bus.start()
    return _global_event_bus


def emit_event(
    category: EventCategory,
    event_type: EventType,
    source: str,
    context: Dict[str, Any],
    **kwargs,
) -> str:
    """Convenience function to emit events to the global bus."""
    bus = get_event_bus()
    return bus.emit(category, event_type, source, context, **kwargs)
