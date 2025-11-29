"""
HyperLearner Outcome Tracker.

Links events to their outcomes and calculates learning value.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import threading
import logging
import hashlib

from ..models import (
    LearningEvent,
    Outcome,
    EventOutcomePair,
    OutcomeType,
    LearningPriority,
    EventType,
    EventCategory,
)
from ..core.event_bus import get_event_bus, EventBus


logger = logging.getLogger(__name__)


class OutcomeTracker:
    """
    Tracks outcomes for all events and creates learning pairs.
    
    Key Features:
    - Pending event tracking with timeout
    - Automatic outcome correlation
    - Learning priority calculation
    - Feature extraction for ML training
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        outcome_timeout_hours: int = 72,
    ):
        self._event_bus = event_bus or get_event_bus()
        self._outcome_timeout = timedelta(hours=outcome_timeout_hours)
        
        self._pending_events: Dict[str, LearningEvent] = {}
        self._outcomes: Dict[str, Outcome] = {}
        self._learning_pairs: List[EventOutcomePair] = []
        self._lock = threading.Lock()
        
        self._total_outcomes = 0
        self._wins = 0
        self._losses = 0
        
        self._event_bus.subscribe(self._on_event)
        
    def _on_event(self, event: LearningEvent):
        """Handle incoming events."""
        if self._is_outcome_event(event.event_type):
            self._process_outcome_event(event)
        else:
            self._track_pending_event(event)
            
    def _is_outcome_event(self, event_type: EventType) -> bool:
        """Check if event type represents an outcome."""
        outcome_types = {
            EventType.TRADE_EXITED,
            EventType.STOP_TRIGGERED,
            EventType.TARGET_HIT,
            EventType.BATTLE_WON,
            EventType.BATTLE_LOST,
            EventType.BATTLE_TIE,
            EventType.PREDICTION_CORRECT,
            EventType.PREDICTION_WRONG,
            EventType.RUNNER_DETECTED,
            EventType.RUNNER_MISSED,
        }
        return event_type in outcome_types
        
    def _track_pending_event(self, event: LearningEvent):
        """Track event pending outcome resolution."""
        with self._lock:
            self._pending_events[event.event_id] = event
            
    def _process_outcome_event(self, event: LearningEvent):
        """Process an outcome event and link to original."""
        correlation_id = event.context.get("correlation_id")
        original_event_id = event.context.get("original_event_id")
        
        if original_event_id:
            with self._lock:
                original_event = self._pending_events.pop(original_event_id, None)
        elif correlation_id:
            related_ids = self._event_bus.get_correlated_events(correlation_id)
            original_event = None
            with self._lock:
                for eid in related_ids:
                    if eid in self._pending_events:
                        original_event = self._pending_events.pop(eid)
                        break
        else:
            original_event = None
            
        if original_event:
            outcome = self._create_outcome(event, original_event)
            self._create_learning_pair(original_event, outcome)
            
    def record_outcome(
        self,
        event_id: str,
        outcome_type: OutcomeType,
        return_pct: Optional[float] = None,
        return_dollars: Optional[float] = None,
        was_correct: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[EventOutcomePair]:
        """
        Manually record an outcome for an event.
        
        Use this for outcomes that don't flow through the event bus.
        """
        with self._lock:
            original_event = self._pending_events.pop(event_id, None)
            
        if not original_event:
            logger.warning(f"[OutcomeTracker] No pending event found: {event_id}")
            return None
            
        outcome_id = hashlib.sha256(
            f"{event_id}-outcome-{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        outcome = Outcome(
            outcome_id=outcome_id,
            event_id=event_id,
            timestamp=datetime.utcnow(),
            outcome_type=outcome_type,
            return_pct=return_pct,
            return_dollars=return_dollars,
            duration_seconds=int((datetime.utcnow() - original_event.timestamp).total_seconds()),
            was_correct=was_correct,
            details=details or {},
        )
        
        pair = self._create_learning_pair(original_event, outcome)
        return pair
        
    def _create_outcome(self, outcome_event: LearningEvent, original_event: LearningEvent) -> Outcome:
        """Create outcome from outcome event."""
        outcome_type = self._map_event_to_outcome_type(outcome_event.event_type)
        
        return_pct = outcome_event.context.get("return_pct")
        return_dollars = outcome_event.context.get("return_dollars")
        was_correct = outcome_event.context.get("was_correct")
        
        if was_correct is None:
            was_correct = outcome_type in {OutcomeType.WIN, OutcomeType.PASS}
            
        outcome_id = hashlib.sha256(
            f"{original_event.event_id}-{outcome_event.event_id}".encode()
        ).hexdigest()[:16]
        
        return Outcome(
            outcome_id=outcome_id,
            event_id=original_event.event_id,
            timestamp=outcome_event.timestamp,
            outcome_type=outcome_type,
            return_pct=return_pct,
            return_dollars=return_dollars,
            duration_seconds=int((outcome_event.timestamp - original_event.timestamp).total_seconds()),
            was_correct=was_correct,
            details=outcome_event.context,
        )
        
    def _map_event_to_outcome_type(self, event_type: EventType) -> OutcomeType:
        """Map event type to outcome type."""
        mapping = {
            EventType.TARGET_HIT: OutcomeType.WIN,
            EventType.BATTLE_WON: OutcomeType.WIN,
            EventType.PREDICTION_CORRECT: OutcomeType.WIN,
            EventType.RUNNER_DETECTED: OutcomeType.WIN,
            EventType.STOP_TRIGGERED: OutcomeType.LOSS,
            EventType.BATTLE_LOST: OutcomeType.LOSS,
            EventType.PREDICTION_WRONG: OutcomeType.LOSS,
            EventType.RUNNER_MISSED: OutcomeType.LOSS,
            EventType.TRADE_EXITED: OutcomeType.NEUTRAL,
            EventType.BATTLE_TIE: OutcomeType.NEUTRAL,
        }
        return mapping.get(event_type, OutcomeType.NEUTRAL)
        
    def _create_learning_pair(self, event: LearningEvent, outcome: Outcome) -> EventOutcomePair:
        """Create learning pair and extract features."""
        priority = self._calculate_priority(event, outcome)
        features = self._extract_features(event, outcome)
        lessons = self._extract_lessons(event, outcome)
        
        pair = EventOutcomePair(
            event=event,
            outcome=outcome,
            learning_priority=priority,
            features_extracted=features,
            lessons=lessons,
        )
        
        with self._lock:
            self._learning_pairs.append(pair)
            self._outcomes[outcome.outcome_id] = outcome
            self._total_outcomes += 1
            
            if outcome.outcome_type == OutcomeType.WIN:
                self._wins += 1
            elif outcome.outcome_type == OutcomeType.LOSS:
                self._losses += 1
                
        logger.debug(f"[OutcomeTracker] Created learning pair: {event.event_type.value} -> {outcome.outcome_type.value}")
        return pair
        
    def _calculate_priority(self, event: LearningEvent, outcome: Outcome) -> LearningPriority:
        """Calculate learning priority based on value and rarity."""
        priority_score = 3
        
        if outcome.outcome_type == OutcomeType.LOSS:
            priority_score += 1
            
        if outcome.return_pct is not None:
            if abs(outcome.return_pct) > 10:
                priority_score += 1
            if abs(outcome.return_pct) > 20:
                priority_score += 1
                
        if event.category in {EventCategory.OMEGA, EventCategory.RISK}:
            priority_score += 1
            
        if event.event_type in {EventType.RUNNER_MISSED, EventType.RUNNER_DETECTED}:
            priority_score += 1
            
        priority_score = min(priority_score, 5)
        
        return LearningPriority(priority_score)
        
    def _extract_features(self, event: LearningEvent, outcome: Outcome) -> Dict[str, float]:
        """Extract numerical features for ML training."""
        features = {}
        
        features["event_confidence"] = event.confidence
        features["category_code"] = hash(event.category.value) % 100 / 100.0
        features["event_type_code"] = hash(event.event_type.value) % 100 / 100.0
        
        ctx = event.context
        
        if "quantrascore" in ctx:
            features["quantrascore"] = float(ctx["quantrascore"])
        if "entropy" in ctx:
            features["entropy"] = float(ctx["entropy"])
        if "suppression" in ctx:
            features["suppression"] = float(ctx["suppression"])
        if "drift" in ctx:
            features["drift"] = float(ctx["drift"])
        if "volatility" in ctx:
            features["volatility"] = float(ctx["volatility"])
        if "volume_ratio" in ctx:
            features["volume_ratio"] = float(ctx["volume_ratio"])
        if "rsi" in ctx:
            features["rsi"] = float(ctx["rsi"])
        if "atr" in ctx:
            features["atr"] = float(ctx["atr"])
            
        if "protocols_fired" in ctx:
            features["protocols_fired_count"] = float(len(ctx["protocols_fired"]))
        if "omega_triggers" in ctx:
            features["omega_trigger_count"] = float(len(ctx["omega_triggers"]))
            
        if outcome.duration_seconds:
            features["duration_hours"] = outcome.duration_seconds / 3600.0
        if outcome.return_pct is not None:
            features["return_pct"] = outcome.return_pct
            
        features["is_win"] = 1.0 if outcome.outcome_type == OutcomeType.WIN else 0.0
        features["is_loss"] = 1.0 if outcome.outcome_type == OutcomeType.LOSS else 0.0
        
        return features
        
    def _extract_lessons(self, event: LearningEvent, outcome: Outcome) -> List[str]:
        """Extract human-readable lessons."""
        lessons = []
        
        if outcome.outcome_type == OutcomeType.LOSS:
            if outcome.return_pct and outcome.return_pct < -5:
                lessons.append(f"Significant loss of {outcome.return_pct:.1f}% - review entry criteria")
            if event.confidence < 0.6:
                lessons.append("Low confidence signal led to loss - tighten confidence threshold")
            if "quantrascore" in event.context and event.context["quantrascore"] < 70:
                lessons.append("Below-threshold QuantraScore preceded loss")
                
        if outcome.outcome_type == OutcomeType.WIN:
            if outcome.return_pct and outcome.return_pct > 5:
                lessons.append(f"Strong win of {outcome.return_pct:.1f}% - reinforce this pattern")
            if event.confidence > 0.8:
                lessons.append("High confidence signal validated - pattern is reliable")
                
        if event.event_type == EventType.RUNNER_MISSED:
            lessons.append("Missed runner opportunity - analyze entry timing")
        if event.event_type == EventType.OMEGA_TRIGGERED:
            lessons.append(f"Safety override triggered: {event.context.get('omega_type', 'unknown')}")
            
        return lessons
        
    def cleanup_expired(self) -> int:
        """Clean up expired pending events."""
        cutoff = datetime.utcnow() - self._outcome_timeout
        expired_count = 0
        
        with self._lock:
            expired_ids = [
                eid for eid, event in self._pending_events.items()
                if event.timestamp < cutoff
            ]
            for eid in expired_ids:
                event = self._pending_events.pop(eid)
                
                timeout_outcome = Outcome(
                    outcome_id=f"timeout-{eid[:8]}",
                    event_id=eid,
                    timestamp=datetime.utcnow(),
                    outcome_type=OutcomeType.NEUTRAL,
                    was_correct=None,
                    details={"reason": "timeout"},
                )
                self._create_learning_pair(event, timeout_outcome)
                expired_count += 1
                
        if expired_count > 0:
            logger.info(f"[OutcomeTracker] Cleaned up {expired_count} expired events")
            
        return expired_count
        
    def get_learning_pairs(
        self,
        limit: Optional[int] = None,
        min_priority: LearningPriority = LearningPriority.LOW,
    ) -> List[EventOutcomePair]:
        """Get learning pairs for training."""
        with self._lock:
            pairs = [p for p in self._learning_pairs if p.learning_priority.value >= min_priority.value]
            
        pairs.sort(key=lambda p: p.learning_priority.value, reverse=True)
        
        if limit:
            pairs = pairs[:limit]
            
        return pairs
        
    def get_stats(self) -> Dict[str, Any]:
        """Get outcome tracking statistics."""
        with self._lock:
            pending = len(self._pending_events)
            pairs = len(self._learning_pairs)
            
        win_rate = self._wins / max(self._total_outcomes, 1) * 100
        
        return {
            "pending_events": pending,
            "total_outcomes": self._total_outcomes,
            "learning_pairs": pairs,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate_pct": round(win_rate, 2),
        }
