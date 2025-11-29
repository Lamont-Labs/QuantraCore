"""
HyperLearner Data Models.

Defines all event types, outcomes, and learning artifacts for
the hyper-velocity learning system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json


class EventCategory(Enum):
    ANALYSIS = "analysis"
    SIGNAL = "signal"
    EXECUTION = "execution"
    OMEGA = "omega"
    BATTLE = "battle"
    PROTOCOL = "protocol"
    PREDICTION = "prediction"
    RISK = "risk"
    DATA = "data"
    SYSTEM = "system"


class EventType(Enum):
    SCAN_COMPLETED = "scan_completed"
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_PASSED = "signal_passed"
    SIGNAL_REJECTED = "signal_rejected"
    TRADE_ENTERED = "trade_entered"
    TRADE_EXITED = "trade_exited"
    STOP_TRIGGERED = "stop_triggered"
    TARGET_HIT = "target_hit"
    OMEGA_TRIGGERED = "omega_triggered"
    OMEGA_OVERRIDE = "omega_override"
    PROTOCOL_FIRED = "protocol_fired"
    PROTOCOL_SILENT = "protocol_silent"
    RUNNER_DETECTED = "runner_detected"
    RUNNER_MISSED = "runner_missed"
    BATTLE_WON = "battle_won"
    BATTLE_LOST = "battle_lost"
    BATTLE_TIE = "battle_tie"
    PREDICTION_CORRECT = "prediction_correct"
    PREDICTION_WRONG = "prediction_wrong"
    REGIME_CHANGE = "regime_change"
    VOLATILITY_SPIKE = "volatility_spike"
    DATA_ANOMALY = "data_anomaly"
    SYSTEM_ERROR = "system_error"
    MODEL_RETRAINED = "model_retrained"


class OutcomeType(Enum):
    WIN = "win"
    LOSS = "loss"
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NEUTRAL = "neutral"
    PENDING = "pending"


class LearningPriority(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class LearningEvent:
    """A single event captured for learning."""
    event_id: str
    timestamp: datetime
    category: EventCategory
    event_type: EventType
    source_component: str
    symbol: Optional[str]
    context: Dict[str, Any]
    decision_made: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "event_type": self.event_type.value,
            "source_component": self.source_component,
            "symbol": self.symbol,
            "context": self.context,
            "decision_made": self.decision_made,
            "confidence": self.confidence,
        }
    
    @classmethod
    def generate_id(cls, timestamp: datetime, event_type: EventType, context: Dict) -> str:
        content = f"{timestamp.isoformat()}-{event_type.value}-{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Outcome:
    """The result of an event/decision."""
    outcome_id: str
    event_id: str
    timestamp: datetime
    outcome_type: OutcomeType
    return_pct: Optional[float] = None
    return_dollars: Optional[float] = None
    duration_seconds: Optional[int] = None
    was_correct: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "outcome_type": self.outcome_type.value,
            "return_pct": self.return_pct,
            "return_dollars": self.return_dollars,
            "duration_seconds": self.duration_seconds,
            "was_correct": self.was_correct,
            "details": self.details,
        }


@dataclass
class EventOutcomePair:
    """Linked event and outcome for learning."""
    event: LearningEvent
    outcome: Outcome
    learning_priority: LearningPriority
    features_extracted: Dict[str, float] = field(default_factory=dict)
    lessons: List[str] = field(default_factory=list)
    
    def to_training_sample(self) -> Dict[str, Any]:
        return {
            "features": self.features_extracted,
            "label": self.outcome.outcome_type.value,
            "return_pct": self.outcome.return_pct or 0.0,
            "was_correct": self.outcome.was_correct,
            "priority": self.learning_priority.value,
        }


@dataclass
class Pattern:
    """A recognized pattern from historical events."""
    pattern_id: str
    pattern_type: str
    description: str
    feature_signature: Dict[str, Any]
    occurrence_count: int
    win_rate: float
    avg_return: float
    confidence: float
    discovered_at: datetime
    last_seen: datetime
    examples: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "feature_signature": self.feature_signature,
            "occurrence_count": self.occurrence_count,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "confidence": self.confidence,
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass
class LearningBatch:
    """A batch of samples ready for model training."""
    batch_id: str
    created_at: datetime
    samples: List[EventOutcomePair]
    total_samples: int
    priority_breakdown: Dict[str, int]
    categories_included: List[str]
    ready_for_training: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "total_samples": self.total_samples,
            "priority_breakdown": self.priority_breakdown,
            "categories_included": self.categories_included,
            "ready_for_training": self.ready_for_training,
        }


@dataclass
class LearningMetrics:
    """Metrics tracking the learning system's performance."""
    total_events_captured: int = 0
    total_outcomes_recorded: int = 0
    total_patterns_discovered: int = 0
    total_retraining_cycles: int = 0
    events_per_hour: float = 0.0
    outcomes_per_hour: float = 0.0
    avg_event_to_outcome_delay: float = 0.0
    model_improvement_rate: float = 0.0
    win_rate_trend: List[float] = field(default_factory=list)
    accuracy_trend: List[float] = field(default_factory=list)
    last_retrain_at: Optional[datetime] = None
    last_pattern_discovered_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events_captured": self.total_events_captured,
            "total_outcomes_recorded": self.total_outcomes_recorded,
            "total_patterns_discovered": self.total_patterns_discovered,
            "total_retraining_cycles": self.total_retraining_cycles,
            "events_per_hour": round(self.events_per_hour, 2),
            "outcomes_per_hour": round(self.outcomes_per_hour, 2),
            "avg_event_to_outcome_delay": round(self.avg_event_to_outcome_delay, 2),
            "model_improvement_rate": round(self.model_improvement_rate, 4),
            "win_rate_trend": self.win_rate_trend[-20:],
            "accuracy_trend": self.accuracy_trend[-20:],
            "last_retrain_at": self.last_retrain_at.isoformat() if self.last_retrain_at else None,
            "last_pattern_discovered_at": self.last_pattern_discovered_at.isoformat() if self.last_pattern_discovered_at else None,
        }


@dataclass
class MetaLearningInsight:
    """Insight about how to improve the learning process itself."""
    insight_id: str
    insight_type: str
    description: str
    recommendation: str
    expected_improvement: float
    confidence: float
    discovered_at: datetime
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "description": self.description,
            "recommendation": self.recommendation,
            "expected_improvement": round(self.expected_improvement, 4),
            "confidence": round(self.confidence, 3),
            "discovered_at": self.discovered_at.isoformat(),
            "applied": self.applied,
        }
