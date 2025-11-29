"""
QuantraCore Apex - HyperLearner.

Hyper-Velocity Learning System that captures EVERYTHING
the system does and learns from it at an accelerated rate.

LEARNING PHILOSOPHY:
- Every action is a lesson
- Every outcome improves the system
- Wins reinforce patterns
- Losses identify weaknesses
- Passes validate quality control
- Fails fix broken processes

COMPONENTS:
- EventBus: Universal event capture
- OutcomeTracker: Links events to results
- PatternMiner: Discovers win/loss patterns
- ContinuousTrainer: Prioritized retraining
- MetaLearner: Optimizes learning itself

INTEGRATION:
Use decorators and hooks to automatically capture
learning opportunities from any component.
"""

from .models import (
    EventCategory,
    EventType,
    OutcomeType,
    LearningPriority,
    LearningEvent,
    Outcome,
    EventOutcomePair,
    Pattern,
    LearningBatch,
    LearningMetrics,
    MetaLearningInsight,
)

from .core.event_bus import EventBus, get_event_bus, emit_event
from .core.hyperlearner import HyperLearner, get_hyperlearner
from .capture.outcome_tracker import OutcomeTracker
from .patterns.pattern_miner import PatternMiner
from .retraining.continuous_trainer import ContinuousTrainer
from .meta.meta_learner import MetaLearner

from .integration.hooks import (
    learn_from_scan,
    learn_from_signal,
    learn_from_execution,
    learn_from_omega,
    LearningContext,
    emit_protocol_result,
    emit_runner_detection,
    emit_regime_change,
    emit_data_anomaly,
)


__all__ = [
    "EventCategory",
    "EventType",
    "OutcomeType",
    "LearningPriority",
    "LearningEvent",
    "Outcome",
    "EventOutcomePair",
    "Pattern",
    "LearningBatch",
    "LearningMetrics",
    "MetaLearningInsight",
    "EventBus",
    "get_event_bus",
    "emit_event",
    "HyperLearner",
    "get_hyperlearner",
    "OutcomeTracker",
    "PatternMiner",
    "ContinuousTrainer",
    "MetaLearner",
    "learn_from_scan",
    "learn_from_signal",
    "learn_from_execution",
    "learn_from_omega",
    "LearningContext",
    "emit_protocol_result",
    "emit_runner_detection",
    "emit_regime_change",
    "emit_data_anomaly",
]
