"""HyperLearner Integration Hooks."""

from .hooks import (
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
