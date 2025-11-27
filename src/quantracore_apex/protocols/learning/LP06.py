"""
LP06 - Continuation Result Label Protocol

Generates continuation outcome labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP06: Generate continuation result label.
    """
    prob = apex_result.continuation_metrics.continuation_probability
    exhaustion = apex_result.continuation_metrics.exhaustion_signal
    
    if exhaustion:
        label_value = 0
        label_name = "exhaustion"
    elif prob > 0.7:
        label_value = 2
        label_name = "likely_continue"
    elif prob > 0.4:
        label_value = 1
        label_name = "neutral"
    else:
        label_value = 0
        label_name = "likely_reverse"
    
    return LearningLabel(
        protocol_id="LP06",
        label_name="continuation_result",
        value=label_value,
        confidence=prob,
        metadata={
            "result_name": label_name,
            "continuation_probability": prob,
            "exhaustion_signal": exhaustion,
            "num_classes": 3,
        }
    )
