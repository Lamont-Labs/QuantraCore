"""
LP05 - Entropy State Label Protocol

Generates entropy state classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP05: Generate entropy state label.
    """
    state_map = {
        "stable": 0,
        "elevated": 1,
        "chaotic": 2,
    }
    
    state_value = apex_result.entropy_state.value
    label_value = state_map.get(state_value, 0)
    
    return LearningLabel(
        protocol_id="LP05",
        label_name="entropy_state",
        value=label_value,
        confidence=0.85,
        metadata={
            "state_name": state_value,
            "combined_entropy": apex_result.entropy_metrics.combined_entropy,
            "num_classes": 3,
        }
    )
