"""
LP04 - Suppression State Label Protocol

Generates suppression state classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP04: Generate suppression state label.
    """
    state_map = {
        "none": 0,
        "light": 1,
        "moderate": 2,
        "heavy": 3,
    }
    
    state_value = apex_result.suppression_state.value
    label_value = state_map.get(state_value, 0)
    
    return LearningLabel(
        protocol_id="LP04",
        label_name="suppression_state",
        value=label_value,
        confidence=0.9,
        metadata={
            "state_name": state_value,
            "suppression_level": apex_result.suppression_metrics.suppression_level,
            "num_classes": 4,
        }
    )
