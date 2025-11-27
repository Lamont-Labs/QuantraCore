"""
LP10 - Drift State Label Protocol

Generates drift state classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP10: Generate drift state label.
    """
    state_map = {
        "none": 0,
        "mild": 1,
        "significant": 2,
        "critical": 3,
    }
    
    state_value = apex_result.drift_state.value
    label_value = state_map.get(state_value, 0)
    
    return LearningLabel(
        protocol_id="LP10",
        label_name="drift_state",
        value=label_value,
        confidence=0.85,
        metadata={
            "state_name": state_value,
            "drift_magnitude": apex_result.drift_metrics.drift_magnitude,
            "num_classes": 4,
        }
    )
