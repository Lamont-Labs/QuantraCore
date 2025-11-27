"""
LP01 - Regime Label Protocol

Generates regime classification labels for training.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP01: Generate regime classification label.
    
    Maps regime to numeric class for model training.
    """
    regime_map = {
        "trending_up": 0,
        "trending_down": 1,
        "range_bound": 2,
        "volatile": 3,
        "compressed": 4,
        "unknown": 5,
    }
    
    regime_value = apex_result.regime.value
    label_value = regime_map.get(regime_value, 5)
    
    return LearningLabel(
        protocol_id="LP01",
        label_name="regime_class",
        value=label_value,
        confidence=apex_result.verdict.confidence,
        metadata={
            "regime_name": regime_value,
            "num_classes": 6,
        }
    )
