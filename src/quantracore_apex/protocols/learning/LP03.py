"""
LP03 - Risk Tier Label Protocol

Generates risk tier classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP03: Generate risk tier label.
    
    Maps risk tier to numeric class for training.
    """
    risk_map = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "extreme": 3,
    }
    
    risk_value = apex_result.risk_tier.value
    label_value = risk_map.get(risk_value, 1)
    
    return LearningLabel(
        protocol_id="LP03",
        label_name="risk_tier",
        value=label_value,
        confidence=apex_result.verdict.confidence,
        metadata={
            "risk_name": risk_value,
            "num_classes": 4,
        }
    )
