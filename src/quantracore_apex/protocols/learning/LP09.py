"""
LP09 - QuantraScore Numeric Label Protocol

Generates QuantraScore as regression target.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP09: Generate QuantraScore numeric label.
    
    Returns QuantraScore as regression target (0-100).
    """
    return LearningLabel(
        protocol_id="LP09",
        label_name="quantrascore_numeric",
        value=apex_result.quantrascore,
        confidence=apex_result.verdict.confidence,
        metadata={
            "score_range": "0-100",
            "task_type": "regression",
        }
    )
