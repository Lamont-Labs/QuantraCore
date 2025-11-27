"""
LP07 - Score Bucket Label Protocol

Generates QuantraScore bucket classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP07: Generate score bucket label.
    """
    bucket_map = {
        "very_low": 0,
        "low": 1,
        "neutral": 2,
        "high": 3,
        "very_high": 4,
    }
    
    bucket_value = apex_result.score_bucket.value
    label_value = bucket_map.get(bucket_value, 2)
    
    return LearningLabel(
        protocol_id="LP07",
        label_name="score_bucket",
        value=label_value,
        confidence=apex_result.verdict.confidence,
        metadata={
            "bucket_name": bucket_value,
            "quantrascore": apex_result.quantrascore,
            "num_classes": 5,
        }
    )
