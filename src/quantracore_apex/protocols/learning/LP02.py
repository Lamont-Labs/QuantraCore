"""
LP02 - Volatility Band Label Protocol

Generates volatility band classification labels.
Category: Core Labels
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP02: Generate volatility band label.
    
    Classifies volatility into bands for model training.
    """
    vol_ratio = apex_result.microtraits.volatility_ratio
    
    if vol_ratio < 0.5:
        band = 0
        band_name = "very_low"
    elif vol_ratio < 0.8:
        band = 1
        band_name = "low"
    elif vol_ratio < 1.2:
        band = 2
        band_name = "normal"
    elif vol_ratio < 1.5:
        band = 3
        band_name = "elevated"
    else:
        band = 4
        band_name = "high"
    
    return LearningLabel(
        protocol_id="LP02",
        label_name="volatility_band",
        value=band,
        confidence=0.85,
        metadata={
            "band_name": band_name,
            "volatility_ratio": vol_ratio,
            "num_classes": 5,
        }
    )
