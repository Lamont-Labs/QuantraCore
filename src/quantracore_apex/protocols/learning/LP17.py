"""
LP17 - Support/Resistance Proximity Label Protocol

Measures proximity to key support/resistance levels.
Category: Advanced Labels - Price Levels
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP17: Generate support/resistance proximity label.
    
    Classifies position relative to S/R:
    - Near resistance
    - Near support
    - Mid-range
    - At boundary (very close)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP17",
            label_name="sr_proximity",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    
    recent_high = np.max(highs[-20:])
    recent_low = np.min(lows[-20:])
    price_range = recent_high - recent_low
    
    current_price = closes[-1]
    
    dist_to_resistance = (recent_high - current_price) / (price_range + 1e-10)
    dist_to_support = (current_price - recent_low) / (price_range + 1e-10)
    
    position_in_range = (current_price - recent_low) / (price_range + 1e-10)
    
    if dist_to_resistance < 0.1:
        proximity = 0
        proximity_name = "at_resistance"
        confidence_mult = 0.9
    elif dist_to_resistance < 0.25:
        proximity = 1
        proximity_name = "near_resistance"
        confidence_mult = 0.75
    elif dist_to_support < 0.1:
        proximity = 3
        proximity_name = "at_support"
        confidence_mult = 0.9
    elif dist_to_support < 0.25:
        proximity = 4
        proximity_name = "near_support"
        confidence_mult = 0.75
    else:
        proximity = 2
        proximity_name = "mid_range"
        confidence_mult = 0.6
    
    range_atr = np.mean(highs - lows)
    range_quality = price_range / (range_atr * 5 + 1e-10)
    range_quality = np.clip(range_quality, 0.3, 1.0)
    
    confidence = min(0.9, confidence_mult * range_quality)
    
    return LearningLabel(
        protocol_id="LP17",
        label_name="sr_proximity",
        value=proximity,
        confidence=confidence,
        metadata={
            "proximity_name": proximity_name,
            "dist_to_resistance": float(dist_to_resistance),
            "dist_to_support": float(dist_to_support),
            "position_in_range": float(position_in_range),
            "recent_high": float(recent_high),
            "recent_low": float(recent_low),
            "num_classes": 5,
        }
    )
