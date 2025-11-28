"""
LP21 - Institutional Activity Label Protocol

Detects signs of institutional participation.
Category: Advanced Labels - Flow Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP21: Generate institutional activity label.
    
    Classifies institutional participation level:
    - High institutional (large blocks, consistent direction)
    - Moderate institutional
    - Retail dominated (noise, inconsistent)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP21",
            label_name="institutional_activity",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    
    avg_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    large_volume_bars = np.sum(volumes > avg_volume + std_volume)
    large_vol_ratio = large_volume_bars / len(volumes)
    
    large_vol_indices = np.where(volumes > avg_volume + std_volume)[0]
    if len(large_vol_indices) > 1:
        returns_on_large = []
        for idx in large_vol_indices:
            if idx > 0:
                ret = (closes[idx] - closes[idx-1]) / closes[idx-1]
                returns_on_large.append(ret)
        if returns_on_large:
            direction_consistency = abs(np.mean(np.sign(returns_on_large)))
        else:
            direction_consistency = 0
    else:
        direction_consistency = 0
    
    ranges = highs - lows
    avg_range = np.mean(ranges)
    range_consistency = 1 - (np.std(ranges) / (avg_range + 1e-10))
    range_consistency = np.clip(range_consistency, 0, 1)
    
    returns = np.diff(closes) / closes[:-1]
    trend_clarity = abs(np.sum(np.sign(returns))) / len(returns)
    
    institutional_score = (
        large_vol_ratio * 0.3 +
        direction_consistency * 0.3 +
        range_consistency * 0.2 +
        trend_clarity * 0.2
    )
    
    if institutional_score > 0.6:
        activity = 0
        activity_name = "high_institutional"
    elif institutional_score > 0.35:
        activity = 1
        activity_name = "moderate_institutional"
    else:
        activity = 2
        activity_name = "retail_dominated"
    
    confidence = min(0.85, 0.4 + institutional_score * 0.5)
    
    return LearningLabel(
        protocol_id="LP21",
        label_name="institutional_activity",
        value=activity,
        confidence=confidence,
        metadata={
            "activity_name": activity_name,
            "institutional_score": float(institutional_score),
            "large_vol_ratio": float(large_vol_ratio),
            "direction_consistency": float(direction_consistency),
            "range_consistency": float(range_consistency),
            "trend_clarity": float(trend_clarity),
            "num_classes": 3,
        }
    )
