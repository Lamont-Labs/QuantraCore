"""
LP16 - Pattern Quality Score Label Protocol

Rates the quality/reliability of detected chart patterns.
Category: Advanced Labels - Pattern Quality
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP16: Generate pattern quality score label.
    
    Rates pattern quality based on:
    - Symmetry and proportions
    - Volume characteristics
    - Price action cleanliness
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP16",
            label_name="pattern_quality",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    
    bodies = np.abs(closes - np.array([b.open for b in bars[-30:]]))
    ranges = highs - lows
    body_ratio = np.mean(bodies / (ranges + 1e-10))
    
    returns = np.diff(closes) / closes[:-1]
    noise = np.std(returns) / (np.abs(np.mean(returns)) + 1e-10)
    
    vol_consistency = 1 - (np.std(volumes) / (np.mean(volumes) + 1e-10))
    vol_consistency = np.clip(vol_consistency, 0, 1)
    
    swing_highs = []
    swing_lows = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    structure_clarity = min(len(swing_highs), len(swing_lows)) / 3
    structure_clarity = np.clip(structure_clarity, 0, 1)
    
    quality_score = (
        body_ratio * 0.25 +
        (1 / (1 + noise)) * 0.25 +
        vol_consistency * 0.25 +
        structure_clarity * 0.25
    )
    
    if quality_score > 0.7:
        quality = 0
        quality_name = "high_quality"
    elif quality_score > 0.4:
        quality = 1
        quality_name = "medium_quality"
    else:
        quality = 2
        quality_name = "low_quality"
    
    confidence = min(0.9, 0.5 + quality_score * 0.4)
    
    return LearningLabel(
        protocol_id="LP16",
        label_name="pattern_quality",
        value=quality,
        confidence=confidence,
        metadata={
            "quality_name": quality_name,
            "quality_score": float(quality_score),
            "body_ratio": float(body_ratio),
            "noise_level": float(noise),
            "vol_consistency": float(vol_consistency),
            "structure_clarity": float(structure_clarity),
            "num_classes": 3,
        }
    )
