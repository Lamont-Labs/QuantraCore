"""
LP11 - Future Price Direction Label Protocol

Generates forward-looking price direction labels for supervised training.
Category: Advanced Labels - Forward Looking
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP11: Generate future price direction label.
    
    Uses the last N bars to compute expected future direction based on:
    - Momentum continuation probability
    - Mean reversion signals
    - Volume confirmation
    """
    bars = window.bars
    if len(bars) < 20:
        return LearningLabel(
            protocol_id="LP11",
            label_name="future_direction",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-20:]])
    volumes = np.array([b.volume for b in bars[-20:]])
    
    returns = np.diff(closes) / closes[:-1]
    momentum = np.sum(returns[-5:])
    
    vol_trend = volumes[-5:].mean() / (volumes[:-5].mean() + 1e-10)
    
    price_vs_sma = closes[-1] / (np.mean(closes) + 1e-10)
    
    score = 0.0
    score += np.clip(momentum * 10, -1, 1) * 0.4
    score += np.clip((vol_trend - 1) * 2, -1, 1) * 0.3
    score += np.clip((price_vs_sma - 1) * 5, -1, 1) * 0.3
    
    if score > 0.2:
        direction = 0
        direction_name = "bullish"
    elif score < -0.2:
        direction = 1
        direction_name = "bearish"
    else:
        direction = 2
        direction_name = "neutral"
    
    confidence = min(0.95, 0.5 + abs(score) * 0.5)
    
    return LearningLabel(
        protocol_id="LP11",
        label_name="future_direction",
        value=direction,
        confidence=confidence,
        metadata={
            "direction_name": direction_name,
            "momentum_score": float(momentum),
            "volume_trend": float(vol_trend),
            "price_vs_sma": float(price_vs_sma),
            "composite_score": float(score),
            "num_classes": 3,
        }
    )
