"""
Continuation Estimator Engine

Estimates trend continuation probability using deterministic heuristics.
Part of the QuantraCore Apex prediction stack.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ..core.schemas import OhlcvBar


@dataclass
class ContinuationEstimate:
    """Continuation estimate result."""
    continuation_probability: float = 0.0
    reversal_probability: float = 0.0
    trend_strength: float = 0.0
    trend_age: int = 0
    exhaustion_level: float = 0.0
    momentum_status: str = "neutral"
    confidence: float = 0.0
    compliance_note: str = "Estimate is structural probability, not trading advice"


def estimate_continuation(
    bars: List[OhlcvBar],
    lookback: int = 20,
) -> ContinuationEstimate:
    """
    Estimate trend continuation probability.
    
    Args:
        bars: OHLCV price bars
        lookback: Analysis period
        
    Returns:
        ContinuationEstimate with continuation/reversal probabilities
        
    Method:
    - Measures trend strength and age
    - Analyzes momentum health
    - Detects exhaustion patterns
    """
    if len(bars) < lookback + 5:
        return ContinuationEstimate()
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    trend_direction = closes[-1] - closes[-lookback]
    price_range = float(np.max(closes[-lookback:]) - np.min(closes[-lookback:]))
    trend_strength = abs(trend_direction) / max(price_range, 1e-10)
    trend_strength = min(trend_strength, 1.0)
    
    trend_age = 0
    if trend_direction > 0:
        for i in range(len(closes) - 2, -1, -1):
            if closes[i] < closes[i + 1]:
                trend_age += 1
            else:
                break
    else:
        for i in range(len(closes) - 2, -1, -1):
            if closes[i] > closes[i + 1]:
                trend_age += 1
            else:
                break
    
    short_momentum = (closes[-1] - closes[-5]) / max(closes[-5], 1e-10)
    long_momentum = (closes[-1] - closes[-lookback]) / max(closes[-lookback], 1e-10)
    
    if abs(short_momentum) > 0.001 and abs(long_momentum) > 0.001:
        if (short_momentum > 0) == (long_momentum > 0):
            if abs(short_momentum) < abs(long_momentum) * 0.5:
                momentum_status = "weakening"
            else:
                momentum_status = "healthy"
        else:
            momentum_status = "diverging"
    else:
        momentum_status = "neutral"
    
    exhaustion_level = 0.0
    
    if trend_age > 10:
        exhaustion_level += 0.2
    if momentum_status == "weakening":
        exhaustion_level += 0.3
    if momentum_status == "diverging":
        exhaustion_level += 0.4
    
    recent_range = float(np.mean(highs[-5:] - lows[-5:]))
    prior_range = float(np.mean(highs[-10:-5] - lows[-10:-5])) if len(highs) >= 10 else recent_range
    if recent_range > prior_range * 1.5:
        exhaustion_level += 0.2
    
    exhaustion_level = min(exhaustion_level, 1.0)
    
    base_continuation = 0.5
    
    continuation_probability = base_continuation + (trend_strength * 0.3) - (exhaustion_level * 0.4)
    
    if momentum_status == "healthy":
        continuation_probability += 0.1
    elif momentum_status == "weakening":
        continuation_probability -= 0.1
    elif momentum_status == "diverging":
        continuation_probability -= 0.2
    
    continuation_probability = float(np.clip(continuation_probability, 0.1, 0.9))
    reversal_probability = 1.0 - continuation_probability
    
    confidence = min(abs(continuation_probability - 0.5) * 150 + 30, 75.0)
    
    return ContinuationEstimate(
        continuation_probability=round(continuation_probability, 4),
        reversal_probability=round(reversal_probability, 4),
        trend_strength=round(trend_strength, 4),
        trend_age=trend_age,
        exhaustion_level=round(exhaustion_level, 4),
        momentum_status=momentum_status,
        confidence=round(confidence, 2),
    )
