"""
Instability Predictor Engine

Predicts early instability signals using deterministic heuristics.
Part of the QuantraCore Apex prediction stack.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ..core.schemas import OhlcvBar


@dataclass
class InstabilityPrediction:
    """Instability prediction result."""
    instability_score: float = 0.0
    instability_type: str = "stable"
    warning_level: str = "none"
    volatility_instability: float = 0.0
    structure_instability: float = 0.0
    momentum_instability: float = 0.0
    time_to_event: int = 0
    confidence: float = 0.0
    compliance_note: str = "Prediction is structural probability, not trading advice"


def predict_instability(
    bars: List[OhlcvBar],
    lookback: int = 20,
) -> InstabilityPrediction:
    """
    Predict early instability signals.
    
    Args:
        bars: OHLCV price bars
        lookback: Analysis period
        
    Returns:
        InstabilityPrediction with instability metrics
        
    Method:
    - Analyzes volatility patterns for instability
    - Detects structural weakness
    - Measures momentum divergence
    """
    if len(bars) < lookback + 10:
        return InstabilityPrediction()
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    np.array([b.volume for b in bars], dtype=float)
    
    returns = np.diff(np.log(closes + 1e-10))
    recent_vol = float(np.std(returns[-10:]))
    prior_vol = float(np.std(returns[-20:-10])) if len(returns) >= 20 else recent_vol
    
    vol_change = abs(recent_vol - prior_vol) / max(prior_vol, 1e-10)
    volatility_instability = min(vol_change, 1.0)
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
    lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
    
    structure_confusion = abs(higher_highs - lower_lows) / (lookback - 1)
    structure_instability = 1.0 - structure_confusion
    
    short_momentum = closes[-1] - closes[-5] if len(closes) >= 5 else 0
    long_momentum = closes[-1] - closes[-lookback]
    
    if (short_momentum > 0 and long_momentum < 0) or (short_momentum < 0 and long_momentum > 0):
        momentum_instability = 0.8
    elif abs(short_momentum) < abs(long_momentum) * 0.3:
        momentum_instability = 0.5
    else:
        momentum_instability = 0.2
    
    instability_score = (
        volatility_instability * 0.35 +
        structure_instability * 0.35 +
        momentum_instability * 0.30
    )
    instability_score = float(np.clip(instability_score, 0.0, 1.0))
    
    if instability_score < 0.3:
        instability_type = "stable"
        warning_level = "none"
    elif instability_score < 0.5:
        instability_type = "mild"
        warning_level = "low"
    elif instability_score < 0.7:
        instability_type = "moderate"
        warning_level = "medium"
    else:
        instability_type = "severe"
        warning_level = "high"
    
    time_to_event = max(1, int(5 * (1 - instability_score)))
    
    confidence = min(instability_score * 80 + 20, 75.0)
    
    return InstabilityPrediction(
        instability_score=round(instability_score, 4),
        instability_type=instability_type,
        warning_level=warning_level,
        volatility_instability=round(volatility_instability, 4),
        structure_instability=round(structure_instability, 4),
        momentum_instability=round(momentum_instability, 4),
        time_to_event=time_to_event,
        confidence=round(confidence, 2),
    )
