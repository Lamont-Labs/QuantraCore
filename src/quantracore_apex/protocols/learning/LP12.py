"""
LP12 - Future Volatility Regime Label Protocol

Predicts upcoming volatility regime for training forward-looking models.
Category: Advanced Labels - Forward Looking
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP12: Generate future volatility regime label.
    
    Predicts volatility expansion/contraction based on:
    - Current compression state
    - Historical volatility patterns
    - Volume dynamics
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP12",
            label_name="future_volatility",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    atr_values = highs - lows
    recent_atr = np.mean(atr_values[-5:])
    historical_atr = np.mean(atr_values[-30:-5])
    
    compression = apex_result.microtraits.compression_score
    
    returns = np.abs(np.diff(closes) / closes[:-1])
    vol_trend = np.polyfit(range(len(returns[-10:])), returns[-10:], 1)[0]
    
    expansion_score = 0.0
    
    if compression > 0.7:
        expansion_score += 0.4
    elif compression < 0.3:
        expansion_score -= 0.3
    
    atr_ratio = recent_atr / (historical_atr + 1e-10)
    if atr_ratio < 0.7:
        expansion_score += 0.3
    elif atr_ratio > 1.3:
        expansion_score -= 0.2
    
    expansion_score += np.clip(vol_trend * 100, -0.3, 0.3)
    
    if expansion_score > 0.3:
        regime = 0
        regime_name = "expansion_likely"
    elif expansion_score < -0.2:
        regime = 2
        regime_name = "contraction_likely"
    else:
        regime = 1
        regime_name = "stable"
    
    confidence = min(0.9, 0.5 + abs(expansion_score))
    
    return LearningLabel(
        protocol_id="LP12",
        label_name="future_volatility",
        value=regime,
        confidence=confidence,
        metadata={
            "regime_name": regime_name,
            "compression_score": float(compression),
            "atr_ratio": float(atr_ratio),
            "vol_trend": float(vol_trend),
            "expansion_score": float(expansion_score),
            "num_classes": 3,
        }
    )
