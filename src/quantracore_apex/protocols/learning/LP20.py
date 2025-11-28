"""
LP20 - Breakout Authenticity Label Protocol

Evaluates authenticity/reliability of breakout moves.
Category: Advanced Labels - Breakout Quality
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP20: Generate breakout authenticity label.
    
    Classifies breakout quality:
    - Authentic breakout (likely to continue)
    - Suspicious breakout (may fail)
    - False breakout (likely reversal)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP20",
            label_name="breakout_authenticity",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    
    prior_high = np.max(highs[-25:-5])
    prior_low = np.min(lows[-25:-5])
    prior_range = prior_high - prior_low
    
    is_bullish_breakout = closes[-1] > prior_high
    is_bearish_breakout = closes[-1] < prior_low
    
    if not is_bullish_breakout and not is_bearish_breakout:
        return LearningLabel(
            protocol_id="LP20",
            label_name="breakout_authenticity",
            value=1,
            confidence=0.3,
            metadata={
                "authenticity_name": "no_breakout",
                "prior_high": float(prior_high),
                "prior_low": float(prior_low),
                "num_classes": 3,
            }
        )
    
    breakout_volume = volumes[-3:].mean()
    prior_volume = volumes[-20:-5].mean()
    volume_ratio = breakout_volume / (prior_volume + 1e-10)
    
    if is_bullish_breakout:
        extension = (closes[-1] - prior_high) / (prior_range + 1e-10)
    else:
        extension = (prior_low - closes[-1]) / (prior_range + 1e-10)
    
    close_strength = 0.0
    for i in range(-3, 0):
        bar_range = highs[i] - lows[i]
        if is_bullish_breakout:
            close_strength += (closes[i] - lows[i]) / (bar_range + 1e-10)
        else:
            close_strength += (highs[i] - closes[i]) / (bar_range + 1e-10)
    close_strength /= 3
    
    authenticity_score = 0.0
    
    if volume_ratio > 1.5:
        authenticity_score += 0.35
    elif volume_ratio > 1.2:
        authenticity_score += 0.2
    elif volume_ratio < 0.8:
        authenticity_score -= 0.2
    
    if extension > 0.3:
        authenticity_score += 0.25
    elif extension > 0.15:
        authenticity_score += 0.15
    elif extension < 0.05:
        authenticity_score -= 0.15
    
    if close_strength > 0.7:
        authenticity_score += 0.2
    elif close_strength < 0.3:
        authenticity_score -= 0.2
    
    if authenticity_score > 0.4:
        authenticity = 0
        auth_name = "authentic"
    elif authenticity_score > 0:
        authenticity = 1
        auth_name = "suspicious"
    else:
        authenticity = 2
        auth_name = "likely_false"
    
    confidence = min(0.85, 0.4 + abs(authenticity_score) * 0.5)
    
    return LearningLabel(
        protocol_id="LP20",
        label_name="breakout_authenticity",
        value=authenticity,
        confidence=confidence,
        metadata={
            "authenticity_name": auth_name,
            "authenticity_score": float(authenticity_score),
            "volume_ratio": float(volume_ratio),
            "extension": float(extension),
            "close_strength": float(close_strength),
            "breakout_direction": "bullish" if is_bullish_breakout else "bearish",
            "num_classes": 3,
        }
    )
