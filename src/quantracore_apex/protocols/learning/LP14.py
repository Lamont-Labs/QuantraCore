"""
LP14 - Compression Breakout Direction Label Protocol

Predicts breakout direction from compression zones.
Category: Advanced Labels - Pattern Prediction
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP14: Generate compression breakout direction label.
    
    Predicts direction of breakout from compressed ranges:
    - Bullish breakout likely
    - Bearish breakout likely
    - No breakout expected
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP14",
            label_name="breakout_direction",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    compression = apex_result.microtraits.compression_score
    
    if compression < 0.5:
        return LearningLabel(
            protocol_id="LP14",
            label_name="breakout_direction",
            value=2,
            confidence=0.3,
            metadata={
                "direction_name": "no_compression",
                "compression_score": float(compression),
                "num_classes": 3,
            }
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    
    trend_before = np.polyfit(range(20), closes[:20], 1)[0]
    
    upper_touches = np.sum(highs[-10:] >= np.percentile(highs[-10:], 90))
    lower_touches = np.sum(lows[-10:] <= np.percentile(lows[-10:], 10))
    
    vol_trend = volumes[-5:].mean() / (volumes[-20:-5].mean() + 1e-10)
    
    bullish_score = 0.0
    
    if trend_before > 0:
        bullish_score += 0.3
    else:
        bullish_score -= 0.3
    
    if upper_touches > lower_touches:
        bullish_score += 0.2
    elif lower_touches > upper_touches:
        bullish_score -= 0.2
    
    if vol_trend > 1.2:
        bullish_score += 0.15 * np.sign(trend_before)
    
    current_pos = (closes[-1] - lows[-10:].min()) / (highs[-10:].max() - lows[-10:].min() + 1e-10)
    if current_pos > 0.6:
        bullish_score += 0.15
    elif current_pos < 0.4:
        bullish_score -= 0.15
    
    if bullish_score > 0.25:
        direction = 0
        direction_name = "bullish_breakout"
    elif bullish_score < -0.25:
        direction = 1
        direction_name = "bearish_breakout"
    else:
        direction = 2
        direction_name = "uncertain"
    
    confidence = min(0.85, 0.4 + abs(bullish_score) + compression * 0.2)
    
    return LearningLabel(
        protocol_id="LP14",
        label_name="breakout_direction",
        value=direction,
        confidence=confidence,
        metadata={
            "direction_name": direction_name,
            "compression_score": float(compression),
            "trend_before": float(trend_before),
            "bullish_score": float(bullish_score),
            "current_position": float(current_pos),
            "num_classes": 3,
        }
    )
