"""
LP24 - Time-to-Event Estimation Label Protocol

Estimates time until significant price event.
Category: Advanced Labels - Temporal Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP24: Generate time-to-event estimation label.
    
    Estimates urgency of potential move:
    - Imminent (1-3 bars)
    - Near-term (4-10 bars)
    - Extended (10+ bars)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP24",
            label_name="time_to_event",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    
    compression = apex_result.microtraits.compression_score
    
    ranges = highs - lows
    range_contraction = ranges[-5:].mean() / (ranges[-20:-5].mean() + 1e-10)
    
    vol_buildup = volumes[-5:].mean() / (volumes[-20:-5].mean() + 1e-10)
    
    prior_high = np.max(highs[-20:-5])
    prior_low = np.min(lows[-20:-5])
    proximity_to_breakout = min(
        abs(closes[-1] - prior_high) / (prior_high + 1e-10),
        abs(closes[-1] - prior_low) / (prior_low + 1e-10)
    )
    
    returns = np.abs(np.diff(closes) / closes[:-1])
    vol_trend = np.polyfit(range(len(returns[-10:])), returns[-10:], 1)[0]
    
    urgency_score = 0.0
    
    if compression > 0.8:
        urgency_score += 0.35
    elif compression > 0.6:
        urgency_score += 0.2
    
    if range_contraction < 0.5:
        urgency_score += 0.25
    elif range_contraction < 0.7:
        urgency_score += 0.15
    
    if vol_buildup > 1.3:
        urgency_score += 0.2
    
    if proximity_to_breakout < 0.02:
        urgency_score += 0.15
    elif proximity_to_breakout < 0.05:
        urgency_score += 0.08
    
    if vol_trend > 0.001:
        urgency_score += 0.1
    
    if urgency_score > 0.6:
        time_class = 0
        time_name = "imminent"
        est_bars = "1-3"
    elif urgency_score > 0.35:
        time_class = 1
        time_name = "near_term"
        est_bars = "4-10"
    else:
        time_class = 2
        time_name = "extended"
        est_bars = "10+"
    
    confidence = min(0.8, 0.35 + urgency_score * 0.5)
    
    return LearningLabel(
        protocol_id="LP24",
        label_name="time_to_event",
        value=time_class,
        confidence=confidence,
        metadata={
            "time_name": time_name,
            "estimated_bars": est_bars,
            "urgency_score": float(urgency_score),
            "compression": float(compression),
            "range_contraction": float(range_contraction),
            "vol_buildup": float(vol_buildup),
            "proximity_to_breakout": float(proximity_to_breakout),
            "num_classes": 3,
        }
    )
