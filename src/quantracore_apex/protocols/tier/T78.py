"""
T78 - Trend Continuation Probability Protocol

Calculates probability of trend continuation.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T78: Calculate trend continuation probability.
    
    Fires when conditions favor trend continuation.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T78",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    trend_10 = (closes[-1] - closes[-11]) / closes[-11] * 100
    trend_20 = (closes[-1] - closes[-21]) / closes[-21] * 100
    
    trend_alignment = np.sign(trend_10) == np.sign(trend_20)
    
    recent_vol = np.mean(volumes[-5:])
    prior_vol = np.mean(volumes[-20:-5])
    volume_support = recent_vol > prior_vol
    
    pullback_depth = 0
    if trend_10 > 0:
        recent_high = np.max(closes[-10:])
        pullback_depth = (recent_high - closes[-1]) / recent_high * 100
    else:
        recent_low = np.min(closes[-10:])
        pullback_depth = (closes[-1] - recent_low) / recent_low * 100
    
    shallow_pullback = pullback_depth < 3
    
    continuation_score: float = 0.0
    if trend_alignment:
        continuation_score += 0.3
    if volume_support:
        continuation_score += 0.25
    if shallow_pullback:
        continuation_score += 0.25
    if abs(trend_10) > 3:
        continuation_score += 0.2
    
    if continuation_score > 0.7 and trend_alignment:
        signal_type = "high_continuation_probability"
        confidence = min(0.9, 0.5 + continuation_score * 0.4)
        fired = True
    elif continuation_score > 0.5:
        signal_type = "moderate_continuation_probability"
        confidence = 0.7
        fired = True
    elif trend_alignment:
        signal_type = "trend_aligned"
        confidence = 0.55
        fired = True
    else:
        signal_type = "uncertain_continuation"
        confidence = 0.35
        fired = False
    
    return ProtocolResult(
        protocol_id="T78",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "trend_10_pct": float(trend_10),
            "trend_20_pct": float(trend_20),
            "trend_alignment": bool(trend_alignment),
            "volume_support": bool(volume_support),
            "pullback_depth_pct": float(pullback_depth),
            "continuation_score": float(continuation_score),
        }
    )
