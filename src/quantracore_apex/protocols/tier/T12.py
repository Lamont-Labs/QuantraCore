"""
T12 - Bearish Continuation Protocol

Detects bearish continuation patterns.
Category: Continuation Logic
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T12: Detect bearish continuation patterns.
    
    Fires when bearish continuation setup identified.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T12",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    in_downtrend = closes[-1] < np.mean(closes[-20:])
    
    recent_bounce = closes[-1] > closes[-5] and closes[-5] < closes[-10]
    
    bearish_recent = microtraits.bullish_pct_last20 < 0.45
    
    volume_confirms = microtraits.volume_intensity > 0.8
    
    if in_downtrend and bearish_recent:
        if recent_bounce:
            signal_type = "bearish_bounce_continuation"
            confidence = 0.7
        else:
            signal_type = "bearish_continuation"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_bearish_continuation"
        confidence = 0.15
        fired = False
    
    if volume_confirms and fired:
        confidence = min(0.9, confidence + 0.1)
    
    return ProtocolResult(
        protocol_id="T12",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "in_downtrend": in_downtrend,
            "recent_bounce": recent_bounce,
            "bearish_pct": float(1 - microtraits.bullish_pct_last20),
        }
    )
