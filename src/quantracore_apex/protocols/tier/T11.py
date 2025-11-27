"""
T11 - Bullish Continuation Protocol

Detects bullish continuation patterns.
Category: Continuation Logic
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T11: Detect bullish continuation patterns.
    
    Fires when bullish continuation setup identified.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T11",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    in_uptrend = closes[-1] > np.mean(closes[-20:])
    
    recent_pullback = closes[-1] < closes[-5] and closes[-5] > closes[-10]
    
    bullish_recent = microtraits.bullish_pct_last20 > 0.55
    
    volume_confirms = microtraits.volume_intensity > 0.8
    
    if in_uptrend and bullish_recent:
        if recent_pullback:
            signal_type = "bullish_pullback_continuation"
            confidence = 0.7
        else:
            signal_type = "bullish_continuation"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_bullish_continuation"
        confidence = 0.15
        fired = False
    
    if volume_confirms and fired:
        confidence = min(0.9, confidence + 0.1)
    
    return ProtocolResult(
        protocol_id="T11",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "in_uptrend": in_uptrend,
            "recent_pullback": recent_pullback,
            "bullish_pct": float(microtraits.bullish_pct_last20),
        }
    )
