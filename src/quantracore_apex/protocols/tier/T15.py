"""
T15 - Wedge Pattern Protocol

Detects wedge patterns (rising/falling).
Category: Continuation Logic
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T15: Detect wedge patterns.
    
    Fires when wedge pattern identified.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T15",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    recent_bars = bars[-20:]
    highs = np.array([b.high for b in recent_bars])
    lows = np.array([b.low for b in recent_bars])
    x = np.arange(len(highs))
    
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    
    both_up = high_slope > 0 and low_slope > 0
    both_down = high_slope < 0 and low_slope < 0
    converging = abs(high_slope - low_slope) < abs(high_slope) * 0.5
    
    if both_up and converging:
        signal_type = "rising_wedge"
        confidence = 0.65
        fired = True
    elif both_down and converging:
        signal_type = "falling_wedge"
        confidence = 0.65
        fired = True
    else:
        signal_type = "no_wedge"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T15",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "high_slope": float(high_slope),
            "low_slope": float(low_slope),
            "both_up": both_up,
            "both_down": both_down,
        }
    )
