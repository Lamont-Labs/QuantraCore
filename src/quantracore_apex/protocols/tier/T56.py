"""
T56 - Wedge Pattern Detection Protocol

Detects rising and falling wedge patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T56: Detect wedge patterns.
    
    Fires when rising or falling wedge patterns are detected.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T56",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars[-20:]])
    lows = np.array([b.low for b in bars[-20:]])
    
    x = np.arange(len(highs))
    
    high_slope, _ = np.polyfit(x, highs, 1)
    low_slope, _ = np.polyfit(x, lows, 1)
    
    both_rising = high_slope > 0 and low_slope > 0
    both_falling = high_slope < 0 and low_slope < 0
    
    converging = abs(high_slope) < abs(low_slope) if both_rising else abs(low_slope) < abs(high_slope)
    
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    narrowing = range_end < range_start * 0.8
    
    if both_rising and converging and narrowing:
        signal_type = "rising_wedge"
        confidence = 0.8
        fired = True
    elif both_falling and converging and narrowing:
        signal_type = "falling_wedge"
        confidence = 0.8
        fired = True
    elif both_rising and narrowing:
        signal_type = "rising_channel"
        confidence = 0.6
        fired = True
    elif both_falling and narrowing:
        signal_type = "falling_channel"
        confidence = 0.6
        fired = True
    else:
        signal_type = "no_wedge"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T56",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "high_slope": float(high_slope),
            "low_slope": float(low_slope),
            "both_rising": bool(both_rising),
            "both_falling": bool(both_falling),
            "narrowing": bool(narrowing),
        }
    )
