"""
T16 - Double Bottom Protocol

Detects double bottom reversal patterns.
Category: Reversal Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T16: Detect double bottom patterns.
    
    Fires when double bottom structure identified.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T16",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    lows = np.array([b.low for b in bars[-30:]])
    
    min_idx1 = np.argmin(lows[:15])
    min_idx2 = np.argmin(lows[15:]) + 15
    
    low1 = lows[min_idx1]
    low2 = lows[min_idx2]
    
    if low1 == 0:
        return ProtocolResult(
            protocol_id="T16",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_low"}
        )
    
    similar_lows = abs(low1 - low2) / low1 < 0.02
    
    middle_section = lows[min_idx1 + 1:min_idx2]
    if len(middle_section) > 0:
        neckline = np.max(middle_section)
        has_bounce = neckline > low1 * 1.02
    else:
        has_bounce = False
        neckline = low1
    
    current = bars[-1].close
    breaking_neckline = current > neckline
    
    if similar_lows and has_bounce:
        if breaking_neckline:
            signal_type = "double_bottom_confirmed"
            confidence = 0.8
        else:
            signal_type = "double_bottom_forming"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_double_bottom"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T16",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "low1": float(low1),
            "low2": float(low2),
            "similar_lows": similar_lows,
            "neckline": float(neckline),
        }
    )
