"""
T17 - Double Top Protocol

Detects double top reversal patterns.
Category: Reversal Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T17: Detect double top patterns.
    
    Fires when double top structure identified.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T17",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars[-30:]])
    
    max_idx1 = np.argmax(highs[:15])
    max_idx2 = np.argmax(highs[15:]) + 15
    
    high1 = highs[max_idx1]
    high2 = highs[max_idx2]
    
    if high1 == 0:
        return ProtocolResult(
            protocol_id="T17",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_high"}
        )
    
    similar_highs = abs(high1 - high2) / high1 < 0.02
    
    middle_section = highs[max_idx1 + 1:max_idx2]
    if len(middle_section) > 0:
        neckline = np.min(middle_section)
        has_dip = neckline < high1 * 0.98
    else:
        has_dip = False
        neckline = high1
    
    current = bars[-1].close
    breaking_neckline = current < neckline
    
    if similar_highs and has_dip:
        if breaking_neckline:
            signal_type = "double_top_confirmed"
            confidence = 0.8
        else:
            signal_type = "double_top_forming"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_double_top"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T17",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "high1": float(high1),
            "high2": float(high2),
            "similar_highs": similar_highs,
            "neckline": float(neckline),
        }
    )
