"""
T58 - Island Reversal Detection Protocol

Detects island reversal patterns formed by gaps.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T58: Detect island reversal patterns.
    
    Fires when island reversal gap patterns are detected.
    """
    bars = window.bars
    if len(bars) < 15:
        return ProtocolResult(
            protocol_id="T58",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    np.array([b.open for b in bars])
    np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    gaps = []
    for i in range(1, len(bars)):
        gap_up = lows[i] > highs[i-1]
        gap_down = highs[i] < lows[i-1]
        if gap_up:
            gaps.append((i, "up", lows[i] - highs[i-1]))
        elif gap_down:
            gaps.append((i, "down", lows[i-1] - highs[i]))
    
    island_top = False
    island_bottom = False
    gap1_idx = 0
    gap2_idx = 0
    
    for i in range(len(gaps) - 1):
        idx1, dir1, size1 = gaps[i]
        for j in range(i + 1, len(gaps)):
            idx2, dir2, size2 = gaps[j]
            
            if dir1 == "up" and dir2 == "down":
                if idx2 - idx1 <= 10 and idx2 - idx1 >= 2:
                    island_top = True
                    gap1_idx = idx1
                    gap2_idx = idx2
                    break
            
            if dir1 == "down" and dir2 == "up":
                if idx2 - idx1 <= 10 and idx2 - idx1 >= 2:
                    island_bottom = True
                    gap1_idx = idx1
                    gap2_idx = idx2
                    break
        
        if island_top or island_bottom:
            break
    
    if island_top:
        signal_type = "island_top_reversal"
        confidence = 0.85
        fired = True
    elif island_bottom:
        signal_type = "island_bottom_reversal"
        confidence = 0.85
        fired = True
    elif len(gaps) >= 2:
        signal_type = "multiple_gaps"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_island"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T58",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "gap_count": len(gaps),
            "gap1_idx": int(gap1_idx),
            "gap2_idx": int(gap2_idx),
        }
    )
