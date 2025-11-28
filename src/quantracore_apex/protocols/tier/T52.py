"""
T52 - Double Top/Bottom Detection Protocol

Detects double top and double bottom patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T52: Detect double top/bottom patterns.
    
    Fires when potential reversal patterns are detected.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T52",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    tolerance = 0.02
    
    local_highs = []
    local_lows = []
    
    for i in range(2, len(bars) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            local_highs.append((i, highs[i]))
        
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            local_lows.append((i, lows[i]))
    
    double_top = False
    double_bottom = False
    pattern_price = 0
    
    for i in range(len(local_highs) - 1):
        idx1, high1 = local_highs[i]
        for j in range(i + 1, len(local_highs)):
            idx2, high2 = local_highs[j]
            if abs(high1 - high2) / high1 < tolerance and idx2 - idx1 >= 5:
                between_low = np.min(lows[idx1:idx2])
                if (high1 - between_low) / high1 > 0.03:
                    double_top = True
                    pattern_price = (high1 + high2) / 2
                    break
        if double_top:
            break
    
    for i in range(len(local_lows) - 1):
        idx1, low1 = local_lows[i]
        for j in range(i + 1, len(local_lows)):
            idx2, low2 = local_lows[j]
            if abs(low1 - low2) / low1 < tolerance and idx2 - idx1 >= 5:
                between_high = np.max(highs[idx1:idx2])
                if (between_high - low1) / low1 > 0.03:
                    double_bottom = True
                    pattern_price = (low1 + low2) / 2
                    break
        if double_bottom:
            break
    
    if double_top:
        signal_type = "double_top"
        confidence = 0.8
        fired = True
    elif double_bottom:
        signal_type = "double_bottom"
        confidence = 0.8
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T52",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "local_highs_count": len(local_highs),
            "local_lows_count": len(local_lows),
            "pattern_price": float(pattern_price),
        }
    )
