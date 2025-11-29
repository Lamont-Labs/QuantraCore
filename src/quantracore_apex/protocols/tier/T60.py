"""
T60 - Three Drives Pattern Detection Protocol

Detects three drives harmonic patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T60: Detect three drives patterns.
    
    Fires when three drives harmonic pattern is detected.
    """
    bars = window.bars
    if len(bars) < 35:
        return ProtocolResult(
            protocol_id="T60",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    np.array([b.close for b in bars])
    
    local_highs = []
    local_lows = []
    
    for i in range(3, len(bars) - 3):
        if highs[i] == max(highs[i-3:i+4]):
            local_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-3:i+4]):
            local_lows.append((i, lows[i]))
    
    bullish_3drives = False
    bearish_3drives = False
    
    if len(local_lows) >= 3:
        for combo in range(len(local_lows) - 2):
            idx1, low1 = local_lows[combo]
            idx2, low2 = local_lows[combo + 1]
            idx3, low3 = local_lows[combo + 2]
            
            if low1 > low2 > low3:
                ratio1 = (low1 - low2) / max(low1, 0.0001)
                ratio2 = (low2 - low3) / max(low2, 0.0001)
                
                if 0.8 < ratio1 / max(ratio2, 0.0001) < 1.25:
                    spacing1 = idx2 - idx1
                    spacing2 = idx3 - idx2
                    if 0.6 < spacing1 / max(spacing2, 1) < 1.7:
                        bullish_3drives = True
                        break
    
    if len(local_highs) >= 3:
        for combo in range(len(local_highs) - 2):
            idx1, high1 = local_highs[combo]
            idx2, high2 = local_highs[combo + 1]
            idx3, high3 = local_highs[combo + 2]
            
            if high1 < high2 < high3:
                ratio1 = (high2 - high1) / max(high1, 0.0001)
                ratio2 = (high3 - high2) / max(high2, 0.0001)
                
                if 0.8 < ratio1 / max(ratio2, 0.0001) < 1.25:
                    spacing1 = idx2 - idx1
                    spacing2 = idx3 - idx2
                    if 0.6 < spacing1 / max(spacing2, 1) < 1.7:
                        bearish_3drives = True
                        break
    
    if bullish_3drives:
        signal_type = "bullish_three_drives"
        confidence = 0.8
        fired = True
    elif bearish_3drives:
        signal_type = "bearish_three_drives"
        confidence = 0.8
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T60",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "local_highs_count": len(local_highs),
            "local_lows_count": len(local_lows),
        }
    )
