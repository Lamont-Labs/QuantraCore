"""
T53 - Head and Shoulders Detection Protocol

Detects head and shoulders patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T53: Detect head and shoulders patterns.
    
    Fires when H&S or inverse H&S patterns are detected.
    """
    bars = window.bars
    if len(bars) < 40:
        return ProtocolResult(
            protocol_id="T53",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    local_highs = []
    local_lows = []
    
    for i in range(3, len(bars) - 3):
        if highs[i] == max(highs[i-3:i+4]):
            local_highs.append((i, highs[i]))
        if lows[i] == min(lows[i-3:i+4]):
            local_lows.append((i, lows[i]))
    
    hs_pattern = False
    inverse_hs = False
    neckline = 0
    
    if len(local_highs) >= 3:
        for i in range(len(local_highs) - 2):
            left_idx, left_high = local_highs[i]
            for j in range(i + 1, len(local_highs) - 1):
                head_idx, head_high = local_highs[j]
                for k in range(j + 1, len(local_highs)):
                    right_idx, right_high = local_highs[k]
                    
                    if head_high > left_high and head_high > right_high:
                        if abs(left_high - right_high) / left_high < 0.05:
                            if head_idx - left_idx >= 5 and right_idx - head_idx >= 5:
                                hs_pattern = True
                                left_trough = np.min(lows[left_idx:head_idx])
                                right_trough = np.min(lows[head_idx:right_idx])
                                neckline = (left_trough + right_trough) / 2
                                break
                if hs_pattern:
                    break
            if hs_pattern:
                break
    
    if not hs_pattern and len(local_lows) >= 3:
        for i in range(len(local_lows) - 2):
            left_idx, left_low = local_lows[i]
            for j in range(i + 1, len(local_lows) - 1):
                head_idx, head_low = local_lows[j]
                for k in range(j + 1, len(local_lows)):
                    right_idx, right_low = local_lows[k]
                    
                    if head_low < left_low and head_low < right_low:
                        if abs(left_low - right_low) / left_low < 0.05:
                            if head_idx - left_idx >= 5 and right_idx - head_idx >= 5:
                                inverse_hs = True
                                left_peak = np.max(highs[left_idx:head_idx])
                                right_peak = np.max(highs[head_idx:right_idx])
                                neckline = (left_peak + right_peak) / 2
                                break
                if inverse_hs:
                    break
            if inverse_hs:
                break
    
    if hs_pattern:
        signal_type = "head_and_shoulders"
        confidence = 0.85
        fired = True
    elif inverse_hs:
        signal_type = "inverse_head_and_shoulders"
        confidence = 0.85
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T53",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "neckline": float(neckline),
            "local_highs_count": len(local_highs),
            "local_lows_count": len(local_lows),
        }
    )
