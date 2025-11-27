"""
T19 - Inverse Head and Shoulders Protocol

Detects inverse head and shoulders reversal patterns.
Category: Reversal Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T19: Detect inverse head and shoulders patterns.
    
    Fires when inverse H&S structure identified.
    """
    bars = window.bars
    if len(bars) < 40:
        return ProtocolResult(
            protocol_id="T19",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    lows = np.array([b.low for b in bars[-40:]])
    
    third = len(lows) // 3
    
    left_shoulder_idx = np.argmin(lows[:third])
    head_idx = np.argmin(lows[third:2*third]) + third
    right_shoulder_idx = np.argmin(lows[2*third:]) + 2*third
    
    left_shoulder = lows[left_shoulder_idx]
    head = lows[head_idx]
    right_shoulder = lows[right_shoulder_idx]
    
    if left_shoulder == 0:
        return ProtocolResult(
            protocol_id="T19",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_shoulder"}
        )
    
    head_lowest = head < left_shoulder and head < right_shoulder
    
    shoulders_similar = abs(left_shoulder - right_shoulder) / left_shoulder < 0.05
    
    neckline_points = [
        np.max(lows[left_shoulder_idx:head_idx]),
        np.max(lows[head_idx:right_shoulder_idx])
    ]
    neckline = np.mean(neckline_points)
    
    current = bars[-1].close
    breaking_neckline = current > neckline
    
    if head_lowest and shoulders_similar:
        if breaking_neckline:
            signal_type = "ihs_confirmed"
            confidence = 0.85
        else:
            signal_type = "ihs_forming"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_ihs"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T19",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "left_shoulder": float(left_shoulder),
            "head": float(head),
            "right_shoulder": float(right_shoulder),
            "neckline": float(neckline),
        }
    )
