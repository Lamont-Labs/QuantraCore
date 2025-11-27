"""
T18 - Head and Shoulders Protocol

Detects head and shoulders reversal patterns.
Category: Reversal Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T18: Detect head and shoulders patterns.
    
    Fires when H&S structure identified.
    """
    bars = window.bars
    if len(bars) < 40:
        return ProtocolResult(
            protocol_id="T18",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars[-40:]])
    
    third = len(highs) // 3
    
    left_shoulder_idx = np.argmax(highs[:third])
    head_idx = np.argmax(highs[third:2*third]) + third
    right_shoulder_idx = np.argmax(highs[2*third:]) + 2*third
    
    left_shoulder = highs[left_shoulder_idx]
    head = highs[head_idx]
    right_shoulder = highs[right_shoulder_idx]
    
    if head == 0:
        return ProtocolResult(
            protocol_id="T18",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_head"}
        )
    
    head_highest = head > left_shoulder and head > right_shoulder
    
    shoulders_similar = abs(left_shoulder - right_shoulder) / left_shoulder < 0.05 if left_shoulder > 0 else False
    
    neckline_points = [
        np.min(highs[left_shoulder_idx:head_idx]),
        np.min(highs[head_idx:right_shoulder_idx])
    ]
    neckline = np.mean(neckline_points)
    
    current = bars[-1].close
    breaking_neckline = current < neckline
    
    if head_highest and shoulders_similar:
        if breaking_neckline:
            signal_type = "hs_confirmed"
            confidence = 0.85
        else:
            signal_type = "hs_forming"
            confidence = 0.6
        fired = True
    else:
        signal_type = "no_hs"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T18",
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
