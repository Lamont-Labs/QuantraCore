"""
T57 - Cup and Handle Detection Protocol

Detects cup and handle continuation patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T57: Detect cup and handle patterns.
    
    Fires when cup and handle bullish continuation is detected.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T57",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    
    left_rim = np.max(closes[-50:-35])
    cup_bottom = np.min(closes[-35:-15])
    right_rim = np.max(closes[-15:-5])
    
    rim_similarity = abs(left_rim - right_rim) / left_rim < 0.05
    cup_depth = (left_rim - cup_bottom) / left_rim
    valid_depth = 0.1 < cup_depth < 0.35
    
    handle_high = np.max(highs[-10:])
    handle_low = np.min(closes[-10:])
    handle_pullback = (right_rim - handle_low) / right_rim
    valid_handle = 0.05 < handle_pullback < 0.15
    
    rounded_bottom = True
    cup_segment = closes[-35:-15]
    mid_point = len(cup_segment) // 2
    left_half_trend = np.mean(cup_segment[:mid_point]) > np.mean(cup_segment[mid_point:])
    
    if rim_similarity and valid_depth and valid_handle:
        signal_type = "cup_and_handle"
        confidence = 0.85
        fired = True
    elif rim_similarity and valid_depth:
        signal_type = "cup_forming"
        confidence = 0.7
        fired = True
    elif valid_depth and cup_bottom < closes[-1] < right_rim:
        signal_type = "potential_cup"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_cup"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T57",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "left_rim": float(left_rim),
            "cup_bottom": float(cup_bottom),
            "right_rim": float(right_rim),
            "cup_depth_pct": float(cup_depth * 100),
            "handle_pullback_pct": float(handle_pullback * 100),
        }
    )
