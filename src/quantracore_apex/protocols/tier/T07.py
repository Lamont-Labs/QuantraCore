"""
T07 - Bollinger Band Squeeze Protocol

Detects Bollinger Band squeezes and breakouts.
Category: Volatility Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T07: Detect Bollinger Band squeeze.
    
    Fires when bands are contracted and ready for expansion.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T07",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    bb_widths = []
    for i in range(20, len(closes) + 1):
        window_closes = closes[i-20:i]
        sma = np.mean(window_closes)
        std = np.std(window_closes)
        if sma > 0:
            bb_width = (4 * std) / sma
            bb_widths.append(bb_width)
    
    if len(bb_widths) < 10:
        return ProtocolResult(
            protocol_id="T07",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    current_width = bb_widths[-1]
    historical_width = np.mean(bb_widths[:-5]) if len(bb_widths) > 5 else np.mean(bb_widths)
    min_width_6m = np.percentile(bb_widths, 10)
    
    squeeze_ratio = current_width / historical_width if historical_width > 0 else 1
    near_minimum = current_width <= min_width_6m * 1.1
    
    if squeeze_ratio < 0.6 and near_minimum:
        signal_type = "extreme_squeeze"
        confidence = 0.85
        fired = True
    elif squeeze_ratio < 0.75:
        signal_type = "moderate_squeeze"
        confidence = 0.6
        fired = True
    else:
        signal_type = "no_squeeze"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T07",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "current_bb_width": float(current_width),
            "squeeze_ratio": float(squeeze_ratio),
            "near_minimum": near_minimum,
        }
    )
