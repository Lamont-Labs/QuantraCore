"""
T74 - Price Position Analysis Protocol

Analyzes price position within historical range.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T74: Analyze price position in range.
    
    Fires when price is at extreme positions in historical range.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T74",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    current_price = closes[-1]
    
    high_20 = np.max(highs[-20:])
    low_20 = np.min(lows[-20:])
    range_20 = high_20 - low_20
    
    high_50 = np.max(highs[-50:])
    low_50 = np.min(lows[-50:])
    range_50 = high_50 - low_50
    
    position_20 = (current_price - low_20) / max(range_20, 0.0001) * 100
    position_50 = (current_price - low_50) / max(range_50, 0.0001) * 100
    
    at_20d_high = current_price >= high_20 * 0.99
    at_20d_low = current_price <= low_20 * 1.01
    at_50d_high = current_price >= high_50 * 0.99
    at_50d_low = current_price <= low_50 * 1.01
    
    if at_50d_high:
        signal_type = "at_50d_high"
        confidence = 0.85
        fired = True
    elif at_50d_low:
        signal_type = "at_50d_low"
        confidence = 0.85
        fired = True
    elif at_20d_high:
        signal_type = "at_20d_high"
        confidence = 0.75
        fired = True
    elif at_20d_low:
        signal_type = "at_20d_low"
        confidence = 0.75
        fired = True
    elif position_20 > 80:
        signal_type = "upper_range"
        confidence = 0.6
        fired = True
    elif position_20 < 20:
        signal_type = "lower_range"
        confidence = 0.6
        fired = True
    else:
        signal_type = "mid_range"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T74",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "position_20_pct": float(position_20),
            "position_50_pct": float(position_50),
            "high_20": float(high_20),
            "low_20": float(low_20),
            "high_50": float(high_50),
            "low_50": float(low_50),
        }
    )
