"""
T70 - Range Boundary Analysis Protocol

Analyzes price behavior at range boundaries.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T70: Analyze range boundaries.
    
    Fires when price tests or breaks range boundaries.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T70",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    range_high = np.max(highs[-20:])
    range_low = np.min(lows[-20:])
    range_size = range_high - range_low
    
    current_price = closes[-1]
    
    upper_zone = range_high - range_size * 0.1
    lower_zone = range_low + range_size * 0.1
    middle = (range_high + range_low) / 2
    
    range_ratio = range_size / middle
    is_ranging = range_ratio < 0.1
    
    at_upper = current_price >= upper_zone
    at_lower = current_price <= lower_zone
    near_middle = abs(current_price - middle) / middle < 0.02
    
    touches_upper = sum(1 for h in highs[-10:] if h >= upper_zone)
    touches_lower = sum(1 for l in lows[-10:] if l <= lower_zone)
    
    if at_upper and touches_upper >= 2:
        signal_type = "testing_range_resistance"
        confidence = 0.8
        fired = True
    elif at_lower and touches_lower >= 2:
        signal_type = "testing_range_support"
        confidence = 0.8
        fired = True
    elif current_price > range_high:
        signal_type = "broke_above_range"
        confidence = 0.85
        fired = True
    elif current_price < range_low:
        signal_type = "broke_below_range"
        confidence = 0.85
        fired = True
    elif is_ranging and near_middle:
        signal_type = "range_midpoint"
        confidence = 0.5
        fired = True
    elif is_ranging:
        signal_type = "in_range"
        confidence = 0.4
        fired = True
    else:
        signal_type = "no_clear_range"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T70",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "range_high": float(range_high),
            "range_low": float(range_low),
            "current_price": float(current_price),
            "range_size_pct": float(range_ratio * 100),
            "touches_upper": int(touches_upper),
            "touches_lower": int(touches_lower),
        }
    )
