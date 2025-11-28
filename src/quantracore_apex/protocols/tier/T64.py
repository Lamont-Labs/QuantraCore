"""
T64 - Fibonacci Retracement Protocol

Analyzes price position relative to Fibonacci levels.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T64: Analyze Fibonacci retracement levels.
    
    Fires when price is at key Fibonacci levels.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T64",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    swing_high = np.max(highs)
    swing_low = np.min(lows)
    price_range = swing_high - swing_low
    
    fib_levels = {
        "0.236": swing_high - 0.236 * price_range,
        "0.382": swing_high - 0.382 * price_range,
        "0.500": swing_high - 0.500 * price_range,
        "0.618": swing_high - 0.618 * price_range,
        "0.786": swing_high - 0.786 * price_range,
    }
    
    current_price = closes[-1]
    
    nearest_fib = None
    min_distance = float('inf')
    
    for level_name, level_price in fib_levels.items():
        distance = abs(current_price - level_price) / current_price
        if distance < min_distance:
            min_distance = distance
            nearest_fib = (level_name, level_price)
    
    at_golden = nearest_fib and nearest_fib[0] == "0.618" and min_distance < 0.02
    at_half = nearest_fib and nearest_fib[0] == "0.500" and min_distance < 0.02
    
    if at_golden:
        signal_type = "at_golden_ratio"
        confidence = 0.85
        fired = True
    elif at_half:
        signal_type = "at_50_retracement"
        confidence = 0.8
        fired = True
    elif min_distance < 0.015 and nearest_fib:
        signal_type = f"at_fib_{nearest_fib[0]}"
        confidence = 0.75
        fired = True
    elif min_distance < 0.03 and nearest_fib:
        signal_type = f"near_fib_{nearest_fib[0]}"
        confidence = 0.6
        fired = True
    else:
        signal_type = "between_fib_levels"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T64",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "swing_high": float(swing_high),
            "swing_low": float(swing_low),
            "nearest_fib_level": nearest_fib[0] if nearest_fib else "",
            "nearest_fib_price": float(nearest_fib[1]) if nearest_fib else 0,
            "distance_pct": float(min_distance * 100),
        }
    )
