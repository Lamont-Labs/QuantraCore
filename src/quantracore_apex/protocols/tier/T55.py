"""
T55 - Flag and Pennant Detection Protocol

Detects flag and pennant continuation patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T55: Detect flag and pennant patterns.
    
    Fires when continuation patterns are detected after strong moves.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T55",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    pole_move = (closes[-15] - closes[-30]) / closes[-30] * 100
    strong_pole = abs(pole_move) > 5
    
    flag_highs = highs[-10:]
    flag_lows = lows[-10:]
    
    x = np.arange(10)
    high_slope, _ = np.polyfit(x, flag_highs, 1)
    low_slope, _ = np.polyfit(x, flag_lows, 1)
    
    flag_range = np.mean(flag_highs - flag_lows)
    consolidation = flag_range < abs(closes[-15] - closes[-30]) * 0.5
    
    parallel_channels = abs(high_slope - low_slope) < 0.01 * np.mean(closes[-10:])
    
    bullish_flag = pole_move > 5 and high_slope < 0 and low_slope < 0 and parallel_channels
    bearish_flag = pole_move < -5 and high_slope > 0 and low_slope > 0 and parallel_channels
    
    converging = (high_slope < 0 and low_slope > 0) or abs(high_slope - low_slope) < 0.005 * np.mean(closes[-10:])
    pennant = strong_pole and converging and consolidation
    
    if bullish_flag and consolidation:
        signal_type = "bullish_flag"
        confidence = 0.8
        fired = True
    elif bearish_flag and consolidation:
        signal_type = "bearish_flag"
        confidence = 0.8
        fired = True
    elif pennant:
        signal_type = "pennant"
        confidence = 0.75
        fired = True
    elif strong_pole and consolidation:
        signal_type = "potential_continuation"
        confidence = 0.6
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T55",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "pole_move_pct": float(pole_move),
            "high_slope": float(high_slope),
            "low_slope": float(low_slope),
            "consolidation": bool(consolidation),
        }
    )
