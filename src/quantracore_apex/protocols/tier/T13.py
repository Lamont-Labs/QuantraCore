"""
T13 - Flag Pattern Protocol

Detects flag consolidation patterns.
Category: Continuation Logic
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T13: Detect flag patterns.
    
    Fires when flag consolidation identified.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T13",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    pole_move = (closes[-20] - closes[-30]) / closes[-30] if closes[-30] != 0 else 0
    has_pole = abs(pole_move) > 0.05
    
    flag_closes = closes[-15:]
    flag_slope = np.polyfit(range(len(flag_closes)), flag_closes, 1)[0]
    
    flag_range = np.max(flag_closes) - np.min(flag_closes)
    flag_tight = flag_range / np.mean(flag_closes) < 0.05 if np.mean(flag_closes) > 0 else False
    
    counter_trend_flag = (pole_move > 0 and flag_slope < 0) or (pole_move < 0 and flag_slope > 0)
    
    if has_pole and flag_tight and counter_trend_flag:
        signal_type = "bull_flag" if pole_move > 0 else "bear_flag"
        confidence = 0.75
        fired = True
    elif has_pole and flag_tight:
        signal_type = "potential_flag"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_flag"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T13",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "pole_move_pct": float(pole_move * 100),
            "flag_slope": float(flag_slope),
            "flag_tight": flag_tight,
        }
    )
