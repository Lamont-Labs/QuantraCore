"""
T14 - Pennant Pattern Protocol

Detects pennant consolidation patterns.
Category: Continuation Logic
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T14: Detect pennant patterns.
    
    Fires when pennant consolidation identified.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T14",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    recent_bars = bars[-15:]
    highs = np.array([b.high for b in recent_bars])
    lows = np.array([b.low for b in recent_bars])
    x = np.arange(len(highs))
    
    high_slope = np.polyfit(x, highs, 1)[0]
    low_slope = np.polyfit(x, lows, 1)[0]
    
    converging = high_slope < 0 and low_slope > 0
    
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    narrowing = range_end < range_start * 0.6
    
    closes = np.array([b.close for b in bars])
    prior_move = (closes[-20] - closes[-30]) / closes[-30] if len(closes) >= 30 and closes[-30] != 0 else 0
    has_prior_move = abs(prior_move) > 0.04
    
    if converging and narrowing and has_prior_move:
        signal_type = "bullish_pennant" if prior_move > 0 else "bearish_pennant"
        confidence = 0.7
        fired = True
    elif converging and narrowing:
        signal_type = "potential_pennant"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_pennant"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T14",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "converging": converging,
            "range_contraction": float(1 - range_end / range_start) if range_start > 0 else 0,
            "prior_move_pct": float(prior_move * 100),
        }
    )
