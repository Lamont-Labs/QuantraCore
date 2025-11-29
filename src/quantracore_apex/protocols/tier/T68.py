"""
T68 - False Breakout Detection Protocol

Detects potential false breakouts (bull/bear traps).
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T68: Detect false breakouts.
    
    Fires when breakout shows signs of failure.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T68",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    prior_high = np.max(highs[-20:-5])
    prior_low = np.min(lows[-20:-5])
    
    recent_highs = highs[-5:]
    recent_lows = lows[-5:]
    closes[-5:]
    
    broke_above = np.any(recent_highs > prior_high)
    broke_below = np.any(recent_lows < prior_low)
    
    bull_trap = False
    bear_trap = False
    
    if broke_above:
        if closes[-1] < prior_high:
            bull_trap = True
    
    if broke_below:
        if closes[-1] > prior_low:
            bear_trap = True
    
    rejection_wick = False
    if bull_trap:
        rejection_wick = (highs[-1] - closes[-1]) > (closes[-1] - lows[-1]) * 2
    elif bear_trap:
        rejection_wick = (closes[-1] - lows[-1]) > (highs[-1] - closes[-1]) * 2
    
    avg_volume = np.mean(volumes[-20:-5])
    breakout_volume = np.mean(volumes[-3:])
    weak_volume = breakout_volume < avg_volume
    
    if bull_trap and rejection_wick:
        signal_type = "bull_trap_confirmed"
        confidence = 0.85
        fired = True
    elif bear_trap and rejection_wick:
        signal_type = "bear_trap_confirmed"
        confidence = 0.85
        fired = True
    elif bull_trap and weak_volume:
        signal_type = "potential_bull_trap"
        confidence = 0.7
        fired = True
    elif bear_trap and weak_volume:
        signal_type = "potential_bear_trap"
        confidence = 0.7
        fired = True
    elif bull_trap or bear_trap:
        signal_type = "possible_false_breakout"
        confidence = 0.55
        fired = True
    else:
        signal_type = "no_false_breakout"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T68",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "prior_high": float(prior_high),
            "prior_low": float(prior_low),
            "bull_trap": bool(bull_trap),
            "bear_trap": bool(bear_trap),
            "rejection_wick": bool(rejection_wick),
            "weak_volume": bool(weak_volume),
        }
    )
