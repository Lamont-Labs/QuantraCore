"""
T67 - Breakout Confirmation Protocol

Confirms breakouts from support/resistance levels.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T67: Confirm breakouts.
    
    Fires when breakout is confirmed with follow-through.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T67",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    prior_high = np.max(highs[-25:-3])
    prior_low = np.min(lows[-25:-3])
    
    current_price = closes[-1]
    avg_volume = np.mean(volumes[-20:-1])
    current_volume = volumes[-1]
    
    upside_breakout = current_price > prior_high
    downside_breakout = current_price < prior_low
    
    volume_confirmation = current_volume > avg_volume * 1.5
    
    follow_through = False
    if upside_breakout:
        follow_through = closes[-1] > closes[-2] > prior_high
    elif downside_breakout:
        follow_through = closes[-1] < closes[-2] < prior_low
    
    breakout_magnitude = 0
    if upside_breakout:
        breakout_magnitude = (current_price - prior_high) / prior_high * 100
    elif downside_breakout:
        breakout_magnitude = (prior_low - current_price) / prior_low * 100
    
    if upside_breakout and volume_confirmation and follow_through:
        signal_type = "confirmed_upside_breakout"
        confidence = 0.9
        fired = True
    elif downside_breakout and volume_confirmation and follow_through:
        signal_type = "confirmed_downside_breakout"
        confidence = 0.9
        fired = True
    elif upside_breakout and volume_confirmation:
        signal_type = "upside_breakout_volume"
        confidence = 0.75
        fired = True
    elif downside_breakout and volume_confirmation:
        signal_type = "downside_breakout_volume"
        confidence = 0.75
        fired = True
    elif upside_breakout or downside_breakout:
        signal_type = "unconfirmed_breakout"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_breakout"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T67",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "prior_high": float(prior_high),
            "prior_low": float(prior_low),
            "current_price": float(current_price),
            "volume_confirmation": bool(volume_confirmation),
            "breakout_magnitude_pct": float(breakout_magnitude),
        }
    )
