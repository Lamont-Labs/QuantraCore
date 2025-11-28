"""
T37 - Williams %R Protocol

Analyzes Williams %R for overbought/oversold conditions.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T37: Analyze Williams %R.
    
    Fires when Williams %R indicates extreme conditions.
    """
    bars = window.bars
    if len(bars) < 15:
        return ProtocolResult(
            protocol_id="T37",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    period = 14
    highest_high = np.max(highs[-period:])
    lowest_low = np.min(lows[-period:])
    
    if highest_high == lowest_low:
        williams_r = -50
    else:
        williams_r = ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100
    
    wr_values = []
    for i in range(min(5, len(closes) - period)):
        hh = np.max(highs[-(period+i):-(i) if i > 0 else None])
        ll = np.min(lows[-(period+i):-(i) if i > 0 else None])
        if hh != ll:
            wr = ((hh - closes[-(i+1)]) / (hh - ll)) * -100
            wr_values.append(wr)
    
    wr_trend = williams_r - np.mean(wr_values) if wr_values else 0
    
    if williams_r > -20:
        signal_type = "overbought"
        confidence = min(0.9, 0.6 + (williams_r + 20) * 0.015)
        fired = True
    elif williams_r < -80:
        signal_type = "oversold"
        confidence = min(0.9, 0.6 + (-80 - williams_r) * 0.015)
        fired = True
    elif abs(wr_trend) > 20:
        signal_type = "momentum_shift"
        confidence = 0.6
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T37",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "williams_r": float(williams_r),
            "wr_trend": float(wr_trend),
            "highest_high": float(highest_high),
            "lowest_low": float(lowest_low),
        }
    )
