"""
T01 - Trend Direction Protocol

Analyzes primary trend direction using multiple timeframe alignment.
Category: Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T01: Determine primary trend direction.
    
    Fires when trend direction is clearly established.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T01",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    current = closes[-1]
    
    above_20 = current > sma20
    above_50 = current > sma50
    sma20_above_50 = sma20 > sma50
    
    trend_strength = microtraits.trend_consistency
    
    if above_20 and above_50 and sma20_above_50 and trend_strength > 0.3:
        signal_type = "uptrend_confirmed"
        confidence = min(0.95, 0.6 + abs(trend_strength) * 0.35)
        fired = True
    elif not above_20 and not above_50 and not sma20_above_50 and trend_strength < -0.3:
        signal_type = "downtrend_confirmed"
        confidence = min(0.95, 0.6 + abs(trend_strength) * 0.35)
        fired = True
    else:
        signal_type = "trend_unclear"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T01",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "current_price": float(current),
            "sma20": float(sma20),
            "sma50": float(sma50),
            "trend_consistency": float(trend_strength),
        }
    )
