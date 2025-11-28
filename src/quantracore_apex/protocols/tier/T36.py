"""
T36 - ADX Trend Strength Protocol

Analyzes ADX for trend strength measurement.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T36: Analyze ADX trend strength.
    
    Fires when ADX indicates strong trending conditions.
    """
    bars = window.bars
    if len(bars) < 28:
        return ProtocolResult(
            protocol_id="T36",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(1, len(bars)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)
        
        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)
        
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    
    period = 14
    smoothed_plus_dm = np.mean(plus_dm[-period:])
    smoothed_minus_dm = np.mean(minus_dm[-period:])
    smoothed_tr = np.mean(tr_values[-period:])
    
    if smoothed_tr == 0:
        plus_di = 0
        minus_di = 0
    else:
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = (di_diff / max(di_sum, 0.0001)) * 100
    
    adx = dx
    
    if adx > 50:
        signal_type = "very_strong_trend"
        confidence = 0.9
        fired = True
    elif adx > 40:
        signal_type = "strong_trend"
        confidence = 0.8
        fired = True
    elif adx > 25:
        signal_type = "trending"
        confidence = 0.65
        fired = True
    elif adx < 20:
        signal_type = "weak_trend"
        confidence = 0.5
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T36",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "adx": float(adx),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "dx": float(dx),
        }
    )
