"""
T33 - Stochastic Momentum Protocol

Analyzes stochastic oscillator for momentum signals.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T33: Analyze stochastic momentum.
    
    Fires when stochastic shows significant momentum conditions.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T33",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    period = 14
    lowest_low = np.min(lows[-period:])
    highest_high = np.max(highs[-period:])
    
    if highest_high == lowest_low:
        stoch_k = 50.0
    else:
        stoch_k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    stoch_k_values = []
    for i in range(min(5, len(closes) - period)):
        ll = np.min(lows[-(period+i):-(i) if i > 0 else None])
        hh = np.max(highs[-(period+i):-(i) if i > 0 else None])
        if hh != ll:
            sk = ((closes[-(i+1)] - ll) / (hh - ll)) * 100
            stoch_k_values.append(sk)
    
    stoch_d = np.mean(stoch_k_values[-3:]) if len(stoch_k_values) >= 3 else stoch_k
    
    bullish_cross = len(stoch_k_values) >= 2 and stoch_k_values[-1] > stoch_d and stoch_k_values[-2] < stoch_d
    bearish_cross = len(stoch_k_values) >= 2 and stoch_k_values[-1] < stoch_d and stoch_k_values[-2] > stoch_d
    
    if stoch_k > 80 and stoch_d > 80:
        signal_type = "overbought"
        confidence = 0.8
        fired = True
    elif stoch_k < 20 and stoch_d < 20:
        signal_type = "oversold"
        confidence = 0.8
        fired = True
    elif bullish_cross and stoch_k < 30:
        signal_type = "bullish_crossover"
        confidence = 0.85
        fired = True
    elif bearish_cross and stoch_k > 70:
        signal_type = "bearish_crossover"
        confidence = 0.85
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T33",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "stoch_k": float(stoch_k),
            "stoch_d": float(stoch_d),
            "lowest_low": float(lowest_low),
            "highest_high": float(highest_high),
        }
    )
