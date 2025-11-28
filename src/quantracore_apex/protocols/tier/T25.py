"""
T25 - Keltner Channel Analysis Protocol

Analyzes price position relative to Keltner Channels.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T25: Analyze Keltner Channel position.
    
    Fires when price breaks above/below Keltner Channels.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T25",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    ema20 = closes[-20:].mean()
    
    tr_values = []
    for i in range(1, min(20, len(bars))):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-i-1]),
            abs(lows[-i] - closes[-i-1])
        )
        tr_values.append(tr)
    
    atr = np.mean(tr_values) if tr_values else (highs[-1] - lows[-1])
    
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    current_price = closes[-1]
    
    if current_price > upper_keltner:
        signal_type = "above_upper_keltner"
        position_score = (current_price - upper_keltner) / max(atr, 0.0001)
        confidence = min(0.9, 0.6 + position_score * 0.1)
        fired = True
    elif current_price < lower_keltner:
        signal_type = "below_lower_keltner"
        position_score = (lower_keltner - current_price) / max(atr, 0.0001)
        confidence = min(0.9, 0.6 + position_score * 0.1)
        fired = True
    else:
        signal_type = "within_keltner"
        position_score = 0
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T25",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "upper_keltner": float(upper_keltner),
            "lower_keltner": float(lower_keltner),
            "ema20": float(ema20),
            "atr": float(atr),
            "current_price": float(current_price),
        }
    )
