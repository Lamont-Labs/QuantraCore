"""
T24 - ATR Breakout Detection Protocol

Detects ATR-based breakout conditions.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T24: Detect ATR-based breakouts.
    
    Fires when price moves exceed ATR thresholds indicating breakout.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T24",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    tr_values = []
    for i in range(1, len(bars)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)
    
    atr14 = np.mean(tr_values[-14:]) if len(tr_values) >= 14 else np.mean(tr_values)
    
    last_move = abs(closes[-1] - closes[-2])
    atr_multiple = last_move / max(atr14, 0.0001)
    
    range_last5 = np.max(highs[-5:]) - np.min(lows[-5:])
    range_atr = range_last5 / max(atr14, 0.0001)
    
    if atr_multiple > 2.0 or range_atr > 3.0:
        signal_type = "strong_breakout"
        confidence = min(0.95, 0.6 + atr_multiple * 0.1)
        fired = True
    elif atr_multiple > 1.5 or range_atr > 2.0:
        signal_type = "moderate_breakout"
        confidence = 0.6
        fired = True
    elif atr_multiple > 1.0:
        signal_type = "mild_breakout"
        confidence = 0.4
        fired = True
    else:
        signal_type = "no_breakout"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T24",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "atr14": float(atr14),
            "last_move": float(last_move),
            "atr_multiple": float(atr_multiple),
            "range_atr": float(range_atr),
        }
    )
