"""
T38 - CCI Momentum Protocol

Analyzes Commodity Channel Index for momentum signals.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T38: Analyze CCI momentum.
    
    Fires when CCI indicates significant momentum conditions.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T38",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    typical_prices = np.array([(b.high + b.low + b.close) / 3 for b in bars])
    
    period = 20
    sma = np.mean(typical_prices[-period:])
    mean_deviation = np.mean(np.abs(typical_prices[-period:] - sma))
    
    if mean_deviation == 0:
        cci = 0
    else:
        cci = (typical_prices[-1] - sma) / (0.015 * mean_deviation)
    
    cci_values = []
    for i in range(min(5, len(typical_prices) - period)):
        segment = typical_prices[-(period+i):-(i) if i > 0 else None]
        s = np.mean(segment)
        md = np.mean(np.abs(segment - s))
        if md > 0:
            c = (typical_prices[-(i+1)] - s) / (0.015 * md)
            cci_values.append(c)
    
    cci_trend = cci - np.mean(cci_values) if cci_values else 0
    
    if cci > 200:
        signal_type = "extremely_overbought"
        confidence = 0.9
        fired = True
    elif cci > 100:
        signal_type = "overbought"
        confidence = 0.75
        fired = True
    elif cci < -200:
        signal_type = "extremely_oversold"
        confidence = 0.9
        fired = True
    elif cci < -100:
        signal_type = "oversold"
        confidence = 0.75
        fired = True
    elif abs(cci_trend) > 100:
        signal_type = "cci_momentum_shift"
        confidence = 0.65
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T38",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "cci": float(cci),
            "cci_trend": float(cci_trend),
            "typical_price": float(typical_prices[-1]),
            "sma": float(sma),
        }
    )
