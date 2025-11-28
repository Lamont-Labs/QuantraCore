"""
T21 - Volatility Expansion Detection Protocol

Detects volatility expansion phases indicating potential breakout conditions.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T21: Detect volatility expansion phases.
    
    Fires when volatility is expanding from compressed state, indicating potential breakout.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T21",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    atr_short = np.mean(highs[-5:] - lows[-5:])
    atr_long = np.mean(highs[-20:] - lows[-20:])
    
    expansion_ratio = atr_short / max(atr_long, 0.0001)
    
    price_range_recent = (np.max(closes[-5:]) - np.min(closes[-5:])) / closes[-1]
    price_range_prior = (np.max(closes[-20:-5]) - np.min(closes[-20:-5])) / closes[-1]
    
    range_expansion = price_range_recent / max(price_range_prior, 0.0001)
    
    if expansion_ratio > 1.5 and range_expansion > 1.3:
        signal_type = "volatility_expanding"
        confidence = min(0.95, 0.5 + (expansion_ratio - 1) * 0.2)
        fired = True
    elif expansion_ratio > 1.2:
        signal_type = "volatility_mildly_expanding"
        confidence = 0.4 + (expansion_ratio - 1) * 0.2
        fired = True
    else:
        signal_type = "volatility_stable"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T21",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "atr_short": float(atr_short),
            "atr_long": float(atr_long),
            "expansion_ratio": float(expansion_ratio),
            "range_expansion": float(range_expansion),
        }
    )
