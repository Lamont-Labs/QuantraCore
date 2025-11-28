"""
T72 - Market Regime Classification Protocol

Classifies current market regime (trending, ranging, volatile).
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T72: Classify market regime.
    
    Fires when market regime is clearly identifiable.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T72",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns[-20:]) * np.sqrt(252)
    
    price_range = (np.max(highs[-20:]) - np.min(lows[-20:])) / closes[-1]
    
    x = np.arange(20)
    slope, _ = np.polyfit(x, closes[-20:], 1)
    trend_strength = abs(slope) / np.mean(closes[-20:])
    
    is_trending = trend_strength > 0.001
    is_volatile = volatility > 0.3
    is_ranging = price_range < 0.1 and not is_trending
    
    directional_moves = sum(1 for i in range(-19, 0) if np.sign(closes[i] - closes[i-1]) == np.sign(slope))
    trend_consistency = directional_moves / 19
    
    if is_trending and trend_consistency > 0.6:
        regime = "strong_trend"
        confidence = 0.85
        fired = True
    elif is_volatile and not is_trending:
        regime = "high_volatility_chop"
        confidence = 0.8
        fired = True
    elif is_ranging:
        regime = "range_bound"
        confidence = 0.75
        fired = True
    elif is_trending:
        regime = "weak_trend"
        confidence = 0.6
        fired = True
    else:
        regime = "transitional"
        confidence = 0.5
        fired = True
    
    return ProtocolResult(
        protocol_id="T72",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=regime,
        details={
            "volatility": float(volatility),
            "trend_strength": float(trend_strength),
            "price_range_pct": float(price_range * 100),
            "trend_consistency": float(trend_consistency),
            "is_trending": bool(is_trending),
            "is_volatile": bool(is_volatile),
            "is_ranging": bool(is_ranging),
        }
    )
