"""
T71 - Trend Strength Analysis Protocol

Analyzes overall trend strength using multiple indicators.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T71: Analyze overall trend strength.
    
    Fires when trend is clearly defined and strong.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T71",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    
    current_price = closes[-1]
    
    above_sma20 = current_price > sma20
    above_sma50 = current_price > sma50
    sma20_above_sma50 = sma20 > sma50
    
    bullish_aligned = above_sma20 and above_sma50 and sma20_above_sma50
    bearish_aligned = not above_sma20 and not above_sma50 and not sma20_above_sma50
    
    higher_highs = sum(1 for i in range(-9, 0) if closes[i] > closes[i-1])
    lower_lows = sum(1 for i in range(-9, 0) if closes[i] < closes[i-1])
    
    trend_consistency = max(higher_highs, lower_lows) / 9
    
    price_distance_sma20 = (current_price - sma20) / sma20 * 100
    price_distance_sma50 = (current_price - sma50) / sma50 * 100
    
    trend_strength_score = 0
    if bullish_aligned:
        trend_strength_score = trend_consistency * 0.5 + min(price_distance_sma20, 10) * 0.05
    elif bearish_aligned:
        trend_strength_score = trend_consistency * 0.5 + min(abs(price_distance_sma20), 10) * 0.05
    
    if bullish_aligned and trend_strength_score > 0.6:
        signal_type = "strong_uptrend"
        confidence = min(0.9, 0.5 + trend_strength_score * 0.4)
        fired = True
    elif bearish_aligned and trend_strength_score > 0.6:
        signal_type = "strong_downtrend"
        confidence = min(0.9, 0.5 + trend_strength_score * 0.4)
        fired = True
    elif bullish_aligned:
        signal_type = "uptrend"
        confidence = 0.65
        fired = True
    elif bearish_aligned:
        signal_type = "downtrend"
        confidence = 0.65
        fired = True
    else:
        signal_type = "no_clear_trend"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T71",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "sma20": float(sma20),
            "sma50": float(sma50),
            "trend_strength_score": float(trend_strength_score),
            "price_distance_sma20": float(price_distance_sma20),
            "bullish_aligned": bool(bullish_aligned),
            "bearish_aligned": bool(bearish_aligned),
        }
    )
