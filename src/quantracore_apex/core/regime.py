"""
Regime classification module for QuantraCore Apex.

Classifies market regime based on structural analysis.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, RegimeType, Microtraits


def compute_trend_score(bars: List[OhlcvBar]) -> float:
    """
    Compute trend strength score.
    Positive = uptrend, Negative = downtrend, Near zero = no trend.
    """
    if len(bars) < 20:
        return 0.0
    
    closes = np.array([bar.close for bar in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-min(50, len(closes)):])
    
    current = closes[-1]
    
    if sma50 == 0:
        return 0.0
    
    trend = (current - sma50) / sma50
    
    direction = 1 if sma20 > sma50 else -1 if sma20 < sma50 else 0
    
    return float(trend * direction)


def compute_range_score(bars: List[OhlcvBar]) -> float:
    """
    Compute how range-bound the price action is.
    Higher score = more range-bound.
    """
    if len(bars) < 20:
        return 0.0
    
    closes = np.array([bar.close for bar in bars[-20:]])
    
    price_range = np.max(closes) - np.min(closes)
    avg_price = np.mean(closes)
    
    if avg_price == 0:
        return 0.0
    
    range_pct = price_range / avg_price
    
    crosses = 0
    mean = np.mean(closes)
    for i in range(1, len(closes)):
        if (closes[i-1] < mean and closes[i] > mean) or \
           (closes[i-1] > mean and closes[i] < mean):
            crosses += 1
    
    cross_rate = crosses / len(closes)
    
    range_score = 0.0
    if range_pct < 0.05:
        range_score += 0.5
    if cross_rate > 0.2:
        range_score += 0.5
    
    return float(min(1.0, range_score))


def compute_volatility_score(bars: List[OhlcvBar]) -> float:
    """
    Compute volatility score.
    Higher = more volatile.
    """
    if len(bars) < 20:
        return 0.5
    
    returns = []
    for i in range(1, len(bars)):
        if bars[i-1].close != 0:
            ret = (bars[i].close - bars[i-1].close) / bars[i-1].close
            returns.append(ret)
    
    if not returns:
        return 0.5
    
    volatility = np.std(returns) * np.sqrt(252)
    
    if volatility < 0.15:
        return 0.2
    elif volatility < 0.30:
        return 0.5
    elif volatility < 0.50:
        return 0.7
    else:
        return 0.9


def classify_regime(window: OhlcvWindow, microtraits: Microtraits) -> RegimeType:
    """
    Classify market regime from window and microtraits.
    """
    bars = window.bars
    
    trend_score = compute_trend_score(bars)
    range_score = compute_range_score(bars)
    volatility_score = compute_volatility_score(bars)
    compression = microtraits.compression_score
    
    if compression > 0.6:
        return RegimeType.COMPRESSED
    
    if volatility_score > 0.7:
        return RegimeType.VOLATILE
    
    if abs(trend_score) > 0.05:
        if trend_score > 0:
            return RegimeType.TRENDING_UP
        else:
            return RegimeType.TRENDING_DOWN
    
    if range_score > 0.5:
        return RegimeType.RANGE_BOUND
    
    return RegimeType.UNKNOWN
