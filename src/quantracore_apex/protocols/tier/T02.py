"""
T02 - Trend Strength Protocol

Measures the strength and quality of the current trend.
Category: Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def compute_adx(bars, period: int = 14) -> float:
    """Compute simplified ADX indicator."""
    if len(bars) < period + 1:
        return 0.0
    
    plus_dm = []
    minus_dm = []
    tr_list = []
    
    for i in range(1, len(bars)):
        high_diff = bars[i].high - bars[i-1].high
        low_diff = bars[i-1].low - bars[i].low
        
        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
        
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        tr_list.append(tr)
    
    if len(tr_list) < period:
        return 0.0
    
    atr = np.mean(tr_list[-period:])
    if atr == 0:
        return 0.0
    
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr
    
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return 0.0
    
    dx = 100 * abs(plus_di - minus_di) / di_sum
    
    return float(dx)


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T02: Measure trend strength using ADX-like computation.
    
    Fires when trend strength exceeds threshold.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T02",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    adx = compute_adx(bars)
    
    body_consistency = microtraits.body_ratio
    directional_strength = abs(microtraits.trend_consistency)
    
    combined_strength = (adx / 100 * 0.5) + (directional_strength * 0.3) + (body_consistency * 0.2)
    
    if adx > 25 and combined_strength > 0.4:
        signal_type = "strong_trend"
        confidence = min(0.95, combined_strength + 0.3)
        fired = True
    elif adx > 20:
        signal_type = "moderate_trend"
        confidence = combined_strength + 0.2
        fired = True
    else:
        signal_type = "weak_trend"
        confidence = combined_strength
        fired = False
    
    return ProtocolResult(
        protocol_id="T02",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "adx": float(adx),
            "combined_strength": float(combined_strength),
            "body_consistency": float(body_consistency),
        }
    )
