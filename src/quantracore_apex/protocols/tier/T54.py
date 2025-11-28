"""
T54 - Triangle Pattern Detection Protocol

Detects ascending, descending, and symmetrical triangle patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T54: Detect triangle patterns.
    
    Fires when triangle consolidation patterns are detected.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T54",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars[-20:]])
    lows = np.array([b.low for b in bars[-20:]])
    
    x = np.arange(len(highs))
    
    high_slope, high_intercept = np.polyfit(x, highs, 1)
    low_slope, low_intercept = np.polyfit(x, lows, 1)
    
    high_fitted = high_slope * x + high_intercept
    low_fitted = low_slope * x + low_intercept
    high_r2 = 1 - np.sum((highs - high_fitted)**2) / np.sum((highs - np.mean(highs))**2)
    low_r2 = 1 - np.sum((lows - low_fitted)**2) / np.sum((lows - np.mean(lows))**2)
    
    trendline_quality = (high_r2 + low_r2) / 2
    
    converging = (high_slope < 0 and low_slope > 0) or \
                 (abs(high_slope) < 0.001 and low_slope > 0) or \
                 (high_slope < 0 and abs(low_slope) < 0.001)
    
    range_start = highs[0] - lows[0]
    range_end = highs[-1] - lows[-1]
    range_contraction = range_end < range_start * 0.7
    
    ascending = abs(high_slope) < 0.001 and low_slope > 0.001
    descending = high_slope < -0.001 and abs(low_slope) < 0.001
    symmetrical = high_slope < -0.001 and low_slope > 0.001
    
    if ascending and range_contraction and trendline_quality > 0.5:
        signal_type = "ascending_triangle"
        confidence = 0.75 + trendline_quality * 0.15
        fired = True
    elif descending and range_contraction and trendline_quality > 0.5:
        signal_type = "descending_triangle"
        confidence = 0.75 + trendline_quality * 0.15
        fired = True
    elif symmetrical and range_contraction and trendline_quality > 0.5:
        signal_type = "symmetrical_triangle"
        confidence = 0.7 + trendline_quality * 0.15
        fired = True
    elif converging and range_contraction:
        signal_type = "consolidating_triangle"
        confidence = 0.6
        fired = True
    else:
        signal_type = "no_triangle"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T54",
        fired=bool(fired),
        confidence=float(min(confidence, 0.95)),
        signal_type=signal_type,
        details={
            "high_slope": float(high_slope),
            "low_slope": float(low_slope),
            "trendline_quality": float(trendline_quality),
            "range_contraction": bool(range_contraction),
        }
    )
