"""
Compression Forecast Engine

Forecasts compression/expansion cycles using deterministic heuristics.
Part of the QuantraCore Apex prediction stack.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ..core.schemas import OhlcvBar


@dataclass
class CompressionForecast:
    """Compression forecast result."""
    current_compression: float = 0.0
    compression_trend: str = "stable"
    breakout_probability: float = 0.0
    consolidation_depth: float = 0.0
    expected_duration: int = 0
    breakout_direction_bias: str = "neutral"
    confidence: float = 0.0
    compliance_note: str = "Forecast is structural probability, not trading advice"


def forecast_compression(
    bars: List[OhlcvBar],
    lookback: int = 20,
) -> CompressionForecast:
    """
    Forecast compression/expansion cycles.
    
    Args:
        bars: OHLCV price bars
        lookback: Analysis period
        
    Returns:
        CompressionForecast with cycle predictions
        
    Method:
    - Measures current compression state
    - Analyzes compression cycle patterns
    - Estimates breakout probability
    """
    if len(bars) < lookback + 10:
        return CompressionForecast()
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    ranges = highs - lows
    recent_range = float(np.mean(ranges[-lookback:]))
    historical_range = float(np.mean(ranges[:-lookback])) if len(ranges) > lookback else recent_range
    
    compression_ratio = recent_range / max(historical_range, 1e-10)
    current_compression = max(0.0, 1.0 - compression_ratio)
    
    short_range = float(np.mean(ranges[-5:]))
    mid_range = float(np.mean(ranges[-10:-5])) if len(ranges) >= 10 else short_range
    
    if short_range < mid_range * 0.9:
        compression_trend = "tightening"
    elif short_range > mid_range * 1.1:
        compression_trend = "expanding"
    else:
        compression_trend = "stable"
    
    breakout_probability = 0.0
    if current_compression > 0.3:
        breakout_probability = min(current_compression * 1.2, 0.8)
    
    if current_compression > 0.5:
        consolidation_depth = min(current_compression * 1.5, 1.0)
    else:
        consolidation_depth = current_compression
    
    if current_compression > 0.3:
        expected_duration = max(1, int(5 * (1 - current_compression)))
    else:
        expected_duration = 10
    
    recent_trend = closes[-1] - closes[-lookback]
    if recent_trend > 0:
        breakout_direction_bias = "bullish"
    elif recent_trend < 0:
        breakout_direction_bias = "bearish"
    else:
        breakout_direction_bias = "neutral"
    
    confidence = min(current_compression * 80 + 20, 75.0)
    
    return CompressionForecast(
        current_compression=round(current_compression, 4),
        compression_trend=compression_trend,
        breakout_probability=round(breakout_probability, 4),
        consolidation_depth=round(consolidation_depth, 4),
        expected_duration=expected_duration,
        breakout_direction_bias=breakout_direction_bias,
        confidence=round(confidence, 2),
    )
