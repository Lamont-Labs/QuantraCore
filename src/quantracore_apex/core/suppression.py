"""
Suppression analysis module for QuantraCore Apex.

Suppression detects coiled/compressed structures that may
precede significant directional moves.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, SuppressionMetrics, SuppressionState


def compute_range_compression(bars: List[OhlcvBar], lookback: int = 20) -> float:
    """
    Compute range compression ratio.
    Higher values indicate more suppressed/compressed ranges.
    """
    if len(bars) < lookback:
        return 0.0
    
    recent = bars[-lookback // 2:]
    historical = bars[:-lookback // 2] if len(bars) > lookback // 2 else bars
    
    recent_avg_range = np.mean([bar.range for bar in recent])
    historical_avg_range = np.mean([bar.range for bar in historical])
    
    if historical_avg_range == 0:
        return 0.0
    
    compression = 1.0 - (recent_avg_range / historical_avg_range)
    return float(max(0.0, min(1.0, compression)))


def compute_volatility_compression(bars: List[OhlcvBar]) -> float:
    """
    Compute volatility compression using ATR comparison.
    """
    if len(bars) < 14:
        return 0.0
    
    true_ranges = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < 14:
        return 0.0
    
    recent_atr = np.mean(true_ranges[-7:])
    historical_atr = np.mean(true_ranges[:-7])
    
    if historical_atr == 0:
        return 0.0
    
    compression = 1.0 - (recent_atr / historical_atr)
    return float(max(0.0, min(1.0, compression)))


def compute_bollinger_squeeze(bars: List[OhlcvBar], period: int = 20) -> float:
    """
    Compute Bollinger Band width squeeze indicator.
    Lower values indicate tighter squeeze (more suppression).
    """
    if len(bars) < period:
        return 0.0
    
    closes = np.array([bar.close for bar in bars[-period:]])
    sma = np.mean(closes)
    std = np.std(closes)
    
    if sma == 0:
        return 0.0
    
    bb_width = (2 * std * 2) / sma
    
    historical_closes = [bar.close for bar in bars[:-period]] if len(bars) > period else closes
    historical_std = np.std(historical_closes) if len(historical_closes) > 1 else std
    
    if historical_std == 0:
        return 0.0
    
    squeeze = 1.0 - (std / historical_std)
    return float(max(0.0, min(1.0, squeeze)))


def compute_coil_factor(suppression_level: float, bars: List[OhlcvBar]) -> float:
    """
    Compute coil factor - the "spring tension" of the suppression.
    Higher values indicate greater stored energy for potential move.
    """
    if len(bars) < 10:
        return 0.0
    
    volumes = [bar.volume for bar in bars[-10:]]
    volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
    volume_building = 1.0 if volume_trend > 0 else 0.5
    
    closes = [bar.close for bar in bars[-10:]]
    price_coil = 1.0 - np.std(closes) / np.mean(closes) if np.mean(closes) > 0 else 0
    
    coil = suppression_level * volume_building * (1 + price_coil)
    return float(min(2.0, coil))


def compute_breakout_probability(suppression_level: float, coil_factor: float) -> float:
    """
    Estimate probability of breakout based on suppression metrics.
    This is a structural probability, NOT a trade signal.
    """
    if suppression_level < 0.3:
        return 0.1
    
    base_prob = suppression_level * 0.4
    coil_boost = coil_factor * 0.2
    
    probability = base_prob + coil_boost
    return float(max(0.0, min(1.0, probability)))


def determine_suppression_state(level: float) -> SuppressionState:
    """Determine suppression state from level."""
    if level < 0.2:
        return SuppressionState.NONE
    elif level < 0.4:
        return SuppressionState.LIGHT
    elif level < 0.6:
        return SuppressionState.MODERATE
    else:
        return SuppressionState.HEAVY


def compute_suppression(window: OhlcvWindow) -> SuppressionMetrics:
    """
    Compute all suppression metrics from an OHLCV window.
    """
    bars = window.bars
    
    range_comp = compute_range_compression(bars)
    vol_comp = compute_volatility_compression(bars)
    bb_squeeze = compute_bollinger_squeeze(bars)
    
    suppression_level = (range_comp * 0.4 + vol_comp * 0.35 + bb_squeeze * 0.25)
    
    coil_factor = compute_coil_factor(suppression_level, bars)
    breakout_probability = compute_breakout_probability(suppression_level, coil_factor)
    suppression_state = determine_suppression_state(suppression_level)
    
    return SuppressionMetrics(
        suppression_level=suppression_level,
        suppression_state=suppression_state,
        coil_factor=coil_factor,
        breakout_probability=breakout_probability,
    )
