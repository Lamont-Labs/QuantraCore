"""
Microtrait computation module for QuantraCore Apex.

Microtraits are fine-grained structural features extracted from OHLCV windows
that form the foundation of deterministic analysis.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, Microtraits


def compute_wick_ratio(bars: List[OhlcvBar]) -> float:
    """Compute average wick-to-range ratio across bars."""
    if not bars:
        return 0.0
    
    ratios = []
    for bar in bars:
        if bar.range > 0:
            total_wick = max(0, bar.upper_wick) + max(0, bar.lower_wick)
            ratio = total_wick / bar.range
            ratios.append(max(0.0, min(1.0, ratio)))
        else:
            ratios.append(0.0)
    
    result = float(np.mean(ratios)) if ratios else 0.0
    return max(0.0, min(1.0, result))


def compute_body_ratio(bars: List[OhlcvBar]) -> float:
    """Compute average body-to-range ratio across bars."""
    if not bars:
        return 0.0
    
    ratios = []
    for bar in bars:
        if bar.range > 0:
            ratio = abs(bar.body) / bar.range
            ratios.append(max(0.0, min(1.0, ratio)))
        else:
            ratios.append(0.0)
    
    result = float(np.mean(ratios)) if ratios else 0.0
    return max(0.0, min(1.0, result))


def compute_bullish_pct(bars: List[OhlcvBar], lookback: int = 20) -> float:
    """Compute percentage of bullish bars in recent lookback period."""
    recent = bars[-lookback:] if len(bars) >= lookback else bars
    if not recent:
        return 0.5
    
    bullish_count = sum(1 for bar in recent if bar.is_bullish)
    return bullish_count / len(recent)


def compute_compression_score(bars: List[OhlcvBar]) -> float:
    """
    Compute compression score based on range contraction.
    Higher score = more compressed (coiled) structure.
    """
    if len(bars) < 10:
        return 0.0
    
    ranges = [bar.range for bar in bars]
    recent_ranges = ranges[-10:]
    historical_ranges = ranges[:-10] if len(ranges) > 10 else ranges
    
    if not historical_ranges or np.mean(historical_ranges) == 0:
        return 0.0
    
    compression = 1.0 - (np.mean(recent_ranges) / np.mean(historical_ranges))
    return float(max(0.0, min(1.0, compression)))


def compute_noise_score(bars: List[OhlcvBar]) -> float:
    """
    Compute noise score based on price action irregularity.
    Higher score = more chaotic/noisy price action.
    """
    if len(bars) < 5:
        return 0.0
    
    closes = [bar.close for bar in bars]
    changes = np.diff(closes)
    
    if len(changes) < 2:
        return 0.0
    
    direction_changes = np.sum(np.diff(np.sign(changes)) != 0)
    max_changes = len(changes) - 1
    
    if max_changes == 0:
        return 0.0
    
    return float(direction_changes / max_changes)


def compute_strength_slope(bars: List[OhlcvBar]) -> float:
    """
    Compute strength slope (trend direction and magnitude).
    Positive = uptrend, Negative = downtrend.
    """
    if len(bars) < 2:
        return 0.0
    
    closes = np.array([bar.close for bar in bars])
    x = np.arange(len(closes))
    
    if np.std(closes) == 0:
        return 0.0
    
    slope = np.polyfit(x, closes, 1)[0]
    normalized_slope = slope / np.mean(closes) * 100
    
    return float(np.clip(normalized_slope, -10, 10))


def compute_range_density(bars: List[OhlcvBar]) -> float:
    """
    Compute range density (how much of total range is utilized).
    """
    if len(bars) < 2:
        return 0.0
    
    total_range = max(bar.high for bar in bars) - min(bar.low for bar in bars)
    if total_range <= 0:
        return 0.0
    
    sum_bar_ranges = sum(max(0, bar.range) for bar in bars)
    density = sum_bar_ranges / (total_range * len(bars))
    
    return float(max(0.0, min(1.0, density)))


def compute_volume_intensity(bars: List[OhlcvBar]) -> float:
    """
    Compute volume intensity relative to historical average.
    """
    if len(bars) < 10:
        return 1.0
    
    volumes = [bar.volume for bar in bars]
    recent_vol = np.mean(volumes[-10:])
    historical_vol = np.mean(volumes[:-10]) if len(volumes) > 10 else np.mean(volumes)
    
    if historical_vol == 0:
        return 1.0
    
    return float(recent_vol / historical_vol)


def compute_trend_consistency(bars: List[OhlcvBar]) -> float:
    """
    Compute trend consistency score.
    +1 = perfectly consistent uptrend
    -1 = perfectly consistent downtrend
    0 = no trend / mixed
    """
    if len(bars) < 5:
        return 0.0
    
    closes = [bar.close for bar in bars]
    changes = np.diff(closes)
    
    if len(changes) == 0:
        return 0.0
    
    positive_changes = np.sum(changes > 0)
    negative_changes = np.sum(changes < 0)
    total_changes = len(changes)
    
    if total_changes == 0:
        return 0.0
    
    consistency = (positive_changes - negative_changes) / total_changes
    return float(consistency)


def compute_volatility_ratio(bars: List[OhlcvBar]) -> float:
    """
    Compute ratio of recent volatility to historical volatility.
    """
    if len(bars) < 20:
        return 1.0
    
    ranges = [bar.range for bar in bars]
    recent_vol = np.std(ranges[-10:])
    historical_vol = np.std(ranges[:-10]) if len(ranges) > 10 else np.std(ranges)
    
    if historical_vol == 0:
        return 1.0
    
    return float(recent_vol / historical_vol)


def compute_microtraits(window: OhlcvWindow) -> Microtraits:
    """
    Compute all microtraits from an OHLCV window.
    
    This is the main entry point for microtrait extraction.
    All computations are deterministic given the same input.
    Values are clamped to valid bounds to handle edge cases.
    """
    bars = window.bars
    
    def clamp_01(val: float) -> float:
        return max(0.0, min(1.0, float(val)))
    
    def clamp_ge0(val: float) -> float:
        return max(0.0, float(val))
    
    return Microtraits(
        wick_ratio=clamp_01(compute_wick_ratio(bars)),
        body_ratio=clamp_01(compute_body_ratio(bars)),
        bullish_pct_last20=clamp_01(compute_bullish_pct(bars, lookback=20)),
        compression_score=clamp_01(compute_compression_score(bars)),
        noise_score=clamp_01(compute_noise_score(bars)),
        strength_slope=float(np.clip(compute_strength_slope(bars), -10, 10)),
        range_density=clamp_01(compute_range_density(bars)),
        volume_intensity=clamp_ge0(compute_volume_intensity(bars)),
        trend_consistency=float(np.clip(compute_trend_consistency(bars), -1, 1)),
        volatility_ratio=clamp_ge0(compute_volatility_ratio(bars)),
    )
