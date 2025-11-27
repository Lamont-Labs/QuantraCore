"""
Volume spike detection module for QuantraCore Apex.

Detects and analyzes significant volume events.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, VolumeMetrics


def detect_volume_spike(bars: List[OhlcvBar], threshold: float = 2.0) -> tuple:
    """
    Detect if current volume represents a spike.
    Returns (is_spike, magnitude).
    """
    if len(bars) < 20:
        return False, 1.0
    
    volumes = np.array([bar.volume for bar in bars])
    current_vol = volumes[-1]
    avg_vol = np.mean(volumes[-20:-1])
    
    if avg_vol == 0:
        return False, 1.0
    
    magnitude = current_vol / avg_vol
    is_spike = magnitude >= threshold
    
    return is_spike, float(magnitude)


def compute_volume_trend(bars: List[OhlcvBar]) -> str:
    """
    Determine volume trend direction.
    """
    if len(bars) < 10:
        return "stable"
    
    volumes = np.array([bar.volume for bar in bars[-10:]])
    
    x = np.arange(len(volumes))
    slope = np.polyfit(x, volumes, 1)[0]
    
    avg_vol = np.mean(volumes)
    if avg_vol == 0:
        return "stable"
    
    normalized_slope = slope / avg_vol
    
    if normalized_slope > 0.05:
        return "increasing"
    elif normalized_slope < -0.05:
        return "decreasing"
    else:
        return "stable"


def compute_relative_volume(bars: List[OhlcvBar]) -> float:
    """
    Compute relative volume compared to historical average.
    """
    if len(bars) < 20:
        return 1.0
    
    current_vol = bars[-1].volume
    avg_vol = np.mean([bar.volume for bar in bars[-20:-1]])
    
    if avg_vol == 0:
        return 1.0
    
    return float(current_vol / avg_vol)


def compute_volume_metrics(window: OhlcvWindow) -> VolumeMetrics:
    """
    Compute all volume metrics from an OHLCV window.
    """
    bars = window.bars
    
    spike_detected, spike_magnitude = detect_volume_spike(bars)
    volume_trend = compute_volume_trend(bars)
    relative_volume = compute_relative_volume(bars)
    
    return VolumeMetrics(
        volume_spike_detected=spike_detected,
        spike_magnitude=spike_magnitude,
        volume_trend=volume_trend,
        relative_volume=relative_volume,
    )
