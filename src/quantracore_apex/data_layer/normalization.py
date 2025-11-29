"""
Data Normalization module for QuantraCore Apex.

Handles timestamp normalization, scaling, and data cleaning.
"""

import numpy as np
from typing import List
from datetime import timezone
from copy import deepcopy

from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


def normalize_timestamps(bars: List[OhlcvBar]) -> List[OhlcvBar]:
    """
    Normalize timestamps to UTC.
    """
    normalized = []
    for bar in bars:
        new_bar = deepcopy(bar)
        if bar.timestamp.tzinfo is None:
            new_bar.timestamp = bar.timestamp.replace(tzinfo=timezone.utc)
        normalized.append(new_bar)
    return normalized


def remove_zero_volume(bars: List[OhlcvBar]) -> List[OhlcvBar]:
    """
    Remove bars with zero or negative volume.
    """
    return [bar for bar in bars if bar.volume > 0]


def remove_invalid_prices(bars: List[OhlcvBar]) -> List[OhlcvBar]:
    """
    Remove bars with invalid OHLC relationships.
    """
    valid = []
    for bar in bars:
        if bar.low <= bar.open <= bar.high and \
           bar.low <= bar.close <= bar.high and \
           bar.low > 0:
            valid.append(bar)
    return valid


def compute_zscore_normalized_prices(bars: List[OhlcvBar], window: int = 20) -> List[float]:
    """
    Compute Z-score normalized closing prices.
    """
    if len(bars) < window:
        return [0.0] * len(bars)
    
    closes = np.array([bar.close for bar in bars])
    zscores = []
    
    for i in range(len(closes)):
        if i < window:
            zscores.append(0.0)
        else:
            window_data = closes[i-window:i]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                zscore = (closes[i] - mean) / std
            else:
                zscore = 0.0
            zscores.append(float(zscore))
    
    return zscores


def apply_volatility_scaling(bars: List[OhlcvBar]) -> List[OhlcvBar]:
    """
    Apply volatility-aware scaling to normalize range comparisons.
    """
    if len(bars) < 20:
        return bars
    
    ranges = np.array([bar.range for bar in bars])
    avg_range = np.mean(ranges)
    
    if avg_range == 0:
        return bars
    
    scaled = []
    for bar in bars:
        new_bar = deepcopy(bar)
        scaled.append(new_bar)
    
    return scaled


def normalize_ohlcv(
    bars: List[OhlcvBar],
    remove_invalid: bool = True,
    zscore_window: int = 20
) -> tuple:
    """
    Full normalization pipeline for OHLCV data.
    
    Args:
        bars: Raw OHLCV bars
        remove_invalid: Whether to remove invalid bars
        zscore_window: Window for Z-score calculation
        
    Returns:
        Tuple of (normalized_bars, zscore_prices)
    """
    normalized = normalize_timestamps(bars)
    
    if remove_invalid:
        normalized = remove_zero_volume(normalized)
        normalized = remove_invalid_prices(normalized)
    
    normalized = apply_volatility_scaling(normalized)
    
    zscores = compute_zscore_normalized_prices(normalized, zscore_window)
    
    return normalized, zscores


def build_windows(
    bars: List[OhlcvBar],
    symbol: str,
    timeframe: str = "1d",
    window_size: int = 100,
    step: int = 1
) -> List[OhlcvWindow]:
    """
    Build rolling windows from OHLCV bars.
    
    Args:
        bars: Normalized OHLCV bars
        symbol: Ticker symbol
        timeframe: Bar timeframe
        window_size: Number of bars per window (default 100)
        step: Step size between windows
        
    Returns:
        List of OhlcvWindow objects
    """
    windows = []
    
    for i in range(0, len(bars) - window_size + 1, step):
        window_bars = bars[i:i + window_size]
        window = OhlcvWindow(
            symbol=symbol,
            timeframe=timeframe,
            bars=window_bars
        )
        windows.append(window)
    
    return windows
