"""
Entropy computation module for QuantraCore Apex.

Entropy measures the disorder/randomness in price action,
helping identify regime transitions and structural instability.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, EntropyMetrics, EntropyState


def compute_shannon_entropy(values: np.ndarray, bins: int = 10) -> float:
    """
    Compute Shannon entropy of a distribution.
    Higher values indicate more disorder/randomness.
    """
    if len(values) < 2:
        return 0.0
    
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    
    if len(hist) == 0:
        return 0.0
    
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    max_entropy = np.log2(bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return float(normalized_entropy)


def compute_price_entropy(bars: List[OhlcvBar]) -> float:
    """
    Compute entropy of price changes.
    Measures the unpredictability of price movements.
    """
    if len(bars) < 5:
        return 0.0
    
    closes = np.array([bar.close for bar in bars])
    returns = np.diff(closes) / closes[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) < 3:
        return 0.0
    
    return compute_shannon_entropy(returns, bins=min(10, len(returns) // 2))


def compute_volume_entropy(bars: List[OhlcvBar]) -> float:
    """
    Compute entropy of volume distribution.
    Measures the irregularity of volume patterns.
    """
    if len(bars) < 5:
        return 0.0
    
    volumes = np.array([bar.volume for bar in bars])
    if np.std(volumes) == 0:
        return 0.0
    
    return compute_shannon_entropy(volumes, bins=min(10, len(volumes) // 2))


def compute_entropy_floor(bars: List[OhlcvBar]) -> float:
    """
    Compute the baseline entropy floor.
    This represents the minimum expected entropy for the asset.
    """
    if len(bars) < 20:
        return 0.0
    
    window_size = min(20, len(bars) // 3)
    entropies = []
    
    for i in range(len(bars) - window_size):
        window = bars[i:i + window_size]
        closes = np.array([bar.close for bar in window])
        returns = np.diff(closes) / closes[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        if len(returns) >= 3:
            entropies.append(compute_shannon_entropy(returns, bins=5))
    
    if not entropies:
        return 0.0
    
    return float(np.percentile(entropies, 10))


def determine_entropy_state(combined_entropy: float, entropy_floor: float) -> EntropyState:
    """
    Determine the entropy state based on current vs floor entropy.
    """
    if entropy_floor == 0:
        entropy_floor = 0.3
    
    ratio = combined_entropy / entropy_floor if entropy_floor > 0 else combined_entropy
    
    if ratio < 1.2:
        return EntropyState.STABLE
    elif ratio < 2.0:
        return EntropyState.ELEVATED
    else:
        return EntropyState.CHAOTIC


def compute_entropy(window: OhlcvWindow) -> EntropyMetrics:
    """
    Compute all entropy metrics from an OHLCV window.
    
    Returns:
        EntropyMetrics with price, volume, combined entropy and state.
    """
    bars = window.bars
    
    price_entropy = compute_price_entropy(bars)
    volume_entropy = compute_volume_entropy(bars)
    
    combined_entropy = (price_entropy * 0.7 + volume_entropy * 0.3)
    
    entropy_floor = compute_entropy_floor(bars)
    entropy_state = determine_entropy_state(combined_entropy, entropy_floor)
    
    return EntropyMetrics(
        price_entropy=price_entropy,
        volume_entropy=volume_entropy,
        combined_entropy=combined_entropy,
        entropy_state=entropy_state,
        entropy_floor=entropy_floor,
    )
