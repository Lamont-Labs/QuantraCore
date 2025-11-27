"""
Drift detection module for QuantraCore Apex.

Drift measures deviation from expected behavior patterns,
helping identify regime shifts and mean reversion opportunities.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, DriftMetrics, DriftState


def compute_price_drift(bars: List[OhlcvBar], lookback: int = 50) -> tuple:
    """
    Compute price drift from moving average.
    Returns (magnitude, direction).
    """
    if len(bars) < lookback:
        return 0.0, 0.0
    
    closes = np.array([bar.close for bar in bars])
    current_price = closes[-1]
    
    sma = np.mean(closes[-lookback:])
    
    if sma == 0:
        return 0.0, 0.0
    
    drift_pct = (current_price - sma) / sma
    magnitude = abs(drift_pct)
    direction = np.sign(drift_pct)
    
    return float(magnitude), float(direction)


def compute_momentum_drift(bars: List[OhlcvBar]) -> float:
    """
    Compute momentum drift - deviation from expected momentum trajectory.
    """
    if len(bars) < 20:
        return 0.0
    
    closes = np.array([bar.close for bar in bars])
    
    short_momentum = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
    long_momentum = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] != 0 else 0
    
    if long_momentum == 0:
        return 0.0
    
    expected_short = long_momentum * (5 / 20)
    drift = abs(short_momentum - expected_short)
    
    return float(min(1.0, drift * 10))


def compute_volatility_drift(bars: List[OhlcvBar]) -> float:
    """
    Compute volatility drift - deviation from expected volatility.
    """
    if len(bars) < 30:
        return 0.0
    
    ranges = np.array([bar.range for bar in bars])
    
    recent_vol = np.std(ranges[-10:])
    historical_vol = np.std(ranges[-30:-10])
    
    if historical_vol == 0:
        return 0.0
    
    drift = abs(recent_vol - historical_vol) / historical_vol
    return float(min(1.0, drift))


def compute_mean_reversion_pressure(drift_magnitude: float, bars: List[OhlcvBar]) -> float:
    """
    Compute mean reversion pressure based on drift and RSI.
    Higher values indicate stronger pull back toward mean.
    """
    if len(bars) < 14:
        return 0.0
    
    closes = np.array([bar.close for bar in bars[-15:]])
    changes = np.diff(closes)
    
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    if rsi > 70:
        rsi_pressure = (rsi - 70) / 30
    elif rsi < 30:
        rsi_pressure = (30 - rsi) / 30
    else:
        rsi_pressure = 0
    
    pressure = (drift_magnitude * 0.5 + rsi_pressure * 0.5)
    return float(min(1.0, pressure))


def determine_drift_state(magnitude: float) -> DriftState:
    """Determine drift state from magnitude."""
    if magnitude < 0.02:
        return DriftState.NONE
    elif magnitude < 0.05:
        return DriftState.MILD
    elif magnitude < 0.10:
        return DriftState.SIGNIFICANT
    else:
        return DriftState.CRITICAL


def compute_drift(window: OhlcvWindow) -> DriftMetrics:
    """
    Compute all drift metrics from an OHLCV window.
    """
    bars = window.bars
    
    price_drift_mag, price_drift_dir = compute_price_drift(bars)
    momentum_drift = compute_momentum_drift(bars)
    volatility_drift = compute_volatility_drift(bars)
    
    combined_magnitude = (
        price_drift_mag * 0.5 +
        momentum_drift * 0.3 +
        volatility_drift * 0.2
    )
    
    mean_reversion_pressure = compute_mean_reversion_pressure(combined_magnitude, bars)
    drift_state = determine_drift_state(combined_magnitude)
    
    return DriftMetrics(
        drift_magnitude=combined_magnitude,
        drift_direction=price_drift_dir,
        drift_state=drift_state,
        mean_reversion_pressure=mean_reversion_pressure,
    )
