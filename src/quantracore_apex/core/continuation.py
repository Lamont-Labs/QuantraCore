"""
Continuation analysis module for QuantraCore Apex.

Analyzes momentum and trend continuation probability.
"""

import numpy as np
from typing import List
from .schemas import OhlcvWindow, OhlcvBar, ContinuationMetrics


def compute_momentum_strength(bars: List[OhlcvBar]) -> float:
    """
    Compute current momentum strength.
    """
    if len(bars) < 10:
        return 0.5
    
    closes = np.array([bar.close for bar in bars])
    
    returns = np.diff(closes[-10:]) / closes[-10:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) == 0:
        return 0.5
    
    momentum = np.mean(returns) * 100
    
    strength = 0.5 + np.clip(momentum * 5, -0.5, 0.5)
    return float(strength)


def detect_exhaustion(bars: List[OhlcvBar]) -> bool:
    """
    Detect potential trend exhaustion signals.
    """
    if len(bars) < 10:
        return False
    
    recent = bars[-5:]
    prior = bars[-10:-5]
    
    recent_body_avg = np.mean([bar.body for bar in recent])
    prior_body_avg = np.mean([bar.body for bar in prior])
    
    if prior_body_avg == 0:
        return False
    
    body_shrinking = recent_body_avg < prior_body_avg * 0.6
    
    recent_wick_avg = np.mean([bar.upper_wick + bar.lower_wick for bar in recent])
    recent_range_avg = np.mean([bar.range for bar in recent])
    
    high_wicks = recent_wick_avg > recent_range_avg * 0.6 if recent_range_avg > 0 else False
    
    recent_vol = np.mean([bar.volume for bar in recent])
    prior_vol = np.mean([bar.volume for bar in prior])
    volume_declining = recent_vol < prior_vol * 0.7 if prior_vol > 0 else False
    
    exhaustion_signals = sum([body_shrinking, high_wicks, volume_declining])
    return exhaustion_signals >= 2


def compute_reversal_risk(bars: List[OhlcvBar], exhaustion: bool) -> float:
    """
    Compute risk of trend reversal.
    """
    if len(bars) < 20:
        return 0.3
    
    base_risk = 0.4 if exhaustion else 0.2
    
    closes = np.array([bar.close for bar in bars[-20:]])
    
    highest = np.max(closes)
    lowest = np.min(closes)
    current = closes[-1]
    
    if highest == lowest:
        return base_risk
    
    position = (current - lowest) / (highest - lowest)
    
    if position > 0.9:
        base_risk += 0.2
    elif position < 0.1:
        base_risk += 0.2
    
    changes = np.diff(closes)
    same_direction = all(c > 0 for c in changes[-5:]) or all(c < 0 for c in changes[-5:])
    if same_direction:
        base_risk += 0.15
    
    return float(min(1.0, base_risk))


def compute_continuation_probability(
    momentum_strength: float,
    exhaustion: bool,
    reversal_risk: float
) -> float:
    """
    Compute overall continuation probability.
    This is a structural probability, NOT a trade signal.
    """
    base = momentum_strength
    
    if exhaustion:
        base *= 0.6
    
    base *= (1 - reversal_risk * 0.5)
    
    return float(max(0.0, min(1.0, base)))


def compute_continuation(window: OhlcvWindow) -> ContinuationMetrics:
    """
    Compute all continuation metrics from an OHLCV window.
    """
    bars = window.bars
    
    momentum_strength = compute_momentum_strength(bars)
    exhaustion = detect_exhaustion(bars)
    reversal_risk = compute_reversal_risk(bars, exhaustion)
    continuation_prob = compute_continuation_probability(
        momentum_strength, exhaustion, reversal_risk
    )
    
    return ContinuationMetrics(
        continuation_probability=continuation_prob,
        momentum_strength=momentum_strength,
        exhaustion_signal=exhaustion,
        reversal_risk=reversal_risk,
    )
