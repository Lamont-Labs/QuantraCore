"""
T39 - Momentum Exhaustion Protocol

Detects momentum exhaustion patterns.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T39: Detect momentum exhaustion.
    
    Fires when momentum shows signs of exhaustion.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T39",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    price_trend = (closes[-1] - closes[-10]) / closes[-10] * 100
    
    recent_momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
    prior_momentum = (closes[-5] - closes[-10]) / closes[-10] * 100
    
    momentum_deceleration = abs(prior_momentum) - abs(recent_momentum)
    
    recent_vol_avg = np.mean(volumes[-5:])
    prior_vol_avg = np.mean(volumes[-15:-5])
    volume_decline = (prior_vol_avg - recent_vol_avg) / max(prior_vol_avg, 1) * 100
    
    range_recent = (np.max(highs[-5:]) - np.min(lows[-5:])) / closes[-1] * 100
    range_prior = (np.max(highs[-15:-5]) - np.min(lows[-15:-5])) / closes[-10] * 100
    range_contraction = range_prior - range_recent
    
    exhaustion_score = 0
    if momentum_deceleration > 0:
        exhaustion_score += momentum_deceleration * 2
    if volume_decline > 20:
        exhaustion_score += volume_decline * 0.5
    if range_contraction > 0:
        exhaustion_score += range_contraction * 3
    
    if exhaustion_score > 30:
        signal_type = "strong_exhaustion"
        confidence = min(0.9, 0.5 + exhaustion_score * 0.01)
        fired = True
    elif exhaustion_score > 15:
        signal_type = "moderate_exhaustion"
        confidence = 0.65
        fired = True
    elif exhaustion_score > 5:
        signal_type = "early_exhaustion"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_exhaustion"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T39",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "price_trend_pct": float(price_trend),
            "momentum_deceleration": float(momentum_deceleration),
            "volume_decline_pct": float(volume_decline),
            "exhaustion_score": float(exhaustion_score),
        }
    )
