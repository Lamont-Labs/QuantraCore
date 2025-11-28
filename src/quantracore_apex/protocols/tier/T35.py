"""
T35 - Momentum Divergence Protocol

Detects price-momentum divergences.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T35: Detect momentum divergences.
    
    Fires when price and momentum diverge significantly.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T35",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    def calc_rsi(g, l):
        ag = np.mean(g) if len(g) > 0 else 0
        al = np.mean(l) if len(l) > 0 else 0
        if al == 0:
            return 100
        return 100 - (100 / (1 + ag/al))
    
    rsi_current = calc_rsi(gains[-14:], losses[-14:])
    rsi_prev = calc_rsi(gains[-28:-14], losses[-28:-14])
    
    price_current = closes[-1]
    price_prev = closes[-15]
    
    price_higher = price_current > price_prev
    rsi_higher = rsi_current > rsi_prev
    
    price_lower = price_current < price_prev
    rsi_lower = rsi_current < rsi_prev
    
    bullish_divergence = price_lower and not rsi_lower
    bearish_divergence = price_higher and not rsi_higher
    
    price_change = (price_current - price_prev) / price_prev
    rsi_change = rsi_current - rsi_prev
    
    divergence_strength = abs(price_change * 100 - rsi_change / 10)
    
    if bullish_divergence and divergence_strength > 5:
        signal_type = "bullish_divergence"
        confidence = min(0.9, 0.5 + divergence_strength * 0.05)
        fired = True
    elif bearish_divergence and divergence_strength > 5:
        signal_type = "bearish_divergence"
        confidence = min(0.9, 0.5 + divergence_strength * 0.05)
        fired = True
    elif bullish_divergence or bearish_divergence:
        signal_type = "weak_divergence"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_divergence"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T35",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "rsi_current": float(rsi_current),
            "rsi_prev": float(rsi_prev),
            "price_change_pct": float(price_change * 100),
            "divergence_strength": float(divergence_strength),
        }
    )
