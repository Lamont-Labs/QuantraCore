"""
T77 - Mean Reversion Probability Protocol

Calculates probability of mean reversion.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T77: Calculate mean reversion probability.
    
    Fires when conditions favor mean reversion.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T77",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    
    current_price = closes[-1]
    
    deviation_sma20 = (current_price - sma20) / sma20 * 100
    deviation_sma50 = (current_price - sma50) / sma50 * 100
    
    returns = np.diff(closes) / closes[:-1]
    vol = np.std(returns[-20:])
    zscore = (current_price - sma20) / (sma20 * vol) if vol > 0 else 0
    
    historical_reversions = 0
    for i in range(20, len(closes) - 5):
        if abs(closes[i] - np.mean(closes[i-20:i])) / np.mean(closes[i-20:i]) > 0.05:
            future_return = (closes[i+5] - closes[i]) / closes[i]
            deviation = (closes[i] - np.mean(closes[i-20:i])) / np.mean(closes[i-20:i])
            if deviation * future_return < 0:
                historical_reversions += 1
    
    reversion_rate = historical_reversions / max(len(closes) - 25, 1)
    
    reversion_score = abs(zscore) * 0.3 + reversion_rate * 0.4 + abs(deviation_sma20) * 0.02
    
    if abs(zscore) > 2.5 and reversion_rate > 0.5:
        signal_type = "high_reversion_probability"
        confidence = min(0.9, 0.5 + reversion_score * 0.2)
        fired = True
    elif abs(zscore) > 2:
        signal_type = "elevated_reversion_probability"
        confidence = 0.75
        fired = True
    elif abs(zscore) > 1.5:
        signal_type = "moderate_reversion_probability"
        confidence = 0.6
        fired = True
    else:
        signal_type = "low_reversion_probability"
        confidence = 0.35
        fired = False
    
    return ProtocolResult(
        protocol_id="T77",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "deviation_sma20_pct": float(deviation_sma20),
            "deviation_sma50_pct": float(deviation_sma50),
            "zscore": float(zscore),
            "reversion_rate": float(reversion_rate),
            "reversion_score": float(reversion_score),
        }
    )
