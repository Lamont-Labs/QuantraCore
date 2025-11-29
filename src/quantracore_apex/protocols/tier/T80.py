"""
T80 - Composite Signal Integration Protocol

Integrates signals from multiple protocols for overall assessment.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T80: Integrate composite signals.
    
    Fires when multiple analysis dimensions align.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T80",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    np.array([b.high for b in bars])
    np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    current_price = closes[-1]
    
    trend_bullish = current_price > sma20 > sma50
    trend_bearish = current_price < sma20 < sma50
    
    momentum = (closes[-1] - closes[-10]) / closes[-10] * 100
    
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns[-20:]) * np.sqrt(252)
    vol_normal = volatility < 0.4
    
    volume_increasing = np.mean(volumes[-5:]) > np.mean(volumes[-20:])
    
    signals = {
        "trend": 1 if trend_bullish else (-1 if trend_bearish else 0),
        "momentum": 1 if momentum > 3 else (-1 if momentum < -3 else 0),
        "volatility": 1 if vol_normal else 0,
        "volume": 1 if volume_increasing else 0,
    }
    
    bullish_count = sum(1 for v in signals.values() if v > 0)
    bearish_count = sum(1 for v in signals.values() if v < 0)
    
    alignment_score = max(bullish_count, bearish_count) / 4
    
    if bullish_count >= 3:
        signal_type = "strongly_bullish_composite"
        confidence = 0.85
        fired = True
    elif bearish_count >= 3:
        signal_type = "strongly_bearish_composite"
        confidence = 0.85
        fired = True
    elif bullish_count >= 2 and bearish_count == 0:
        signal_type = "bullish_composite"
        confidence = 0.7
        fired = True
    elif bearish_count >= 2 and bullish_count == 0:
        signal_type = "bearish_composite"
        confidence = 0.7
        fired = True
    elif bullish_count == bearish_count and bullish_count > 0:
        signal_type = "mixed_signals"
        confidence = 0.4
        fired = False
    else:
        signal_type = "neutral_composite"
        confidence = 0.35
        fired = False
    
    return ProtocolResult(
        protocol_id="T80",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "trend_signal": signals["trend"],
            "momentum_signal": signals["momentum"],
            "volatility_signal": signals["volatility"],
            "volume_signal": signals["volume"],
            "bullish_count": int(bullish_count),
            "bearish_count": int(bearish_count),
            "alignment_score": float(alignment_score),
        }
    )
