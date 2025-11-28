"""
T43 - Volume Price Confirmation Protocol

Analyzes volume-price relationship for trend confirmation.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T43: Analyze volume-price confirmation.
    
    Fires when volume confirms or contradicts price movement.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T43",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    price_up_days = closes[1:] > closes[:-1]
    price_down_days = closes[1:] < closes[:-1]
    
    up_volume = np.sum(volumes[1:][price_up_days])
    down_volume = np.sum(volumes[1:][price_down_days])
    
    volume_ratio = up_volume / max(down_volume, 1)
    
    recent_price_trend = (closes[-1] - closes[-10]) / closes[-10] * 100
    recent_volume_trend = (np.mean(volumes[-5:]) - np.mean(volumes[-15:-5])) / max(np.mean(volumes[-15:-5]), 1) * 100
    
    price_up = recent_price_trend > 2
    price_down = recent_price_trend < -2
    volume_up = recent_volume_trend > 20
    volume_down = recent_volume_trend < -20
    
    if price_up and volume_up:
        signal_type = "bullish_confirmation"
        confidence = 0.85
        fired = True
    elif price_down and volume_up:
        signal_type = "bearish_confirmation"
        confidence = 0.85
        fired = True
    elif price_up and volume_down:
        signal_type = "weak_rally"
        confidence = 0.7
        fired = True
    elif price_down and volume_down:
        signal_type = "weak_decline"
        confidence = 0.7
        fired = True
    elif volume_ratio > 1.5:
        signal_type = "buying_pressure"
        confidence = 0.65
        fired = True
    elif volume_ratio < 0.67:
        signal_type = "selling_pressure"
        confidence = 0.65
        fired = True
    else:
        signal_type = "no_confirmation"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T43",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "up_volume": float(up_volume),
            "down_volume": float(down_volume),
            "volume_ratio": float(volume_ratio),
            "recent_price_trend": float(recent_price_trend),
            "recent_volume_trend": float(recent_volume_trend),
        }
    )
