"""
T46 - VWAP Analysis Protocol

Analyzes price position relative to VWAP.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T46: Analyze VWAP position.
    
    Fires when price has significant deviation from VWAP.
    """
    bars = window.bars
    if len(bars) < 15:
        return ProtocolResult(
            protocol_id="T46",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    typical_prices = np.array([(b.high + b.low + b.close) / 3 for b in bars])
    volumes = np.array([b.volume for b in bars])
    closes = np.array([b.close for b in bars])
    
    cumulative_tp_vol = np.cumsum(typical_prices * volumes)
    cumulative_vol = np.cumsum(volumes)
    
    vwap = cumulative_tp_vol / np.maximum(cumulative_vol, 1)
    
    current_price = closes[-1]
    current_vwap = vwap[-1]
    
    deviation = (current_price - current_vwap) / current_vwap * 100
    
    price_vol_product = typical_prices * volumes
    vwap_std = np.std(price_vol_product[-20:] / np.maximum(volumes[-20:], 1))
    
    std_deviation = (current_price - current_vwap) / max(vwap_std, 0.0001)
    
    above_vwap_streak = 0
    for i in range(-1, -min(10, len(closes)), -1):
        if closes[i] > vwap[i]:
            above_vwap_streak += 1
        else:
            break
    
    below_vwap_streak = 0
    for i in range(-1, -min(10, len(closes)), -1):
        if closes[i] < vwap[i]:
            below_vwap_streak += 1
        else:
            break
    
    if deviation > 2.0 and above_vwap_streak >= 5:
        signal_type = "strong_above_vwap"
        confidence = 0.85
        fired = True
    elif deviation < -2.0 and below_vwap_streak >= 5:
        signal_type = "strong_below_vwap"
        confidence = 0.85
        fired = True
    elif abs(std_deviation) > 2:
        signal_type = "extended_from_vwap"
        confidence = 0.7
        fired = True
    elif abs(deviation) < 0.5:
        signal_type = "at_vwap"
        confidence = 0.6
        fired = True
    else:
        signal_type = "normal_vwap_position"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T46",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "vwap": float(current_vwap),
            "deviation_pct": float(deviation),
            "above_vwap_streak": int(above_vwap_streak),
            "below_vwap_streak": int(below_vwap_streak),
        }
    )
