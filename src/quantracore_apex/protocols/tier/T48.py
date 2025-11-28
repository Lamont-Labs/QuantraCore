"""
T48 - Volume Profile Analysis Protocol

Analyzes volume distribution across price levels.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T48: Analyze volume profile.
    
    Fires when price is at significant volume nodes.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T48",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    price_min = np.min(lows)
    price_max = np.max(highs)
    price_range = price_max - price_min
    
    num_levels = 20
    level_size = price_range / num_levels
    
    volume_profile = np.zeros(num_levels)
    for i in range(len(bars)):
        level = int((closes[i] - price_min) / max(level_size, 0.0001))
        level = min(level, num_levels - 1)
        volume_profile[level] += volumes[i]
    
    poc_level = np.argmax(volume_profile)
    poc_price = price_min + (poc_level + 0.5) * level_size
    
    current_price = closes[-1]
    current_level = int((current_price - price_min) / max(level_size, 0.0001))
    current_level = min(current_level, num_levels - 1)
    
    current_volume_node = volume_profile[current_level]
    avg_volume_node = np.mean(volume_profile)
    
    high_volume_node = current_volume_node > avg_volume_node * 1.5
    low_volume_node = current_volume_node < avg_volume_node * 0.5
    
    distance_to_poc = abs(current_price - poc_price) / current_price * 100
    
    if high_volume_node and distance_to_poc < 1:
        signal_type = "at_poc"
        confidence = 0.85
        fired = True
    elif high_volume_node:
        signal_type = "high_volume_node"
        confidence = 0.75
        fired = True
    elif low_volume_node:
        signal_type = "low_volume_node"
        confidence = 0.7
        fired = True
    elif distance_to_poc > 3:
        signal_type = "extended_from_poc"
        confidence = 0.65
        fired = True
    else:
        signal_type = "normal_profile_position"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T48",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "poc_price": float(poc_price),
            "distance_to_poc_pct": float(distance_to_poc),
            "current_volume_node": float(current_volume_node),
            "avg_volume_node": float(avg_volume_node),
        }
    )
