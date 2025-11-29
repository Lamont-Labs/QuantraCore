"""
T62 - Key Resistance Level Detection Protocol

Detects significant resistance levels from price action.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T62: Detect key resistance levels.
    
    Fires when price approaches or tests significant resistance.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T62",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    closes = np.array([b.close for b in bars])
    
    tolerance = 0.02
    resistance_levels = []
    
    for i in range(3, len(bars) - 1):
        if highs[i] >= max(highs[i-3:i]) and highs[i] >= max(highs[i+1:min(i+4, len(highs))]):
            resistance_levels.append((i, highs[i]))
    
    resistance_clusters: list = []
    for idx, level in resistance_levels:
        found_cluster = False
        for cluster in resistance_clusters:
            if abs(level - cluster["price"]) / level < tolerance:
                cluster["touches"] += 1
                cluster["indices"].append(idx)
                found_cluster = True
                break
        if not found_cluster:
            resistance_clusters.append({"price": level, "touches": 1, "indices": [idx]})
    
    current_price = closes[-1]
    
    nearest_resistance = None
    min_distance = float('inf')
    
    for cluster in resistance_clusters:
        if cluster["price"] > current_price:
            distance = (cluster["price"] - current_price) / current_price
            if distance < min_distance:
                min_distance = distance
                nearest_resistance = cluster
    
    if nearest_resistance and min_distance < 0.02:
        signal_type = "at_resistance"
        confidence = min(0.9, 0.6 + nearest_resistance["touches"] * 0.1)
        fired = True
    elif nearest_resistance and min_distance < 0.05:
        signal_type = "approaching_resistance"
        confidence = 0.7
        fired = True
    elif nearest_resistance and nearest_resistance["touches"] >= 3:
        signal_type = "strong_resistance_above"
        confidence = 0.65
        fired = True
    else:
        signal_type = "no_significant_resistance"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T62",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "nearest_resistance": float(nearest_resistance["price"]) if nearest_resistance else 0,
            "resistance_touches": int(nearest_resistance["touches"]) if nearest_resistance else 0,
            "distance_pct": float(min_distance * 100),
            "total_resistance_levels": len(resistance_clusters),
        }
    )
