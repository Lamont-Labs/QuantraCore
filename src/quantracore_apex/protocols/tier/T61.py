"""
T61 - Key Support Level Detection Protocol

Detects significant support levels from price action.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T61: Detect key support levels.
    
    Fires when price approaches or tests significant support.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T61",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    np.array([b.volume for b in bars])
    
    tolerance = 0.02
    support_levels = []
    
    for i in range(3, len(bars) - 1):
        if lows[i] <= min(lows[i-3:i]) and lows[i] <= min(lows[i+1:min(i+4, len(lows))]):
            support_levels.append((i, lows[i]))
    
    support_clusters = []
    for idx, level in support_levels:
        found_cluster = False
        for cluster in support_clusters:
            if abs(level - cluster["price"]) / level < tolerance:
                cluster["touches"] += 1
                cluster["indices"].append(idx)
                cluster["price"] = np.mean([level] + [lvl for _, lvl in support_levels if abs(lvl - cluster["price"]) / lvl < tolerance])
                found_cluster = True
                break
        if not found_cluster:
            support_clusters.append({"price": level, "touches": 1, "indices": [idx]})
    
    current_price = closes[-1]
    
    nearest_support = None
    min_distance = float('inf')
    
    for cluster in support_clusters:
        if cluster["price"] < current_price:
            distance = (current_price - cluster["price"]) / current_price
            if distance < min_distance:
                min_distance = distance
                nearest_support = cluster
    
    if nearest_support and min_distance < 0.02:
        signal_type = "at_support"
        confidence = min(0.9, 0.6 + nearest_support["touches"] * 0.1)
        fired = True
    elif nearest_support and min_distance < 0.05:
        signal_type = "approaching_support"
        confidence = 0.7
        fired = True
    elif nearest_support and nearest_support["touches"] >= 3:
        signal_type = "strong_support_below"
        confidence = 0.65
        fired = True
    else:
        signal_type = "no_significant_support"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T61",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "nearest_support": float(nearest_support["price"]) if nearest_support else 0,
            "support_touches": int(nearest_support["touches"]) if nearest_support else 0,
            "distance_pct": float(min_distance * 100),
            "total_support_levels": len(support_clusters),
        }
    )
