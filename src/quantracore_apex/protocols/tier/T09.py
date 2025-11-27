"""
T09 - Range Compression Protocol

Detects price range compression patterns.
Category: Volatility Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T09: Detect range compression.
    
    Fires when consecutive ranges narrow significantly.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T09",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    ranges = [b.range for b in bars]
    
    narrowing_count = 0
    for i in range(-1, -min(10, len(ranges)), -1):
        if ranges[i] < ranges[i-1]:
            narrowing_count += 1
    
    recent_range = np.mean(ranges[-5:])
    prior_range = np.mean(ranges[-15:-5])
    
    if prior_range == 0:
        compression_pct = 0
    else:
        compression_pct = 1 - (recent_range / prior_range)
    
    if narrowing_count >= 5 and compression_pct > 0.3:
        signal_type = "strong_range_compression"
        confidence = min(0.9, 0.5 + compression_pct * 0.5)
        fired = True
    elif compression_pct > 0.2:
        signal_type = "moderate_range_compression"
        confidence = 0.45
        fired = True
    else:
        signal_type = "no_compression"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T09",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "narrowing_bars": narrowing_count,
            "compression_pct": float(compression_pct),
            "microtraits_compression": float(microtraits.compression_score),
        }
    )
