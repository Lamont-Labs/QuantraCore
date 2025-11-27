"""
T06 - Volatility Expansion Protocol

Detects volatility expansion from compressed states.
Category: Volatility Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T06: Detect volatility expansion.
    
    Fires when volatility expands from compression.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T06",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    ranges = np.array([b.range for b in bars])
    
    recent_range = np.mean(ranges[-5:])
    prior_range = np.mean(ranges[-20:-5])
    historical_range = np.mean(ranges[-30:-20])
    
    if prior_range == 0 or historical_range == 0:
        return ProtocolResult(
            protocol_id="T06",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_range"}
        )
    
    was_compressed = prior_range < historical_range * 0.7
    
    expansion_ratio = recent_range / prior_range
    is_expanding = expansion_ratio > 1.5
    
    if was_compressed and is_expanding:
        signal_type = "volatility_expansion"
        confidence = min(0.9, 0.5 + (expansion_ratio - 1.5) * 0.2)
        fired = True
    elif is_expanding:
        signal_type = "expansion_without_compression"
        confidence = 0.4
        fired = True
    else:
        signal_type = "no_expansion"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T06",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "expansion_ratio": float(expansion_ratio),
            "was_compressed": was_compressed,
            "compression_score": float(microtraits.compression_score),
        }
    )
