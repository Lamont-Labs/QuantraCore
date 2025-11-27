"""
T03 - Trend Momentum Protocol

Analyzes momentum within the trend using rate of change.
Category: Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T03: Analyze trend momentum.
    
    Fires when momentum aligns with trend direction.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T03",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if closes[-6] != 0 else 0
    roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if closes[-11] != 0 else 0
    roc_20 = (closes[-1] - closes[-21]) / closes[-21] * 100 if len(closes) > 20 and closes[-21] != 0 else 0
    
    momentum_aligned = (roc_5 > 0 and roc_10 > 0 and roc_20 > 0) or \
                       (roc_5 < 0 and roc_10 < 0 and roc_20 < 0)
    
    momentum_strength = (abs(roc_5) + abs(roc_10) + abs(roc_20)) / 30
    
    if momentum_aligned and momentum_strength > 0.3:
        signal_type = "momentum_aligned" if roc_5 > 0 else "momentum_aligned_down"
        confidence = min(0.9, 0.5 + momentum_strength * 0.4)
        fired = True
    else:
        signal_type = "momentum_diverging"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T03",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "roc_5": float(roc_5),
            "roc_10": float(roc_10),
            "roc_20": float(roc_20),
            "momentum_aligned": momentum_aligned,
        }
    )
