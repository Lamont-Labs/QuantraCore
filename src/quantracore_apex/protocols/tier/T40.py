"""
T40 - Multi-Timeframe Momentum Alignment Protocol

Analyzes momentum alignment across multiple timeframes.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T40: Analyze multi-timeframe momentum alignment.
    
    Fires when momentum aligns across timeframes.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T40",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    mom_5 = (closes[-1] - closes[-5]) / closes[-5] * 100
    mom_10 = (closes[-1] - closes[-10]) / closes[-10] * 100
    mom_20 = (closes[-1] - closes[-20]) / closes[-20] * 100
    mom_50 = (closes[-1] - closes[-50]) / closes[-50] * 100
    
    momentums = [mom_5, mom_10, mom_20, mom_50]
    
    all_positive = all(m > 0 for m in momentums)
    all_negative = all(m < 0 for m in momentums)
    
    strength_order = (
        (abs(mom_5) >= abs(mom_10) >= abs(mom_20) >= abs(mom_50)) or
        (abs(mom_5) <= abs(mom_10) <= abs(mom_20) <= abs(mom_50))
    )
    
    alignment_score = sum(1 for m in momentums if m > 0) / 4
    
    avg_momentum = np.mean(momentums)
    momentum_strength = abs(avg_momentum)
    
    if all_positive and strength_order:
        signal_type = "bullish_alignment"
        confidence = min(0.95, 0.6 + momentum_strength * 0.02)
        fired = True
    elif all_negative and strength_order:
        signal_type = "bearish_alignment"
        confidence = min(0.95, 0.6 + momentum_strength * 0.02)
        fired = True
    elif all_positive or all_negative:
        signal_type = "directional_alignment"
        confidence = 0.7
        fired = True
    elif abs(alignment_score - 0.5) < 0.1:
        signal_type = "mixed_signals"
        confidence = 0.4
        fired = False
    else:
        signal_type = "partial_alignment"
        confidence = 0.5
        fired = True
    
    return ProtocolResult(
        protocol_id="T40",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "mom_5": float(mom_5),
            "mom_10": float(mom_10),
            "mom_20": float(mom_20),
            "mom_50": float(mom_50),
            "alignment_score": float(alignment_score),
            "avg_momentum": float(avg_momentum),
        }
    )
