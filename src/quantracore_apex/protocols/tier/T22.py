"""
T22 - Volatility Contraction Detection Protocol

Detects volatility contraction phases indicating potential consolidation before move.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T22: Detect volatility contraction phases.
    
    Fires when volatility is contracting, indicating potential coiling before major move.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T22",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    atr_recent = np.mean(highs[-5:] - lows[-5:])
    atr_prior = np.mean(highs[-20:-5] - lows[-20:-5])
    
    contraction_ratio = atr_recent / max(atr_prior, 0.0001)
    compression_score = float(microtraits.compression_score)
    
    if contraction_ratio < 0.6 and compression_score > 0.6:
        signal_type = "strong_contraction"
        confidence = min(0.95, 0.7 + compression_score * 0.2)
        fired = True
    elif contraction_ratio < 0.8 and compression_score > 0.4:
        signal_type = "moderate_contraction"
        confidence = 0.5 + compression_score * 0.2
        fired = True
    else:
        signal_type = "no_contraction"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T22",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "atr_recent": float(atr_recent),
            "atr_prior": float(atr_prior),
            "contraction_ratio": float(contraction_ratio),
            "compression_score": compression_score,
        }
    )
