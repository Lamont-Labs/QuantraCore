"""
T10 - Implied Volatility Structure Protocol

Analyzes structural volatility patterns.
Category: Volatility Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T10: Analyze volatility structure.
    
    Fires when volatility shows specific structural patterns.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T10",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    returns = np.diff(closes) / closes[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) < 20:
        return ProtocolResult(
            protocol_id="T10",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_returns"}
        )
    
    realized_vol = np.std(returns) * np.sqrt(252)
    
    recent_vol = np.std(returns[-10:]) * np.sqrt(252)
    historical_vol = np.std(returns[:-10]) * np.sqrt(252)
    
    if historical_vol == 0:
        vol_ratio = 1
    else:
        vol_ratio = recent_vol / historical_vol
    
    vol_skew = np.mean([r for r in returns if r < 0]) / np.mean([abs(r) for r in returns]) if any(r < 0 for r in returns) else 0
    
    if vol_ratio < 0.6:
        signal_type = "vol_contraction"
        confidence = 0.7
        fired = True
    elif vol_ratio > 1.5:
        signal_type = "vol_expansion"
        confidence = 0.6
        fired = True
    else:
        signal_type = "vol_stable"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T10",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "realized_vol": float(realized_vol),
            "vol_ratio": float(vol_ratio),
            "vol_skew": float(vol_skew),
        }
    )
