"""
T30 - Volatility Mean Reversion Protocol

Detects volatility mean reversion opportunities.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T30: Detect volatility mean reversion.
    
    Fires when volatility is extended and likely to revert.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T30",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    returns = np.diff(closes) / closes[:-1]
    
    vol_5d = np.std(returns[-5:]) * np.sqrt(252)
    vol_20d = np.std(returns[-20:]) * np.sqrt(252)
    vol_50d = np.std(returns[-50:]) * np.sqrt(252)
    
    vol_ratio_short = vol_5d / max(vol_50d, 0.0001)
    vol_ratio_medium = vol_20d / max(vol_50d, 0.0001)
    
    z_score = (vol_5d - vol_50d) / max(np.std([vol_5d, vol_20d, vol_50d]), 0.0001)
    
    if vol_ratio_short > 2.0 and z_score > 2.0:
        signal_type = "vol_extended_high"
        confidence = min(0.9, 0.5 + vol_ratio_short * 0.1)
        fired = True
        reversion_direction = "expect_contraction"
    elif vol_ratio_short < 0.5 and z_score < -2.0:
        signal_type = "vol_extended_low"
        confidence = min(0.9, 0.5 + (1/vol_ratio_short) * 0.1)
        fired = True
        reversion_direction = "expect_expansion"
    elif abs(z_score) > 1.5:
        signal_type = "vol_moderately_extended"
        confidence = 0.6
        fired = True
        reversion_direction = "expect_normalization"
    else:
        signal_type = "vol_normal"
        confidence = 0.3
        fired = False
        reversion_direction = "none"
    
    return ProtocolResult(
        protocol_id="T30",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "vol_5d": float(vol_5d),
            "vol_20d": float(vol_20d),
            "vol_50d": float(vol_50d),
            "vol_ratio_short": float(vol_ratio_short),
            "z_score": float(z_score),
            "reversion_direction": reversion_direction,
        }
    )
