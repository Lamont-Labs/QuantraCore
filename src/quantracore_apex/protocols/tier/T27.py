"""
T27 - Volatility Skew Detection Protocol

Detects asymmetric volatility patterns (upside vs downside).
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T27: Detect volatility skew.
    
    Fires when volatility is asymmetric between up and down moves.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T27",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    returns = np.diff(closes) / closes[:-1]
    
    up_returns = returns[returns > 0]
    down_returns = returns[returns < 0]
    
    if len(up_returns) > 0 and len(down_returns) > 0:
        upside_vol = np.std(up_returns)
        downside_vol = np.std(np.abs(down_returns))
        
        skew_ratio = upside_vol / max(downside_vol, 0.0001)
    else:
        skew_ratio = 1.0
        upside_vol = 0.0
        downside_vol = 0.0
    
    if skew_ratio > 1.5:
        signal_type = "upside_skew"
        confidence = min(0.9, 0.5 + (skew_ratio - 1) * 0.2)
        fired = True
    elif skew_ratio < 0.67:
        signal_type = "downside_skew"
        confidence = min(0.9, 0.5 + (1/skew_ratio - 1) * 0.2)
        fired = True
    else:
        signal_type = "symmetric_volatility"
        confidence = 0.4
        fired = False
    
    return ProtocolResult(
        protocol_id="T27",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "upside_vol": float(upside_vol),
            "downside_vol": float(downside_vol),
            "skew_ratio": float(skew_ratio),
            "up_count": int(len(up_returns)),
            "down_count": int(len(down_returns)),
        }
    )
