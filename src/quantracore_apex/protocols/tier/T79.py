"""
T79 - Risk Environment Assessment Protocol

Assesses overall risk environment.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T79: Assess risk environment.
    
    Fires when risk environment is clearly elevated or subdued.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T79",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    np.array([b.high for b in bars])
    np.array([b.low for b in bars])
    
    returns = np.diff(closes) / closes[:-1]
    
    vol_5 = np.std(returns[-5:]) * np.sqrt(252) * 100
    vol_20 = np.std(returns[-20:]) * np.sqrt(252) * 100
    vol_50 = np.std(returns[-50:]) * np.sqrt(252) * 100
    
    vol_expansion = vol_5 > vol_20 * 1.5
    vol_contraction = vol_5 < vol_20 * 0.5
    
    negative_days = sum(1 for r in returns[-10:] if r < 0)
    large_moves = sum(1 for r in returns[-10:] if abs(r) > 0.02)
    
    gap_count = sum(1 for i in range(1, min(10, len(bars))) if abs(bars[-i].open - bars[-i-1].close) / bars[-i-1].close > 0.01)
    
    risk_score = 0
    risk_score += vol_5 / 30 * 0.3
    risk_score += large_moves / 10 * 0.3
    risk_score += gap_count / 10 * 0.2
    risk_score += negative_days / 10 * 0.2
    
    if risk_score > 0.7 or vol_expansion:
        signal_type = "high_risk_environment"
        confidence = min(0.9, 0.5 + risk_score * 0.4)
        fired = True
    elif risk_score > 0.4:
        signal_type = "elevated_risk"
        confidence = 0.7
        fired = True
    elif vol_contraction and risk_score < 0.3:
        signal_type = "low_risk_environment"
        confidence = 0.75
        fired = True
    elif risk_score < 0.3:
        signal_type = "subdued_risk"
        confidence = 0.6
        fired = True
    else:
        signal_type = "normal_risk"
        confidence = 0.4
        fired = False
    
    return ProtocolResult(
        protocol_id="T79",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "vol_5_annualized": float(vol_5),
            "vol_20_annualized": float(vol_20),
            "vol_50_annualized": float(vol_50),
            "risk_score": float(risk_score),
            "large_moves_count": int(large_moves),
            "gap_count": int(gap_count),
        }
    )
