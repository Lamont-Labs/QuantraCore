"""
T34 - Rate of Change Protocol

Analyzes price rate of change for momentum signals.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T34: Analyze rate of change momentum.
    
    Fires when ROC indicates significant momentum.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T34",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    roc_10 = ((closes[-1] - closes[-11]) / closes[-11]) * 100 if len(closes) >= 11 else 0
    roc_5 = ((closes[-1] - closes[-6]) / closes[-6]) * 100 if len(closes) >= 6 else 0
    roc_1 = ((closes[-1] - closes[-2]) / closes[-2]) * 100
    
    roc_values = []
    for i in range(min(20, len(closes) - 10)):
        if closes[-(11+i)] != 0:
            roc = ((closes[-(1+i)] - closes[-(11+i)]) / closes[-(11+i)]) * 100
            roc_values.append(roc)
    
    roc_mean = np.mean(roc_values) if roc_values else 0
    roc_std = np.std(roc_values) if roc_values else 1
    
    roc_zscore = (roc_10 - roc_mean) / max(roc_std, 0.0001)
    
    roc_acceleration = roc_5 - (roc_10 - roc_5)
    
    if abs(roc_zscore) > 2.5:
        signal_type = "extreme_momentum"
        confidence = min(0.95, 0.6 + abs(roc_zscore) * 0.1)
        fired = True
    elif abs(roc_zscore) > 1.5:
        signal_type = "strong_momentum"
        confidence = 0.75
        fired = True
    elif abs(roc_acceleration) > abs(roc_mean) * 2:
        signal_type = "momentum_acceleration"
        confidence = 0.65
        fired = True
    else:
        signal_type = "normal_momentum"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T34",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "roc_10": float(roc_10),
            "roc_5": float(roc_5),
            "roc_1": float(roc_1),
            "roc_zscore": float(roc_zscore),
            "roc_acceleration": float(roc_acceleration),
        }
    )
