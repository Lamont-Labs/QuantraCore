"""
T31 - RSI Momentum Protocol

Analyzes RSI for momentum signals and overbought/oversold conditions.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T31: Analyze RSI momentum.
    
    Fires when RSI indicates significant momentum conditions.
    """
    bars = window.bars
    if len(bars) < 15:
        return ProtocolResult(
            protocol_id="T31",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    deltas = np.diff(closes)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    rsi_5d = []
    for i in range(5):
        if len(gains) >= 14 + i:
            ag = np.mean(gains[-(14+i):-(i) if i > 0 else None])
            al = np.mean(losses[-(14+i):-(i) if i > 0 else None])
            if al > 0:
                rsi_5d.append(100 - (100 / (1 + ag/al)))
    
    rsi_trend = rsi - np.mean(rsi_5d) if rsi_5d else 0
    
    if rsi > 80:
        signal_type = "overbought_extreme"
        confidence = min(0.95, 0.6 + (rsi - 80) * 0.02)
        fired = True
    elif rsi > 70:
        signal_type = "overbought"
        confidence = 0.7
        fired = True
    elif rsi < 20:
        signal_type = "oversold_extreme"
        confidence = min(0.95, 0.6 + (20 - rsi) * 0.02)
        fired = True
    elif rsi < 30:
        signal_type = "oversold"
        confidence = 0.7
        fired = True
    elif abs(rsi_trend) > 10:
        signal_type = "momentum_shift"
        confidence = 0.6
        fired = True
    else:
        signal_type = "neutral_momentum"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T31",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "rsi": float(rsi),
            "rsi_trend": float(rsi_trend),
            "avg_gain": float(avg_gain),
            "avg_loss": float(avg_loss),
        }
    )
