"""
T49 - Force Index Protocol

Analyzes Force Index for trend strength with volume.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T49: Analyze Force Index.
    
    Fires when Force Index shows significant trend strength.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T49",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    force_index = []
    for i in range(1, len(bars)):
        fi = (closes[i] - closes[i-1]) * volumes[i]
        force_index.append(fi)
    
    force_index = np.array(force_index)
    
    fi_13_ema = np.mean(force_index[-13:])
    
    fi_normalized = fi_13_ema / max(np.std(force_index), 1)
    
    fi_trend = np.mean(force_index[-3:]) - np.mean(force_index[-10:-3])
    
    positive_streak = 0
    for fi in reversed(force_index[-10:]):
        if fi > 0:
            positive_streak += 1
        else:
            break
    
    negative_streak = 0
    for fi in reversed(force_index[-10:]):
        if fi < 0:
            negative_streak += 1
        else:
            break
    
    if fi_13_ema > 0 and fi_normalized > 2:
        signal_type = "strong_bullish_force"
        confidence = min(0.9, 0.5 + fi_normalized * 0.15)
        fired = True
    elif fi_13_ema < 0 and fi_normalized < -2:
        signal_type = "strong_bearish_force"
        confidence = min(0.9, 0.5 + abs(fi_normalized) * 0.15)
        fired = True
    elif positive_streak >= 5:
        signal_type = "sustained_buying"
        confidence = 0.75
        fired = True
    elif negative_streak >= 5:
        signal_type = "sustained_selling"
        confidence = 0.75
        fired = True
    elif abs(fi_trend) > np.std(force_index):
        signal_type = "force_shift"
        confidence = 0.6
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T49",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "fi_13_ema": float(fi_13_ema),
            "fi_normalized": float(fi_normalized),
            "positive_streak": int(positive_streak),
            "negative_streak": int(negative_streak),
        }
    )
