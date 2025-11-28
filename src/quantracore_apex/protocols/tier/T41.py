"""
T41 - Volume Spike Detection Protocol

Detects significant volume spikes indicating institutional activity.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T41: Detect volume spikes.
    
    Fires when volume significantly exceeds historical average.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T41",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    volumes = np.array([b.volume for b in bars])
    closes = np.array([b.close for b in bars])
    
    current_volume = volumes[-1]
    avg_volume_20 = np.mean(volumes[-20:])
    avg_volume_5 = np.mean(volumes[-5:])
    
    volume_ratio = current_volume / max(avg_volume_20, 1)
    
    volume_std = np.std(volumes[-20:])
    volume_zscore = (current_volume - avg_volume_20) / max(volume_std, 1)
    
    price_change = (closes[-1] - closes[-2]) / closes[-2] * 100
    
    if volume_ratio > 3.0 and volume_zscore > 2.5:
        signal_type = "extreme_volume_spike"
        confidence = min(0.95, 0.6 + volume_ratio * 0.1)
        fired = True
    elif volume_ratio > 2.0:
        signal_type = "significant_volume_spike"
        confidence = 0.8
        fired = True
    elif volume_ratio > 1.5:
        signal_type = "moderate_volume_spike"
        confidence = 0.65
        fired = True
    elif volume_ratio < 0.5:
        signal_type = "volume_dry_up"
        confidence = 0.6
        fired = True
    else:
        signal_type = "normal_volume"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T41",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_volume": float(current_volume),
            "avg_volume_20": float(avg_volume_20),
            "volume_ratio": float(volume_ratio),
            "volume_zscore": float(volume_zscore),
            "price_change_pct": float(price_change),
        }
    )
