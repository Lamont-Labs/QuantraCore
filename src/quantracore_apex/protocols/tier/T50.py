"""
T50 - Volume Climax Detection Protocol

Detects volume climax events indicating potential reversals.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T50: Detect volume climax events.
    
    Fires when volume climax suggests potential trend exhaustion.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T50",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    current_volume = volumes[-1]
    avg_volume = np.mean(volumes[-20:])
    volume_std = np.std(volumes[-20:])
    
    volume_zscore = (current_volume - avg_volume) / max(volume_std, 1)
    
    current_range = highs[-1] - lows[-1]
    avg_range = np.mean(highs[-20:] - lows[-20:])
    
    range_expansion = current_range / max(avg_range, 0.0001)
    
    price_change = (closes[-1] - closes[-2]) / closes[-2] * 100
    
    climax_score = volume_zscore * 0.5 + range_expansion * 0.3 + abs(price_change) * 0.2
    
    recent_trend = (closes[-1] - closes[-10]) / closes[-10] * 100
    
    selling_climax = climax_score > 3 and recent_trend < -5
    buying_climax = climax_score > 3 and recent_trend > 5
    
    if selling_climax:
        signal_type = "selling_climax"
        confidence = min(0.95, 0.5 + climax_score * 0.1)
        fired = True
    elif buying_climax:
        signal_type = "buying_climax"
        confidence = min(0.95, 0.5 + climax_score * 0.1)
        fired = True
    elif volume_zscore > 3:
        signal_type = "extreme_volume"
        confidence = 0.8
        fired = True
    elif volume_zscore > 2 and range_expansion > 1.5:
        signal_type = "potential_climax"
        confidence = 0.7
        fired = True
    else:
        signal_type = "no_climax"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T50",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "volume_zscore": float(volume_zscore),
            "range_expansion": float(range_expansion),
            "price_change_pct": float(price_change),
            "climax_score": float(climax_score),
            "recent_trend_pct": float(recent_trend),
        }
    )
