"""
T20 - Volume Climax Protocol

Detects volume climax events that often precede reversals.
Category: Volume Engines
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T20: Detect volume climax events.
    
    Fires when extreme volume with specific price behavior detected.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T20",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    volumes = np.array([b.volume for b in bars])
    
    current_vol = volumes[-1]
    avg_vol = np.mean(volumes[-30:-1])
    vol_std = np.std(volumes[-30:-1])
    
    if avg_vol == 0:
        return ProtocolResult(
            protocol_id="T20",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_volume"}
        )
    
    vol_zscore = (current_vol - avg_vol) / vol_std if vol_std > 0 else 0
    extreme_volume = vol_zscore > 2.5
    
    last_bar = bars[-1]
    wide_range = last_bar.range > np.mean([b.range for b in bars[-30:-1]]) * 1.5
    
    reversal_wick = (last_bar.upper_wick > last_bar.body and last_bar.close < last_bar.open) or \
                    (last_bar.lower_wick > last_bar.body and last_bar.close > last_bar.open)
    
    if extreme_volume and wide_range:
        if reversal_wick:
            signal_type = "volume_climax_reversal"
            confidence = 0.85
        else:
            signal_type = "volume_climax"
            confidence = 0.7
        fired = True
    elif extreme_volume:
        signal_type = "high_volume_alert"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_climax"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T20",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "volume_zscore": float(vol_zscore),
            "relative_volume": float(current_vol / avg_vol),
            "wide_range": wide_range,
            "reversal_wick": reversal_wick,
        }
    )
