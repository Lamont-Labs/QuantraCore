"""
T23 - Bollinger Band Squeeze Protocol

Detects Bollinger Band squeeze conditions indicating low volatility compression.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T23: Detect Bollinger Band squeeze.
    
    Fires when bands are squeezed, indicating potential explosive move ahead.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T23",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    band_width = (upper_band - lower_band) / sma20
    
    historical_widths = []
    for i in range(min(50, len(closes) - 20)):
        idx = -20 - i
        if abs(idx) > len(closes):
            break
        segment = closes[idx:idx+20] if idx+20 <= 0 else closes[idx:]
        if len(segment) >= 20:
            hw = 4 * np.std(segment) / np.mean(segment)
            historical_widths.append(hw)
    
    if historical_widths:
        avg_width = np.mean(historical_widths)
        squeeze_ratio = band_width / max(avg_width, 0.0001)
    else:
        squeeze_ratio = 1.0
    
    if squeeze_ratio < 0.5:
        signal_type = "extreme_squeeze"
        confidence = 0.9
        fired = True
    elif squeeze_ratio < 0.7:
        signal_type = "moderate_squeeze"
        confidence = 0.7
        fired = True
    elif squeeze_ratio < 0.85:
        signal_type = "mild_squeeze"
        confidence = 0.5
        fired = True
    else:
        signal_type = "no_squeeze"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T23",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "band_width": float(band_width),
            "squeeze_ratio": float(squeeze_ratio),
            "upper_band": float(upper_band),
            "lower_band": float(lower_band),
        }
    )
