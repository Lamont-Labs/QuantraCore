"""
T04 - Trend Channel Protocol

Detects trend channels and position within them.
Category: Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T04: Detect trend channels.
    
    Fires when price is within a defined channel.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T04",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars[-20:]])
    lows = np.array([b.low for b in bars[-20:]])
    closes = np.array([b.close for b in bars[-20:]])
    x = np.arange(20)
    
    high_slope, high_intercept = np.polyfit(x, highs, 1)
    low_slope, low_intercept = np.polyfit(x, lows, 1)
    
    slope_diff = abs(high_slope - low_slope) / (abs(high_slope) + abs(low_slope) + 1e-10)
    parallel_channel = slope_diff < 0.3
    
    current_price = closes[-1]
    channel_top = high_slope * 19 + high_intercept
    channel_bottom = low_slope * 19 + low_intercept
    channel_width = channel_top - channel_bottom
    
    if channel_width > 0:
        position = (current_price - channel_bottom) / channel_width
    else:
        position = 0.5
    
    avg_slope = (high_slope + low_slope) / 2
    channel_direction = "ascending" if avg_slope > 0 else "descending" if avg_slope < 0 else "horizontal"
    
    if parallel_channel and 0.1 < position < 0.9:
        signal_type = f"channel_{channel_direction}"
        confidence = 0.7 - abs(position - 0.5) * 0.4
        fired = True
    else:
        signal_type = "no_clear_channel"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T04",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "channel_direction": channel_direction,
            "position_in_channel": float(position),
            "channel_width": float(channel_width),
            "parallel_score": float(1 - slope_diff),
        }
    )
