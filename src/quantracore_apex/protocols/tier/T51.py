"""
T51 - Candlestick Pattern Recognition Protocol

Detects common candlestick patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T51: Detect candlestick patterns.
    
    Fires when significant candlestick patterns are detected.
    """
    bars = window.bars
    if len(bars) < 5:
        return ProtocolResult(
            protocol_id="T51",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    last = bars[-1]
    prev = bars[-2]
    
    body = abs(last.close - last.open)
    upper_shadow = last.high - max(last.close, last.open)
    lower_shadow = min(last.close, last.open) - last.low
    total_range = last.high - last.low
    
    is_bullish = last.close > last.open
    is_bearish = last.close < last.open
    
    patterns_detected = []
    
    if total_range > 0:
        body_ratio = body / total_range
        
        if body_ratio < 0.1 and upper_shadow > body * 2 and lower_shadow > body * 2:
            patterns_detected.append("doji")
        
        if is_bullish and lower_shadow > body * 2 and upper_shadow < body * 0.5:
            patterns_detected.append("hammer")
        
        if is_bearish and upper_shadow > body * 2 and lower_shadow < body * 0.5:
            patterns_detected.append("shooting_star")
        
        if is_bullish and body_ratio > 0.7:
            patterns_detected.append("bullish_marubozu")
        
        if is_bearish and body_ratio > 0.7:
            patterns_detected.append("bearish_marubozu")
    
    prev_body = abs(prev.close - prev.open)
    if prev.close < prev.open and last.close > last.open:
        if last.open < prev.close and last.close > prev.open:
            patterns_detected.append("bullish_engulfing")
    
    if prev.close > prev.open and last.close < last.open:
        if last.open > prev.close and last.close < prev.open:
            patterns_detected.append("bearish_engulfing")
    
    if patterns_detected:
        signal_type = patterns_detected[0]
        confidence = 0.7 + len(patterns_detected) * 0.05
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T51",
        fired=bool(fired),
        confidence=float(min(confidence, 0.95)),
        signal_type=signal_type,
        details={
            "patterns_detected": patterns_detected,
            "body_size": float(body),
            "upper_shadow": float(upper_shadow),
            "lower_shadow": float(lower_shadow),
            "is_bullish": bool(is_bullish),
        }
    )
