"""
T75 - Trend Maturity Analysis Protocol

Analyzes how mature/extended the current trend is.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T75: Analyze trend maturity.
    
    Fires when trend appears young, mature, or extended.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T75",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma50 = np.mean(closes[-50:])
    current_price = closes[-1]
    
    trend_up = current_price > sma50
    
    days_in_trend = 0
    for i in range(1, len(closes)):
        if trend_up and closes[-i] > np.mean(closes[-(i+20):]):
            days_in_trend += 1
        elif not trend_up and closes[-i] < np.mean(closes[-(i+20):]):
            days_in_trend += 1
        else:
            break
    
    total_move = (closes[-1] - closes[-days_in_trend-1]) / closes[-days_in_trend-1] * 100 if days_in_trend > 0 else 0
    
    pullbacks = 0
    for i in range(1, min(days_in_trend, len(closes) - 1)):
        if trend_up and closes[-i] < closes[-i-1]:
            pullbacks += 1
        elif not trend_up and closes[-i] > closes[-i-1]:
            pullbacks += 1
    
    pullback_ratio = pullbacks / max(days_in_trend, 1)
    
    if days_in_trend < 10:
        maturity = "young_trend"
        confidence = 0.75
        fired = True
    elif days_in_trend < 25 and abs(total_move) < 15:
        maturity = "developing_trend"
        confidence = 0.7
        fired = True
    elif days_in_trend > 40 or abs(total_move) > 30:
        maturity = "extended_trend"
        confidence = 0.85
        fired = True
    elif pullback_ratio > 0.4:
        maturity = "weakening_trend"
        confidence = 0.7
        fired = True
    else:
        maturity = "mature_trend"
        confidence = 0.65
        fired = True
    
    return ProtocolResult(
        protocol_id="T75",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=maturity,
        details={
            "days_in_trend": int(days_in_trend),
            "total_move_pct": float(total_move),
            "pullback_ratio": float(pullback_ratio),
            "trend_direction": "up" if trend_up else "down",
        }
    )
