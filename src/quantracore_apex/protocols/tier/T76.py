"""
T76 - Seasonality Pattern Protocol

Detects potential seasonality patterns.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T76: Detect seasonality patterns.
    
    Fires when day-of-week or cyclical patterns are detected.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T76",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    daily_returns = np.diff(closes) / closes[:-1] * 100
    
    weekly_pattern = []
    for i in range(5):
        day_returns = daily_returns[i::5]
        if len(day_returns) > 0:
            weekly_pattern.append({
                "day": i,
                "avg_return": np.mean(day_returns),
                "win_rate": sum(1 for r in day_returns if r > 0) / len(day_returns)
            })
    
    monthly_position = len(bars) % 20
    
    best_day = max(weekly_pattern, key=lambda x: x["avg_return"]) if weekly_pattern else None
    worst_day = min(weekly_pattern, key=lambda x: x["avg_return"]) if weekly_pattern else None
    
    day_spread = (best_day["avg_return"] - worst_day["avg_return"]) if best_day and worst_day else 0
    
    cycle_5 = np.mean(daily_returns[-5:]) if len(daily_returns) >= 5 else 0
    cycle_10 = np.mean(daily_returns[-10:]) if len(daily_returns) >= 10 else 0
    cycle_20 = np.mean(daily_returns[-20:]) if len(daily_returns) >= 20 else 0
    
    if day_spread > 0.5:
        signal_type = "weekly_seasonality_detected"
        confidence = min(0.8, 0.5 + day_spread * 0.3)
        fired = True
    elif monthly_position < 5:
        signal_type = "early_month"
        confidence = 0.6
        fired = True
    elif monthly_position > 15:
        signal_type = "late_month"
        confidence = 0.6
        fired = True
    elif abs(cycle_5 - cycle_20) > 0.5:
        signal_type = "cycle_divergence"
        confidence = 0.55
        fired = True
    else:
        signal_type = "no_clear_seasonality"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T76",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "day_spread": float(day_spread),
            "monthly_position": int(monthly_position),
            "cycle_5_avg": float(cycle_5),
            "cycle_10_avg": float(cycle_10),
            "cycle_20_avg": float(cycle_20),
        }
    )
