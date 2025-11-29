"""
T63 - Moving Average Support/Resistance Protocol

Analyzes moving averages as dynamic support/resistance.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T63: Analyze MA support/resistance.
    
    Fires when price interacts with key moving averages.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T63",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma20 = np.mean(closes[-20:])
    sma50 = np.mean(closes[-50:])
    
    ema20 = closes[-20:].copy()
    alpha = 2 / 21
    for i in range(1, len(ema20)):
        ema20[i] = alpha * ema20[i] + (1 - alpha) * ema20[i-1]
    ema20_val = ema20[-1]
    
    current_price = closes[-1]
    
    ma_levels = [
        ("sma20", sma20),
        ("sma50", sma50),
        ("ema20", ema20_val),
    ]
    
    nearest_ma = None
    min_distance = float('inf')
    
    for name, level in ma_levels:
        distance = abs(current_price - level) / current_price
        if distance < min_distance:
            min_distance = distance
            nearest_ma = (name, level)
    
    above_all = all(current_price > level for _, level in ma_levels)
    below_all = all(current_price < level for _, level in ma_levels)
    
    crossover_sma20 = closes[-2] < sma20 and closes[-1] > sma20
    crossunder_sma20 = closes[-2] > sma20 and closes[-1] < sma20
    
    if min_distance < 0.01 and nearest_ma:
        signal_type = f"at_{nearest_ma[0]}"
        confidence = 0.8
        fired = True
    elif crossover_sma20:
        signal_type = "bullish_ma_cross"
        confidence = 0.75
        fired = True
    elif crossunder_sma20:
        signal_type = "bearish_ma_cross"
        confidence = 0.75
        fired = True
    elif above_all:
        signal_type = "above_all_mas"
        confidence = 0.7
        fired = True
    elif below_all:
        signal_type = "below_all_mas"
        confidence = 0.7
        fired = True
    else:
        signal_type = "between_mas"
        confidence = 0.4
        fired = False
    
    return ProtocolResult(
        protocol_id="T63",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "sma20": float(sma20),
            "sma50": float(sma50),
            "ema20": float(ema20_val),
            "nearest_ma_distance_pct": float(min_distance * 100),
        }
    )
