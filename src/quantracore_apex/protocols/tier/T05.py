"""
T05 - Moving Average Crossover Protocol

Detects moving average crossovers for trend changes.
Category: Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T05: Detect moving average crossovers.
    
    Fires when recent crossover detected.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T05",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    sma_fast = []
    sma_slow = []
    
    for i in range(20, len(closes) + 1):
        sma_fast.append(np.mean(closes[i-10:i]))
        sma_slow.append(np.mean(closes[i-20:i]))
    
    if len(sma_fast) < 5:
        return ProtocolResult(
            protocol_id="T05",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    crossover_detected = False
    crossover_type = None
    bars_since_cross = 0
    
    for i in range(len(sma_fast) - 1, max(0, len(sma_fast) - 10), -1):
        prev_diff = sma_fast[i-1] - sma_slow[i-1]
        curr_diff = sma_fast[i] - sma_slow[i]
        
        if prev_diff <= 0 and curr_diff > 0:
            crossover_detected = True
            crossover_type = "bullish_crossover"
            bars_since_cross = len(sma_fast) - 1 - i
            break
        elif prev_diff >= 0 and curr_diff < 0:
            crossover_detected = True
            crossover_type = "bearish_crossover"
            bars_since_cross = len(sma_fast) - 1 - i
            break
    
    if crossover_detected:
        recency_factor = max(0.3, 1 - bars_since_cross * 0.1)
        confidence = 0.6 * recency_factor
        fired = True
    else:
        crossover_type = "no_crossover"
        confidence = 0.1
        fired = False
    
    return ProtocolResult(
        protocol_id="T05",
        fired=fired,
        confidence=confidence,
        signal_type=crossover_type,
        details={
            "current_fast_ma": float(sma_fast[-1]) if sma_fast else 0,
            "current_slow_ma": float(sma_slow[-1]) if sma_slow else 0,
            "bars_since_cross": bars_since_cross,
        }
    )
