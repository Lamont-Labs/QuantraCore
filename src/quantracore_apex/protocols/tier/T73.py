"""
T73 - Relative Performance Analysis Protocol

Analyzes performance relative to recent history.
Category: Market Context
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T73: Analyze relative performance.
    
    Fires when performance is significantly above or below normal.
    """
    bars = window.bars
    if len(bars) < 60:
        return ProtocolResult(
            protocol_id="T73",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    perf_5d = (closes[-1] - closes[-6]) / closes[-6] * 100
    perf_20d = (closes[-1] - closes[-21]) / closes[-21] * 100
    perf_60d = (closes[-1] - closes[-61]) / closes[-61] * 100
    
    historical_5d_returns = []
    for i in range(20, len(closes) - 5):
        ret = (closes[i+5] - closes[i]) / closes[i] * 100
        historical_5d_returns.append(ret)
    
    if historical_5d_returns:
        mean_5d = np.mean(historical_5d_returns)
        std_5d = np.std(historical_5d_returns)
        zscore_5d = (perf_5d - mean_5d) / max(std_5d, 0.0001)
    else:
        zscore_5d = 0
    
    relative_momentum = perf_5d / max(abs(perf_20d), 0.01)
    
    if zscore_5d > 2:
        signal_type = "strong_outperformance"
        confidence = min(0.9, 0.6 + zscore_5d * 0.1)
        fired = True
    elif zscore_5d < -2:
        signal_type = "strong_underperformance"
        confidence = min(0.9, 0.6 + abs(zscore_5d) * 0.1)
        fired = True
    elif zscore_5d > 1:
        signal_type = "outperforming"
        confidence = 0.7
        fired = True
    elif zscore_5d < -1:
        signal_type = "underperforming"
        confidence = 0.7
        fired = True
    else:
        signal_type = "average_performance"
        confidence = 0.4
        fired = False
    
    return ProtocolResult(
        protocol_id="T73",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "perf_5d_pct": float(perf_5d),
            "perf_20d_pct": float(perf_20d),
            "perf_60d_pct": float(perf_60d),
            "zscore_5d": float(zscore_5d),
            "relative_momentum": float(relative_momentum),
        }
    )
