"""
T28 - Intraday Range Analysis Protocol

Analyzes intraday range patterns for volatility signals.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T28: Analyze intraday range patterns.
    
    Fires when intraday ranges show significant patterns.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T28",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    ranges = np.array([b.high - b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    avg_range = np.mean(ranges[-20:])
    recent_range = np.mean(ranges[-5:])
    
    range_ratio = recent_range / max(avg_range, 0.0001)
    
    range_pct = ranges / closes * 100
    avg_range_pct = np.mean(range_pct[-20:])
    recent_range_pct = np.mean(range_pct[-5:])
    
    hl_consistency = np.std(ranges[-10:]) / max(np.mean(ranges[-10:]), 0.0001)
    
    if range_ratio > 1.5 and recent_range_pct > 3.0:
        signal_type = "range_expansion"
        confidence = min(0.9, 0.5 + range_ratio * 0.15)
        fired = True
    elif range_ratio < 0.6 and recent_range_pct < 1.0:
        signal_type = "range_contraction"
        confidence = min(0.9, 0.5 + (1/range_ratio) * 0.15)
        fired = True
    elif hl_consistency < 0.3:
        signal_type = "consistent_ranges"
        confidence = 0.6
        fired = True
    else:
        signal_type = "normal_ranges"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T28",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "avg_range": float(avg_range),
            "recent_range": float(recent_range),
            "range_ratio": float(range_ratio),
            "avg_range_pct": float(avg_range_pct),
            "hl_consistency": float(hl_consistency),
        }
    )
