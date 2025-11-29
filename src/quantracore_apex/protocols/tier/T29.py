"""
T29 - Gap Volatility Protocol

Analyzes gap patterns and their volatility implications.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T29: Analyze gap patterns.
    
    Fires when significant gaps indicate volatility events.
    """
    bars = window.bars
    if len(bars) < 20:
        return ProtocolResult(
            protocol_id="T29",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    gaps_list = []
    for i in range(1, len(bars)):
        gap = bars[i].open - bars[i-1].close
        gap_pct = gap / bars[i-1].close * 100
        gaps_list.append(gap_pct)
    
    gaps = np.array(gaps_list)
    
    recent_gaps = gaps[-5:]
    historical_gaps = gaps[:-5] if len(gaps) > 5 else gaps
    
    avg_gap_size = np.mean(np.abs(recent_gaps))
    hist_avg_gap = np.mean(np.abs(historical_gaps)) if len(historical_gaps) > 0 else avg_gap_size
    
    last_gap = gaps[-1] if len(gaps) > 0 else 0
    
    gap_direction_consistency = np.sum(recent_gaps > 0) / max(len(recent_gaps), 1)
    
    if abs(last_gap) > 2.0:
        signal_type = "large_gap" if last_gap > 0 else "large_gap_down"
        confidence = min(0.95, 0.6 + abs(last_gap) * 0.1)
        fired = True
    elif avg_gap_size > hist_avg_gap * 1.5:
        signal_type = "elevated_gap_activity"
        confidence = 0.7
        fired = True
    elif gap_direction_consistency > 0.8 or gap_direction_consistency < 0.2:
        signal_type = "consistent_gap_direction"
        confidence = 0.6
        fired = True
    else:
        signal_type = "normal_gaps"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T29",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "last_gap_pct": float(last_gap),
            "avg_gap_size": float(avg_gap_size),
            "hist_avg_gap": float(hist_avg_gap),
            "gap_direction_consistency": float(gap_direction_consistency),
        }
    )
