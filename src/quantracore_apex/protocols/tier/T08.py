"""
T08 - ATR Contraction Protocol

Detects ATR-based volatility contraction.
Category: Volatility Structures
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def compute_atr(bars, period: int = 14) -> list:
    """Compute ATR series."""
    if len(bars) < period + 1:
        return []
    
    tr_list = []
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i-1].close),
            abs(bars[i].low - bars[i-1].close)
        )
        tr_list.append(tr)
    
    atr_series = []
    for i in range(period, len(tr_list) + 1):
        atr_series.append(np.mean(tr_list[i-period:i]))
    
    return atr_series


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T08: Detect ATR contraction.
    
    Fires when ATR contracts significantly.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T08",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    atr_series = compute_atr(bars)
    
    if len(atr_series) < 10:
        return ProtocolResult(
            protocol_id="T08",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_atr_data"}
        )
    
    current_atr = atr_series[-1]
    avg_atr = np.mean(atr_series[:-5]) if len(atr_series) > 5 else np.mean(atr_series)
    
    if avg_atr == 0:
        return ProtocolResult(
            protocol_id="T08",
            fired=False,
            confidence=0.0,
            details={"reason": "zero_atr"}
        )
    
    contraction_ratio = current_atr / avg_atr
    
    atr_trend = np.polyfit(range(min(10, len(atr_series))), atr_series[-10:], 1)[0]
    declining = atr_trend < 0
    
    if contraction_ratio < 0.6 and declining:
        signal_type = "severe_atr_contraction"
        confidence = 0.8
        fired = True
    elif contraction_ratio < 0.75:
        signal_type = "moderate_atr_contraction"
        confidence = 0.55
        fired = True
    else:
        signal_type = "no_contraction"
        confidence = 0.15
        fired = False
    
    return ProtocolResult(
        protocol_id="T08",
        fired=fired,
        confidence=confidence,
        signal_type=signal_type,
        details={
            "current_atr": float(current_atr),
            "average_atr": float(avg_atr),
            "contraction_ratio": float(contraction_ratio),
            "atr_declining": declining,
        }
    )
