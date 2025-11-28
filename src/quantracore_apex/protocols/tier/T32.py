"""
T32 - MACD Momentum Protocol

Analyzes MACD for trend momentum and crossover signals.
Category: Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate exponential moving average."""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T32: Analyze MACD momentum.
    
    Fires when MACD shows significant momentum signals.
    """
    bars = window.bars
    if len(bars) < 35:
        return ProtocolResult(
            protocol_id="T32",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line[-9:], 9) if len(macd_line) >= 9 else macd_line
    
    macd_current = macd_line[-1]
    macd_prev = macd_line[-2]
    signal_current = signal_line[-1] if len(signal_line) > 0 else 0
    
    histogram = macd_current - signal_current
    histogram_prev = macd_prev - (signal_line[-2] if len(signal_line) > 1 else 0)
    
    bullish_cross = macd_prev < histogram_prev and macd_current > histogram
    bearish_cross = macd_prev > histogram_prev and macd_current < histogram
    
    macd_slope = macd_current - macd_prev
    
    if bullish_cross and histogram > 0:
        signal_type = "bullish_crossover"
        confidence = 0.85
        fired = True
    elif bearish_cross and histogram < 0:
        signal_type = "bearish_crossover"
        confidence = 0.85
        fired = True
    elif abs(histogram) > abs(np.mean(macd_line[-20:])) * 2:
        signal_type = "strong_momentum"
        confidence = 0.75
        fired = True
    elif abs(macd_slope) > abs(np.std(np.diff(macd_line[-20:]))):
        signal_type = "momentum_acceleration"
        confidence = 0.65
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T32",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "macd": float(macd_current),
            "signal": float(signal_current),
            "histogram": float(histogram),
            "macd_slope": float(macd_slope),
        }
    )
