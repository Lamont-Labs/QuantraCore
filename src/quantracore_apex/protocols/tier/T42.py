"""
T42 - On-Balance Volume Protocol

Analyzes OBV trends for accumulation/distribution signals.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T42: Analyze On-Balance Volume.
    
    Fires when OBV shows significant accumulation or distribution.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T42",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    obv_list = [0]
    for i in range(1, len(bars)):
        if closes[i] > closes[i-1]:
            obv_list.append(obv_list[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv_list.append(obv_list[-1] - volumes[i])
        else:
            obv_list.append(obv_list[-1])
    
    obv = np.array(obv_list)
    
    obv_sma20 = np.mean(obv[-20:])
    obv_current = obv[-1]
    
    obv_trend = obv[-1] - obv[-10]
    price_trend = closes[-1] - closes[-10]
    
    obv_normalized = obv / max(np.std(obv), 1)
    obv_slope = (obv_normalized[-1] - obv_normalized[-5]) / 5
    
    bullish_divergence = obv_trend > 0 and price_trend < 0
    bearish_divergence = obv_trend < 0 and price_trend > 0
    
    if bullish_divergence:
        signal_type = "bullish_divergence"
        confidence = 0.85
        fired = True
    elif bearish_divergence:
        signal_type = "bearish_divergence"
        confidence = 0.85
        fired = True
    elif obv_current > obv_sma20 * 1.2:
        signal_type = "strong_accumulation"
        confidence = 0.75
        fired = True
    elif obv_current < obv_sma20 * 0.8:
        signal_type = "strong_distribution"
        confidence = 0.75
        fired = True
    elif abs(obv_slope) > 0.5:
        signal_type = "obv_trending"
        confidence = 0.6
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T42",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "obv_current": float(obv_current),
            "obv_sma20": float(obv_sma20),
            "obv_trend": float(obv_trend),
            "obv_slope": float(obv_slope),
        }
    )
