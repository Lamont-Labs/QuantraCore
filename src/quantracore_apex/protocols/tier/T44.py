"""
T44 - Accumulation Distribution Protocol

Analyzes Accumulation/Distribution line for money flow signals.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T44: Analyze Accumulation/Distribution.
    
    Fires when A/D line shows significant money flow patterns.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T44",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    ad_line_list: list = []
    for i in range(len(bars)):
        hl_range = highs[i] - lows[i]
        if hl_range == 0:
            mf_multiplier = 0
        else:
            mf_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range
        
        mf_volume = mf_multiplier * volumes[i]
        
        if ad_line_list:
            ad_line_list.append(ad_line_list[-1] + mf_volume)
        else:
            ad_line_list.append(mf_volume)
    
    ad_line = np.array(ad_line_list)
    
    ad_current = ad_line[-1]
    ad_sma10 = np.mean(ad_line[-10:])
    ad_sma20 = np.mean(ad_line[-20:])
    
    ad_trend = ad_line[-1] - ad_line[-10]
    price_trend = closes[-1] - closes[-10]
    
    bullish_divergence = ad_trend > 0 and price_trend < 0
    bearish_divergence = ad_trend < 0 and price_trend > 0
    
    if bullish_divergence:
        signal_type = "bullish_divergence"
        confidence = 0.85
        fired = True
    elif bearish_divergence:
        signal_type = "bearish_divergence"
        confidence = 0.85
        fired = True
    elif ad_sma10 > ad_sma20 * 1.1:
        signal_type = "accumulation_phase"
        confidence = 0.75
        fired = True
    elif ad_sma10 < ad_sma20 * 0.9:
        signal_type = "distribution_phase"
        confidence = 0.75
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T44",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "ad_current": float(ad_current),
            "ad_sma10": float(ad_sma10),
            "ad_sma20": float(ad_sma20),
            "ad_trend": float(ad_trend),
        }
    )
