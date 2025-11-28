"""
T47 - Chaikin Money Flow Protocol

Analyzes CMF for money flow signals.
Category: Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T47: Analyze Chaikin Money Flow.
    
    Fires when CMF shows significant buying/selling pressure.
    """
    bars = window.bars
    if len(bars) < 25:
        return ProtocolResult(
            protocol_id="T47",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars])
    
    mf_multipliers = []
    for i in range(len(bars)):
        hl_range = highs[i] - lows[i]
        if hl_range == 0:
            mf_multipliers.append(0)
        else:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range
            mf_multipliers.append(mfm)
    
    mf_volumes = np.array(mf_multipliers) * volumes
    
    period = 20
    cmf = np.sum(mf_volumes[-period:]) / max(np.sum(volumes[-period:]), 1)
    
    cmf_values = []
    for offset in range(5):
        cmf_val = np.sum(mf_volumes[-(period+offset):-(offset) if offset > 0 else None]) / \
                  max(np.sum(volumes[-(period+offset):-(offset) if offset > 0 else None]), 1)
        cmf_values.append(cmf_val)
    
    cmf_trend = cmf - np.mean(cmf_values) if cmf_values else 0
    
    if cmf > 0.25:
        signal_type = "strong_buying_pressure"
        confidence = min(0.9, 0.5 + cmf * 1.5)
        fired = True
    elif cmf > 0.1:
        signal_type = "moderate_buying_pressure"
        confidence = 0.7
        fired = True
    elif cmf < -0.25:
        signal_type = "strong_selling_pressure"
        confidence = min(0.9, 0.5 + abs(cmf) * 1.5)
        fired = True
    elif cmf < -0.1:
        signal_type = "moderate_selling_pressure"
        confidence = 0.7
        fired = True
    elif abs(cmf_trend) > 0.1:
        signal_type = "cmf_shift"
        confidence = 0.6
        fired = True
    else:
        signal_type = "neutral"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T47",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "cmf": float(cmf),
            "cmf_trend": float(cmf_trend),
        }
    )
