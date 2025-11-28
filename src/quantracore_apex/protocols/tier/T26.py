"""
T26 - Historical Volatility Regime Protocol

Classifies current volatility regime based on historical comparison.
Category: Volatility Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T26: Classify volatility regime.
    
    Fires when volatility regime is clearly identifiable.
    """
    bars = window.bars
    if len(bars) < 50:
        return ProtocolResult(
            protocol_id="T26",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars])
    
    returns = np.diff(closes) / closes[:-1]
    
    vol_5d = np.std(returns[-5:]) * np.sqrt(252)
    vol_20d = np.std(returns[-20:]) * np.sqrt(252)
    vol_50d = np.std(returns[-50:]) * np.sqrt(252)
    
    vol_percentile = 0
    for period in [20, 30, 40, 50]:
        if len(returns) >= period:
            rolling_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-period)]
            if rolling_vols:
                current_vol = np.std(returns[-20:])
                vol_percentile = (sum(1 for v in rolling_vols if v < current_vol) / len(rolling_vols)) * 100
                break
    
    if vol_percentile > 80:
        regime = "high_volatility"
        confidence = 0.85
        fired = True
    elif vol_percentile > 60:
        regime = "elevated_volatility"
        confidence = 0.7
        fired = True
    elif vol_percentile < 20:
        regime = "low_volatility"
        confidence = 0.85
        fired = True
    elif vol_percentile < 40:
        regime = "subdued_volatility"
        confidence = 0.7
        fired = True
    else:
        regime = "normal_volatility"
        confidence = 0.5
        fired = False
    
    return ProtocolResult(
        protocol_id="T26",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=regime,
        details={
            "vol_5d": float(vol_5d),
            "vol_20d": float(vol_20d),
            "vol_50d": float(vol_50d),
            "vol_percentile": float(vol_percentile),
        }
    )
