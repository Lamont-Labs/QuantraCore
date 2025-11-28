"""
T69 - Support/Resistance Flip Protocol

Detects when support becomes resistance or vice versa.
Category: Support/Resistance
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T69: Detect S/R level flips.
    
    Fires when support flips to resistance or vice versa.
    """
    bars = window.bars
    if len(bars) < 30:
        return ProtocolResult(
            protocol_id="T69",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    prior_support = np.min(lows[-30:-15])
    prior_resistance = np.max(highs[-30:-15])
    
    current_price = closes[-1]
    tolerance = 0.02
    
    support_to_resistance = False
    resistance_to_support = False
    
    if current_price < prior_support:
        recent_highs_near_support = np.any(
            (highs[-10:] > prior_support * (1 - tolerance)) &
            (highs[-10:] < prior_support * (1 + tolerance))
        )
        if recent_highs_near_support:
            support_to_resistance = True
    
    if current_price > prior_resistance:
        recent_lows_near_resistance = np.any(
            (lows[-10:] > prior_resistance * (1 - tolerance)) &
            (lows[-10:] < prior_resistance * (1 + tolerance))
        )
        if recent_lows_near_resistance:
            resistance_to_support = True
    
    retest = False
    if support_to_resistance:
        retest = abs(highs[-1] - prior_support) / prior_support < tolerance
    elif resistance_to_support:
        retest = abs(lows[-1] - prior_resistance) / prior_resistance < tolerance
    
    if support_to_resistance and retest:
        signal_type = "support_now_resistance_retest"
        confidence = 0.85
        fired = True
    elif resistance_to_support and retest:
        signal_type = "resistance_now_support_retest"
        confidence = 0.85
        fired = True
    elif support_to_resistance:
        signal_type = "support_now_resistance"
        confidence = 0.7
        fired = True
    elif resistance_to_support:
        signal_type = "resistance_now_support"
        confidence = 0.7
        fired = True
    else:
        signal_type = "no_flip"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T69",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "prior_support": float(prior_support),
            "prior_resistance": float(prior_resistance),
            "current_price": float(current_price),
            "support_to_resistance": bool(support_to_resistance),
            "resistance_to_support": bool(resistance_to_support),
        }
    )
