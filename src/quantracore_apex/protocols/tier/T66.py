"""
T66 - Psychological Level Analysis Protocol

Analyzes price relative to psychological round numbers.
Category: Support/Resistance
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T66: Analyze psychological price levels.
    
    Fires when price is near round number levels.
    """
    bars = window.bars
    if len(bars) < 5:
        return ProtocolResult(
            protocol_id="T66",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    current_price = bars[-1].close
    
    if current_price >= 1000:
        major_round = round(current_price / 100) * 100
        minor_round = round(current_price / 50) * 50
    elif current_price >= 100:
        major_round = round(current_price / 10) * 10
        minor_round = round(current_price / 5) * 5
    elif current_price >= 10:
        major_round = round(current_price)
        minor_round = round(current_price * 2) / 2
    else:
        major_round = round(current_price, 1)
        minor_round = round(current_price, 2)
    
    distance_major = abs(current_price - major_round) / current_price
    distance_minor = abs(current_price - minor_round) / current_price
    
    recent_crosses = 0
    for i in range(-1, -min(10, len(bars)), -1):
        if (bars[i].low <= major_round <= bars[i].high):
            recent_crosses += 1
    
    if distance_major < 0.005:
        signal_type = "at_major_level"
        confidence = 0.8
        fired = True
    elif distance_major < 0.01:
        signal_type = "near_major_level"
        confidence = 0.7
        fired = True
    elif distance_minor < 0.005:
        signal_type = "at_minor_level"
        confidence = 0.6
        fired = True
    elif recent_crosses >= 3:
        signal_type = "testing_round_level"
        confidence = 0.65
        fired = True
    else:
        signal_type = "away_from_levels"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T66",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "major_round": float(major_round),
            "minor_round": float(minor_round),
            "distance_major_pct": float(distance_major * 100),
            "recent_crosses": int(recent_crosses),
        }
    )
