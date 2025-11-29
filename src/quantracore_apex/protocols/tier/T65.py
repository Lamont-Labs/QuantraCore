"""
T65 - Pivot Point Analysis Protocol

Analyzes price relative to pivot points.
Category: Support/Resistance
"""

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T65: Analyze pivot point levels.
    
    Fires when price interacts with pivot points.
    """
    bars = window.bars
    if len(bars) < 10:
        return ProtocolResult(
            protocol_id="T65",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    prev_high = bars[-2].high
    prev_low = bars[-2].low
    prev_close = bars[-2].close
    
    pivot = (prev_high + prev_low + prev_close) / 3
    
    r1 = 2 * pivot - prev_low
    r2 = pivot + (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    
    s1 = 2 * pivot - prev_high
    s2 = pivot - (prev_high - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    current_price = bars[-1].close
    
    levels = {
        "R3": r3, "R2": r2, "R1": r1,
        "Pivot": pivot,
        "S1": s1, "S2": s2, "S3": s3
    }
    
    nearest_level = None
    min_distance = float('inf')
    
    for name, price in levels.items():
        distance = abs(current_price - price) / current_price
        if distance < min_distance:
            min_distance = distance
            nearest_level = (name, price)
    
    above_pivot = current_price > pivot
    
    if min_distance < 0.005 and nearest_level:
        signal_type = f"at_{nearest_level[0].lower()}"
        confidence = 0.8
        fired = True
    elif min_distance < 0.015 and nearest_level:
        signal_type = f"near_{nearest_level[0].lower()}"
        confidence = 0.65
        fired = True
    elif above_pivot and current_price > r1:
        signal_type = "above_r1"
        confidence = 0.6
        fired = True
    elif not above_pivot and current_price < s1:
        signal_type = "below_s1"
        confidence = 0.6
        fired = True
    else:
        signal_type = "between_levels"
        confidence = 0.3
        fired = False
    
    return ProtocolResult(
        protocol_id="T65",
        fired=bool(fired),
        confidence=float(confidence),
        signal_type=signal_type,
        details={
            "current_price": float(current_price),
            "pivot": float(pivot),
            "r1": float(r1), "r2": float(r2), "r3": float(r3),
            "s1": float(s1), "s2": float(s2), "s3": float(s3),
            "nearest_level": nearest_level[0] if nearest_level else "",
        }
    )
