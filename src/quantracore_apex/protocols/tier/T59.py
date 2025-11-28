"""
T59 - Rounding Bottom Detection Protocol

Detects rounding bottom (saucer) reversal patterns.
Category: Pattern Recognition
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


def run(window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
    """
    T59: Detect rounding bottom patterns.
    
    Fires when gradual U-shaped reversal pattern is detected.
    """
    bars = window.bars
    if len(bars) < 40:
        return ProtocolResult(
            protocol_id="T59",
            fired=False,
            confidence=0.0,
            details={"reason": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-40:]])
    
    x = np.arange(len(closes))
    
    coeffs = np.polyfit(x, closes, 2)
    a, b, c = coeffs
    
    fitted = a * x**2 + b * x + c
    r2 = 1 - np.sum((closes - fitted)**2) / np.sum((closes - np.mean(closes))**2)
    
    is_u_shaped = a > 0
    
    vertex_x = -b / (2 * a) if a != 0 else len(closes) / 2
    vertex_in_range = 10 < vertex_x < len(closes) - 10
    
    depth = np.min(closes) - np.max([closes[0], closes[-1]])
    depth_pct = abs(depth) / np.max(closes) * 100
    
    left_higher = closes[0] > np.min(closes[10:20])
    right_higher = closes[-1] > np.min(closes[-20:-10])
    
    if is_u_shaped and r2 > 0.6 and vertex_in_range and depth_pct > 5:
        signal_type = "rounding_bottom"
        confidence = 0.7 + r2 * 0.2
        fired = True
    elif is_u_shaped and r2 > 0.4 and left_higher and right_higher:
        signal_type = "potential_rounding_bottom"
        confidence = 0.6
        fired = True
    else:
        signal_type = "no_pattern"
        confidence = 0.2
        fired = False
    
    return ProtocolResult(
        protocol_id="T59",
        fired=bool(fired),
        confidence=float(min(confidence, 0.95)),
        signal_type=signal_type,
        details={
            "curvature_a": float(a),
            "r_squared": float(r2),
            "vertex_x": float(vertex_x),
            "depth_pct": float(depth_pct),
        }
    )
