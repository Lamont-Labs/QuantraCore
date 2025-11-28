"""
LP15 - Volume Confirmation Strength Label Protocol

Measures volume confirmation strength for price moves.
Category: Advanced Labels - Volume Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP15: Generate volume confirmation strength label.
    
    Classifies how strongly volume confirms price action:
    - Strong confirmation (volume supports move)
    - Weak confirmation (marginal support)
    - Divergence (volume contradicts move)
    """
    bars = window.bars
    if len(bars) < 20:
        return LearningLabel(
            protocol_id="LP15",
            label_name="volume_confirmation",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-20:]])
    volumes = np.array([b.volume for b in bars[-20:]])
    
    returns = np.diff(closes) / closes[:-1]
    
    up_days = returns > 0
    down_days = returns < 0
    
    avg_up_volume = volumes[1:][up_days].mean() if up_days.any() else 0
    avg_down_volume = volumes[1:][down_days].mean() if down_days.any() else 0
    avg_volume = volumes.mean()
    
    recent_price_change = closes[-1] / closes[0] - 1
    
    if recent_price_change > 0:
        if avg_up_volume > avg_down_volume * 1.2:
            confirmation = 0
            conf_name = "strong_bullish"
        elif avg_up_volume > avg_down_volume:
            confirmation = 1
            conf_name = "weak_bullish"
        else:
            confirmation = 2
            conf_name = "bearish_divergence"
    else:
        if avg_down_volume > avg_up_volume * 1.2:
            confirmation = 0
            conf_name = "strong_bearish"
        elif avg_down_volume > avg_up_volume:
            confirmation = 1
            conf_name = "weak_bearish"
        else:
            confirmation = 2
            conf_name = "bullish_divergence"
    
    vol_ratio = max(avg_up_volume, avg_down_volume) / (min(avg_up_volume, avg_down_volume) + 1e-10)
    confidence = min(0.9, 0.4 + min(vol_ratio - 1, 1) * 0.5)
    
    return LearningLabel(
        protocol_id="LP15",
        label_name="volume_confirmation",
        value=confirmation,
        confidence=confidence,
        metadata={
            "confirmation_name": conf_name,
            "avg_up_volume": float(avg_up_volume),
            "avg_down_volume": float(avg_down_volume),
            "volume_ratio": float(vol_ratio),
            "price_change": float(recent_price_change),
            "num_classes": 3,
        }
    )
