"""
LP18 - Trend Exhaustion Label Protocol

Detects trend exhaustion probability for reversal prediction.
Category: Advanced Labels - Trend Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP18: Generate trend exhaustion label.
    
    Classifies trend exhaustion state:
    - Fresh trend (early stages)
    - Mature trend (mid-stage)
    - Exhausted trend (late stage, reversal likely)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP18",
            label_name="trend_exhaustion",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    
    returns = np.diff(closes) / closes[:-1]
    trend_direction = 1 if np.mean(returns) > 0 else -1
    
    cumulative_move = (closes[-1] / closes[0] - 1) * trend_direction
    
    acceleration = returns[-5:].mean() - returns[-15:-5].mean()
    acceleration *= trend_direction
    
    vol_decay = volumes[-5:].mean() / (volumes[-20:-5].mean() + 1e-10)
    
    wicks = highs - lows
    bodies = np.abs(closes - np.array([b.open for b in bars[-30:]]))
    wick_ratio = np.mean((wicks - bodies) / (wicks + 1e-10))
    
    exhaustion_score = 0.0
    
    if cumulative_move > 0.15:
        exhaustion_score += 0.3
    elif cumulative_move > 0.08:
        exhaustion_score += 0.15
    
    if acceleration < -0.002:
        exhaustion_score += 0.25
    elif acceleration < 0:
        exhaustion_score += 0.1
    
    if vol_decay < 0.7:
        exhaustion_score += 0.2
    elif vol_decay < 0.9:
        exhaustion_score += 0.1
    
    if wick_ratio > 0.5:
        exhaustion_score += 0.15
    
    if exhaustion_score > 0.5:
        exhaustion = 2
        exhaustion_name = "exhausted"
    elif exhaustion_score > 0.25:
        exhaustion = 1
        exhaustion_name = "mature"
    else:
        exhaustion = 0
        exhaustion_name = "fresh"
    
    confidence = min(0.85, 0.4 + exhaustion_score * 0.6)
    
    return LearningLabel(
        protocol_id="LP18",
        label_name="trend_exhaustion",
        value=exhaustion,
        confidence=confidence,
        metadata={
            "exhaustion_name": exhaustion_name,
            "exhaustion_score": float(exhaustion_score),
            "cumulative_move": float(cumulative_move),
            "acceleration": float(acceleration),
            "vol_decay": float(vol_decay),
            "wick_ratio": float(wick_ratio),
            "num_classes": 3,
        }
    )
