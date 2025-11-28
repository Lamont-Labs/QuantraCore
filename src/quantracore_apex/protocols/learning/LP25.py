"""
LP25 - Composite Conviction Score Label Protocol

Generates overall conviction score combining all signals.
Category: Advanced Labels - Composite Score
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP25: Generate composite conviction score label.
    
    Combines multiple factors into overall conviction:
    - High conviction (strong alignment)
    - Moderate conviction
    - Low conviction (mixed signals)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP25",
            label_name="composite_conviction",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    volumes = np.array([b.volume for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    
    mt = apex_result.microtraits
    
    trend_score = 0.0
    returns = np.diff(closes) / closes[:-1]
    if np.mean(returns[-5:]) > 0.002:
        trend_score = 0.8
    elif np.mean(returns[-5:]) > 0:
        trend_score = 0.5
    elif np.mean(returns[-5:]) > -0.002:
        trend_score = 0.3
    else:
        trend_score = 0.1
    
    vol_score = 0.0
    recent_vol = volumes[-5:].mean()
    avg_vol = volumes.mean()
    if recent_vol > avg_vol * 1.5:
        vol_score = 0.8
    elif recent_vol > avg_vol:
        vol_score = 0.6
    else:
        vol_score = 0.3
    
    structure_score = 0.0
    if mt.compression_score > 0.7:
        structure_score = 0.8
    elif mt.compression_score > 0.5:
        structure_score = 0.5
    else:
        structure_score = 0.3
    
    consistency_score = 0.0
    sign_consistency = np.sum(np.sign(returns[-10:]) == np.sign(returns[-1])) / 10
    consistency_score = sign_consistency
    
    entropy_factor = 1 - min(apex_result.entropy_metrics.combined_entropy, 1.0)
    
    verdict_confidence = apex_result.verdict.confidence
    
    composite = (
        trend_score * 0.20 +
        vol_score * 0.15 +
        structure_score * 0.15 +
        consistency_score * 0.15 +
        entropy_factor * 0.15 +
        verdict_confidence * 0.20
    )
    
    if composite > 0.65:
        conviction = 0
        conviction_name = "high_conviction"
    elif composite > 0.45:
        conviction = 1
        conviction_name = "moderate_conviction"
    else:
        conviction = 2
        conviction_name = "low_conviction"
    
    direction = "bullish" if np.mean(returns[-5:]) > 0 else "bearish"
    
    confidence = min(0.9, 0.4 + composite * 0.5)
    
    return LearningLabel(
        protocol_id="LP25",
        label_name="composite_conviction",
        value=conviction,
        confidence=confidence,
        metadata={
            "conviction_name": conviction_name,
            "composite_score": float(composite),
            "direction": direction,
            "components": {
                "trend": float(trend_score),
                "volume": float(vol_score),
                "structure": float(structure_score),
                "consistency": float(consistency_score),
                "entropy_factor": float(entropy_factor),
                "verdict_confidence": float(verdict_confidence),
            },
            "num_classes": 3,
        }
    )
