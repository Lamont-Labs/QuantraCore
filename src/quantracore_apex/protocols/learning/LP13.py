"""
LP13 - Momentum Persistence Label Protocol

Measures momentum persistence probability for training.
Category: Advanced Labels - Momentum Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP13: Generate momentum persistence label.
    
    Classifies momentum strength and persistence:
    - Strong persistent (trending consistently)
    - Weak/fading (losing steam)
    - Reversing (momentum flipping)
    """
    bars = window.bars
    if len(bars) < 20:
        return LearningLabel(
            protocol_id="LP13",
            label_name="momentum_persistence",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-20:]])
    returns = np.diff(closes) / closes[:-1]
    
    recent_momentum = np.mean(returns[-5:])
    prior_momentum = np.mean(returns[-10:-5])
    
    momentum_consistency = np.sum(np.sign(returns[-10:]) == np.sign(recent_momentum)) / 10
    
    momentum_acceleration = recent_momentum - prior_momentum
    
    if abs(recent_momentum) > 0.01 and momentum_consistency > 0.7:
        if momentum_acceleration > 0:
            persistence = 0
            persistence_name = "strong_accelerating"
        else:
            persistence = 1
            persistence_name = "strong_decelerating"
    elif np.sign(recent_momentum) != np.sign(prior_momentum) and abs(recent_momentum) > 0.005:
        persistence = 3
        persistence_name = "reversing"
    else:
        persistence = 2
        persistence_name = "weak_fading"
    
    confidence = min(0.9, 0.4 + momentum_consistency * 0.5)
    
    return LearningLabel(
        protocol_id="LP13",
        label_name="momentum_persistence",
        value=persistence,
        confidence=confidence,
        metadata={
            "persistence_name": persistence_name,
            "recent_momentum": float(recent_momentum),
            "prior_momentum": float(prior_momentum),
            "consistency": float(momentum_consistency),
            "acceleration": float(momentum_acceleration),
            "num_classes": 4,
        }
    )
