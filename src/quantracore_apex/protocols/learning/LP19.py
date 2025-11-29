"""
LP19 - Mean Reversion Potential Label Protocol

Measures mean reversion probability and potential.
Category: Advanced Labels - Reversion Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP19: Generate mean reversion potential label.
    
    Classifies mean reversion opportunity:
    - High potential (extended from mean)
    - Moderate potential
    - Low potential (near mean)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP19",
            label_name="mean_reversion",
            value=2,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    
    sma_20 = np.mean(closes[-20:])
    np.mean(closes) if len(closes) >= 50 else np.mean(closes)
    
    std_20 = np.std(closes[-20:])
    
    z_score = (closes[-1] - sma_20) / (std_20 + 1e-10)
    
    pct_from_mean = (closes[-1] - sma_20) / (sma_20 + 1e-10)
    
    entropy = apex_result.entropy_metrics.combined_entropy
    
    recent_closes = closes[-10:]
    recent_returns = np.diff(recent_closes) / recent_closes[:-1]
    recent_volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0
    
    all_returns = np.diff(closes) / closes[:-1]
    historical_volatility = np.std(all_returns) if len(all_returns) > 0 else 0
    vol_ratio = recent_volatility / (historical_volatility + 1e-10)
    
    reversion_score = 0.0
    
    if abs(z_score) > 2.5:
        reversion_score += 0.4
    elif abs(z_score) > 2.0:
        reversion_score += 0.3
    elif abs(z_score) > 1.5:
        reversion_score += 0.2
    
    if abs(pct_from_mean) > 0.08:
        reversion_score += 0.25
    elif abs(pct_from_mean) > 0.05:
        reversion_score += 0.15
    
    if vol_ratio > 1.5:
        reversion_score += 0.15
    
    if entropy < 0.3:
        reversion_score += 0.1
    
    if reversion_score > 0.5:
        potential = 0
        potential_name = "high_potential"
    elif reversion_score > 0.25:
        potential = 1
        potential_name = "moderate_potential"
    else:
        potential = 2
        potential_name = "low_potential"
    
    reversion_direction = "down" if z_score > 0 else "up"
    
    confidence = min(0.85, 0.4 + reversion_score * 0.5)
    
    return LearningLabel(
        protocol_id="LP19",
        label_name="mean_reversion",
        value=potential,
        confidence=confidence,
        metadata={
            "potential_name": potential_name,
            "reversion_direction": reversion_direction,
            "z_score": float(z_score),
            "pct_from_mean": float(pct_from_mean),
            "reversion_score": float(reversion_score),
            "vol_ratio": float(vol_ratio),
            "num_classes": 3,
        }
    )
