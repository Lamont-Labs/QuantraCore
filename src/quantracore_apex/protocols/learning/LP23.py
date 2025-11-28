"""
LP23 - Risk-Adjusted Opportunity Score Label Protocol

Generates risk-adjusted opportunity scores for trade selection.
Category: Advanced Labels - Risk Analysis
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP23: Generate risk-adjusted opportunity score label.
    
    Classifies opportunity quality:
    - High risk/reward (favorable setup)
    - Moderate risk/reward
    - Poor risk/reward (unfavorable)
    """
    bars = window.bars
    if len(bars) < 30:
        return LearningLabel(
            protocol_id="LP23",
            label_name="risk_adjusted_score",
            value=1,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-30:]])
    highs = np.array([b.high for b in bars[-30:]])
    lows = np.array([b.low for b in bars[-30:]])
    
    atr = np.mean(highs - lows)
    current_price = closes[-1]
    
    recent_high = np.max(highs[-10:])
    recent_low = np.min(lows[-10:])
    
    dist_to_resistance = recent_high - current_price
    dist_to_support = current_price - recent_low
    
    if dist_to_support > 0:
        rr_long = dist_to_resistance / dist_to_support
    else:
        rr_long = 0
    
    if dist_to_resistance > 0:
        rr_short = dist_to_support / dist_to_resistance
    else:
        rr_short = 0
    
    best_rr = max(rr_long, rr_short)
    trade_direction = "long" if rr_long > rr_short else "short"
    
    volatility = np.std(np.diff(closes) / closes[:-1])
    
    compression = apex_result.microtraits.compression_score
    
    stop_distance = atr * 1.5
    risk_pct = stop_distance / current_price
    
    opportunity_score = 0.0
    
    if best_rr > 3:
        opportunity_score += 0.4
    elif best_rr > 2:
        opportunity_score += 0.25
    elif best_rr > 1.5:
        opportunity_score += 0.1
    
    if compression > 0.6:
        opportunity_score += 0.2
    
    if volatility < 0.02:
        opportunity_score += 0.15
    elif volatility > 0.04:
        opportunity_score -= 0.15
    
    if risk_pct < 0.02:
        opportunity_score += 0.15
    elif risk_pct > 0.05:
        opportunity_score -= 0.1
    
    if opportunity_score > 0.5:
        quality = 0
        quality_name = "high_opportunity"
    elif opportunity_score > 0.2:
        quality = 1
        quality_name = "moderate_opportunity"
    else:
        quality = 2
        quality_name = "poor_opportunity"
    
    confidence = min(0.85, 0.4 + opportunity_score * 0.5)
    
    return LearningLabel(
        protocol_id="LP23",
        label_name="risk_adjusted_score",
        value=quality,
        confidence=confidence,
        metadata={
            "quality_name": quality_name,
            "opportunity_score": float(opportunity_score),
            "best_rr_ratio": float(best_rr),
            "trade_direction": trade_direction,
            "risk_pct": float(risk_pct),
            "compression": float(compression),
            "num_classes": 3,
        }
    )
