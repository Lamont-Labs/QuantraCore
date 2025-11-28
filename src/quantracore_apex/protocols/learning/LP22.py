"""
LP22 - Market Phase Classification Label Protocol

Classifies current market phase/cycle.
Category: Advanced Labels - Market Structure
"""

import numpy as np
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from .learning_loader import LearningLabel


def generate_label(window: OhlcvWindow, apex_result: ApexResult) -> LearningLabel:
    """
    LP22: Generate market phase classification label.
    
    Classifies market phase (Wyckoff-inspired):
    - Accumulation (base building)
    - Markup (uptrend)
    - Distribution (topping)
    - Markdown (downtrend)
    """
    bars = window.bars
    if len(bars) < 40:
        return LearningLabel(
            protocol_id="LP22",
            label_name="market_phase",
            value=0,
            confidence=0.0,
            metadata={"status": "insufficient_data"}
        )
    
    closes = np.array([b.close for b in bars[-40:]])
    volumes = np.array([b.volume for b in bars[-40:]])
    highs = np.array([b.high for b in bars[-40:]])
    lows = np.array([b.low for b in bars[-40:]])
    
    trend = np.polyfit(range(len(closes)), closes, 1)[0]
    trend_pct = trend / (np.mean(closes) + 1e-10)
    
    range_expansion = (highs.max() - lows.min()) / (np.mean(closes) + 1e-10)
    
    vol_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
    vol_trend_pct = vol_trend / (np.mean(volumes) + 1e-10)
    
    compression = apex_result.microtraits.compression_score
    
    volatility = np.std(np.diff(closes) / closes[:-1])
    
    phase_scores = {
        "accumulation": 0.0,
        "markup": 0.0,
        "distribution": 0.0,
        "markdown": 0.0,
    }
    
    if trend_pct > 0.001:
        phase_scores["markup"] += 0.4
        if vol_trend_pct > 0:
            phase_scores["markup"] += 0.2
    elif trend_pct < -0.001:
        phase_scores["markdown"] += 0.4
        if vol_trend_pct > 0:
            phase_scores["markdown"] += 0.2
    
    if compression > 0.6 and abs(trend_pct) < 0.0005:
        if closes[-1] < np.mean(closes):
            phase_scores["accumulation"] += 0.5
        else:
            phase_scores["distribution"] += 0.5
    
    if volatility < 0.015 and compression > 0.5:
        if trend_pct > 0:
            phase_scores["accumulation"] += 0.2
        else:
            phase_scores["distribution"] += 0.2
    
    if closes[-1] > np.percentile(closes, 75) and volatility > 0.02:
        phase_scores["distribution"] += 0.3
    elif closes[-1] < np.percentile(closes, 25) and volatility > 0.02:
        phase_scores["accumulation"] += 0.3
    
    best_phase = max(phase_scores, key=phase_scores.get)
    phase_map = {
        "accumulation": 0,
        "markup": 1,
        "distribution": 2,
        "markdown": 3,
    }
    
    phase = phase_map[best_phase]
    confidence = min(0.85, 0.3 + phase_scores[best_phase])
    
    return LearningLabel(
        protocol_id="LP22",
        label_name="market_phase",
        value=phase,
        confidence=confidence,
        metadata={
            "phase_name": best_phase,
            "phase_scores": {k: float(v) for k, v in phase_scores.items()},
            "trend_pct": float(trend_pct),
            "vol_trend_pct": float(vol_trend_pct),
            "compression": float(compression),
            "num_classes": 4,
        }
    )
