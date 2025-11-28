"""
MR01 — Compression Explosion Detector

Detects extreme price compression followed by potential explosive breakout.
Analyzes ATR contraction, Bollinger Band squeeze, and range tightening.

This is Stage 1 deterministic implementation using heuristic rules.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR01Result:
    """Result of MR01 Compression Explosion detection."""
    protocol_id: str = "MR01"
    fired: bool = False
    compression_score: float = 0.0
    atr_contraction_ratio: float = 0.0
    range_tightness: float = 0.0
    bollinger_squeeze: float = 0.0
    explosion_probability: float = 0.0
    direction_bias: str = "neutral"
    confidence: float = 0.0
    notes: str = ""


def run_MR01(bars: List[OhlcvBar], lookback: int = 20) -> MR01Result:
    """
    Execute MR01 Compression Explosion Detector.
    
    Identifies extreme compression states that often precede
    large directional moves. Uses deterministic rules only.
    
    Args:
        bars: OHLCV price bars (minimum 50 required)
        lookback: Period for compression calculation
        
    Returns:
        MR01Result with compression metrics and explosion probability
        
    Protocol Logic:
    1. Calculate ATR contraction vs historical average
    2. Measure Bollinger Band width squeeze
    3. Compute price range tightening
    4. Combine into compression score
    5. Estimate explosion probability based on compression depth
    """
    if len(bars) < 50:
        return MR01Result(notes="Insufficient data (need 50+ bars)")
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    
    if len(tr) < lookback:
        return MR01Result(notes="Insufficient TR data")
    
    current_atr = float(np.mean(tr[-lookback:]))
    historical_atr = float(np.mean(tr[:-lookback])) if len(tr) > lookback else current_atr
    
    atr_contraction_ratio = current_atr / max(historical_atr, 1e-10)
    
    recent_closes = closes[-lookback:]
    bb_std = float(np.std(recent_closes))
    bb_mean = float(np.mean(recent_closes))
    bollinger_width = (2 * bb_std) / max(bb_mean, 1e-10)
    
    historical_std = float(np.std(closes[:-lookback])) if len(closes) > lookback else bb_std
    historical_mean = float(np.mean(closes[:-lookback])) if len(closes) > lookback else bb_mean
    historical_bb_width = (2 * historical_std) / max(historical_mean, 1e-10)
    
    bollinger_squeeze = bollinger_width / max(historical_bb_width, 1e-10)
    
    recent_range = float(np.max(highs[-lookback:]) - np.min(lows[-lookback:]))
    historical_range = float(np.max(highs[:-lookback]) - np.min(lows[:-lookback])) if len(highs) > lookback else recent_range
    range_tightness = recent_range / max(historical_range, 1e-10)
    
    compression_score = (
        (1.0 - min(atr_contraction_ratio, 1.0)) * 0.4 +
        (1.0 - min(bollinger_squeeze, 1.0)) * 0.35 +
        (1.0 - min(range_tightness, 1.0)) * 0.25
    )
    compression_score = float(np.clip(compression_score, 0.0, 1.0))
    
    explosion_probability = 0.0
    if compression_score > 0.3:
        explosion_probability = min(compression_score * 1.2, 0.85)
    
    recent_trend = closes[-1] - closes[-lookback] if len(closes) >= lookback else 0
    if recent_trend > 0:
        direction_bias = "bullish"
    elif recent_trend < 0:
        direction_bias = "bearish"
    else:
        direction_bias = "neutral"
    
    fired = compression_score >= 0.5 and explosion_probability >= 0.4
    
    confidence = min(compression_score * 100, 90.0)
    
    return MR01Result(
        fired=fired,
        compression_score=round(compression_score, 4),
        atr_contraction_ratio=round(atr_contraction_ratio, 4),
        range_tightness=round(range_tightness, 4),
        bollinger_squeeze=round(bollinger_squeeze, 4),
        explosion_probability=round(explosion_probability, 4),
        direction_bias=direction_bias,
        confidence=round(confidence, 2),
        notes="Compression explosion precursor detected" if fired else "No significant compression",
    )
