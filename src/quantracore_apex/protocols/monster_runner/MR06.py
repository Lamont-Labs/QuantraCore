"""
MR06 — Bollinger Breakout Detector

Detects Bollinger Band breakouts after squeeze conditions.
Identifies compression followed by explosive directional moves.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR06Result:
    """Result of MR06 Bollinger Breakout detection."""
    protocol_id: str = "MR06"
    fired: bool = False
    breakout_score: float = 0.0
    squeeze_depth: float = 0.0
    breakout_direction: str = "neutral"
    band_width: float = 0.0
    price_vs_upper: float = 0.0
    price_vs_lower: float = 0.0
    confidence: float = 0.0
    notes: str = ""


def run_MR06(bars: List[OhlcvBar], lookback: int = 20) -> MR06Result:
    """
    Execute MR06 Bollinger Breakout Detector.
    
    Identifies Bollinger Band breakouts after squeeze conditions
    that often precede large directional moves.
    
    Args:
        bars: OHLCV price bars (minimum 50 required)
        lookback: Period for Bollinger calculation
        
    Returns:
        MR06Result with breakout metrics
        
    Protocol Logic:
    1. Calculate Bollinger Bands (20-period, 2 std dev)
    2. Measure band width (squeeze indicator)
    3. Detect breakout above upper or below lower band
    4. Score based on squeeze depth and breakout strength
    """
    if len(bars) < 50:
        return MR06Result(notes="Insufficient data (need 50+ bars)")
    
    closes = np.array([b.close for b in bars])
    
    sma = np.convolve(closes, np.ones(lookback)/lookback, mode='valid')
    
    if len(sma) < lookback:
        return MR06Result(notes="Insufficient data for SMA calculation")
    
    rolling_std = np.array([np.std(closes[i:i+lookback]) for i in range(len(closes)-lookback+1)])
    
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std
    
    current_price = closes[-1]
    current_upper = upper_band[-1]
    current_lower = lower_band[-1]
    current_sma = sma[-1]
    
    band_width = (current_upper - current_lower) / max(current_sma, 1e-10)
    
    historical_widths = (upper_band - lower_band) / np.maximum(sma, 1e-10)
    min_width = np.min(historical_widths[-10:]) if len(historical_widths) >= 10 else band_width
    squeeze_depth = 1.0 - (min_width / max(np.mean(historical_widths), 1e-10))
    squeeze_depth = float(np.clip(squeeze_depth, 0.0, 1.0))
    
    price_vs_upper = (current_price - current_upper) / max(abs(current_upper), 1e-10)
    price_vs_lower = (current_price - current_lower) / max(abs(current_lower), 1e-10)
    
    breakout_direction = "neutral"
    breakout_score = 0.0
    
    if current_price > current_upper:
        breakout_direction = "bullish"
        breakout_score = min(0.5 + squeeze_depth * 0.3 + price_vs_upper * 2, 1.0)
    elif current_price < current_lower:
        breakout_direction = "bearish"
        breakout_score = min(0.5 + squeeze_depth * 0.3 + abs(price_vs_lower) * 2, 1.0)
    else:
        if squeeze_depth > 0.7:
            breakout_score = squeeze_depth * 0.4
    
    breakout_score = float(np.clip(breakout_score, 0.0, 1.0))
    fired = breakout_score >= 0.6
    
    confidence = min(breakout_score * 100, 89.0)
    
    notes_parts = []
    if breakout_direction != "neutral":
        notes_parts.append(f"{breakout_direction.capitalize()} breakout")
    if squeeze_depth > 0.7:
        notes_parts.append("Post-squeeze")
    if squeeze_depth > 0.5:
        notes_parts.append(f"Squeeze depth: {squeeze_depth:.2f}")
    
    return MR06Result(
        fired=fired,
        breakout_score=round(breakout_score, 4),
        squeeze_depth=round(squeeze_depth, 4),
        breakout_direction=breakout_direction,
        band_width=round(float(band_width), 4),
        price_vs_upper=round(float(price_vs_upper), 4),
        price_vs_lower=round(float(price_vs_lower), 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No breakout detected",
    )
