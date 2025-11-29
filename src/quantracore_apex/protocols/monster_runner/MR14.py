"""
MR14 — Fractal Explosion Detector

Detects breakouts to new 20-day highs/lows with momentum.
Identifies fractal breakout patterns.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR14Result:
    """Result of MR14 Fractal Explosion detection."""
    protocol_id: str = "MR14"
    fired: bool = False
    fractal_score: float = 0.0
    is_new_high: bool = False
    is_new_low: bool = False
    breakout_strength: float = 0.0
    days_since_level: int = 0
    confidence: float = 0.0
    notes: str = ""


def run_MR14(bars: List[OhlcvBar], lookback: int = 20) -> MR14Result:
    """
    Execute MR14 Fractal Explosion Detector.
    
    Identifies price breaking to new highs or lows with
    strong momentum.
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for high/low comparison
        
    Returns:
        MR14Result with fractal breakout metrics
        
    Protocol Logic:
    1. Check if current price is new 20-day high/low
    2. Calculate breakout strength (% above/below prior level)
    3. Determine how long the level held
    4. Score fractal breakout probability
    """
    if len(bars) < 30:
        return MR14Result(notes="Insufficient data (need 30+ bars)")
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    prior_high = np.max(highs[-lookback-1:-1])
    prior_low = np.min(lows[-lookback-1:-1])
    
    current_high = highs[-1]
    current_low = lows[-1]
    current_close = closes[-1]
    
    is_new_high = current_high > prior_high
    is_new_low = current_low < prior_low
    
    if is_new_high:
        breakout_strength = (current_high - prior_high) / max(prior_high, 1e-10)
    elif is_new_low:
        breakout_strength = (prior_low - current_low) / max(prior_low, 1e-10)
    else:
        breakout_strength = 0.0
    
    days_since_level = 0
    if is_new_high:
        for i in range(2, min(lookback + 1, len(highs))):
            if highs[-i] == prior_high:
                days_since_level = i - 1
                break
    elif is_new_low:
        for i in range(2, min(lookback + 1, len(lows))):
            if lows[-i] == prior_low:
                days_since_level = i - 1
                break
    
    fractal_score = 0.0
    
    if is_new_high or is_new_low:
        fractal_score += 0.4
    
    if breakout_strength > 0.02:
        fractal_score += min(breakout_strength * 10, 0.3)
    
    if days_since_level >= 15:
        fractal_score += 0.2
    elif days_since_level >= 10:
        fractal_score += 0.15
    elif days_since_level >= 5:
        fractal_score += 0.1
    
    fractal_score = float(np.clip(fractal_score, 0.0, 1.0))
    fired = fractal_score >= 0.5
    
    confidence = min(fractal_score * 100, 85.0)
    
    notes_parts = []
    if is_new_high:
        notes_parts.append(f"New {lookback}-day high")
    elif is_new_low:
        notes_parts.append(f"New {lookback}-day low")
    if breakout_strength > 0.02:
        notes_parts.append(f"Breakout strength: {breakout_strength*100:.2f}%")
    if days_since_level > 0:
        notes_parts.append(f"Level held for {days_since_level} days")
    
    return MR14Result(
        fired=fired,
        fractal_score=round(fractal_score, 4),
        is_new_high=is_new_high,
        is_new_low=is_new_low,
        breakout_strength=round(float(breakout_strength), 4),
        days_since_level=days_since_level,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No fractal breakout detected",
    )
