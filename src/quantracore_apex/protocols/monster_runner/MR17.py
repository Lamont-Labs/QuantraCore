"""
MR17 — Meme Stock Frenzy Detector

Detects meme stock frenzy characteristics based on price action.
Identifies retail-driven momentum patterns.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR17Result:
    """Result of MR17 Meme Stock Frenzy detection."""
    protocol_id: str = "MR17"
    fired: bool = False
    frenzy_score: float = 0.0
    volatility_ratio: float = 0.0
    volume_intensity: float = 0.0
    price_swing: float = 0.0
    frenzy_characteristics: int = 0
    confidence: float = 0.0
    notes: str = ""


def run_MR17(bars: List[OhlcvBar], lookback: int = 10) -> MR17Result:
    """
    Execute MR17 Meme Stock Frenzy Detector.
    
    Identifies meme stock frenzy patterns based on
    extreme volatility and volume characteristics.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for baseline
        
    Returns:
        MR17Result with frenzy metrics
        
    Protocol Logic:
    1. Calculate volatility ratio (recent vs historical)
    2. Measure volume intensity
    3. Track price swing magnitude
    4. Count frenzy characteristics
    """
    if len(bars) < 20:
        return MR17Result(notes="Insufficient data (need 20+ bars)")
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    recent_volatility = np.std(closes[-lookback:]) / max(np.mean(closes[-lookback:]), 1e-10)
    historical_volatility = np.std(closes[:-lookback]) / max(np.mean(closes[:-lookback]), 1e-10)
    volatility_ratio = recent_volatility / max(historical_volatility, 1e-10)
    
    avg_volume = float(np.mean(volumes[:-lookback]))
    recent_avg_volume = float(np.mean(volumes[-lookback:]))
    volume_intensity = recent_avg_volume / max(avg_volume, 1.0)
    
    price_swing = (np.max(highs[-lookback:]) - np.min(lows[-lookback:])) / max(np.mean(closes[-lookback:]), 1e-10)
    
    frenzy_characteristics = 0
    
    if volatility_ratio > 2.0:
        frenzy_characteristics += 1
    if volatility_ratio > 3.0:
        frenzy_characteristics += 1
    
    if volume_intensity > 3.0:
        frenzy_characteristics += 1
    if volume_intensity > 5.0:
        frenzy_characteristics += 1
    
    if price_swing > 0.30:
        frenzy_characteristics += 1
    if price_swing > 0.50:
        frenzy_characteristics += 1
    
    frenzy_score = 0.0
    
    if volatility_ratio > 3.0:
        frenzy_score += 0.35
    elif volatility_ratio > 2.0:
        frenzy_score += 0.2
    
    if volume_intensity > 5.0:
        frenzy_score += 0.35
    elif volume_intensity > 3.0:
        frenzy_score += 0.2
    
    if price_swing > 0.50:
        frenzy_score += 0.3
    elif price_swing > 0.30:
        frenzy_score += 0.2
    
    frenzy_score = float(np.clip(frenzy_score, 0.0, 1.0))
    fired = frenzy_score >= 0.6
    
    confidence = min(frenzy_score * 100, 99.0)
    
    notes_parts = []
    if frenzy_characteristics >= 4:
        notes_parts.append("MEME FRENZY DETECTED")
    if volatility_ratio > 3.0:
        notes_parts.append(f"Volatility {volatility_ratio:.1f}x normal")
    if volume_intensity > 5.0:
        notes_parts.append(f"Volume {volume_intensity:.1f}x average")
    if price_swing > 0.50:
        notes_parts.append(f"Price swing: {price_swing*100:.1f}%")
    
    return MR17Result(
        fired=fired,
        frenzy_score=round(frenzy_score, 4),
        volatility_ratio=round(float(volatility_ratio), 4),
        volume_intensity=round(float(volume_intensity), 4),
        price_swing=round(float(price_swing), 4),
        frenzy_characteristics=frenzy_characteristics,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No frenzy pattern detected",
    )
