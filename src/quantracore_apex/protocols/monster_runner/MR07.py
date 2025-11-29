"""
MR07 — Volume Explosion Detector

Detects simultaneous volume and price explosion events.
Identifies institutional-level buying/selling pressure.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR07Result:
    """Result of MR07 Volume Explosion detection."""
    protocol_id: str = "MR07"
    fired: bool = False
    explosion_score: float = 0.0
    volume_multiple: float = 0.0
    price_change_pct: float = 0.0
    direction: str = "neutral"
    institutional_signature: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR07(bars: List[OhlcvBar], lookback: int = 20) -> MR07Result:
    """
    Execute MR07 Volume Explosion Detector.
    
    Identifies combined volume and price explosions that
    indicate strong directional momentum.
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for baseline calculation
        
    Returns:
        MR07Result with explosion metrics
        
    Protocol Logic:
    1. Calculate volume multiple vs average
    2. Calculate price change percentage
    3. Detect institutional signature (high volume + directional)
    4. Score explosion intensity
    """
    if len(bars) < 30:
        return MR07Result(notes="Insufficient data (need 30+ bars)")
    
    volumes = np.array([b.volume for b in bars], dtype=float)
    closes = np.array([b.close for b in bars])
    
    if np.all(volumes == 0):
        return MR07Result(notes="No volume data available")
    
    avg_volume = float(np.mean(volumes[-lookback:-1]))
    current_volume = volumes[-1]
    
    volume_multiple = current_volume / max(avg_volume, 1.0)
    
    price_change_pct = (closes[-1] - closes[-2]) / max(closes[-2], 1e-10)
    
    direction = "bullish" if price_change_pct > 0 else "bearish" if price_change_pct < 0 else "neutral"
    
    institutional_signature = volume_multiple > 4.0 and abs(price_change_pct) > 0.03
    
    explosion_score = 0.0
    
    if volume_multiple > 2.0:
        explosion_score += min(volume_multiple * 0.1, 0.4)
    
    if abs(price_change_pct) > 0.05:
        explosion_score += min(abs(price_change_pct) * 5, 0.4)
    
    if institutional_signature:
        explosion_score += 0.2
    
    explosion_score = float(np.clip(explosion_score, 0.0, 1.0))
    fired = explosion_score >= 0.6
    
    confidence = min(explosion_score * 100, 93.0)
    
    notes_parts = []
    if volume_multiple > 4.0:
        notes_parts.append(f"Volume {volume_multiple:.1f}x average")
    if abs(price_change_pct) > 0.08:
        notes_parts.append(f"Price move {price_change_pct*100:.1f}%")
    if institutional_signature:
        notes_parts.append("Institutional signature detected")
    
    return MR07Result(
        fired=fired,
        explosion_score=round(explosion_score, 4),
        volume_multiple=round(float(volume_multiple), 4),
        price_change_pct=round(float(price_change_pct), 4),
        direction=direction,
        institutional_signature=institutional_signature,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No explosion detected",
    )
