"""
MR13 — News Catalyst Detector

Detects large gap moves indicative of news catalysts.
Identifies potential M&A, earnings, or regulatory news events.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR13Result:
    """Result of MR13 News Catalyst detection."""
    protocol_id: str = "MR13"
    fired: bool = False
    catalyst_score: float = 0.0
    gap_magnitude: float = 0.0
    gap_direction: str = "neutral"
    volume_spike: bool = False
    potential_catalyst: str = "unknown"
    confidence: float = 0.0
    notes: str = ""


def run_MR13(bars: List[OhlcvBar], lookback: int = 20) -> MR13Result:
    """
    Execute MR13 News Catalyst Detector.
    
    Identifies large overnight gaps that suggest significant
    news catalysts.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for volume comparison
        
    Returns:
        MR13Result with catalyst detection metrics
        
    Protocol Logic:
    1. Calculate overnight gap percentage
    2. Assess volume spike
    3. Classify potential catalyst type based on gap size
    4. Score catalyst probability
    """
    if len(bars) < 20:
        return MR13Result(notes="Insufficient data (need 20+ bars)")
    
    current_bar = bars[-1]
    prior_bar = bars[-2]
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    gap_magnitude = abs(current_bar.open - prior_bar.close) / max(prior_bar.close, 1e-10)
    gap_direction = "up" if current_bar.open > prior_bar.close else "down" if current_bar.open < prior_bar.close else "neutral"
    
    avg_volume = float(np.mean(volumes[-lookback:-1]))
    current_volume = volumes[-1]
    volume_spike = current_volume > 3.0 * avg_volume
    
    if gap_magnitude > 0.30:
        potential_catalyst = "major_news_or_ma"
    elif gap_magnitude > 0.15:
        potential_catalyst = "earnings_or_guidance"
    elif gap_magnitude > 0.08:
        potential_catalyst = "analyst_rating"
    elif gap_magnitude > 0.03:
        potential_catalyst = "sector_news"
    else:
        potential_catalyst = "none_detected"
    
    catalyst_score = 0.0
    
    if gap_magnitude > 0.15:
        catalyst_score += 0.5
    elif gap_magnitude > 0.08:
        catalyst_score += 0.35
    elif gap_magnitude > 0.03:
        catalyst_score += 0.2
    
    if volume_spike:
        catalyst_score += 0.3
    
    if gap_magnitude > 0.10 and volume_spike:
        catalyst_score += 0.2
    
    catalyst_score = float(np.clip(catalyst_score, 0.0, 1.0))
    fired = catalyst_score >= 0.6
    
    confidence = min(catalyst_score * 100, 90.0)
    
    notes_parts = []
    if gap_magnitude > 0.15:
        notes_parts.append(f"Major gap: {gap_direction} {gap_magnitude*100:.1f}%")
    elif gap_magnitude > 0.08:
        notes_parts.append(f"Significant gap: {gap_direction} {gap_magnitude*100:.1f}%")
    if potential_catalyst != "none_detected":
        notes_parts.append(f"Potential: {potential_catalyst.replace('_', ' ')}")
    if volume_spike:
        notes_parts.append("High volume confirmation")
    
    return MR13Result(
        fired=fired,
        catalyst_score=round(catalyst_score, 4),
        gap_magnitude=round(float(gap_magnitude), 4),
        gap_direction=gap_direction,
        volume_spike=volume_spike,
        potential_catalyst=potential_catalyst,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No catalyst pattern detected",
    )
