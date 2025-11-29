"""
MR19 — FOMO Cascade Detector

Detects FOMO (Fear Of Missing Out) cascade patterns.
Identifies late-stage momentum chasing behavior.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR19Result:
    """Result of MR19 FOMO Cascade detection."""
    protocol_id: str = "MR19"
    fired: bool = False
    fomo_score: float = 0.0
    cumulative_return_10d: float = 0.0
    volume_growth: float = 0.0
    consecutive_ups: int = 0
    fomo_stage: str = "none"
    confidence: float = 0.0
    notes: str = ""


def run_MR19(bars: List[OhlcvBar], lookback: int = 10) -> MR19Result:
    """
    Execute MR19 FOMO Cascade Detector.
    
    Identifies FOMO-driven price action characterized by
    accelerating returns and growing volume.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for analysis
        
    Returns:
        MR19Result with FOMO cascade metrics
        
    Protocol Logic:
    1. Calculate cumulative 10-day return
    2. Measure volume growth rate
    3. Count consecutive up days
    4. Classify FOMO stage
    """
    if len(bars) < 20:
        return MR19Result(notes="Insufficient data (need 20+ bars)")
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    cumulative_return_10d = (closes[-1] - closes[-11]) / max(closes[-11], 1e-10)
    
    early_volume = float(np.mean(volumes[-lookback-5:-lookback]))
    late_volume = float(np.mean(volumes[-5:]))
    volume_growth = late_volume / max(early_volume, 1.0)
    
    consecutive_ups = 0
    for i in range(1, min(lookback + 1, len(closes))):
        if closes[-i] > closes[-i-1]:
            consecutive_ups += 1
        else:
            break
    
    if cumulative_return_10d > 0.60 and volume_growth > 2.0:
        fomo_stage = "extreme"
    elif cumulative_return_10d > 0.40 and volume_growth > 1.5:
        fomo_stage = "late"
    elif cumulative_return_10d > 0.25 and consecutive_ups >= 5:
        fomo_stage = "developing"
    elif cumulative_return_10d > 0.15:
        fomo_stage = "early"
    else:
        fomo_stage = "none"
    
    fomo_score = 0.0
    
    if cumulative_return_10d > 0.60:
        fomo_score += 0.5
    elif cumulative_return_10d > 0.40:
        fomo_score += 0.35
    elif cumulative_return_10d > 0.25:
        fomo_score += 0.25
    
    if volume_growth > 2.0:
        fomo_score += 0.3
    elif volume_growth > 1.5:
        fomo_score += 0.2
    
    if consecutive_ups >= 7:
        fomo_score += 0.2
    elif consecutive_ups >= 5:
        fomo_score += 0.15
    
    fomo_score = float(np.clip(fomo_score, 0.0, 1.0))
    fired = fomo_score >= 0.6
    
    confidence = min(fomo_score * 100, 95.0)
    
    notes_parts = []
    if fomo_stage == "extreme":
        notes_parts.append("EXTREME FOMO CASCADE")
    elif fomo_stage == "late":
        notes_parts.append("Late-stage FOMO")
    if cumulative_return_10d > 0.60:
        notes_parts.append(f"10-day return: +{cumulative_return_10d*100:.1f}%")
    if volume_growth > 2.0:
        notes_parts.append(f"Volume growing {volume_growth:.1f}x")
    if consecutive_ups >= 7:
        notes_parts.append(f"{consecutive_ups} consecutive up days")
    
    return MR19Result(
        fired=fired,
        fomo_score=round(fomo_score, 4),
        cumulative_return_10d=round(float(cumulative_return_10d), 4),
        volume_growth=round(float(volume_growth), 4),
        consecutive_ups=consecutive_ups,
        fomo_stage=fomo_stage,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No FOMO cascade detected",
    )
