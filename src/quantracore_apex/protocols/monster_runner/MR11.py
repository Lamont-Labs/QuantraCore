"""
MR11 — Short Squeeze Gamma Detector

Detects short squeeze and gamma squeeze conditions.
Identifies rapid price acceleration with extreme volume.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR11Result:
    """Result of MR11 Short Squeeze Gamma detection."""
    protocol_id: str = "MR11"
    fired: bool = False
    squeeze_score: float = 0.0
    price_acceleration: float = 0.0
    volume_surge: float = 0.0
    consecutive_up_days: int = 0
    parabolic_move: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR11(bars: List[OhlcvBar], lookback: int = 10) -> MR11Result:
    """
    Execute MR11 Short Squeeze Gamma Detector.
    
    Identifies parabolic price acceleration patterns typical
    of short squeeze or gamma squeeze events.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for analysis
        
    Returns:
        MR11Result with squeeze metrics
        
    Protocol Logic:
    1. Calculate price acceleration over multiple periods
    2. Measure volume surge relative to average
    3. Count consecutive up days
    4. Detect parabolic move characteristics
    """
    if len(bars) < 20:
        return MR11Result(notes="Insufficient data (need 20+ bars)")
    
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    pct_change_3d = (closes[-1] - closes[-4]) / max(closes[-4], 1e-10)
    pct_change_5d = (closes[-1] - closes[-6]) / max(closes[-6], 1e-10)
    
    price_acceleration = pct_change_3d + pct_change_5d * 0.5
    
    avg_volume = float(np.mean(volumes[-lookback:-1]))
    recent_volume = float(np.mean(volumes[-3:]))
    volume_surge = recent_volume / max(avg_volume, 1.0)
    
    consecutive_up_days = 0
    for i in range(1, min(11, len(closes))):
        if closes[-i] > closes[-i-1]:
            consecutive_up_days += 1
        else:
            break
    
    parabolic_move = (
        pct_change_3d > 0.25 and 
        volume_surge > 5.0 and 
        consecutive_up_days >= 3
    )
    
    squeeze_score = 0.0
    
    if pct_change_3d > 0.25:
        squeeze_score += 0.4
    elif pct_change_3d > 0.15:
        squeeze_score += 0.25
    elif pct_change_3d > 0.08:
        squeeze_score += 0.15
    
    if volume_surge > 5.0:
        squeeze_score += 0.3
    elif volume_surge > 3.0:
        squeeze_score += 0.2
    
    if consecutive_up_days >= 5:
        squeeze_score += 0.2
    elif consecutive_up_days >= 3:
        squeeze_score += 0.1
    
    if parabolic_move:
        squeeze_score += 0.1
    
    squeeze_score = float(np.clip(squeeze_score, 0.0, 1.0))
    fired = squeeze_score >= 0.6
    
    confidence = min(squeeze_score * 100, 94.0)
    
    notes_parts = []
    if parabolic_move:
        notes_parts.append("Parabolic squeeze detected")
    if pct_change_3d > 0.25:
        notes_parts.append(f"3-day move: +{pct_change_3d*100:.1f}%")
    if volume_surge > 5.0:
        notes_parts.append(f"Volume {volume_surge:.1f}x surge")
    if consecutive_up_days >= 5:
        notes_parts.append(f"{consecutive_up_days} consecutive up days")
    
    return MR11Result(
        fired=fired,
        squeeze_score=round(squeeze_score, 4),
        price_acceleration=round(float(price_acceleration), 4),
        volume_surge=round(float(volume_surge), 4),
        consecutive_up_days=consecutive_up_days,
        parabolic_move=parabolic_move,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No squeeze pattern detected",
    )
