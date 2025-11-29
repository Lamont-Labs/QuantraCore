"""
MR16 — Parabolic Phase 3 Detector

Detects late-stage parabolic moves (Phase 3 blow-off tops).
Identifies unsustainable acceleration patterns.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR16Result:
    """Result of MR16 Parabolic Phase 3 detection."""
    protocol_id: str = "MR16"
    fired: bool = False
    parabolic_score: float = 0.0
    five_day_return: float = 0.0
    acceleration_rate: float = 0.0
    is_blow_off: bool = False
    exhaustion_risk: float = 0.0
    confidence: float = 0.0
    notes: str = ""


def run_MR16(bars: List[OhlcvBar], lookback: int = 20) -> MR16Result:
    """
    Execute MR16 Parabolic Phase 3 Detector.
    
    Identifies late-stage parabolic moves that often
    precede sharp reversals.
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for trend analysis
        
    Returns:
        MR16Result with parabolic phase metrics
        
    Protocol Logic:
    1. Calculate 5-day cumulative return
    2. Measure acceleration (return rate increasing)
    3. Detect blow-off characteristics
    4. Score exhaustion risk
    """
    if len(bars) < 30:
        return MR16Result(notes="Insufficient data (need 30+ bars)")
    
    closes = np.array([b.close for b in bars])
    
    five_day_return = (closes[-1] - closes[-6]) / max(closes[-6], 1e-10)
    
    returns_3d = (closes[-1] - closes[-4]) / max(closes[-4], 1e-10)
    returns_3d_prior = (closes[-4] - closes[-7]) / max(closes[-7], 1e-10)
    
    acceleration_rate = returns_3d - returns_3d_prior
    
    is_blow_off = five_day_return > 0.40 and acceleration_rate > 0.10
    
    if five_day_return > 0.40:
        exhaustion_risk = min(five_day_return * 1.5 + acceleration_rate * 2, 1.0)
    elif five_day_return > 0.20:
        exhaustion_risk = five_day_return + acceleration_rate
    else:
        exhaustion_risk = 0.0
    
    exhaustion_risk = float(np.clip(exhaustion_risk, 0.0, 1.0))
    
    parabolic_score = 0.0
    
    if five_day_return > 0.40:
        parabolic_score += 0.5
    elif five_day_return > 0.25:
        parabolic_score += 0.35
    elif five_day_return > 0.15:
        parabolic_score += 0.2
    
    if acceleration_rate > 0.10:
        parabolic_score += 0.3
    elif acceleration_rate > 0.05:
        parabolic_score += 0.2
    
    if is_blow_off:
        parabolic_score += 0.2
    
    parabolic_score = float(np.clip(parabolic_score, 0.0, 1.0))
    fired = parabolic_score >= 0.6
    
    confidence = min(parabolic_score * 100, 92.0)
    
    notes_parts = []
    if is_blow_off:
        notes_parts.append("BLOW-OFF TOP WARNING")
    if five_day_return > 0.40:
        notes_parts.append(f"Parabolic 5-day: +{five_day_return*100:.1f}%")
    if exhaustion_risk > 0.7:
        notes_parts.append("High exhaustion risk")
    
    return MR16Result(
        fired=fired,
        parabolic_score=round(parabolic_score, 4),
        five_day_return=round(float(five_day_return), 4),
        acceleration_rate=round(float(acceleration_rate), 4),
        is_blow_off=is_blow_off,
        exhaustion_risk=round(exhaustion_risk, 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No parabolic pattern detected",
    )
