"""
MR12 — Crypto Pump Detector

Detects extreme price pumps with massive volume.
Identifies parabolic moves typical in high-volatility assets.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR12Result:
    """Result of MR12 Crypto Pump detection."""
    protocol_id: str = "MR12"
    fired: bool = False
    pump_score: float = 0.0
    intraday_move: float = 0.0
    volume_ratio: float = 0.0
    is_pump: bool = False
    is_dump: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR12(bars: List[OhlcvBar], lookback: int = 5) -> MR12Result:
    """
    Execute MR12 Crypto Pump Detector.
    
    Identifies extreme intraday moves with massive volume
    that characterize pump events.
    
    Args:
        bars: OHLCV price bars (minimum 10 required)
        lookback: Period for volume baseline
        
    Returns:
        MR12Result with pump detection metrics
        
    Protocol Logic:
    1. Calculate intraday price move percentage
    2. Compare current volume to recent average
    3. Classify as pump or dump based on direction
    4. Score based on magnitude and volume
    """
    if len(bars) < 10:
        return MR12Result(notes="Insufficient data (need 10+ bars)")
    
    current_bar = bars[-1]
    closes = np.array([b.close for b in bars])
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    intraday_move = (current_bar.close - current_bar.open) / max(current_bar.open, 1e-10)
    
    avg_volume = float(np.mean(volumes[-lookback-1:-1]))
    current_volume = volumes[-1]
    volume_ratio = current_volume / max(avg_volume, 1.0)
    
    is_pump = intraday_move > 0.20 and volume_ratio > 10.0
    is_dump = intraday_move < -0.20 and volume_ratio > 10.0
    
    pump_score = 0.0
    
    if abs(intraday_move) > 0.20:
        pump_score += min(abs(intraday_move) * 2, 0.5)
    elif abs(intraday_move) > 0.10:
        pump_score += 0.25
    
    if volume_ratio > 10.0:
        pump_score += 0.4
    elif volume_ratio > 5.0:
        pump_score += 0.25
    elif volume_ratio > 3.0:
        pump_score += 0.15
    
    if is_pump or is_dump:
        pump_score += 0.1
    
    pump_score = float(np.clip(pump_score, 0.0, 1.0))
    fired = pump_score >= 0.6
    
    confidence = min(pump_score * 100, 96.0)
    
    notes_parts = []
    if is_pump:
        notes_parts.append(f"PUMP detected: +{intraday_move*100:.1f}%")
    elif is_dump:
        notes_parts.append(f"DUMP detected: {intraday_move*100:.1f}%")
    if volume_ratio > 10.0:
        notes_parts.append(f"Volume {volume_ratio:.0f}x average")
    
    return MR12Result(
        fired=fired,
        pump_score=round(pump_score, 4),
        intraday_move=round(float(intraday_move), 4),
        volume_ratio=round(float(volume_ratio), 4),
        is_pump=is_pump,
        is_dump=is_dump,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No pump pattern detected",
    )
