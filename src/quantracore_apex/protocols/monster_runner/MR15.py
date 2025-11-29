"""
MR15 — 100% Day Detector

Detects extreme moves where price doubles or more intraday.
Identifies rare explosive events.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR15Result:
    """Result of MR15 100% Day detection."""
    protocol_id: str = "MR15"
    fired: bool = False
    extreme_score: float = 0.0
    total_return: float = 0.0
    intraday_return: float = 0.0
    is_doubler: bool = False
    is_halver: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR15(bars: List[OhlcvBar], lookback: int = 5) -> MR15Result:
    """
    Execute MR15 100% Day Detector.
    
    Identifies extreme moves where price doubles (or halves)
    in a single session.
    
    Args:
        bars: OHLCV price bars (minimum 10 required)
        lookback: Period for context
        
    Returns:
        MR15Result with extreme move metrics
        
    Protocol Logic:
    1. Calculate total return (close vs prior close)
    2. Calculate intraday return (close vs open)
    3. Check for 100%+ moves
    4. Score based on magnitude
    """
    if len(bars) < 10:
        return MR15Result(notes="Insufficient data (need 10+ bars)")
    
    current_bar = bars[-1]
    prior_bar = bars[-2]
    
    total_return = (current_bar.close - prior_bar.close) / max(prior_bar.close, 1e-10)
    
    intraday_return = (current_bar.close - current_bar.open) / max(current_bar.open, 1e-10)
    
    is_doubler = current_bar.close >= 2.0 * current_bar.open
    is_halver = current_bar.close <= 0.5 * current_bar.open
    
    extreme_score = 0.0
    
    if is_doubler or is_halver:
        extreme_score = 1.0
    elif abs(total_return) > 0.50:
        extreme_score = 0.85
    elif abs(total_return) > 0.30:
        extreme_score = 0.7
    elif abs(total_return) > 0.20:
        extreme_score = 0.5
    elif abs(total_return) > 0.10:
        extreme_score = 0.3
    
    extreme_score = float(np.clip(extreme_score, 0.0, 1.0))
    fired = extreme_score >= 0.6
    
    confidence = min(extreme_score * 100, 98.0)
    
    notes_parts = []
    if is_doubler:
        notes_parts.append("100%+ DAY - DOUBLER")
    elif is_halver:
        notes_parts.append("50%+ DROP - HALVER")
    elif abs(total_return) > 0.50:
        direction = "up" if total_return > 0 else "down"
        notes_parts.append(f"Extreme move: {total_return*100:.1f}% {direction}")
    
    return MR15Result(
        fired=fired,
        extreme_score=round(extreme_score, 4),
        total_return=round(float(total_return), 4),
        intraday_return=round(float(intraday_return), 4),
        is_doubler=is_doubler,
        is_halver=is_halver,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No extreme move detected",
    )
