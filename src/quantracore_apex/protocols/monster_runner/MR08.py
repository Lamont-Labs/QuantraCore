"""
MR08 — Earnings Gap Runner Detector

Detects large gap moves (often earnings-related) with continuation.
Identifies gap-and-go patterns vs gap-and-fill setups.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR08Result:
    """Result of MR08 Earnings Gap Runner detection."""
    protocol_id: str = "MR08"
    fired: bool = False
    runner_score: float = 0.0
    gap_pct: float = 0.0
    gap_direction: str = "neutral"
    continuation: bool = False
    intraday_strength: float = 0.0
    fill_probability: float = 0.0
    confidence: float = 0.0
    notes: str = ""


def run_MR08(bars: List[OhlcvBar], lookback: int = 20) -> MR08Result:
    """
    Execute MR08 Earnings Gap Runner Detector.
    
    Identifies large gap openings with continuation patterns
    that often occur after earnings or major news events.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for analysis
        
    Returns:
        MR08Result with gap runner metrics
        
    Protocol Logic:
    1. Calculate gap percentage from prior close
    2. Assess intraday continuation vs fill
    3. Score runner probability based on gap size and continuation
    """
    if len(bars) < 20:
        return MR08Result(notes="Insufficient data (need 20+ bars)")
    
    current_bar = bars[-1]
    prior_bar = bars[-2]
    
    gap_pct = (current_bar.open - prior_bar.close) / max(prior_bar.close, 1e-10)
    
    gap_direction = "up" if gap_pct > 0 else "down" if gap_pct < 0 else "neutral"
    
    intraday_move = current_bar.close - current_bar.open
    intraday_direction = 1 if intraday_move > 0 else -1 if intraday_move < 0 else 0
    gap_sign = 1 if gap_pct > 0 else -1 if gap_pct < 0 else 0
    
    continuation = (gap_sign != 0) and (gap_sign == intraday_direction)
    
    bar_range = current_bar.high - current_bar.low
    intraday_strength = abs(intraday_move) / max(bar_range, 1e-10) if bar_range > 0 else 0
    
    if gap_direction == "up":
        filled = current_bar.low <= prior_bar.close
    elif gap_direction == "down":
        filled = current_bar.high >= prior_bar.close
    else:
        filled = False
    
    fill_probability = 0.7 if filled else 0.3
    if abs(gap_pct) > 0.15:
        fill_probability *= 0.5
    
    runner_score = 0.0
    
    if abs(gap_pct) > 0.10:
        runner_score += 0.4
    elif abs(gap_pct) > 0.05:
        runner_score += 0.25
    elif abs(gap_pct) > 0.03:
        runner_score += 0.15
    
    if continuation:
        runner_score += 0.3
    
    if intraday_strength > 0.7:
        runner_score += 0.2
    
    if not filled:
        runner_score += 0.1
    
    runner_score = float(np.clip(runner_score, 0.0, 1.0))
    fired = runner_score >= 0.6
    
    confidence = min(runner_score * 100, 91.0)
    
    notes_parts = []
    if abs(gap_pct) > 0.10:
        notes_parts.append(f"Large gap {gap_pct*100:.1f}%")
    if continuation:
        notes_parts.append("Gap continuation")
    else:
        notes_parts.append("Gap fade risk")
    if not filled:
        notes_parts.append("Gap unfilled")
    
    return MR08Result(
        fired=fired,
        runner_score=round(runner_score, 4),
        gap_pct=round(float(gap_pct), 4),
        gap_direction=gap_direction,
        continuation=continuation,
        intraday_strength=round(float(intraday_strength), 4),
        fill_probability=round(float(fill_probability), 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No significant gap detected",
    )
