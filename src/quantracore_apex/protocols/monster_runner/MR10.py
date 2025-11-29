"""
MR10 — NR7 Breakout Detector

Detects breakouts from Narrow Range 7 (NR7) patterns.
Identifies extreme compression followed by expansion.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR10Result:
    """Result of MR10 NR7 Breakout detection."""
    protocol_id: str = "MR10"
    fired: bool = False
    nr7_score: float = 0.0
    is_nr7: bool = False
    range_rank: int = 0
    breakout_detected: bool = False
    breakout_direction: str = "neutral"
    confidence: float = 0.0
    notes: str = ""


def run_MR10(bars: List[OhlcvBar], lookback: int = 7) -> MR10Result:
    """
    Execute MR10 NR7 Breakout Detector.
    
    Identifies NR7 pattern (narrowest range in 7 days) and
    subsequent breakout conditions.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for NR comparison (default 7)
        
    Returns:
        MR10Result with NR7 breakout metrics
        
    Protocol Logic:
    1. Calculate daily ranges for past N bars
    2. Identify if current range is narrowest in lookback
    3. Detect breakout above/below prior high/low
    4. Score breakout probability
    """
    if len(bars) < 20:
        return MR10Result(notes="Insufficient data (need 20+ bars)")
    
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    ranges = highs - lows
    
    recent_ranges = ranges[-lookback:]
    current_range = ranges[-1]
    
    is_nr7 = current_range == np.min(recent_ranges)
    
    sorted_ranges = np.sort(recent_ranges)
    range_rank = int(np.where(sorted_ranges == current_range)[0][0]) + 1
    
    prior_high = highs[-2]
    prior_low = lows[-2]
    current_close = closes[-1]
    
    breakout_detected = current_close > prior_high or current_close < prior_low
    breakout_direction = "bullish" if current_close > prior_high else "bearish" if current_close < prior_low else "neutral"
    
    nr7_score = 0.0
    
    if is_nr7:
        nr7_score += 0.4
    elif range_rank <= 2:
        nr7_score += 0.25
    
    if breakout_detected:
        nr7_score += 0.4
    
    if is_nr7 and breakout_detected:
        nr7_score += 0.2
    
    nr7_score = float(np.clip(nr7_score, 0.0, 1.0))
    fired = nr7_score >= 0.6
    
    confidence = min(nr7_score * 100, 88.0)
    
    notes_parts = []
    if is_nr7:
        notes_parts.append("NR7 pattern detected")
    elif range_rank <= 2:
        notes_parts.append(f"NR{range_rank} pattern")
    if breakout_detected:
        notes_parts.append(f"{breakout_direction.capitalize()} breakout")
    
    return MR10Result(
        fired=fired,
        nr7_score=round(nr7_score, 4),
        is_nr7=is_nr7,
        range_rank=range_rank,
        breakout_detected=breakout_detected,
        breakout_direction=breakout_direction,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No NR7 pattern detected",
    )
