"""
MR20 — Nuclear Runner Detector

Detects extreme multi-day runners (3x+ in 10 days).
Identifies rare, nuclear-level explosive moves.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR20Result:
    """Result of MR20 Nuclear Runner detection."""
    protocol_id: str = "MR20"
    fired: bool = False
    nuclear_score: float = 0.0
    total_return_10d: float = 0.0
    max_drawdown: float = 0.0
    is_nuclear: bool = False
    multiplier: float = 1.0
    confidence: float = 0.0
    notes: str = ""


def run_MR20(bars: List[OhlcvBar], lookback: int = 10) -> MR20Result:
    """
    Execute MR20 Nuclear Runner Detector.
    
    Identifies extreme multi-day moves where price
    triples or more in 10 sessions.
    
    Args:
        bars: OHLCV price bars (minimum 15 required)
        lookback: Period for analysis
        
    Returns:
        MR20Result with nuclear runner metrics
        
    Protocol Logic:
    1. Calculate 10-day total return
    2. Compute price multiplier
    3. Check max drawdown during run
    4. Score nuclear characteristics
    """
    if len(bars) < 15:
        return MR20Result(notes="Insufficient data (need 15+ bars)")
    
    closes = np.array([b.close for b in bars])
    
    start_price = closes[-lookback - 1]
    end_price = closes[-1]
    
    total_return_10d = (end_price - start_price) / max(start_price, 1e-10)
    multiplier = end_price / max(start_price, 1e-10)
    
    peak = closes[-lookback - 1]
    max_drawdown = 0.0
    for price in closes[-lookback:]:
        if price > peak:
            peak = price
        drawdown = (peak - price) / max(peak, 1e-10)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    is_nuclear = multiplier >= 3.0
    
    nuclear_score = 0.0
    
    if multiplier >= 3.0:
        nuclear_score = 1.0
    elif multiplier >= 2.0:
        nuclear_score = 0.85
    elif multiplier >= 1.5:
        nuclear_score = 0.7
    elif multiplier >= 1.3:
        nuclear_score = 0.5
    elif multiplier >= 1.2:
        nuclear_score = 0.3
    
    if max_drawdown < 0.1 and nuclear_score > 0:
        nuclear_score = min(nuclear_score + 0.1, 1.0)
    
    nuclear_score = float(np.clip(nuclear_score, 0.0, 1.0))
    fired = nuclear_score >= 0.6
    
    confidence = min(nuclear_score * 100, 99.0)
    
    notes_parts = []
    if is_nuclear:
        notes_parts.append(f"NUCLEAR RUNNER: {multiplier:.1f}x in 10 days")
    elif multiplier >= 2.0:
        notes_parts.append(f"Major runner: {multiplier:.1f}x")
    elif multiplier >= 1.5:
        notes_parts.append(f"Strong runner: {multiplier:.1f}x")
    
    if max_drawdown < 0.05:
        notes_parts.append("Minimal pullbacks")
    elif max_drawdown > 0.20:
        notes_parts.append(f"Volatile: {max_drawdown*100:.1f}% max drawdown")
    
    return MR20Result(
        fired=fired,
        nuclear_score=round(nuclear_score, 4),
        total_return_10d=round(float(total_return_10d), 4),
        max_drawdown=round(float(max_drawdown), 4),
        is_nuclear=is_nuclear,
        multiplier=round(float(multiplier), 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No nuclear runner detected",
    )
