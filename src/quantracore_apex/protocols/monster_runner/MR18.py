"""
MR18 — Options Gamma Ramp Detector

Detects gamma squeeze conditions from options market pressure.
Identifies potential dealer hedging-driven moves.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR18Result:
    """Result of MR18 Options Gamma Ramp detection."""
    protocol_id: str = "MR18"
    fired: bool = False
    gamma_score: float = 0.0
    volatility_spike: float = 0.0
    intraday_range_ratio: float = 0.0
    directional_persistence: float = 0.0
    gamma_signature: bool = False
    confidence: float = 0.0
    notes: str = ""


def run_MR18(bars: List[OhlcvBar], lookback: int = 10) -> MR18Result:
    """
    Execute MR18 Options Gamma Ramp Detector.
    
    Identifies gamma squeeze signatures based on
    intraday volatility and directional persistence.
    
    Args:
        bars: OHLCV price bars (minimum 20 required)
        lookback: Period for baseline
        
    Returns:
        MR18Result with gamma ramp metrics
        
    Protocol Logic:
    1. Calculate rolling 3-day volatility spike
    2. Measure intraday range expansion
    3. Track directional persistence
    4. Detect gamma signature pattern
    """
    if len(bars) < 20:
        return MR18Result(notes="Insufficient data (need 20+ bars)")
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    recent_std = np.std(closes[-3:]) / max(np.mean(closes[-3:]), 1e-10)
    historical_std = np.std(closes[-lookback:-3]) / max(np.mean(closes[-lookback:-3]), 1e-10)
    volatility_spike = recent_std / max(historical_std, 1e-10)
    
    recent_ranges = (highs[-3:] - lows[-3:]) / closes[-3:]
    historical_ranges = (highs[-lookback:-3] - lows[-lookback:-3]) / closes[-lookback:-3]
    intraday_range_ratio = np.mean(recent_ranges) / max(np.mean(historical_ranges), 1e-10)
    
    returns = np.diff(closes[-5:]) / closes[-5:-1]
    same_direction = np.sum(returns > 0) if np.mean(returns) > 0 else np.sum(returns < 0)
    directional_persistence = same_direction / len(returns)
    
    gamma_signature = bool(
        volatility_spike > 1.5 and
        intraday_range_ratio > 1.5 and
        directional_persistence >= 0.75
    )
    
    gamma_score = 0.0
    
    if volatility_spike > 2.0:
        gamma_score += 0.35
    elif volatility_spike > 1.5:
        gamma_score += 0.25
    
    if intraday_range_ratio > 2.0:
        gamma_score += 0.35
    elif intraday_range_ratio > 1.5:
        gamma_score += 0.25
    
    if directional_persistence >= 0.8:
        gamma_score += 0.2
    elif directional_persistence >= 0.6:
        gamma_score += 0.1
    
    if gamma_signature:
        gamma_score += 0.1
    
    gamma_score = float(np.clip(gamma_score, 0.0, 1.0))
    fired = gamma_score >= 0.6
    
    confidence = min(gamma_score * 100, 89.0)
    
    notes_parts = []
    if gamma_signature:
        notes_parts.append("Gamma ramp signature detected")
    if volatility_spike > 2.0:
        notes_parts.append(f"Volatility spike: {volatility_spike:.1f}x")
    if intraday_range_ratio > 2.0:
        notes_parts.append(f"Range expansion: {intraday_range_ratio:.1f}x")
    
    return MR18Result(
        fired=fired,
        gamma_score=round(gamma_score, 4),
        volatility_spike=round(float(volatility_spike), 4),
        intraday_range_ratio=round(float(intraday_range_ratio), 4),
        directional_persistence=round(float(directional_persistence), 4),
        gamma_signature=gamma_signature,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No gamma ramp detected",
    )
