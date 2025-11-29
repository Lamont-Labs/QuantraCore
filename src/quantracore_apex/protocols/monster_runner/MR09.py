"""
MR09 — VWAP Breakout Detector

Detects VWAP breakouts with high volume confirmation.
Identifies institutional accumulation/distribution patterns.

This is Stage 2 deterministic implementation.

Version: 9.0-A
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR09Result:
    """Result of MR09 VWAP Breakout detection."""
    protocol_id: str = "MR09"
    fired: bool = False
    vwap_score: float = 0.0
    vwap_deviation: float = 0.0
    breakout_direction: str = "neutral"
    volume_confirmation: bool = False
    price_vs_vwap: float = 0.0
    confidence: float = 0.0
    notes: str = ""


def run_MR09(bars: List[OhlcvBar], lookback: int = 20) -> MR09Result:
    """
    Execute MR09 VWAP Breakout Detector.
    
    Identifies price breakouts above/below VWAP with
    strong volume confirmation.
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for volume comparison
        
    Returns:
        MR09Result with VWAP breakout metrics
        
    Protocol Logic:
    1. Calculate VWAP (Volume-Weighted Average Price)
    2. Measure price deviation from VWAP
    3. Check volume confirmation (above average)
    4. Score breakout strength
    """
    if len(bars) < 30:
        return MR09Result(notes="Insufficient data (need 30+ bars)")
    
    volumes = np.array([b.volume for b in bars], dtype=float)
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    closes = np.array([b.close for b in bars])
    
    if np.all(volumes == 0):
        return MR09Result(notes="No volume data available")
    
    typical_price = (highs + lows + closes) / 3
    cumulative_tpv = np.cumsum(typical_price * volumes)
    cumulative_volume = np.cumsum(volumes)
    
    vwap = cumulative_tpv / np.maximum(cumulative_volume, 1.0)
    
    current_price = closes[-1]
    current_vwap = vwap[-1]
    
    price_vs_vwap = (current_price - current_vwap) / max(current_vwap, 1e-10)
    vwap_deviation = abs(price_vs_vwap)
    
    breakout_direction = "bullish" if price_vs_vwap > 0.03 else "bearish" if price_vs_vwap < -0.03 else "neutral"
    
    avg_volume = float(np.mean(volumes[-lookback:-1]))
    current_volume = volumes[-1]
    volume_confirmation = current_volume > 3.0 * avg_volume
    
    vwap_score = 0.0
    
    if vwap_deviation > 0.03:
        vwap_score += min(vwap_deviation * 10, 0.5)
    
    if volume_confirmation:
        vwap_score += 0.3
    elif current_volume > 2.0 * avg_volume:
        vwap_score += 0.15
    
    if breakout_direction != "neutral":
        vwap_score += 0.2
    
    vwap_score = float(np.clip(vwap_score, 0.0, 1.0))
    fired = vwap_score >= 0.6
    
    confidence = min(vwap_score * 100, 87.0)
    
    notes_parts = []
    if breakout_direction != "neutral":
        notes_parts.append(f"VWAP {breakout_direction} breakout")
    if volume_confirmation:
        notes_parts.append("High volume confirmation")
    if vwap_deviation > 0.05:
        notes_parts.append(f"Strong deviation: {vwap_deviation*100:.1f}%")
    
    return MR09Result(
        fired=fired,
        vwap_score=round(vwap_score, 4),
        vwap_deviation=round(float(vwap_deviation), 4),
        breakout_direction=breakout_direction,
        volume_confirmation=volume_confirmation,
        price_vs_vwap=round(float(price_vs_vwap), 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No VWAP breakout detected",
    )
