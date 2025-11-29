"""
MR04 — Institutional Footprint Detector

Detects patterns consistent with institutional accumulation
or distribution that may precede large moves.

This is Stage 1 deterministic implementation using heuristic rules.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR04Result:
    """Result of MR04 Institutional Footprint detection."""
    protocol_id: str = "MR04"
    fired: bool = False
    institutional_score: float = 0.0
    large_bar_ratio: float = 0.0
    price_volume_divergence: float = 0.0
    stealth_accumulation: float = 0.0
    stealth_distribution: float = 0.0
    footprint_type: str = "none"
    confidence: float = 0.0
    notes: str = ""


def run_MR04(bars: List[OhlcvBar], lookback: int = 20) -> MR04Result:
    """
    Execute MR04 Institutional Footprint Detector.
    
    Identifies patterns suggesting institutional activity:
    - Large bars with above-average volume
    - Price/volume divergence
    - Stealth accumulation/distribution
    
    Args:
        bars: OHLCV price bars (minimum 30 required)
        lookback: Period for analysis
        
    Returns:
        MR04Result with institutional footprint metrics
        
    Protocol Logic:
    1. Identify large bars (above-average range)
    2. Calculate price/volume divergence
    3. Detect stealth patterns (gradual accumulation/distribution)
    4. Score institutional activity likelihood
    """
    if len(bars) < 30:
        return MR04Result(notes="Insufficient data (need 30+ bars)")
    
    closes = np.array([b.close for b in bars])
    np.array([b.open for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    volumes = np.array([b.volume for b in bars], dtype=float)
    
    ranges = highs - lows
    avg_range = float(np.mean(ranges[-lookback:]))
    
    large_bars = ranges[-lookback:] > (avg_range * 1.5)
    large_bar_ratio = float(np.sum(large_bars)) / lookback
    
    price_change = closes[-1] - closes[-lookback]
    volume_change = float(np.mean(volumes[-5:])) - float(np.mean(volumes[-lookback:-5]))
    
    avg_volume = float(np.mean(volumes[-lookback:])) + 1e-10
    price_trend = price_change / max(abs(closes[-lookback]), 1e-10)
    volume_trend = volume_change / avg_volume
    
    if price_trend > 0 and volume_trend < 0:
        price_volume_divergence = min(abs(price_trend - volume_trend), 1.0)
    elif price_trend < 0 and volume_trend > 0:
        price_volume_divergence = min(abs(price_trend - volume_trend), 1.0)
    else:
        price_volume_divergence = 0.0
    
    stealth_accumulation = 0.0
    stealth_distribution = 0.0
    
    higher_lows = 0
    lower_highs = 0
    
    for i in range(-lookback + 1, 0):
        if lows[i] > lows[i-1]:
            higher_lows += 1
        if highs[i] < highs[i-1]:
            lower_highs += 1
    
    if higher_lows > lookback * 0.6 and price_change > 0:
        stealth_accumulation = higher_lows / lookback
    if lower_highs > lookback * 0.6 and price_change < 0:
        stealth_distribution = lower_highs / lookback
    
    institutional_score = 0.0
    
    if large_bar_ratio > 0.2:
        institutional_score += large_bar_ratio * 0.3
    
    if price_volume_divergence > 0.1:
        institutional_score += price_volume_divergence * 0.3
    
    if stealth_accumulation > 0.5:
        institutional_score += stealth_accumulation * 0.25
    if stealth_distribution > 0.5:
        institutional_score += stealth_distribution * 0.25
    
    institutional_score = float(np.clip(institutional_score, 0.0, 1.0))
    
    footprint_type = "none"
    if stealth_accumulation > 0.5:
        footprint_type = "accumulation"
    elif stealth_distribution > 0.5:
        footprint_type = "distribution"
    elif large_bar_ratio > 0.3:
        footprint_type = "large_player_activity"
    
    fired = institutional_score >= 0.4
    
    confidence = min(institutional_score * 100, 75.0)
    
    notes_parts = []
    if footprint_type != "none":
        notes_parts.append(f"Footprint type: {footprint_type}")
    if large_bar_ratio > 0.25:
        notes_parts.append(f"Large bar ratio: {large_bar_ratio:.1%}")
    if price_volume_divergence > 0.2:
        notes_parts.append("Price/volume divergence detected")
    
    return MR04Result(
        fired=fired,
        institutional_score=round(institutional_score, 4),
        large_bar_ratio=round(large_bar_ratio, 4),
        price_volume_divergence=round(price_volume_divergence, 4),
        stealth_accumulation=round(stealth_accumulation, 4),
        stealth_distribution=round(stealth_distribution, 4),
        footprint_type=footprint_type,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No significant institutional footprint",
    )
