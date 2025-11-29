"""
MR05 — Multi-Timeframe Alignment Detector

Detects alignment of structural conditions across multiple
timeframes, which often precedes significant moves.

This is Stage 1 deterministic implementation using heuristic rules.
Note: Currently operates on single timeframe with simulated MTF analysis.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR05Result:
    """Result of MR05 Multi-Timeframe Alignment detection."""
    protocol_id: str = "MR05"
    fired: bool = False
    alignment_score: float = 0.0
    trend_alignment: float = 0.0
    momentum_alignment: float = 0.0
    structure_alignment: float = 0.0
    timeframe_scores: Optional[Dict[str, float]] = None
    dominant_direction: str = "neutral"
    confidence: float = 0.0
    notes: str = ""
    
    def __post_init__(self):
        if self.timeframe_scores is None:
            self.timeframe_scores = {}


def run_MR05(bars: List[OhlcvBar], lookback: int = 20) -> MR05Result:
    """
    Execute MR05 Multi-Timeframe Alignment Detector.
    
    Analyzes trend, momentum, and structure alignment across
    simulated timeframes (using different lookback periods).
    
    Args:
        bars: OHLCV price bars (minimum 100 required)
        lookback: Base period for analysis
        
    Returns:
        MR05Result with alignment metrics
        
    Protocol Logic:
    1. Calculate trend direction at multiple "timeframes" (lookback multiples)
    2. Calculate momentum at each "timeframe"
    3. Analyze structural support/resistance alignment
    4. Score overall alignment
    """
    if len(bars) < 100:
        return MR05Result(notes="Insufficient data (need 100+ bars)")
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    timeframes = {
        "short": lookback,
        "medium": lookback * 2,
        "long": lookback * 4,
    }
    
    def calculate_trend(data: np.ndarray, period: int) -> float:
        """Calculate trend direction and strength."""
        if len(data) < period:
            return 0.0
        
        recent = data[-period:]
        slope = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
        return float(np.clip(slope * 10, -1.0, 1.0))
    
    def calculate_momentum(data: np.ndarray, period: int) -> float:
        """Calculate momentum using ROC."""
        if len(data) < period:
            return 0.0
        
        roc = (data[-1] - data[-period]) / max(data[-period], 1e-10)
        return float(np.clip(roc * 5, -1.0, 1.0))
    
    timeframe_scores = {}
    trends = []
    momentums = []
    
    for tf_name, period in timeframes.items():
        trend = calculate_trend(closes, period)
        momentum = calculate_momentum(closes, period)
        
        score = (abs(trend) + abs(momentum)) / 2
        timeframe_scores[tf_name] = round(score, 4)
        
        trends.append(trend)
        momentums.append(momentum)
    
    if all(t > 0 for t in trends):
        trend_alignment = float(np.mean([abs(t) for t in trends]))
    elif all(t < 0 for t in trends):
        trend_alignment = float(np.mean([abs(t) for t in trends]))
    else:
        trend_alignment = 0.0
    
    if all(m > 0 for m in momentums):
        momentum_alignment = float(np.mean([abs(m) for m in momentums]))
    elif all(m < 0 for m in momentums):
        momentum_alignment = float(np.mean([abs(m) for m in momentums]))
    else:
        momentum_alignment = 0.0
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
    higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
    lower_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
    lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
    
    bullish_structure = (higher_highs + higher_lows) / (2 * (lookback - 1))
    bearish_structure = (lower_highs + lower_lows) / (2 * (lookback - 1))
    
    structure_alignment = max(bullish_structure, bearish_structure)
    
    alignment_score = (
        trend_alignment * 0.4 +
        momentum_alignment * 0.35 +
        structure_alignment * 0.25
    )
    alignment_score = float(np.clip(alignment_score, 0.0, 1.0))
    
    avg_trend = np.mean(trends)
    if avg_trend > 0.2:
        dominant_direction = "bullish"
    elif avg_trend < -0.2:
        dominant_direction = "bearish"
    else:
        dominant_direction = "neutral"
    
    fired = alignment_score >= 0.5 and dominant_direction != "neutral"
    
    confidence = min(alignment_score * 100, 85.0)
    
    notes_parts = []
    if trend_alignment > 0.5:
        notes_parts.append("Strong trend alignment")
    if momentum_alignment > 0.5:
        notes_parts.append("Strong momentum alignment")
    if structure_alignment > 0.6:
        notes_parts.append("Clear structural pattern")
    notes_parts.append(f"Direction: {dominant_direction}")
    
    return MR05Result(
        fired=fired,
        alignment_score=round(alignment_score, 4),
        trend_alignment=round(trend_alignment, 4),
        momentum_alignment=round(momentum_alignment, 4),
        structure_alignment=round(structure_alignment, 4),
        timeframe_scores=timeframe_scores,
        dominant_direction=dominant_direction,
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts),
    )
