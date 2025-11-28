"""
MR03 — Volatility Regime Shift Detector

Detects transitions between volatility regimes that often
precede significant directional moves.

This is Stage 1 deterministic implementation using heuristic rules.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from ...core.schemas import OhlcvBar


@dataclass
class MR03Result:
    """Result of MR03 Volatility Regime Shift detection."""
    protocol_id: str = "MR03"
    fired: bool = False
    regime_shift_score: float = 0.0
    current_regime: str = "normal"
    previous_regime: str = "normal"
    transition_strength: float = 0.0
    volatility_expansion: float = 0.0
    volatility_contraction: float = 0.0
    confidence: float = 0.0
    notes: str = ""


def run_MR03(bars: List[OhlcvBar], short_lookback: int = 10, long_lookback: int = 50) -> MR03Result:
    """
    Execute MR03 Volatility Regime Shift Detector.
    
    Identifies transitions between low, normal, and high
    volatility regimes. Uses deterministic analysis only.
    
    Args:
        bars: OHLCV price bars (minimum 60 required)
        short_lookback: Recent period for volatility
        long_lookback: Historical period for baseline
        
    Returns:
        MR03Result with regime shift metrics
        
    Protocol Logic:
    1. Calculate short-term vs long-term volatility
    2. Classify current and previous regimes
    3. Detect regime transitions
    4. Score transition strength
    """
    if len(bars) < long_lookback + 10:
        return MR03Result(notes=f"Insufficient data (need {long_lookback + 10}+ bars)")
    
    closes = np.array([b.close for b in bars])
    
    returns = np.diff(np.log(closes + 1e-10))
    
    short_vol = float(np.std(returns[-short_lookback:]))
    long_vol = float(np.std(returns[-long_lookback:]))
    
    prev_short_vol = float(np.std(returns[-2*short_lookback:-short_lookback]))
    
    def classify_regime(vol: float, baseline: float) -> str:
        ratio = vol / max(baseline, 1e-10)
        if ratio < 0.5:
            return "low"
        elif ratio > 1.5:
            return "high"
        else:
            return "normal"
    
    current_regime = classify_regime(short_vol, long_vol)
    previous_regime = classify_regime(prev_short_vol, long_vol)
    
    vol_ratio = short_vol / max(long_vol, 1e-10)
    volatility_expansion = max(0.0, vol_ratio - 1.0)
    volatility_contraction = max(0.0, 1.0 - vol_ratio)
    
    regime_changed = current_regime != previous_regime
    transition_strength = 0.0
    
    if regime_changed:
        vol_change = abs(short_vol - prev_short_vol) / max(long_vol, 1e-10)
        transition_strength = min(vol_change * 2, 1.0)
    
    regime_shift_score = 0.0
    
    if regime_changed:
        regime_shift_score += 0.4
        regime_shift_score += transition_strength * 0.3
    
    if current_regime == "low" and volatility_contraction > 0.3:
        regime_shift_score += 0.2
    elif current_regime == "high" and volatility_expansion > 0.5:
        regime_shift_score += 0.2
    
    regime_shift_score = float(np.clip(regime_shift_score, 0.0, 1.0))
    
    fired = regime_shift_score >= 0.5 or (regime_changed and transition_strength > 0.4)
    
    confidence = min(regime_shift_score * 100, 80.0)
    
    notes_parts = []
    if regime_changed:
        notes_parts.append(f"Regime shift: {previous_regime} -> {current_regime}")
    if volatility_expansion > 0.5:
        notes_parts.append("Volatility expanding")
    if volatility_contraction > 0.3:
        notes_parts.append("Volatility contracting")
    
    return MR03Result(
        fired=fired,
        regime_shift_score=round(regime_shift_score, 4),
        current_regime=current_regime,
        previous_regime=previous_regime,
        transition_strength=round(transition_strength, 4),
        volatility_expansion=round(volatility_expansion, 4),
        volatility_contraction=round(volatility_contraction, 4),
        confidence=round(confidence, 2),
        notes="; ".join(notes_parts) if notes_parts else "No significant regime shift",
    )
