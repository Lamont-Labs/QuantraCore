"""
Volatility Projection Engine

Projects forward volatility states using deterministic heuristics.
Part of the QuantraCore Apex prediction stack.

Version: 8.1
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from ..core.schemas import OhlcvBar


@dataclass
class VolatilityProjection:
    """Volatility projection result."""
    current_volatility: float = 0.0
    projected_volatility: float = 0.0
    volatility_direction: str = "stable"
    expansion_probability: float = 0.0
    contraction_probability: float = 0.0
    regime_forecast: str = "normal"
    confidence: float = 0.0
    lookforward_periods: int = 5
    compliance_note: str = "Projection is structural probability, not trading advice"


def project_volatility(
    bars: List[OhlcvBar],
    lookback: int = 20,
    lookforward: int = 5,
) -> VolatilityProjection:
    """
    Project forward volatility using deterministic heuristics.
    
    Args:
        bars: OHLCV price bars
        lookback: Historical period for analysis
        lookforward: Periods to project forward
        
    Returns:
        VolatilityProjection with forecasted volatility state
        
    Method:
    - Analyzes ATR trend and acceleration
    - Measures volatility regime transitions
    - Projects based on mean-reversion and momentum
    """
    if len(bars) < lookback + 10:
        return VolatilityProjection()
    
    closes = np.array([b.close for b in bars])
    highs = np.array([b.high for b in bars])
    lows = np.array([b.low for b in bars])
    
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    
    current_atr = float(np.mean(tr[-lookback:]))
    historical_atr = float(np.mean(tr[:-lookback])) if len(tr) > lookback else current_atr
    
    recent_atr = float(np.mean(tr[-5:]))
    prior_atr = float(np.mean(tr[-10:-5])) if len(tr) >= 10 else recent_atr
    
    atr_momentum = (recent_atr - prior_atr) / max(prior_atr, 1e-10)
    
    atr_ratio = current_atr / max(historical_atr, 1e-10)
    mean_reversion_factor = 0.0
    
    if atr_ratio > 1.5:
        mean_reversion_factor = -0.3
    elif atr_ratio < 0.5:
        mean_reversion_factor = 0.3
    
    projected_change = (atr_momentum * 0.6) + (mean_reversion_factor * 0.4)
    projected_volatility = current_atr * (1 + projected_change)
    
    if projected_change > 0.1:
        volatility_direction = "expanding"
        expansion_probability = min(0.5 + projected_change, 0.85)
        contraction_probability = 1 - expansion_probability
    elif projected_change < -0.1:
        volatility_direction = "contracting"
        contraction_probability = min(0.5 - projected_change, 0.85)
        expansion_probability = 1 - contraction_probability
    else:
        volatility_direction = "stable"
        expansion_probability = 0.3
        contraction_probability = 0.3
    
    vol_ratio = projected_volatility / max(historical_atr, 1e-10)
    if vol_ratio < 0.5:
        regime_forecast = "low"
    elif vol_ratio > 1.5:
        regime_forecast = "high"
    else:
        regime_forecast = "normal"
    
    confidence = min(abs(projected_change) * 100 + 30, 75.0)
    
    return VolatilityProjection(
        current_volatility=round(current_atr, 6),
        projected_volatility=round(projected_volatility, 6),
        volatility_direction=volatility_direction,
        expansion_probability=round(expansion_probability, 4),
        contraction_probability=round(contraction_probability, 4),
        regime_forecast=regime_forecast,
        confidence=round(confidence, 2),
        lookforward_periods=lookforward,
    )
