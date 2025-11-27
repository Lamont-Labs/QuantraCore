"""
Expected Move Predictor for QuantraCore Apex.

Computes normalized expected move using volatility and compression metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits


@dataclass
class ExpectedMoveOutput:
    """Output from expected move calculation."""
    expected_move_pct: float
    move_direction_bias: float
    confidence: float
    volatility_component: float
    compression_component: float
    momentum_component: float


class ExpectedMovePredictor:
    """
    Predicts expected price move magnitude.
    
    This is a structural probability estimator, NOT a trade signal.
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
    
    def predict(
        self,
        window: OhlcvWindow,
        microtraits: Optional[Microtraits] = None
    ) -> ExpectedMoveOutput:
        """
        Compute expected move from OHLCV window.
        """
        bars = window.bars
        
        if microtraits is None:
            from src.quantracore_apex.core.microtraits import compute_microtraits
            microtraits = compute_microtraits(window)
        
        ranges = np.array([b.range for b in bars[-self.lookback:]])
        closes = np.array([b.close for b in bars[-self.lookback:]])
        
        if len(closes) < 2 or closes[-1] == 0:
            return ExpectedMoveOutput(
                expected_move_pct=0.0,
                move_direction_bias=0.0,
                confidence=0.0,
                volatility_component=0.0,
                compression_component=0.0,
                momentum_component=0.0
            )
        
        avg_range_pct = np.mean(ranges) / closes[-1]
        vol_component = avg_range_pct * 100
        
        compression = microtraits.compression_score
        compression_multiplier = 1 + (compression * 0.5)
        comp_component = compression * compression_multiplier
        
        returns = np.diff(closes) / closes[:-1]
        momentum = np.mean(returns) * 100
        mom_component = abs(momentum)
        
        expected_move = (vol_component * 0.6 + comp_component * 0.3 + mom_component * 0.1)
        
        direction_bias = np.sign(momentum) * min(0.8, abs(microtraits.trend_consistency))
        
        confidence = 0.5 + (compression * 0.2) + (1 - microtraits.noise_score) * 0.2
        confidence = min(0.9, max(0.3, confidence))
        
        return ExpectedMoveOutput(
            expected_move_pct=float(expected_move),
            move_direction_bias=float(direction_bias),
            confidence=float(confidence),
            volatility_component=float(vol_component),
            compression_component=float(comp_component),
            momentum_component=float(mom_component)
        )
