"""
Estimated Move Engine for QuantraCore Apex.

Core computation logic for estimated move ranges.
This is a RESEARCH-ONLY module that provides statistical move ranges,
NOT predictions or trading signals.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .schemas import (
    EstimatedMoveInput,
    EstimatedMoveOutput,
    HorizonWindow,
    MoveRange,
    MoveConfidence,
)


class EstimatedMoveEngine:
    """
    Engine for computing estimated move ranges.
    
    Combines deterministic analysis with model-assisted predictions
    to generate statistical move ranges for research purposes.
    
    IMPORTANT: This is NOT a prediction engine. Outputs are statistical
    distributions derived from historical patterns, not guarantees.
    """
    
    # Configuration
    HORIZONS = [
        HorizonWindow.SHORT_TERM,
        HorizonWindow.MEDIUM_TERM,
        HorizonWindow.EXTENDED_TERM,
        HorizonWindow.RESEARCH_TERM,
    ]
    
    # Horizon days mapping
    HORIZON_DAYS = {
        HorizonWindow.SHORT_TERM: 1,
        HorizonWindow.MEDIUM_TERM: 3,
        HorizonWindow.EXTENDED_TERM: 5,
        HorizonWindow.RESEARCH_TERM: 10,
    }
    
    # Safety thresholds
    AVOID_TRADE_THRESHOLD = 0.35
    UNCERTAINTY_THRESHOLD = 0.7
    RUNNER_BOOST_THRESHOLD = 0.6
    
    # Base volatility by market cap band (annualized %)
    BASE_VOLATILITY = {
        "mega": 15.0,
        "large": 20.0,
        "mid": 28.0,
        "small": 38.0,
        "micro": 50.0,
        "nano": 65.0,
        "penny": 80.0,
    }
    
    # Regime multipliers
    REGIME_MULTIPLIERS = {
        "trending": 1.2,
        "ranging": 0.8,
        "volatile": 1.5,
        "quiet": 0.6,
        "breakout": 1.4,
        "breakdown": 1.3,
        "consolidation": 0.7,
    }
    
    # Volatility band multipliers
    VOLATILITY_BAND_MULTIPLIERS = {
        "ultra_low": 0.5,
        "low": 0.7,
        "normal": 1.0,
        "elevated": 1.3,
        "high": 1.6,
        "extreme": 2.0,
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the Estimated Move Engine."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def compute(self, input_data: EstimatedMoveInput) -> EstimatedMoveOutput:
        """
        Compute estimated move ranges for all horizons.
        
        Args:
            input_data: Input features for computation
            
        Returns:
            EstimatedMoveOutput with move ranges per horizon
        """
        # Check safety gates first
        safety_clamped = False
        computation_mode = "hybrid"
        
        # Gate 1: High avoid_trade_prob clamps to negative bias
        if input_data.avoid_trade_prob > self.AVOID_TRADE_THRESHOLD:
            safety_clamped = True
        
        # Gate 2: Suppression state blocks expansion
        if input_data.suppression_state == "blocked":
            safety_clamped = True
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(input_data)
        
        # Gate 3: High uncertainty returns neutral
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            return self._create_neutral_output(input_data, uncertainty)
        
        # Determine computation mode
        if input_data.ensemble_disagreement > 0.3:
            computation_mode = "deterministic"
        elif input_data.runner_prob > 0 or input_data.model_quantra_score > 0:
            computation_mode = "hybrid"
        else:
            computation_mode = "deterministic"
        
        # Calculate base volatility
        base_vol = self._get_base_volatility(input_data)
        
        # Apply modifiers
        modified_vol = self._apply_modifiers(base_vol, input_data)
        
        # Calculate runner boost
        runner_boost_applied = False
        if input_data.runner_prob > self.RUNNER_BOOST_THRESHOLD:
            runner_boost_applied = True
            modified_vol *= (1.0 + input_data.runner_prob * 0.5)
        
        # Calculate quality modifier
        quality_modifier = self._calculate_quality_modifier(input_data)
        
        # Apply safety clamping if needed
        if safety_clamped:
            modified_vol *= 0.5  # Reduce expected move
            quality_modifier *= 0.7  # Reduce upside bias
        
        # Generate ranges for each horizon
        ranges = {}
        for horizon in self.HORIZONS:
            move_range = self._calculate_horizon_range(
                horizon=horizon,
                base_volatility=modified_vol,
                quality_modifier=quality_modifier,
                uncertainty=uncertainty,
                safety_clamped=safety_clamped,
                input_data=input_data,
            )
            ranges[horizon.value] = move_range
        
        # Determine confidence level
        confidence = self._determine_confidence(uncertainty, input_data)
        
        return EstimatedMoveOutput(
            symbol=input_data.symbol,
            timestamp=datetime.utcnow(),
            ranges=ranges,
            overall_uncertainty=uncertainty,
            runner_boost_applied=runner_boost_applied,
            quality_modifier=quality_modifier,
            safety_clamped=safety_clamped,
            confidence=confidence,
            computation_mode=computation_mode,
        )
    
    def _calculate_uncertainty(self, input_data: EstimatedMoveInput) -> float:
        """Calculate overall uncertainty score."""
        components = []
        
        # Ensemble disagreement contributes to uncertainty
        components.append(input_data.ensemble_disagreement)
        
        # Vision uncertainty if available
        if input_data.visual_uncertainty is not None:
            components.append(input_data.visual_uncertainty)
        
        # Low quantra score = higher uncertainty
        score_uncertainty = max(0, (50 - input_data.quantra_score) / 100)
        components.append(score_uncertainty)
        
        # High avoid_trade_prob = higher uncertainty
        components.append(input_data.avoid_trade_prob * 0.5)
        
        if components:
            return min(1.0, np.mean(components))
        return 0.5
    
    def _get_base_volatility(self, input_data: EstimatedMoveInput) -> float:
        """Get base volatility for market cap band."""
        return self.BASE_VOLATILITY.get(
            input_data.market_cap_band.lower(),
            self.BASE_VOLATILITY["mid"]
        )
    
    def _apply_modifiers(
        self,
        base_vol: float,
        input_data: EstimatedMoveInput
    ) -> float:
        """Apply regime and volatility band modifiers."""
        vol = base_vol
        
        # Apply regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(
            input_data.regime_type.lower(),
            1.0
        )
        vol *= regime_mult
        
        # Apply volatility band multiplier
        vol_band_mult = self.VOLATILITY_BAND_MULTIPLIERS.get(
            input_data.volatility_band.lower(),
            1.0
        )
        vol *= vol_band_mult
        
        # Apply entropy adjustment
        if input_data.entropy_band.lower() == "high":
            vol *= 1.2
        elif input_data.entropy_band.lower() == "low":
            vol *= 0.85
        
        return vol
    
    def _calculate_quality_modifier(self, input_data: EstimatedMoveInput) -> float:
        """Calculate quality modifier based on score and model outputs."""
        modifier = 1.0
        
        # Higher quantra score = slightly higher upside bias
        if input_data.quantra_score > 70:
            modifier *= 1.1
        elif input_data.quantra_score > 80:
            modifier *= 1.2
        elif input_data.quantra_score < 40:
            modifier *= 0.9
        
        # Runner probability shifts distribution
        if input_data.runner_prob > 0.5:
            modifier *= (1.0 + input_data.runner_prob * 0.3)
        
        return modifier
    
    def _calculate_horizon_range(
        self,
        horizon: HorizonWindow,
        base_volatility: float,
        quality_modifier: float,
        uncertainty: float,
        safety_clamped: bool,
        input_data: EstimatedMoveInput,
    ) -> MoveRange:
        """Calculate move range for a specific horizon."""
        days = self.HORIZON_DAYS[horizon]
        
        # Scale volatility to horizon (sqrt of time)
        horizon_vol = base_volatility * np.sqrt(days / 252)
        
        # Calculate percentiles assuming roughly normal distribution
        # with some skew based on quality modifier
        base_std = horizon_vol / 100  # Convert to decimal
        
        # Shift mean slightly based on quality
        mean_shift = (quality_modifier - 1.0) * base_std * 0.5
        
        # Calculate percentiles
        min_move = mean_shift + base_std * (-1.65)  # 5th percentile
        low_move = mean_shift + base_std * (-0.84)  # 20th percentile
        median_move = mean_shift                     # 50th percentile
        high_move = mean_shift + base_std * 0.84     # 80th percentile
        max_move = mean_shift + base_std * 1.65      # 95th percentile
        
        # Apply quality skew to upside
        if quality_modifier > 1.0:
            high_move *= quality_modifier
            max_move *= quality_modifier
        
        # Safety clamp reduces upside
        if safety_clamped:
            median_move *= 0.8
            high_move *= 0.6
            max_move *= 0.5
        
        # Convert to percentages
        min_pct = min_move * 100
        low_pct = low_move * 100
        median_pct = median_move * 100
        high_pct = high_move * 100
        max_pct = max_move * 100
        
        # Estimate sample count based on similarity
        sample_count = self._estimate_sample_count(input_data)
        
        return MoveRange(
            horizon=horizon,
            min_move_pct=min_pct,
            low_move_pct=low_pct,
            median_move_pct=median_pct,
            high_move_pct=high_pct,
            max_move_pct=max_pct,
            uncertainty_score=uncertainty,
            sample_count=sample_count,
        )
    
    def _estimate_sample_count(self, input_data: EstimatedMoveInput) -> int:
        """Estimate number of similar historical samples."""
        # Base count varies by market cap (more data for larger caps)
        base_counts = {
            "mega": 5000,
            "large": 3000,
            "mid": 2000,
            "small": 1000,
            "micro": 500,
            "nano": 200,
            "penny": 100,
        }
        base = base_counts.get(input_data.market_cap_band.lower(), 1000)
        
        # Reduce for specific regimes/conditions
        if input_data.regime_type.lower() in ["breakout", "breakdown"]:
            base = int(base * 0.3)
        
        if input_data.volatility_band.lower() in ["extreme", "high"]:
            base = int(base * 0.5)
        
        return max(50, base)
    
    def _determine_confidence(
        self,
        uncertainty: float,
        input_data: EstimatedMoveInput
    ) -> MoveConfidence:
        """Determine overall confidence level."""
        if uncertainty > 0.6:
            return MoveConfidence.UNCERTAIN
        elif uncertainty > 0.4:
            return MoveConfidence.LOW
        elif uncertainty > 0.2:
            return MoveConfidence.MODERATE
        else:
            return MoveConfidence.HIGH
    
    def _create_neutral_output(
        self,
        input_data: EstimatedMoveInput,
        uncertainty: float
    ) -> EstimatedMoveOutput:
        """Create neutral output when uncertainty is too high."""
        ranges = {}
        for horizon in self.HORIZONS:
            ranges[horizon.value] = MoveRange(
                horizon=horizon,
                min_move_pct=0.0,
                low_move_pct=0.0,
                median_move_pct=0.0,
                high_move_pct=0.0,
                max_move_pct=0.0,
                uncertainty_score=uncertainty,
                sample_count=0,
            )
        
        return EstimatedMoveOutput(
            symbol=input_data.symbol,
            timestamp=datetime.utcnow(),
            ranges=ranges,
            overall_uncertainty=uncertainty,
            runner_boost_applied=False,
            quality_modifier=1.0,
            safety_clamped=True,
            confidence=MoveConfidence.UNCERTAIN,
            computation_mode="neutral",
            compliance_note="High uncertainty - estimated move suppressed for safety",
        )
    
    def compute_batch(
        self,
        inputs: List[EstimatedMoveInput]
    ) -> List[EstimatedMoveOutput]:
        """Compute estimated moves for multiple symbols."""
        return [self.compute(inp) for inp in inputs]
