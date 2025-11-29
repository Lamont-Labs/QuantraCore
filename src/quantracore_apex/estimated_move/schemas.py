"""
Schemas for Estimated Move Module.

Defines input/output structures for the estimated move computation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime


class HorizonWindow(str, Enum):
    """Time horizon windows for estimated move calculation."""
    SHORT_TERM = "1d"
    MEDIUM_TERM = "3d"
    EXTENDED_TERM = "5d"
    RESEARCH_TERM = "10d"


class MoveConfidence(str, Enum):
    """Confidence level for estimated move."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class MoveRange:
    """Percentile-based move range for a single horizon."""
    horizon: HorizonWindow
    min_move_pct: float      # 5th percentile
    low_move_pct: float      # 20th percentile
    median_move_pct: float   # 50th percentile
    high_move_pct: float     # 80th percentile
    max_move_pct: float      # 95th percentile
    uncertainty_score: float  # 0-1, higher = less certain
    sample_count: int         # Historical samples used
    
    def to_dict(self) -> Dict:
        return {
            "horizon": self.horizon.value,
            "min_move_pct": round(self.min_move_pct, 2),
            "low_move_pct": round(self.low_move_pct, 2),
            "median_move_pct": round(self.median_move_pct, 2),
            "high_move_pct": round(self.high_move_pct, 2),
            "max_move_pct": round(self.max_move_pct, 2),
            "uncertainty_score": round(self.uncertainty_score, 3),
            "sample_count": self.sample_count,
        }


@dataclass
class EstimatedMoveInput:
    """Input features for estimated move calculation."""
    # Deterministic inputs
    symbol: str
    quantra_score: float
    risk_tier: str
    volatility_band: str
    entropy_band: str
    regime_type: str
    suppression_state: str
    protocol_vector: List[float]
    float_pressure: float = 0.0
    liquidity_score: float = 0.5
    market_cap_band: str = "mid"
    
    # Model outputs
    runner_prob: float = 0.0
    quality_tier_logits: Optional[List[float]] = None
    avoid_trade_prob: float = 0.0
    model_quantra_score: float = 0.0
    ensemble_disagreement: float = 0.0
    
    # Vision outputs (optional)
    visual_runner_score: Optional[float] = None
    visual_pattern_logits: Optional[List[float]] = None
    visual_uncertainty: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantra_score": self.quantra_score,
            "risk_tier": self.risk_tier,
            "volatility_band": self.volatility_band,
            "entropy_band": self.entropy_band,
            "regime_type": self.regime_type,
            "suppression_state": self.suppression_state,
            "runner_prob": self.runner_prob,
            "avoid_trade_prob": self.avoid_trade_prob,
            "ensemble_disagreement": self.ensemble_disagreement,
            "market_cap_band": self.market_cap_band,
        }


@dataclass
class EstimatedMoveOutput:
    """Output of estimated move calculation."""
    symbol: str
    timestamp: datetime
    
    # Move ranges per horizon
    ranges: Dict[str, MoveRange] = field(default_factory=dict)
    
    # Summary metrics
    overall_uncertainty: float = 0.0
    runner_boost_applied: bool = False
    quality_modifier: float = 1.0
    safety_clamped: bool = False
    
    # Confidence and status
    confidence: MoveConfidence = MoveConfidence.MODERATE
    computation_mode: str = "hybrid"  # "deterministic", "model", "hybrid"
    
    # Compliance
    compliance_note: str = "Structural research output only - not a price target or trading signal"
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "ranges": {k: v.to_dict() for k, v in self.ranges.items()},
            "overall_uncertainty": round(self.overall_uncertainty, 3),
            "runner_boost_applied": self.runner_boost_applied,
            "quality_modifier": round(self.quality_modifier, 2),
            "safety_clamped": self.safety_clamped,
            "confidence": self.confidence.value,
            "computation_mode": self.computation_mode,
            "compliance_note": self.compliance_note,
        }
    
    def get_horizon_range(self, horizon: HorizonWindow) -> Optional[MoveRange]:
        """Get move range for a specific horizon."""
        return self.ranges.get(horizon.value)
    
    def is_high_uncertainty(self) -> bool:
        """Check if overall uncertainty is too high for reliable output."""
        return self.overall_uncertainty > 0.7
