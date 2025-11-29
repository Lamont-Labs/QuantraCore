"""Core data schemas for QuantraCore Apex."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from enum import Enum
from datetime import datetime
import hashlib
import json


class RegimeType(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    VOLATILE = "volatile"
    COMPRESSED = "compressed"
    UNKNOWN = "unknown"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class SuppressionState(str, Enum):
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"


class EntropyState(str, Enum):
    STABLE = "stable"
    ELEVATED = "elevated"
    CHAOTIC = "chaotic"


class DriftState(str, Enum):
    NONE = "none"
    MILD = "mild"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"


class ScoreBucket(str, Enum):
    VERY_LOW = "very_low"      # 0-20
    LOW = "low"                 # 21-40
    NEUTRAL = "neutral"         # 41-60
    HIGH = "high"               # 61-80
    VERY_HIGH = "very_high"     # 81-100


class OhlcvBar(BaseModel):
    """Single OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


class OhlcvWindow(BaseModel):
    """100-bar OHLCV window for analysis."""
    symbol: str
    timeframe: str
    bars: List[OhlcvBar]
    
    @property
    def window_size(self) -> int:
        return len(self.bars)
    
    def get_hash(self) -> str:
        data = json.dumps([b.model_dump(mode="json") for b in self.bars], default=str)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class Microtraits(BaseModel):
    """Computed microtraits from OHLCV window."""
    wick_ratio: float = Field(ge=0, le=1)
    body_ratio: float = Field(ge=0, le=1)
    bullish_pct_last20: float = Field(ge=0, le=1)
    compression_score: float = Field(ge=0, le=1)
    noise_score: float = Field(ge=0, le=1)
    strength_slope: float
    range_density: float = Field(ge=0)
    volume_intensity: float = Field(ge=0)
    trend_consistency: float = Field(ge=-1, le=1)
    volatility_ratio: float = Field(ge=0)


class EntropyMetrics(BaseModel):
    """Entropy computation results."""
    price_entropy: float = Field(ge=0)
    volume_entropy: float = Field(ge=0)
    combined_entropy: float = Field(ge=0)
    entropy_state: EntropyState
    entropy_floor: float = Field(ge=0)


class SuppressionMetrics(BaseModel):
    """Suppression analysis results."""
    suppression_level: float = Field(ge=0, le=1)
    suppression_state: SuppressionState
    coil_factor: float = Field(ge=0)
    breakout_probability: float = Field(ge=0, le=1)
    is_suppressed: bool = False
    suppression_score: float = Field(default=0.0, ge=0, le=1)


class DriftMetrics(BaseModel):
    """Drift detection results."""
    drift_magnitude: float = Field(ge=0)
    drift_direction: float  # -1 to 1
    drift_state: DriftState
    mean_reversion_pressure: float = Field(ge=0, le=1)


class ContinuationMetrics(BaseModel):
    """Continuation analysis results."""
    continuation_probability: float = Field(ge=0, le=1)
    momentum_strength: float = Field(ge=0, le=1)
    exhaustion_signal: bool
    reversal_risk: float = Field(ge=0, le=1)


class VolumeMetrics(BaseModel):
    """Volume analysis results."""
    volume_spike_detected: bool
    spike_magnitude: float = Field(ge=0)
    volume_trend: str  # "increasing", "decreasing", "stable"
    relative_volume: float = Field(ge=0)


class ProtocolResult(BaseModel):
    """Result from a single protocol execution."""
    protocol_id: str
    fired: bool
    confidence: float = Field(ge=0, le=1)
    signal_type: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ApexContext(BaseModel):
    """Context for Apex engine execution."""
    seed: int = 42
    run_id: Optional[str] = None
    sector: Optional[str] = None
    market_hours: bool = True
    compliance_mode: bool = True


class Verdict(BaseModel):
    """Final verdict from Apex analysis."""
    action: str  # "structural_probability_elevated", "neutral", "caution"
    confidence: float = Field(ge=0, le=1)
    primary_signal: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    compliance_note: str = "Analysis only - not trading advice"


class ApexResult(BaseModel):
    """Complete result from Apex engine execution."""
    symbol: str
    timestamp: datetime
    window_hash: str
    
    quantrascore: float = Field(ge=0, le=100)
    score_bucket: ScoreBucket
    
    regime: RegimeType
    risk_tier: RiskTier
    entropy_state: EntropyState
    suppression_state: SuppressionState
    drift_state: DriftState
    
    microtraits: Microtraits
    entropy_metrics: EntropyMetrics
    suppression_metrics: SuppressionMetrics
    drift_metrics: DriftMetrics
    continuation_metrics: ContinuationMetrics
    volume_metrics: VolumeMetrics
    
    protocol_results: List[ProtocolResult] = Field(default_factory=list)
    monster_runner_results: Dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict
    
    omega_overrides: Dict[str, bool] = Field(default_factory=dict)
    proof_hash: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()
