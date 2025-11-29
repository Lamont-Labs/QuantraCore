"""
EEO Engine Data Models

Defines the EntryExitPlan and all its component structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from .enums import (
    EntryMode,
    ExitMode,
    TrailingStopMode,
    SignalDirection,
    OrderTypeEEO,
    ExitStyle,
    QualityTier,
)


@dataclass
class EntryZone:
    """Price zone for valid entries."""
    lower: float
    upper: float
    
    def contains(self, price: float) -> bool:
        """Check if price is within entry zone."""
        return self.lower <= price <= self.upper
    
    def width(self) -> float:
        """Get the width of the entry zone."""
        return self.upper - self.lower
    
    def midpoint(self) -> float:
        """Get the midpoint of the entry zone."""
        return (self.lower + self.upper) / 2
    
    def to_dict(self) -> Dict[str, float]:
        return {"lower": self.lower, "upper": self.upper}


@dataclass
class BaseEntry:
    """Base entry configuration."""
    order_type: OrderTypeEEO
    entry_price: float
    entry_zone: EntryZone
    time_window_bars: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_type": self.order_type.value,
            "entry_price": self.entry_price,
            "entry_zone": self.entry_zone.to_dict(),
            "time_window_bars": self.time_window_bars,
        }


@dataclass
class ScaledEntry:
    """Single scaled entry in a multi-entry strategy."""
    fraction_of_size: float
    order_type: OrderTypeEEO
    entry_price: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraction_of_size": self.fraction_of_size,
            "order_type": self.order_type.value,
            "entry_price": self.entry_price,
        }


@dataclass
class ProtectiveStop:
    """Protective stop-loss configuration."""
    enabled: bool
    stop_price: float
    stop_type: OrderTypeEEO = OrderTypeEEO.STOP
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stop_price": self.stop_price,
            "stop_type": self.stop_type.value,
            "rationale": self.rationale,
        }


@dataclass
class ProfitTarget:
    """Single profit target."""
    target_price: float
    fraction_to_exit: float
    style: ExitStyle = ExitStyle.LIMIT
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_price": self.target_price,
            "fraction_to_exit": self.fraction_to_exit,
            "style": self.style.value,
            "rationale": self.rationale,
        }


@dataclass
class TrailingStop:
    """Trailing stop configuration."""
    enabled: bool
    mode: TrailingStopMode = TrailingStopMode.ATR
    atr_multiple: float = 2.0
    percent_trail: float = 0.02
    structural_reference: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "atr_multiple": self.atr_multiple,
            "percent_trail": self.percent_trail,
            "structural_reference": self.structural_reference,
        }


@dataclass
class TimeBasedExit:
    """Time-based exit rules."""
    enabled: bool
    max_bars_in_trade: int = 50
    end_of_day_exit: bool = False
    rationale: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_bars_in_trade": self.max_bars_in_trade,
            "end_of_day_exit": self.end_of_day_exit,
            "rationale": self.rationale,
        }


@dataclass
class AbortCondition:
    """Condition that would abort the entry."""
    condition_type: str
    threshold: float
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_type": self.condition_type,
            "threshold": self.threshold,
            "description": self.description,
        }


@dataclass
class PlanMetadata:
    """Metadata about the plan's predictions and confidence."""
    quality_label: QualityTier = QualityTier.C
    runner_bucket: str = "low"
    confidence: float = 0.5
    profile_used: str = "balanced"
    generation_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_label": self.quality_label.value,
            "runner_bucket": self.runner_bucket,
            "confidence": self.confidence,
            "profile_used": self.profile_used,
            "generation_timestamp": self.generation_timestamp.isoformat(),
        }


@dataclass
class EntryExitPlan:
    """
    Complete entry/exit optimization plan.
    
    This is the primary output of the EEO Engine, containing all
    information needed to execute a trade with optimal entry and exit points.
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    
    entry_mode: EntryMode = EntryMode.SINGLE
    exit_mode: ExitMode = ExitMode.SINGLE
    
    size_notional: float = 0.0
    
    base_entry: Optional[BaseEntry] = None
    scaled_entries: List[ScaledEntry] = field(default_factory=list)
    
    protective_stop: Optional[ProtectiveStop] = None
    profit_targets: List[ProfitTarget] = field(default_factory=list)
    trailing_stop: Optional[TrailingStop] = None
    time_based_exit: Optional[TimeBasedExit] = None
    
    abort_conditions: List[AbortCondition] = field(default_factory=list)
    metadata: PlanMetadata = field(default_factory=PlanMetadata)
    
    source_signal_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    compliance_note: str = "Entry/exit plan is research-only - not trading advice"
    
    def is_valid(self) -> bool:
        """Check if the plan has minimum required components."""
        if not self.symbol:
            return False
        if not self.base_entry and not self.scaled_entries:
            return False
        if self.size_notional <= 0:
            return False
        return True
    
    def has_protective_stop(self) -> bool:
        """Check if plan has an active protective stop."""
        return self.protective_stop is not None and self.protective_stop.enabled
    
    def total_target_fraction(self) -> float:
        """Calculate total fraction to exit across all targets."""
        return sum(t.fraction_to_exit for t in self.profit_targets)
    
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if stop and first target exist."""
        if not self.base_entry or not self.protective_stop or not self.profit_targets:
            return None
        
        entry = self.base_entry.entry_price
        stop = self.protective_stop.stop_price
        target = self.profit_targets[0].target_price
        
        if self.direction == SignalDirection.LONG:
            risk = entry - stop
            reward = target - entry
        else:
            risk = stop - entry
            reward = entry - target
        
        if risk <= 0:
            return None
        
        return reward / risk
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_mode": self.entry_mode.value,
            "exit_mode": self.exit_mode.value,
            "size_notional": self.size_notional,
            "base_entry": self.base_entry.to_dict() if self.base_entry else None,
            "scaled_entries": [e.to_dict() for e in self.scaled_entries],
            "protective_stop": self.protective_stop.to_dict() if self.protective_stop else None,
            "profit_targets": [t.to_dict() for t in self.profit_targets],
            "trailing_stop": self.trailing_stop.to_dict() if self.trailing_stop else None,
            "time_based_exit": self.time_based_exit.to_dict() if self.time_based_exit else None,
            "abort_conditions": [a.to_dict() for a in self.abort_conditions],
            "metadata": self.metadata.to_dict(),
            "source_signal_id": self.source_signal_id,
            "created_at": self.created_at.isoformat(),
            "compliance_note": self.compliance_note,
            "is_valid": self.is_valid(),
            "risk_reward_ratio": self.risk_reward_ratio(),
        }
