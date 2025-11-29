"""
EEO Engine Input Contexts

Defines the input data structures that feed into the EEO optimizer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from .enums import (
    SignalDirection,
    VolatilityBand,
    LiquidityBand,
    RegimeType,
    SuppressionState,
    QualityTier,
)


@dataclass
class SignalContext:
    """
    Context from the Apex signal engine.
    
    Contains all relevant information about the signal being optimized.
    """
    symbol: str
    timeframe: str = "1D"
    signal_time: datetime = field(default_factory=datetime.utcnow)
    direction: SignalDirection = SignalDirection.LONG
    quantra_score: float = 50.0
    protocol_trace: List[str] = field(default_factory=list)
    regime_type: RegimeType = RegimeType.CHOP
    volatility_band: VolatilityBand = VolatilityBand.MID
    liquidity_band: LiquidityBand = LiquidityBand.MID
    suppression_state: SuppressionState = SuppressionState.NONE
    zde_label: bool = False
    signal_id: str = ""
    
    def is_favorable_regime(self) -> bool:
        """Check if regime is favorable for trading."""
        return self.regime_type in (RegimeType.TREND_UP, RegimeType.TREND_DOWN)
    
    def is_high_conviction(self) -> bool:
        """Check if signal has high conviction (score > 70)."""
        return self.quantra_score >= 70.0
    
    def is_suppressed(self) -> bool:
        """Check if signal is suppressed or blocked."""
        return self.suppression_state != SuppressionState.NONE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "signal_time": self.signal_time.isoformat(),
            "direction": self.direction.value,
            "quantra_score": self.quantra_score,
            "protocol_trace": self.protocol_trace,
            "regime_type": self.regime_type.value,
            "volatility_band": self.volatility_band.value,
            "liquidity_band": self.liquidity_band.value,
            "suppression_state": self.suppression_state.value,
            "zde_label": self.zde_label,
            "signal_id": self.signal_id,
        }


@dataclass
class PredictiveContext:
    """
    Context from ApexCore v2 predictive models.
    
    Contains model predictions that inform entry/exit optimization.
    """
    runner_prob: float = 0.0
    future_quality_tier: QualityTier = QualityTier.C
    avoid_trade_prob: float = 0.0
    ensemble_disagreement: float = 0.0
    
    estimated_move_min: Optional[float] = None
    estimated_move_median: Optional[float] = None
    estimated_move_max: Optional[float] = None
    estimated_move_uncertainty: float = 0.5
    
    def is_runner_candidate(self) -> bool:
        """Check if this might be a runner (high momentum) trade."""
        return self.runner_prob >= 0.6
    
    def is_high_quality(self) -> bool:
        """Check if quality tier is A or A+."""
        return self.future_quality_tier in (QualityTier.A_PLUS, QualityTier.A)
    
    def should_avoid(self) -> bool:
        """Check if model recommends avoiding this trade."""
        return self.avoid_trade_prob >= 0.6
    
    def has_estimated_move(self) -> bool:
        """Check if estimated move data is available."""
        return self.estimated_move_median is not None
    
    def get_runner_bucket(self) -> str:
        """Get runner probability bucket."""
        if self.runner_prob >= 0.8:
            return "very_high"
        elif self.runner_prob >= 0.6:
            return "high"
        elif self.runner_prob >= 0.4:
            return "medium"
        elif self.runner_prob >= 0.2:
            return "low"
        return "very_low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "runner_prob": self.runner_prob,
            "future_quality_tier": self.future_quality_tier.value,
            "avoid_trade_prob": self.avoid_trade_prob,
            "ensemble_disagreement": self.ensemble_disagreement,
            "estimated_move_min": self.estimated_move_min,
            "estimated_move_median": self.estimated_move_median,
            "estimated_move_max": self.estimated_move_max,
            "estimated_move_uncertainty": self.estimated_move_uncertainty,
            "runner_bucket": self.get_runner_bucket(),
        }


@dataclass
class MarketMicrostructure:
    """
    Real-time market microstructure data.
    
    Contains current pricing and volatility information.
    """
    current_bid: float = 0.0
    current_ask: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    
    atr_14: float = 0.0
    recent_bar_ranges: List[float] = field(default_factory=list)
    
    depth_bid_size: float = 0.0
    depth_ask_size: float = 0.0
    
    def calculate_spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return 0.0
    
    def is_wide_spread(self, threshold_pct: float = 0.5) -> bool:
        """Check if spread is wider than threshold."""
        return self.spread_pct > threshold_pct
    
    def is_liquid(self) -> bool:
        """Check if market appears liquid based on spread and depth."""
        return self.spread_pct < 0.3 and self.depth_bid_size > 0
    
    def get_avg_bar_range(self) -> float:
        """Get average of recent bar ranges."""
        if not self.recent_bar_ranges:
            return self.atr_14
        return sum(self.recent_bar_ranges) / len(self.recent_bar_ranges)
    
    @classmethod
    def from_price(cls, price: float, atr: float = 0.0) -> "MarketMicrostructure":
        """Create from a simple price (estimates bid/ask)."""
        spread_estimate = price * 0.001
        return cls(
            current_bid=price - spread_estimate / 2,
            current_ask=price + spread_estimate / 2,
            mid_price=price,
            spread=spread_estimate,
            spread_pct=0.1,
            atr_14=atr if atr > 0 else price * 0.02,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "atr_14": self.atr_14,
            "recent_bar_ranges": self.recent_bar_ranges,
            "depth_bid_size": self.depth_bid_size,
            "depth_ask_size": self.depth_ask_size,
        }


@dataclass
class RiskContext:
    """
    Account and risk management context.
    
    Contains risk limits and account information for position sizing.
    """
    account_equity: float = 100000.0
    per_trade_risk_fraction: float = 0.01
    max_notional_per_symbol: float = 10000.0
    max_gap_tolerance: float = 0.05
    
    current_exposure: float = 0.0
    open_position_count: int = 0
    max_positions: int = 10
    
    def max_risk_dollars(self) -> float:
        """Calculate maximum dollars at risk per trade."""
        return self.account_equity * self.per_trade_risk_fraction
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return self.open_position_count < self.max_positions
    
    def available_capital(self) -> float:
        """Calculate available capital for new positions."""
        return max(0, self.account_equity - self.current_exposure)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_equity": self.account_equity,
            "per_trade_risk_fraction": self.per_trade_risk_fraction,
            "max_notional_per_symbol": self.max_notional_per_symbol,
            "max_gap_tolerance": self.max_gap_tolerance,
            "current_exposure": self.current_exposure,
            "open_position_count": self.open_position_count,
            "max_positions": self.max_positions,
            "max_risk_dollars": self.max_risk_dollars(),
        }


@dataclass
class EEOContext:
    """
    Complete context for entry/exit optimization.
    
    Aggregates all context sources into a single structure.
    """
    signal: SignalContext
    predictive: PredictiveContext
    microstructure: MarketMicrostructure
    risk: RiskContext
    
    def should_proceed(self) -> bool:
        """Check if conditions are favorable to proceed with optimization."""
        if self.signal.is_suppressed():
            return False
        if self.predictive.should_avoid():
            return False
        if not self.risk.can_open_position():
            return False
        return True
    
    def get_optimization_priority(self) -> str:
        """Determine optimization priority based on context."""
        if self.predictive.is_runner_candidate() and self.predictive.is_high_quality():
            return "high"
        elif self.signal.is_high_conviction():
            return "medium"
        return "low"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.to_dict(),
            "predictive": self.predictive.to_dict(),
            "microstructure": self.microstructure.to_dict(),
            "risk": self.risk.to_dict(),
            "should_proceed": self.should_proceed(),
            "optimization_priority": self.get_optimization_priority(),
        }
