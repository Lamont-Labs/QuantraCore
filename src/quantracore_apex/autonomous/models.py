"""
Autonomous Trading System Models.

Core data structures for the trading orchestrator,
position monitoring, and trade outcome tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class TradingState(str, Enum):
    """Orchestrator state machine."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    RUNNING = "running"
    PAUSED = "paused"
    MARKET_CLOSED = "market_closed"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class PositionStatus(str, Enum):
    """Status of a tracked position."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL_FILL = "partial_fill"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    ERROR = "error"


class ExitReason(str, Enum):
    """Why a position was closed."""
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET_1 = "profit_target_1"
    PROFIT_TARGET_2 = "profit_target_2"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    OMEGA_OVERRIDE = "omega_override"
    MANUAL = "manual"
    EOD_EXIT = "eod_exit"
    RISK_LIMIT = "risk_limit"
    UNKNOWN = "unknown"


class FilterRejectionReason(str, Enum):
    """Why a signal was rejected by quality filter."""
    QUANTRASCORE_TOO_LOW = "quantrascore_below_threshold"
    QUALITY_TIER_INSUFFICIENT = "quality_tier_not_a_or_a_plus"
    RISK_TIER_TOO_HIGH = "risk_tier_extreme"
    OMEGA_DIRECTIVE_BLOCKED = "omega_directive_blocked"
    LIQUIDITY_INSUFFICIENT = "liquidity_below_minimum"
    MAX_POSITIONS_REACHED = "max_positions_reached"
    MAX_EXPOSURE_REACHED = "max_exposure_reached"
    MARKET_CLOSED = "market_closed"
    SYMBOL_RESTRICTED = "symbol_restricted"
    COOLDOWN_ACTIVE = "cooldown_active"
    RUNNER_PROB_LOW = "runner_probability_too_low"
    AVOID_FLAG_SET = "avoid_trade_flag_set"


@dataclass
class QualityThresholds:
    """
    Institutional-grade quality thresholds for signal filtering.
    
    These are strict by design - only the highest quality signals
    should be considered for autonomous execution.
    """
    min_quantrascore: float = 75.0
    required_quality_tiers: List[str] = field(default_factory=lambda: ["A+", "A"])
    max_risk_tier: str = "high"
    min_liquidity_band: str = "medium"
    min_runner_probability: float = 0.0
    max_avoid_probability: float = 0.3
    require_regime_alignment: bool = True
    allowed_regimes: List[str] = field(default_factory=lambda: [
        "trending_up", "trending_down", "range_bound"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_quantrascore": self.min_quantrascore,
            "required_quality_tiers": self.required_quality_tiers,
            "max_risk_tier": self.max_risk_tier,
            "min_liquidity_band": self.min_liquidity_band,
            "min_runner_probability": self.min_runner_probability,
            "max_avoid_probability": self.max_avoid_probability,
            "require_regime_alignment": self.require_regime_alignment,
            "allowed_regimes": self.allowed_regimes,
        }


@dataclass
class OrchestratorConfig:
    """
    Configuration for the TradingOrchestrator.
    
    Controls behavior of the autonomous trading loop.
    """
    watchlist: List[str] = field(default_factory=lambda: [
        "AAPL", "NVDA", "TSLA", "SPY", "QQQ", 
        "AMZN", "GOOG", "META", "MSFT", "AMD"
    ])
    
    max_concurrent_positions: int = 5
    max_portfolio_exposure: float = 0.5
    max_single_position_pct: float = 0.15
    
    scan_interval_seconds: float = 60.0
    position_check_interval_seconds: float = 5.0
    broker_sync_interval_seconds: float = 30.0
    
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    
    respect_market_hours: bool = True
    premarket_allowed: bool = False
    afterhours_allowed: bool = False
    
    symbol_cooldown_seconds: float = 300.0
    
    enable_trailing_stops: bool = True
    enable_time_exits: bool = True
    
    paper_mode: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "watchlist": self.watchlist,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_portfolio_exposure": self.max_portfolio_exposure,
            "max_single_position_pct": self.max_single_position_pct,
            "scan_interval_seconds": self.scan_interval_seconds,
            "position_check_interval_seconds": self.position_check_interval_seconds,
            "broker_sync_interval_seconds": self.broker_sync_interval_seconds,
            "quality_thresholds": self.quality_thresholds.to_dict(),
            "respect_market_hours": self.respect_market_hours,
            "paper_mode": self.paper_mode,
        }


@dataclass
class FilterResult:
    """Result of signal quality filtering."""
    passed: bool
    signal_id: str
    symbol: str
    quantrascore: float
    quality_tier: str
    risk_tier: str
    rejection_reasons: List[FilterRejectionReason] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "quantrascore": self.quantrascore,
            "quality_tier": self.quality_tier,
            "risk_tier": self.risk_tier,
            "rejection_reasons": [r.value for r in self.rejection_reasons],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SignalDecision:
    """
    Decision made by the orchestrator for a signal.
    
    Captures the full context of why a signal was traded or rejected.
    """
    signal_id: str
    symbol: str
    decision: str
    filter_result: FilterResult
    execution_result: Optional[Dict[str, Any]] = None
    eeo_plan: Optional[Dict[str, Any]] = None
    omega_statuses: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "decision": self.decision,
            "filter_result": self.filter_result.to_dict(),
            "execution_result": self.execution_result,
            "eeo_plan": self.eeo_plan,
            "omega_statuses": self.omega_statuses,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PositionState:
    """
    Complete state of a tracked position.
    
    Maintained by PositionMonitor for active trade management.
    """
    position_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    direction: str = "long"
    
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    quantity: float = 0.0
    notional_value: float = 0.0
    
    current_price: float = 0.0
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    protective_stop: float = 0.0
    trailing_stop: float = 0.0
    profit_target_1: float = 0.0
    profit_target_2: float = 0.0
    
    bars_in_trade: int = 0
    max_bars_allowed: int = 50
    
    status: PositionStatus = PositionStatus.PENDING
    
    source_signal_id: str = ""
    broker_order_id: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, price: float) -> None:
        """Update position with new price data."""
        self.current_price = price
        
        if self.direction == "long":
            self.current_pnl = (price - self.entry_price) * self.quantity
            self.current_pnl_pct = (price - self.entry_price) / self.entry_price
            excursion = price - self.entry_price
        else:
            self.current_pnl = (self.entry_price - price) * self.quantity
            self.current_pnl_pct = (self.entry_price - price) / self.entry_price
            excursion = self.entry_price - price
        
        if excursion > 0:
            self.max_favorable_excursion = max(self.max_favorable_excursion, excursion)
        else:
            self.max_adverse_excursion = max(self.max_adverse_excursion, abs(excursion))
    
    def check_stop_hit(self) -> bool:
        """Check if protective stop has been hit."""
        if self.protective_stop <= 0:
            return False
        
        if self.direction == "long":
            return self.current_price <= self.protective_stop
        else:
            return self.current_price >= self.protective_stop
    
    def check_trailing_stop_hit(self) -> bool:
        """Check if trailing stop has been hit."""
        if self.trailing_stop <= 0:
            return False
        
        if self.direction == "long":
            return self.current_price <= self.trailing_stop
        else:
            return self.current_price >= self.trailing_stop
    
    def check_target_hit(self, target_num: int = 1) -> bool:
        """Check if profit target has been hit."""
        target = self.profit_target_1 if target_num == 1 else self.profit_target_2
        if target <= 0:
            return False
        
        if self.direction == "long":
            return self.current_price >= target
        else:
            return self.current_price <= target
    
    def check_time_exit(self) -> bool:
        """Check if max bars exceeded."""
        return self.bars_in_trade >= self.max_bars_allowed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "current_price": self.current_price,
            "current_pnl": self.current_pnl,
            "current_pnl_pct": self.current_pnl_pct,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "protective_stop": self.protective_stop,
            "trailing_stop": self.trailing_stop,
            "profit_target_1": self.profit_target_1,
            "profit_target_2": self.profit_target_2,
            "bars_in_trade": self.bars_in_trade,
            "max_bars_allowed": self.max_bars_allowed,
            "status": self.status.value,
            "source_signal_id": self.source_signal_id,
            "broker_order_id": self.broker_order_id,
            "metadata": self.metadata,
        }


@dataclass
class TradeOutcome:
    """
    Complete outcome record of a closed trade.
    
    Used for feedback loop training and performance analysis.
    """
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    symbol: str = ""
    direction: str = "long"
    
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_price: float = 0.0
    exit_time: datetime = field(default_factory=datetime.utcnow)
    
    quantity: float = 0.0
    notional_value: float = 0.0
    
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    bars_held: int = 0
    
    exit_reason: ExitReason = ExitReason.UNKNOWN
    
    original_quantrascore: float = 0.0
    original_quality_tier: str = ""
    original_risk_tier: str = ""
    original_runner_prob: float = 0.0
    
    source_signal_id: str = ""
    position_id: str = ""
    
    was_profitable: bool = False
    hit_first_target: bool = False
    was_runner: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.was_profitable = self.realized_pnl > 0
        self.was_runner = self.realized_pnl_pct > 0.05
    
    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to ApexLab training sample format."""
        return {
            "symbol": self.symbol,
            "timestamp": self.entry_time.isoformat(),
            "quantrascore": self.original_quantrascore,
            "quality_tier": self.original_quality_tier,
            "risk_tier": self.original_risk_tier,
            "runner_prob": self.original_runner_prob,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_pct": self.realized_pnl_pct,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason.value,
            "was_profitable": self.was_profitable,
            "hit_target": self.hit_first_target,
            "was_runner": self.was_runner,
            "trade_id": self.trade_id,
            "signal_id": self.source_signal_id,
            "source": "autonomous_trading",
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat(),
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_pct": self.realized_pnl_pct,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "bars_held": self.bars_held,
            "exit_reason": self.exit_reason.value,
            "was_profitable": self.was_profitable,
            "hit_first_target": self.hit_first_target,
            "was_runner": self.was_runner,
            "original_quantrascore": self.original_quantrascore,
            "original_quality_tier": self.original_quality_tier,
            "source_signal_id": self.source_signal_id,
            "metadata": self.metadata,
        }


@dataclass
class OrchestratorMetrics:
    """Runtime metrics for the orchestrator."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    total_signals_scanned: int = 0
    signals_passed_filter: int = 0
    signals_rejected: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0
    total_realized_pnl: float = 0.0
    current_positions: int = 0
    current_exposure: float = 0.0
    omega_blocks: int = 0
    reconnection_count: int = 0
    last_scan_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    
    def win_rate(self) -> float:
        total = self.profitable_trades + self.losing_trades
        return self.profitable_trades / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "total_signals_scanned": self.total_signals_scanned,
            "signals_passed_filter": self.signals_passed_filter,
            "signals_rejected": self.signals_rejected,
            "pass_rate": self.signals_passed_filter / max(1, self.total_signals_scanned),
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "profitable_trades": self.profitable_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate(),
            "total_realized_pnl": self.total_realized_pnl,
            "current_positions": self.current_positions,
            "current_exposure": self.current_exposure,
            "omega_blocks": self.omega_blocks,
            "reconnection_count": self.reconnection_count,
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
        }
