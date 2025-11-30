"""
Data Models for QuantraCore Apex Broker Layer.

Defines OrderTicket, ExecutionResult, BrokerPosition, and related models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import hashlib
import json

from .enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderIntent,
    OrderStatus,
    PositionSide,
    RiskDecisionType,
    SignalDirection,
)


@dataclass
class OrderMetadata:
    """Metadata attached to an order from Apex signals."""
    quantra_score: float = 0.0
    runner_prob: float = 0.0
    estimated_move: Optional[Dict] = None
    regime: str = ""
    volatility_band: str = ""
    notes: Optional[str] = None
    company_name: str = ""
    sector: str = "Unknown"
    score_bucket: str = "neutral"
    confidence: float = 0.5
    monster_runner_score: float = 0.0
    monster_runner_fired: bool = False
    runner_probability: float = 0.0
    avoid_trade_probability: float = 0.0
    quality_tier: str = "C"
    entropy_state: str = "mid"
    suppression_state: str = "none"
    drift_state: str = "none"
    vix_level: float = 20.0
    vix_percentile: float = 50.0
    sector_momentum: str = "neutral"
    market_breadth: float = 0.5
    spy_change_pct: float = 0.0
    risk_tier: str = "medium"
    risk_score: float = 50.0
    stop_loss_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    take_profit_price: Optional[float] = None
    take_profit_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    protocols_fired: List[str] = field(default_factory=list)
    tier_protocols: Optional[List[str]] = None
    monster_runner_protocols: Optional[List[str]] = None
    omega_alerts: List[str] = field(default_factory=list)
    omega_blocked: bool = False
    consensus_direction: str = "neutral"
    protocol_confidence: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "quantra_score": self.quantra_score,
            "runner_prob": self.runner_prob,
            "estimated_move": self.estimated_move,
            "regime": self.regime,
            "volatility_band": self.volatility_band,
            "notes": self.notes,
            "company_name": self.company_name,
            "sector": self.sector,
            "score_bucket": self.score_bucket,
            "confidence": self.confidence,
            "monster_runner_score": self.monster_runner_score,
            "monster_runner_fired": self.monster_runner_fired,
            "runner_probability": self.runner_probability,
            "avoid_trade_probability": self.avoid_trade_probability,
            "quality_tier": self.quality_tier,
            "risk_tier": self.risk_tier,
            "protocols_fired": self.protocols_fired,
            "omega_alerts": self.omega_alerts,
            "omega_blocked": self.omega_blocked,
        }


@dataclass
class OrderTicket:
    """Order ticket to be sent to broker."""
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    intent: OrderIntent = OrderIntent.OPEN_LONG
    source_signal_id: str = ""
    strategy_id: str = "default"
    metadata: OrderMetadata = field(default_factory=OrderMetadata)
    ticket_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "ticket_id": self.ticket_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "intent": self.intent.value,
            "source_signal_id": self.source_signal_id,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
        }
    
    def hash(self) -> str:
        """Generate deterministic hash for audit."""
        data = {
            "ticket_id": self.ticket_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "created_at": self.created_at.isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class ExecutionResult:
    """Result of order execution from broker."""
    order_id: str
    broker: str
    status: OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    timestamp_utc: str = ""
    raw_broker_payload: Dict = field(default_factory=dict)
    ticket_id: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "broker": self.broker,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "timestamp_utc": self.timestamp_utc,
            "ticket_id": self.ticket_id,
            "error_message": self.error_message,
        }
    
    @property
    def is_success(self) -> bool:
        return self.status in [OrderStatus.NEW, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]


@dataclass
class BrokerPosition:
    """Position from broker."""
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float = 0.0
    unrealized_pl: float = 0.0
    side: PositionSide = PositionSide.FLAT
    
    def __post_init__(self):
        if self.qty > 0:
            self.side = PositionSide.LONG
        elif self.qty < 0:
            self.side = PositionSide.SHORT
        else:
            self.side = PositionSide.FLAT
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "avg_entry_price": self.avg_entry_price,
            "market_value": self.market_value,
            "unrealized_pl": self.unrealized_pl,
            "side": self.side.value,
        }


@dataclass
class ApexSignal:
    """Signal from Apex engine to be converted to order."""
    signal_id: str
    symbol: str
    direction: SignalDirection
    quantra_score: float = 0.0
    runner_prob: float = 0.0
    regime: str = ""
    volatility_band: str = ""
    estimated_move: Optional[Dict] = None
    size_hint: Optional[float] = None  # Fraction of capital
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantra_score": self.quantra_score,
            "runner_prob": self.runner_prob,
            "regime": self.regime,
            "volatility_band": self.volatility_band,
            "estimated_move": self.estimated_move,
            "size_hint": self.size_hint,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskDecision:
    """Decision from risk engine."""
    decision: RiskDecisionType
    reason: str = ""
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def approved(self) -> bool:
        return self.decision == RiskDecisionType.APPROVE
    
    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "timestamp": self.timestamp.isoformat(),
        }
