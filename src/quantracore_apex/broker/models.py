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
    
    def to_dict(self) -> Dict:
        return {
            "quantra_score": self.quantra_score,
            "runner_prob": self.runner_prob,
            "estimated_move": self.estimated_move,
            "regime": self.regime,
            "volatility_band": self.volatility_band,
            "notes": self.notes,
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
