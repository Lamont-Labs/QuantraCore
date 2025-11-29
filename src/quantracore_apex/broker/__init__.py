"""
QuantraCore Apex Broker Layer.

Provides execution infrastructure for RESEARCH, PAPER, and LIVE trading modes.
Includes broker adapters, risk engine, and execution engine.

SAFETY: Live trading is DISABLED by default. Paper trading only.
"""

from .oms import OrderManagementSystem, Order
from .oms import OrderStatus as LegacyOrderStatus
from .oms import OrderSide as LegacyOrderSide
from .oms import OrderType as LegacyOrderType

from .enums import (
    ExecutionMode,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderIntent,
    OrderStatus,
    PositionSide,
    RiskDecisionType,
    SignalDirection,
)
from .models import (
    OrderTicket,
    ExecutionResult,
    BrokerPosition,
    ApexSignal,
    RiskDecision,
    OrderMetadata,
)
from .config import BrokerConfig, RiskConfig, AlpacaConfig, load_broker_config, create_default_config_file
from .risk_engine import RiskEngine
from .execution_engine import ExecutionEngine
from .router import BrokerRouter
from .execution_logger import ExecutionLogger

__all__ = [
    # Legacy OMS (backward compatibility)
    "OrderManagementSystem",
    "Order",
    "LegacyOrderStatus",
    "LegacyOrderSide",
    "LegacyOrderType",
    # New Broker Layer
    "ExecutionMode",
    "OrderSide",
    "OrderType",
    "TimeInForce",
    "OrderIntent",
    "OrderStatus",
    "PositionSide",
    "RiskDecisionType",
    "SignalDirection",
    "OrderTicket",
    "ExecutionResult",
    "BrokerPosition",
    "ApexSignal",
    "RiskDecision",
    "OrderMetadata",
    "BrokerConfig",
    "RiskConfig",
    "AlpacaConfig",
    "load_broker_config",
    "create_default_config_file",
    "RiskEngine",
    "ExecutionEngine",
    "BrokerRouter",
    "ExecutionLogger",
]
