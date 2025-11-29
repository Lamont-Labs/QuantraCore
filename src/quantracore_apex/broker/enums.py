"""
Enums for QuantraCore Apex Broker Layer.

Defines execution modes, order types, and related enumerations.
"""

from enum import Enum


class ExecutionMode(str, Enum):
    """Execution mode for the trading system."""
    RESEARCH = "RESEARCH"  # No orders, only signals/logs (default)
    PAPER = "PAPER"        # Orders routed to paper simulator / Alpaca paper
    LIVE = "LIVE"          # Orders routed to real brokers (DISABLED by default)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    """Time in force for orders."""
    DAY = "DAY"   # Day order
    GTC = "GTC"   # Good till cancelled
    IOC = "IOC"   # Immediate or cancel
    FOK = "FOK"   # Fill or kill


class OrderIntent(str, Enum):
    """Intent/purpose of the order."""
    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"
    REDUCE = "REDUCE"
    EXIT_ALL = "EXIT_ALL"


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "NEW"
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    """Position side."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class RiskDecisionType(str, Enum):
    """Risk engine decision."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class SignalDirection(str, Enum):
    """Signal direction from Apex engine."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"
