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


class BrokerType(str, Enum):
    """Supported broker types for universal routing."""
    ALPACA_PAPER = "alpaca_paper"           # Alpaca paper trading (default)
    ALPACA_LIVE = "alpaca_live"             # Alpaca live trading
    BINANCE = "binance"                     # Binance spot live
    BINANCE_TESTNET = "binance_testnet"     # Binance testnet (paper)
    BINANCE_FUTURES = "binance_futures"     # Binance USDT-M futures
    IBKR = "ibkr"                           # Interactive Brokers TWS/Gateway
    IBKR_PAPER = "ibkr_paper"               # IBKR paper (port 7497)
    BYBIT = "bybit"                         # Bybit live
    BYBIT_TESTNET = "bybit_testnet"         # Bybit testnet (paper)
    TRADIER = "tradier"                     # Tradier live
    TRADIER_SANDBOX = "tradier_sandbox"     # Tradier sandbox (paper)
    PAPER_SIM = "paper_sim"                 # Internal paper simulator


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
