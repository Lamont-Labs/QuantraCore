"""
EEO Engine Enumerations

Defines all enumeration types for entry/exit optimization.
"""

from enum import Enum


class EntryMode(str, Enum):
    """How the position is entered."""
    SINGLE = "SINGLE"
    SCALED_IN = "SCALED_IN"


class ExitMode(str, Enum):
    """How the position is exited."""
    SINGLE = "SINGLE"
    SCALED_OUT = "SCALED_OUT"


class EntryAggressiveness(str, Enum):
    """How aggressive the entry strategy is."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TrailingStopMode(str, Enum):
    """Mode for trailing stop calculation."""
    ATR = "ATR"
    PERCENT = "PERCENT"
    STRUCTURAL = "STRUCTURAL"


class VolatilityBand(str, Enum):
    """Volatility classification."""
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class LiquidityBand(str, Enum):
    """Liquidity classification."""
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class RegimeType(str, Enum):
    """Market regime classification."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    SQUEEZE = "squeeze"
    CRASH = "crash"


class SuppressionState(str, Enum):
    """Signal suppression state."""
    NONE = "none"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"


class SignalDirection(str, Enum):
    """Direction of the signal."""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderTypeEEO(str, Enum):
    """Order types for EEO plans."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class ExitStyle(str, Enum):
    """Style for profit target exits."""
    LIMIT = "LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class QualityTier(str, Enum):
    """Quality tier for signals."""
    A_PLUS = "A_PLUS"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
