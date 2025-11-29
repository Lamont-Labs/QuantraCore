"""
QuantraCore Apex Autonomous Trading System.

Institutional-grade autonomous trading orchestration with:
- Real-time data streaming (Polygon WebSocket)
- Signal quality filtering (A+/A tier only)
- Position monitoring and management
- Trade outcome tracking for self-learning
- Full Omega Directive compliance

MODES:
- PAPER: Routes to Alpaca paper trading
- RESEARCH: Logs signals only (no execution)
- LIVE: DISABLED by default

Classification: Research/Paper Trading Only
"""

from .models import (
    TradingState,
    PositionState,
    TradeOutcome,
    SignalDecision,
    FilterResult,
    OrchestratorConfig,
    QualityThresholds,
)
from .signal_quality_filter import SignalQualityFilter
from .position_monitor import PositionMonitor
from .trade_outcome_tracker import TradeOutcomeTracker
from .trading_orchestrator import TradingOrchestrator


__all__ = [
    "TradingState",
    "PositionState",
    "TradeOutcome",
    "SignalDecision",
    "FilterResult",
    "OrchestratorConfig",
    "QualityThresholds",
    "SignalQualityFilter",
    "PositionMonitor",
    "TradeOutcomeTracker",
    "TradingOrchestrator",
]
