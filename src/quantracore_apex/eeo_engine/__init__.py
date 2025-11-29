"""
Entry/Exit Optimization Engine (EEO Engine)

Calculates optimal entry zones, exit targets, stops, and execution strategies
for trading signals in a deterministic, model-assisted, institution-safe way.

This module sits between the Apex signal engine and the Execution Engine.
"""

from .enums import (
    EntryMode,
    ExitMode,
    EntryAggressiveness,
    TrailingStopMode,
    VolatilityBand,
    LiquidityBand,
    SignalDirection,
    QualityTier,
    RegimeType,
    SuppressionState,
    OrderTypeEEO,
    ExitStyle,
)
from .models import (
    EntryExitPlan,
    EntryZone,
    BaseEntry,
    ScaledEntry,
    ProtectiveStop,
    ProfitTarget,
    TrailingStop,
    TimeBasedExit,
    AbortCondition,
    PlanMetadata,
)
from .contexts import (
    SignalContext,
    PredictiveContext,
    MarketMicrostructure,
    RiskContext,
    EEOContext,
)
from .profiles import (
    EEOProfile,
    ProfileType,
    CONSERVATIVE_PROFILE,
    BALANCED_PROFILE,
    AGGRESSIVE_RESEARCH_PROFILE,
    get_profile,
)
from .entry_optimizer import EntryOptimizer
from .exit_optimizer import ExitOptimizer
from .eeo_optimizer import EntryExitOptimizer

__all__ = [
    "EntryMode",
    "ExitMode",
    "EntryAggressiveness",
    "TrailingStopMode",
    "VolatilityBand",
    "LiquidityBand",
    "SignalDirection",
    "QualityTier",
    "RegimeType",
    "SuppressionState",
    "OrderTypeEEO",
    "ExitStyle",
    "EntryExitPlan",
    "EntryZone",
    "BaseEntry",
    "ScaledEntry",
    "ProtectiveStop",
    "ProfitTarget",
    "TrailingStop",
    "TimeBasedExit",
    "AbortCondition",
    "PlanMetadata",
    "SignalContext",
    "PredictiveContext",
    "MarketMicrostructure",
    "RiskContext",
    "EEOContext",
    "EEOProfile",
    "ProfileType",
    "CONSERVATIVE_PROFILE",
    "BALANCED_PROFILE",
    "AGGRESSIVE_RESEARCH_PROFILE",
    "get_profile",
    "EntryOptimizer",
    "ExitOptimizer",
    "EntryExitOptimizer",
]
