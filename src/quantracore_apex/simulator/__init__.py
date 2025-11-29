"""
MarketSimulator - Chaos Training Engine for QuantraCore Apex.

Simulates extreme market conditions to stress-test and train the trading system:
- Flash crashes, volatility spikes, gap events
- Liquidity voids, squeeze scenarios, correlation breakdowns
- Black swan events and multi-sigma moves

Simulated trades flow into the feedback loop for automatic learning.
"""

from .scenarios import (
    ChaosScenario,
    ScenarioType,
    FlashCrashScenario,
    VolatilitySpikeScenario,
    GapEventScenario,
    LiquidityVoidScenario,
    MomentumExhaustionScenario,
    SqueezeScenario,
    CorrelationBreakdownScenario,
    BlackSwanScenario,
    get_scenario_by_type,
    get_all_scenarios,
)

from .simulator import (
    MarketSimulator,
    SimulationConfig,
    SimulatedBar,
    SimulationResult,
)

from .runner import (
    SimulatedTradeRunner,
    RunnerConfig,
    BatchSimulationResult,
    run_quick_chaos_training,
)


__all__ = [
    "ChaosScenario",
    "ScenarioType",
    "FlashCrashScenario",
    "VolatilitySpikeScenario",
    "GapEventScenario",
    "LiquidityVoidScenario",
    "MomentumExhaustionScenario",
    "SqueezeScenario",
    "CorrelationBreakdownScenario",
    "BlackSwanScenario",
    "get_scenario_by_type",
    "get_all_scenarios",
    "MarketSimulator",
    "SimulationConfig",
    "SimulatedBar",
    "SimulationResult",
    "SimulatedTradeRunner",
    "RunnerConfig",
    "BatchSimulationResult",
    "run_quick_chaos_training",
]
