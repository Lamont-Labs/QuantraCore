"""
MarketSimulator - Core Engine for Chaos Training.

Generates synthetic market data for various extreme scenarios
and provides the infrastructure for running simulated trades.
"""

import random
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Tuple
from enum import Enum
import numpy as np

from .scenarios import (
    ChaosScenario,
    ScenarioType,
    ScenarioParameters,
    get_scenario_by_type,
    get_all_scenarios,
)


logger = logging.getLogger(__name__)


@dataclass
class SimulatedBar:
    """Single OHLCV bar from simulation."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    scenario_type: ScenarioType = ScenarioType.NORMAL
    phase: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "scenario_type": self.scenario_type.value,
            "phase": self.phase,
            "metadata": self.metadata,
        }


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    symbol: str = "SIM"
    initial_price: float = 100.0
    num_bars: int = 100
    bar_interval_minutes: int = 1
    scenario_type: ScenarioType = ScenarioType.FLASH_CRASH
    intensity: float = 1.0
    random_seed: Optional[int] = None
    
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_scenario_params(self) -> ScenarioParameters:
        """Convert to ScenarioParameters."""
        return ScenarioParameters(
            intensity=self.intensity,
            duration_bars=self.num_bars,
            extra=self.extra_params,
        )


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    config: SimulationConfig
    bars: List[SimulatedBar]
    
    start_price: float = 0.0
    end_price: float = 0.0
    max_price: float = 0.0
    min_price: float = 0.0
    total_volume: float = 0.0
    
    max_drawdown_pct: float = 0.0
    max_runup_pct: float = 0.0
    
    scenario_name: str = ""
    risk_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.bars:
            self._compute_stats()
    
    def _compute_stats(self):
        """Compute summary statistics."""
        closes = [b.close for b in self.bars]
        highs = [b.high for b in self.bars]
        lows = [b.low for b in self.bars]
        volumes = [b.volume for b in self.bars]
        
        self.start_price = closes[0]
        self.end_price = closes[-1]
        self.max_price = max(highs)
        self.min_price = min(lows)
        self.total_volume = sum(volumes)
        
        peak = closes[0]
        max_dd = 0.0
        
        for close in closes:
            if close > peak:
                peak = close
            dd = (peak - close) / peak * 100
            max_dd = max(max_dd, dd)
        
        self.max_drawdown_pct = max_dd
        
        trough = closes[0]
        max_ru = 0.0
        
        for close in closes:
            if close < trough:
                trough = close
            ru = (close - trough) / trough * 100 if trough > 0 else 0
            max_ru = max(max_ru, ru)
        
        self.max_runup_pct = max_ru
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "config": {
                "symbol": self.config.symbol,
                "initial_price": self.config.initial_price,
                "num_bars": self.config.num_bars,
                "scenario_type": self.config.scenario_type.value,
                "intensity": self.config.intensity,
            },
            "stats": {
                "start_price": self.start_price,
                "end_price": self.end_price,
                "return_pct": (self.end_price - self.start_price) / self.start_price * 100,
                "max_price": self.max_price,
                "min_price": self.min_price,
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_runup_pct": self.max_runup_pct,
                "total_volume": self.total_volume,
            },
            "scenario": {
                "name": self.scenario_name,
                "risk_multiplier": self.risk_multiplier,
            },
            "num_bars": len(self.bars),
        }


class MarketSimulator:
    """
    Core market simulation engine.
    
    Generates synthetic OHLCV data for various extreme market scenarios.
    The generated data can be fed into the trading system for training.
    
    Usage:
        simulator = MarketSimulator()
        
        # Run single scenario
        result = simulator.run_scenario(
            scenario_type=ScenarioType.FLASH_CRASH,
            symbol="AAPL",
            initial_price=150.0,
            num_bars=100,
        )
        
        # Generate batch of scenarios
        results = simulator.run_chaos_batch(
            num_scenarios=100,
            symbols=["AAPL", "NVDA", "TSLA"],
        )
        
        # Stream bars for real-time simulation
        for bar in simulator.stream_scenario(...):
            process(bar)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self._seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self._scenarios = {s.scenario_type: s for s in get_all_scenarios()}
    
    def run_scenario(
        self,
        scenario_type: ScenarioType,
        symbol: str = "SIM",
        initial_price: float = 100.0,
        num_bars: int = 100,
        intensity: float = 1.0,
        start_time: Optional[datetime] = None,
        bar_interval_minutes: int = 1,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Run a single chaos scenario simulation.
        
        Args:
            scenario_type: Type of chaos scenario to simulate
            symbol: Stock symbol for the simulation
            initial_price: Starting price
            num_bars: Number of bars to generate
            intensity: Scenario intensity (0.5-2.0 typical)
            start_time: Starting timestamp (defaults to now)
            bar_interval_minutes: Minutes between bars
            extra_params: Additional scenario-specific parameters
        
        Returns:
            SimulationResult with generated bars and statistics
        """
        if scenario_type not in self._scenarios:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        scenario = self._scenarios[scenario_type]
        
        params = ScenarioParameters(
            intensity=intensity,
            duration_bars=num_bars,
            extra=extra_params or {},
        )
        
        price_path = scenario.generate_price_path(
            initial_price=initial_price,
            num_bars=num_bars,
            params=params,
        )
        
        if start_time is None:
            start_time = datetime.utcnow()
        
        bars = []
        current_time = start_time
        
        for i, bar_data in enumerate(price_path):
            bar = SimulatedBar(
                timestamp=current_time,
                symbol=symbol,
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"],
                scenario_type=scenario_type,
                phase=bar_data.get("phase", ""),
                metadata={k: v for k, v in bar_data.items() 
                         if k not in ("open", "high", "low", "close", "volume", "phase")},
            )
            bars.append(bar)
            current_time += timedelta(minutes=bar_interval_minutes)
        
        result = SimulationResult(
            config=SimulationConfig(
                symbol=symbol,
                initial_price=initial_price,
                num_bars=num_bars,
                bar_interval_minutes=bar_interval_minutes,
                scenario_type=scenario_type,
                intensity=intensity,
                extra_params=extra_params or {},
            ),
            bars=bars,
            scenario_name=scenario.name,
            risk_multiplier=scenario.get_risk_multiplier(),
        )
        
        logger.info(
            f"[Simulator] Generated {scenario.name} scenario: "
            f"{symbol} @ ${initial_price:.2f} → ${result.end_price:.2f} "
            f"(DD: {result.max_drawdown_pct:.1f}%, RU: {result.max_runup_pct:.1f}%)"
        )
        
        return result
    
    def stream_scenario(
        self,
        scenario_type: ScenarioType,
        symbol: str = "SIM",
        initial_price: float = 100.0,
        num_bars: int = 100,
        intensity: float = 1.0,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[SimulatedBar]:
        """
        Stream bars one at a time for real-time simulation.
        
        Yields:
            SimulatedBar objects one at a time
        """
        result = self.run_scenario(
            scenario_type=scenario_type,
            symbol=symbol,
            initial_price=initial_price,
            num_bars=num_bars,
            intensity=intensity,
            extra_params=extra_params,
        )
        
        for bar in result.bars:
            yield bar
    
    def run_chaos_batch(
        self,
        num_scenarios: int = 100,
        symbols: Optional[List[str]] = None,
        scenario_types: Optional[List[ScenarioType]] = None,
        intensity_range: Tuple[float, float] = (0.8, 1.5),
        price_range: Tuple[float, float] = (50.0, 500.0),
        bars_range: Tuple[int, int] = (50, 200),
    ) -> List[SimulationResult]:
        """
        Generate a batch of diverse chaos scenarios.
        
        This is useful for training - generates many varied scenarios
        to expose the model to a wide range of market conditions.
        
        Args:
            num_scenarios: Number of scenarios to generate
            symbols: List of symbols to use (randomly selected)
            scenario_types: List of scenario types (all if None)
            intensity_range: Min/max intensity values
            price_range: Min/max initial prices
            bars_range: Min/max number of bars
        
        Returns:
            List of SimulationResult objects
        """
        if symbols is None:
            symbols = ["AAPL", "NVDA", "TSLA", "AMD", "META", "GOOGL", "MSFT", "AMZN"]
        
        if scenario_types is None:
            scenario_types = list(self._scenarios.keys())
        
        results = []
        
        for i in range(num_scenarios):
            symbol = random.choice(symbols)
            scenario_type = random.choice(scenario_types)
            intensity = random.uniform(*intensity_range)
            initial_price = random.uniform(*price_range)
            num_bars = random.randint(*bars_range)
            
            try:
                result = self.run_scenario(
                    scenario_type=scenario_type,
                    symbol=symbol,
                    initial_price=initial_price,
                    num_bars=num_bars,
                    intensity=intensity,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"[Simulator] Scenario {i} failed: {e}")
        
        logger.info(
            f"[Simulator] Generated {len(results)} chaos scenarios "
            f"({len(scenario_types)} types, {len(symbols)} symbols)"
        )
        
        return results
    
    def run_stress_test(
        self,
        symbol: str = "TEST",
        initial_price: float = 100.0,
        scenarios_per_type: int = 10,
        intensity_levels: List[float] = [0.5, 1.0, 1.5, 2.0],
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress test across all scenarios.
        
        Useful for evaluating how a trading strategy performs
        under various extreme conditions.
        
        Returns:
            Dictionary with stress test results by scenario type
        """
        results = {}
        
        for scenario_type in self._scenarios.keys():
            scenario_results = []
            
            for intensity in intensity_levels:
                for _ in range(scenarios_per_type):
                    try:
                        result = self.run_scenario(
                            scenario_type=scenario_type,
                            symbol=symbol,
                            initial_price=initial_price,
                            intensity=intensity,
                        )
                        scenario_results.append({
                            "intensity": intensity,
                            "max_drawdown": result.max_drawdown_pct,
                            "max_runup": result.max_runup_pct,
                            "return_pct": (result.end_price - result.start_price) / result.start_price * 100,
                        })
                    except Exception as e:
                        logger.warning(f"[Simulator] Stress test failed: {e}")
            
            if scenario_results:
                avg_dd = sum(r["max_drawdown"] for r in scenario_results) / len(scenario_results)
                avg_return = sum(r["return_pct"] for r in scenario_results) / len(scenario_results)
                
                results[scenario_type.value] = {
                    "num_runs": len(scenario_results),
                    "avg_max_drawdown": avg_dd,
                    "avg_return": avg_return,
                    "worst_drawdown": max(r["max_drawdown"] for r in scenario_results),
                    "risk_multiplier": self._scenarios[scenario_type].get_risk_multiplier(),
                }
        
        logger.info(f"[Simulator] Stress test complete: {len(results)} scenario types tested")
        
        return results
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available scenario types with descriptions."""
        return [
            {
                "type": s.scenario_type.value,
                "name": s.name,
                "description": s.description,
                "risk_multiplier": s.get_risk_multiplier(),
            }
            for s in self._scenarios.values()
        ]


_simulator: Optional[MarketSimulator] = None


def get_simulator() -> MarketSimulator:
    """Get global simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = MarketSimulator()
    return _simulator


__all__ = [
    "SimulatedBar",
    "SimulationConfig",
    "SimulationResult",
    "MarketSimulator",
    "get_simulator",
]
