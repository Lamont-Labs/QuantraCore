"""
Parallel Battle Simulation Cluster.

Runs 100+ simulated trades simultaneously across multiple strategies.
Each simulation makes predictions and compares against actual outcomes.
"""

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from .models import (
    BattleSimulation,
    SimulationStrategy,
    HyperspeedConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for a simulation strategy."""
    strategy: SimulationStrategy
    entry_threshold: float = 60.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_hold_bars: int = 10
    position_size_pct: float = 0.02
    allow_shorts: bool = False
    
    regime_filter: Optional[List[int]] = None
    min_volume_ratio: float = 1.0


STRATEGY_CONFIGS: Dict[SimulationStrategy, StrategyConfig] = {
    SimulationStrategy.CONSERVATIVE: StrategyConfig(
        strategy=SimulationStrategy.CONSERVATIVE,
        entry_threshold=75.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        max_hold_bars=5,
        position_size_pct=0.01,
        regime_filter=[0, 1],
    ),
    SimulationStrategy.MODERATE: StrategyConfig(
        strategy=SimulationStrategy.MODERATE,
        entry_threshold=65.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_hold_bars=10,
        position_size_pct=0.02,
    ),
    SimulationStrategy.AGGRESSIVE: StrategyConfig(
        strategy=SimulationStrategy.AGGRESSIVE,
        entry_threshold=55.0,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
        max_hold_bars=15,
        position_size_pct=0.04,
    ),
    SimulationStrategy.SCALPING: StrategyConfig(
        strategy=SimulationStrategy.SCALPING,
        entry_threshold=70.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_hold_bars=3,
        position_size_pct=0.03,
        min_volume_ratio=2.0,
    ),
    SimulationStrategy.SWING: StrategyConfig(
        strategy=SimulationStrategy.SWING,
        entry_threshold=60.0,
        stop_loss_pct=0.07,
        take_profit_pct=0.12,
        max_hold_bars=20,
        position_size_pct=0.02,
        regime_filter=[0],
    ),
    SimulationStrategy.CONTRARIAN: StrategyConfig(
        strategy=SimulationStrategy.CONTRARIAN,
        entry_threshold=50.0,
        stop_loss_pct=0.06,
        take_profit_pct=0.08,
        max_hold_bars=8,
        position_size_pct=0.02,
        regime_filter=[3, 4],
    ),
    SimulationStrategy.MOMENTUM: StrategyConfig(
        strategy=SimulationStrategy.MOMENTUM,
        entry_threshold=65.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        max_hold_bars=10,
        position_size_pct=0.03,
        regime_filter=[0, 1],
        min_volume_ratio=1.5,
    ),
    SimulationStrategy.MEAN_REVERSION: StrategyConfig(
        strategy=SimulationStrategy.MEAN_REVERSION,
        entry_threshold=55.0,
        stop_loss_pct=0.04,
        take_profit_pct=0.05,
        max_hold_bars=5,
        position_size_pct=0.02,
    ),
}


class ParallelBattleCluster:
    """
    Runs parallel battle simulations across multiple strategies.
    
    For each historical data point, simulates what would have happened
    if we had traded using different strategies and parameters.
    """
    
    def __init__(self, config: Optional[HyperspeedConfig] = None):
        self.config = config or HyperspeedConfig()
        
        self._simulations: List[BattleSimulation] = []
        self._strategy_stats: Dict[SimulationStrategy, Dict[str, float]] = {
            s: {"wins": 0, "losses": 0, "total_return": 0.0, "trades": 0}
            for s in SimulationStrategy
        }
        
        self._active = False
        
        logger.info(f"[BattleCluster] Initialized with {self.config.parallel_simulations} parallel sims")
    
    def simulate_trade(
        self,
        window: OhlcvWindow,
        future_bars: List[OhlcvBar],
        predictions: Dict[str, float],
        strategy: SimulationStrategy,
    ) -> Optional[BattleSimulation]:
        """
        Simulate a single trade with known future outcome.
        
        Args:
            window: Historical window at entry point
            future_bars: Known future price bars
            predictions: Model predictions at entry
            strategy: Strategy to simulate
        
        Returns:
            BattleSimulation result
        """
        if not window.bars or not future_bars:
            return None
        
        config = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS[SimulationStrategy.MODERATE])
        
        quantrascore = predictions.get("quantrascore", 0)
        regime = int(predictions.get("regime", 2))
        volume_ratio = predictions.get("volume", 1.0)
        
        if quantrascore < config.entry_threshold:
            return None
        
        if config.regime_filter and regime not in config.regime_filter:
            return None
        
        if volume_ratio < config.min_volume_ratio:
            return None
        
        entry_price = window.bars[-1].close
        stop_price = entry_price * (1 - config.stop_loss_pct)
        target_price = entry_price * (1 + config.take_profit_pct)
        
        exit_price = entry_price
        exit_bar = 0
        stop_triggered = False
        target_reached = False
        
        for i, bar in enumerate(future_bars[:config.max_hold_bars]):
            if bar.low <= stop_price:
                exit_price = stop_price
                exit_bar = i + 1
                stop_triggered = True
                break
            
            if bar.high >= target_price:
                exit_price = target_price
                exit_bar = i + 1
                target_reached = True
                break
            
            exit_price = bar.close
            exit_bar = i + 1
        
        simulated_return = (exit_price - entry_price) / entry_price
        
        actual_final = future_bars[-1].close if future_bars else entry_price
        actual_return = (actual_final - entry_price) / entry_price
        
        direction_correct = (simulated_return > 0 and predictions.get("direction", 0.5) > 0.5) or \
                          (simulated_return < 0 and predictions.get("direction", 0.5) <= 0.5)
        
        predicted_runup = predictions.get("runup", 0) / 100
        actual_max = max(b.high for b in future_bars) if future_bars else entry_price
        actual_runup = (actual_max - entry_price) / entry_price
        
        runup_accuracy = 1.0 - min(1.0, abs(predicted_runup - actual_runup) / max(actual_runup, 0.01))
        
        prediction_accuracy = (
            0.4 * (1.0 if direction_correct else 0.0) +
            0.4 * runup_accuracy +
            0.2 * (1.0 if target_reached else 0.5 if not stop_triggered else 0.0)
        )
        
        lessons = []
        
        if stop_triggered and quantrascore > 70:
            lessons.append("High score but stopped out - check volatility prediction")
        
        if target_reached and quantrascore < 60:
            lessons.append("Low score but hit target - signal underestimation")
        
        if abs(predicted_runup - actual_runup) > 0.05:
            lessons.append(f"Runup prediction off by {abs(predicted_runup - actual_runup)*100:.1f}%")
        
        if not direction_correct:
            lessons.append("Direction prediction incorrect")
        
        sim = BattleSimulation(
            strategy=strategy,
            symbol=window.symbol,
            entry_date=window.bars[-1].timestamp.date(),
            entry_price=entry_price,
            exit_price=exit_price,
            direction="long",
            quantrascore=quantrascore,
            prediction_heads=predictions,
            simulated_return_pct=simulated_return * 100,
            actual_return_pct=actual_return * 100,
            prediction_accuracy=prediction_accuracy,
            trade_duration_bars=exit_bar,
            stop_triggered=stop_triggered,
            target_reached=target_reached,
            lessons=lessons,
        )
        
        self._simulations.append(sim)
        
        stats = self._strategy_stats[strategy]
        stats["trades"] += 1
        stats["total_return"] += simulated_return * 100
        if simulated_return > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        
        return sim
    
    def run_parallel_simulations(
        self,
        window: OhlcvWindow,
        future_bars: List[OhlcvBar],
        predictions: Dict[str, float],
        strategies: Optional[List[SimulationStrategy]] = None,
        variations_per_strategy: int = 10,
    ) -> List[BattleSimulation]:
        """
        Run multiple simulations in parallel across strategies.
        
        Args:
            window: Historical window at entry point
            future_bars: Known future price bars
            predictions: Model predictions at entry
            strategies: Strategies to simulate
            variations_per_strategy: Number of parameter variations per strategy
        
        Returns:
            List of simulation results
        """
        strategies = strategies or self.config.simulation_strategies
        results = []
        
        for strategy in strategies:
            base_sim = self.simulate_trade(window, future_bars, predictions, strategy)
            if base_sim:
                results.append(base_sim)
            
            for _ in range(variations_per_strategy - 1):
                varied_predictions = self._vary_predictions(predictions)
                varied_sim = self.simulate_trade(window, future_bars, varied_predictions, strategy)
                if varied_sim:
                    results.append(varied_sim)
        
        return results
    
    def _vary_predictions(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Create a variation of predictions for sensitivity analysis."""
        varied = dict(predictions)
        
        noise = random.gauss(0, 5)
        varied["quantrascore"] = max(0, min(100, predictions.get("quantrascore", 50) + noise))
        
        if random.random() < 0.1:
            varied["regime"] = random.randint(0, 4)
        
        volume_noise = random.gauss(1.0, 0.2)
        varied["volume"] = max(0.1, predictions.get("volume", 1.0) * volume_noise)
        
        return varied
    
    def run_batch_simulations(
        self,
        samples: List[Tuple[OhlcvWindow, List[OhlcvBar], Dict[str, float]]],
        max_workers: int = 8,
    ) -> List[BattleSimulation]:
        """
        Run simulations on a batch of samples in parallel.
        
        Args:
            samples: List of (window, future_bars, predictions) tuples
            max_workers: Number of parallel workers
        
        Returns:
            All simulation results
        """
        self._active = True
        all_results = []
        
        def process_sample(sample):
            window, future_bars, predictions = sample
            return self.run_parallel_simulations(window, future_bars, predictions)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_sample, s) for s in samples]
            
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"[BattleCluster] Simulation error: {e}")
        
        self._active = False
        logger.info(f"[BattleCluster] Completed {len(all_results)} simulations from {len(samples)} samples")
        
        return all_results
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each strategy."""
        performance = {}
        
        for strategy, stats in self._strategy_stats.items():
            trades = stats["trades"]
            if trades == 0:
                continue
            
            win_rate = stats["wins"] / trades * 100 if trades > 0 else 0
            avg_return = stats["total_return"] / trades if trades > 0 else 0
            
            performance[strategy.value] = {
                "trades": trades,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate_pct": round(win_rate, 2),
                "avg_return_pct": round(avg_return, 2),
                "total_return_pct": round(stats["total_return"], 2),
            }
        
        return performance
    
    def get_lessons_learned(self, min_occurrences: int = 5) -> List[Dict[str, Any]]:
        """Extract common lessons from simulations."""
        lesson_counts: Dict[str, int] = {}
        lesson_contexts: Dict[str, List[str]] = {}
        
        for sim in self._simulations:
            for lesson in sim.lessons:
                lesson_counts[lesson] = lesson_counts.get(lesson, 0) + 1
                if lesson not in lesson_contexts:
                    lesson_contexts[lesson] = []
                lesson_contexts[lesson].append(sim.symbol)
        
        lessons = []
        for lesson, count in sorted(lesson_counts.items(), key=lambda x: -x[1]):
            if count >= min_occurrences:
                lessons.append({
                    "lesson": lesson,
                    "occurrences": count,
                    "example_symbols": list(set(lesson_contexts[lesson]))[:5],
                })
        
        return lessons
    
    def get_simulation_count(self) -> int:
        """Get total number of simulations run."""
        return len(self._simulations)
    
    def get_recent_simulations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent simulations."""
        return [s.to_dict() for s in self._simulations[-limit:]]
    
    def is_active(self) -> bool:
        """Check if cluster is actively running simulations."""
        return self._active
    
    def reset_stats(self):
        """Reset all statistics and simulations."""
        self._simulations.clear()
        for stats in self._strategy_stats.values():
            stats["wins"] = 0
            stats["losses"] = 0
            stats["total_return"] = 0.0
            stats["trades"] = 0
        logger.info("[BattleCluster] Stats reset")
