"""
SimulatedTradeRunner - Executes trades against simulated market data.

Runs the ApexEngine against chaos scenarios and records outcomes
for learning via the feedback loop.
"""

import logging
import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import threading

from .simulator import (
    MarketSimulator,
    SimulationResult,
    SimulatedBar,
    get_simulator,
)
from .scenarios import ScenarioType, get_all_scenarios

try:
    from src.quantracore_apex.alpha_factory.feedback_loop import (
        get_feedback_tracker,
        trigger_retrain_if_ready,
    )
    FEEDBACK_LOOP_AVAILABLE = True
except ImportError:
    FEEDBACK_LOOP_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for simulation runner."""
    num_scenarios: int = 100
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "NVDA", "TSLA", "AMD", "META", "GOOGL", "MSFT", "AMZN"
    ])
    scenario_types: Optional[List[ScenarioType]] = None
    
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    
    intensity_range: tuple = (0.7, 1.5)
    price_range: tuple = (50.0, 400.0)
    bars_range: tuple = (60, 150)
    
    save_results: bool = True
    results_dir: str = "data/simulator"
    
    connect_feedback_loop: bool = True


@dataclass
class SimulatedTrade:
    """Record of a simulated trade."""
    trade_id: str
    symbol: str
    scenario_type: str
    
    entry_bar: int
    entry_price: float
    entry_time: datetime
    side: str
    quantity: float
    
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    pnl_dollars: float = 0.0
    pnl_percent: float = 0.0
    
    quantra_score: float = 0.0
    protocol_flags: List[str] = field(default_factory=list)
    feature_snapshot: Dict[str, float] = field(default_factory=dict)
    
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def is_closed(self) -> bool:
        return self.exit_bar is not None
    
    def compute_label(self) -> str:
        """Compute training label from P&L."""
        if self.pnl_percent >= 5.0:
            return "STRONG_WIN"
        elif self.pnl_percent >= 2.0:
            return "WIN"
        elif self.pnl_percent >= 0.5:
            return "MARGINAL_WIN"
        elif self.pnl_percent >= -0.5:
            return "SCRATCH"
        elif self.pnl_percent >= -2.0:
            return "LOSS"
        else:
            return "STRONG_LOSS"
    
    def to_training_sample(self) -> Dict[str, Any]:
        """Convert to ApexLab training format."""
        return {
            "symbol": self.symbol,
            "timestamp": self.entry_time.isoformat(),
            "features": self.feature_snapshot,
            "label": self.compute_label(),
            "metadata": {
                "source": "chaos_simulator",
                "trade_id": self.trade_id,
                "scenario_type": self.scenario_type,
                "quantra_score": self.quantra_score,
                "protocols": self.protocol_flags,
                "pnl_percent": self.pnl_percent,
                "exit_reason": self.exit_reason,
                "max_favorable": self.max_favorable_excursion,
                "max_adverse": self.max_adverse_excursion,
            }
        }


@dataclass
class BatchSimulationResult:
    """Aggregate results from batch simulation."""
    num_scenarios: int
    num_trades: int
    
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    by_scenario: Dict[str, Dict[str, Any]]
    by_label: Dict[str, int]
    
    trades: List[SimulatedTrade]
    training_samples: List[Dict[str, Any]]
    
    run_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "num_scenarios": self.num_scenarios,
                "num_trades": self.num_trades,
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "profit_factor": self.profit_factor,
                "run_time_seconds": self.run_time_seconds,
            },
            "by_scenario": self.by_scenario,
            "by_label": self.by_label,
            "num_training_samples": len(self.training_samples),
        }


class SimulatedTradeRunner:
    """
    Runs trading simulation against chaos scenarios.
    
    Executes a simplified trading strategy against simulated market data
    and captures outcomes for the feedback loop.
    
    Usage:
        runner = SimulatedTradeRunner()
        
        # Run single scenario
        trades = runner.run_single_scenario(result)
        
        # Run chaos training batch
        batch_result = runner.run_chaos_training(config)
        
        # Results automatically flow to feedback loop
    """
    
    def __init__(
        self,
        simulator: Optional[MarketSimulator] = None,
        feedback_tracker: Optional[Any] = None,
    ):
        self._simulator = simulator or get_simulator()
        self._feedback_tracker = feedback_tracker
        self._trade_counter = 0
        self._lock = threading.Lock()
    
    def run_single_scenario(
        self,
        simulation: SimulationResult,
        initial_capital: float = 100_000.0,
        max_position_pct: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
    ) -> List[SimulatedTrade]:
        """
        Run trading simulation on a single scenario.
        
        Uses a simple momentum-based strategy to generate trades
        that can be used for learning.
        """
        trades = []
        bars = simulation.bars
        
        if len(bars) < 20:
            return trades
        
        capital = initial_capital
        position: Optional[SimulatedTrade] = None
        
        for i in range(20, len(bars)):
            bar = bars[i]
            
            if position:
                if position.side == "LONG":
                    pnl_pct = (bar.close - position.entry_price) / position.entry_price
                    if bar.high > position.entry_price:
                        mfe = (bar.high - position.entry_price) / position.entry_price
                        position.max_favorable_excursion = max(
                            position.max_favorable_excursion, mfe
                        )
                    if bar.low < position.entry_price:
                        mae = (position.entry_price - bar.low) / position.entry_price
                        position.max_adverse_excursion = max(
                            position.max_adverse_excursion, mae
                        )
                else:
                    pnl_pct = (position.entry_price - bar.close) / position.entry_price
                    if bar.low < position.entry_price:
                        mfe = (position.entry_price - bar.low) / position.entry_price
                        position.max_favorable_excursion = max(
                            position.max_favorable_excursion, mfe
                        )
                    if bar.high > position.entry_price:
                        mae = (bar.high - position.entry_price) / position.entry_price
                        position.max_adverse_excursion = max(
                            position.max_adverse_excursion, mae
                        )
                
                exit_reason = None
                
                if pnl_pct <= -stop_loss_pct:
                    exit_reason = "stop_loss"
                elif pnl_pct >= take_profit_pct:
                    exit_reason = "take_profit"
                elif i >= len(bars) - 1:
                    exit_reason = "end_of_data"
                elif random.random() < 0.03:
                    exit_reason = "signal_exit"
                
                if exit_reason:
                    position.exit_bar = i
                    position.exit_price = bar.close
                    position.exit_time = bar.timestamp
                    position.exit_reason = exit_reason
                    position.pnl_percent = pnl_pct * 100
                    position.pnl_dollars = pnl_pct * position.entry_price * position.quantity
                    
                    trades.append(position)
                    capital += position.pnl_dollars
                    position = None
            
            else:
                if random.random() > 0.05:
                    continue
                
                lookback = [bars[j].close for j in range(i-10, i)]
                momentum = (lookback[-1] - lookback[0]) / lookback[0]
                volatility = sum(abs(lookback[j] - lookback[j-1]) / lookback[j-1] 
                               for j in range(1, len(lookback))) / (len(lookback) - 1)
                
                position_size = capital * max_position_pct
                risk_adj = 1.0 / (1 + simulation.risk_multiplier * 0.3)
                position_size *= risk_adj
                
                quantity = position_size / bar.close
                
                if momentum > 0.005:
                    side = "LONG"
                elif momentum < -0.005:
                    side = "SHORT"
                else:
                    continue
                
                with self._lock:
                    self._trade_counter += 1
                    trade_id = f"SIM_{simulation.config.symbol}_{self._trade_counter}"
                
                features = self._compute_features(bars[:i+1])
                score = self._compute_mock_score(features, simulation.risk_multiplier)
                protocols = self._generate_mock_protocols(features, simulation.config.scenario_type)
                
                position = SimulatedTrade(
                    trade_id=trade_id,
                    symbol=simulation.config.symbol,
                    scenario_type=simulation.config.scenario_type.value,
                    entry_bar=i,
                    entry_price=bar.close,
                    entry_time=bar.timestamp,
                    side=side,
                    quantity=quantity,
                    quantra_score=score,
                    protocol_flags=protocols,
                    feature_snapshot=features,
                )
        
        if position and not position.is_closed():
            last_bar = bars[-1]
            if position.side == "LONG":
                pnl_pct = (last_bar.close - position.entry_price) / position.entry_price
            else:
                pnl_pct = (position.entry_price - last_bar.close) / position.entry_price
            
            position.exit_bar = len(bars) - 1
            position.exit_price = last_bar.close
            position.exit_time = last_bar.timestamp
            position.exit_reason = "end_of_simulation"
            position.pnl_percent = pnl_pct * 100
            position.pnl_dollars = pnl_pct * position.entry_price * position.quantity
            trades.append(position)
        
        return trades
    
    def run_chaos_training(
        self,
        config: Optional[RunnerConfig] = None,
    ) -> BatchSimulationResult:
        """
        Run full chaos training batch.
        
        Generates multiple scenarios, runs trading simulation on each,
        and aggregates results for learning.
        """
        config = config or RunnerConfig()
        start_time = datetime.utcnow()
        
        logger.info(
            f"[ChaosTrainer] Starting chaos training: "
            f"{config.num_scenarios} scenarios, {len(config.symbols)} symbols"
        )
        
        scenario_types = config.scenario_types
        if scenario_types is None:
            scenario_types = [s.scenario_type for s in get_all_scenarios()]
        
        simulations = self._simulator.run_chaos_batch(
            num_scenarios=config.num_scenarios,
            symbols=config.symbols,
            scenario_types=scenario_types,
            intensity_range=config.intensity_range,
            price_range=config.price_range,
            bars_range=config.bars_range,
        )
        
        all_trades = []
        
        for sim in simulations:
            trades = self.run_single_scenario(
                simulation=sim,
                initial_capital=config.initial_capital,
                max_position_pct=config.max_position_pct,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct,
            )
            all_trades.extend(trades)
        
        result = self._aggregate_results(
            simulations=simulations,
            trades=all_trades,
            start_time=start_time,
        )
        
        if config.connect_feedback_loop:
            self._send_to_feedback_loop(result.training_samples)
        
        if config.save_results:
            self._save_results(result, config.results_dir)
        
        logger.info(
            f"[ChaosTrainer] Completed: {result.num_trades} trades, "
            f"WR: {result.win_rate:.1f}%, PF: {result.profit_factor:.2f}, "
            f"{len(result.training_samples)} samples generated"
        )
        
        return result
    
    def _compute_features(self, bars: List[SimulatedBar]) -> Dict[str, float]:
        """Compute features from price history."""
        if len(bars) < 20:
            return {}
        
        closes = [b.close for b in bars[-20:]]
        highs = [b.high for b in bars[-20:]]
        lows = [b.low for b in bars[-20:]]
        volumes = [b.volume for b in bars[-20:]]
        
        sma_5 = sum(closes[-5:]) / 5
        sma_20 = sum(closes) / 20
        
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        
        highest_high = max(highs)
        lowest_low = min(lows)
        current = closes[-1]
        
        if highest_high != lowest_low:
            relative_position = (current - lowest_low) / (highest_high - lowest_low)
        else:
            relative_position = 0.5
        
        avg_volume = sum(volumes) / len(volumes)
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
        
        momentum = (closes[-1] - closes[0]) / closes[0]
        
        return {
            "close": current,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "sma_ratio": sma_5 / sma_20 if sma_20 > 0 else 1.0,
            "volatility": volatility * 100,
            "relative_position": relative_position,
            "volume_ratio": volume_ratio,
            "momentum_20": momentum * 100,
            "atr_pct": (highest_high - lowest_low) / current * 100 if current > 0 else 0,
        }
    
    def _compute_mock_score(
        self,
        features: Dict[str, float],
        risk_multiplier: float,
    ) -> float:
        """Compute mock QuantraScore based on features."""
        base_score = 50.0
        
        sma_ratio = features.get("sma_ratio", 1.0)
        if sma_ratio > 1.02:
            base_score += 15
        elif sma_ratio < 0.98:
            base_score -= 10
        
        vol = features.get("volatility", 2.0)
        if vol > 5:
            base_score -= 15
        elif vol < 1.5:
            base_score += 10
        
        momentum = features.get("momentum_20", 0)
        if momentum > 5:
            base_score += 10
        elif momentum < -5:
            base_score -= 5
        
        base_score -= risk_multiplier * 5
        
        base_score += random.uniform(-10, 10)
        
        return max(0, min(100, base_score))
    
    def _generate_mock_protocols(
        self,
        features: Dict[str, float],
        scenario_type: ScenarioType,
    ) -> List[str]:
        """Generate mock protocol flags based on scenario."""
        protocols = []
        
        tier_protocols = ["T01", "T05", "T10", "T15", "T20"]
        protocols.extend(random.sample(tier_protocols, k=random.randint(1, 3)))
        
        scenario_omega_map = {
            ScenarioType.FLASH_CRASH: ["O1", "O10", "O11"],
            ScenarioType.VOLATILITY_SPIKE: ["O1", "O2", "O5"],
            ScenarioType.GAP_EVENT: ["O7", "O8"],
            ScenarioType.LIQUIDITY_VOID: ["O9", "O12"],
            ScenarioType.MOMENTUM_EXHAUSTION: ["O3", "O6"],
            ScenarioType.SQUEEZE: ["O4", "O6"],
            ScenarioType.CORRELATION_BREAKDOWN: ["O10", "O15"],
            ScenarioType.BLACK_SWAN: ["O11", "O20"],
        }
        
        omega_flags = scenario_omega_map.get(scenario_type, ["O1"])
        if random.random() < 0.7:
            protocols.extend(random.sample(omega_flags, k=min(2, len(omega_flags))))
        
        return protocols
    
    def _aggregate_results(
        self,
        simulations: List[SimulationResult],
        trades: List[SimulatedTrade],
        start_time: datetime,
    ) -> BatchSimulationResult:
        """Aggregate results from batch run."""
        if not trades:
            return BatchSimulationResult(
                num_scenarios=len(simulations),
                num_trades=0,
                total_pnl=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                by_scenario={},
                by_label={},
                trades=[],
                training_samples=[],
                run_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )
        
        wins = [t for t in trades if t.pnl_percent > 0]
        losses = [t for t in trades if t.pnl_percent <= 0]
        
        total_pnl = sum(t.pnl_dollars for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = sum(t.pnl_percent for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl_percent for t in losses) / len(losses) if losses else 0
        
        gross_profit = sum(t.pnl_dollars for t in wins)
        gross_loss = abs(sum(t.pnl_dollars for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        by_scenario = {}
        for trade in trades:
            st = trade.scenario_type
            if st not in by_scenario:
                by_scenario[st] = {
                    "count": 0,
                    "wins": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                }
            by_scenario[st]["count"] += 1
            if trade.pnl_percent > 0:
                by_scenario[st]["wins"] += 1
            by_scenario[st]["total_pnl"] += trade.pnl_dollars
        
        for st in by_scenario:
            if by_scenario[st]["count"] > 0:
                by_scenario[st]["win_rate"] = by_scenario[st]["wins"] / by_scenario[st]["count"] * 100
                by_scenario[st]["avg_pnl"] = by_scenario[st]["total_pnl"] / by_scenario[st]["count"]
        
        by_label = {}
        for trade in trades:
            label = trade.compute_label()
            by_label[label] = by_label.get(label, 0) + 1
        
        training_samples = [t.to_training_sample() for t in trades]
        
        return BatchSimulationResult(
            num_scenarios=len(simulations),
            num_trades=len(trades),
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            by_scenario=by_scenario,
            by_label=by_label,
            trades=trades,
            training_samples=training_samples,
            run_time_seconds=(datetime.utcnow() - start_time).total_seconds(),
        )
    
    def _send_to_feedback_loop(self, samples: List[Dict[str, Any]]):
        """Send training samples to feedback loop and trigger retraining if ready."""
        if not samples:
            return
        
        try:
            samples_file = Path("data/apexlab/chaos_simulation_samples.json")
            samples_file.parent.mkdir(parents=True, exist_ok=True)
            
            existing = []
            if samples_file.exists():
                with open(samples_file) as f:
                    existing = json.load(f)
            
            existing.extend(samples)
            
            with open(samples_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            
            logger.info(
                f"[ChaosTrainer] Added {len(samples)} samples to training queue. "
                f"Total: {len(existing)} chaos samples ready."
            )
            
            if FEEDBACK_LOOP_AVAILABLE:
                feedback_samples_file = Path("data/apexlab/feedback_samples.json")
                feedback_samples_file.parent.mkdir(parents=True, exist_ok=True)
                
                fb_existing = []
                if feedback_samples_file.exists():
                    with open(feedback_samples_file) as f:
                        fb_existing = json.load(f)
                
                fb_existing.extend(samples)
                
                with open(feedback_samples_file, "w") as f:
                    json.dump(fb_existing, f, indent=2, default=str)
                
                logger.info(
                    f"[ChaosTrainer] Merged {len(samples)} samples into ApexLab feedback queue. "
                    f"Total feedback samples: {len(fb_existing)}"
                )
                
                if trigger_retrain_if_ready():
                    logger.info("[ChaosTrainer] Triggered ApexCore retraining from chaos samples!")
            
        except Exception as e:
            logger.error(f"[ChaosTrainer] Failed to save samples: {e}")
    
    def _save_results(self, result: BatchSimulationResult, results_dir: str):
        """Save batch results to disk."""
        try:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            summary_file = results_path / f"chaos_batch_{timestamp}.json"
            with open(summary_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            logger.info(f"[ChaosTrainer] Results saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"[ChaosTrainer] Failed to save results: {e}")


_runner: Optional[SimulatedTradeRunner] = None


def get_runner() -> SimulatedTradeRunner:
    """Get global runner instance."""
    global _runner
    if _runner is None:
        _runner = SimulatedTradeRunner()
    return _runner


def run_quick_chaos_training(
    num_scenarios: int = 50,
    symbols: Optional[List[str]] = None,
) -> BatchSimulationResult:
    """
    Quick helper to run chaos training.
    
    Usage:
        from quantracore_apex.simulator import run_quick_chaos_training
        result = run_quick_chaos_training(num_scenarios=100)
    """
    runner = get_runner()
    config = RunnerConfig(
        num_scenarios=num_scenarios,
        symbols=symbols or ["AAPL", "NVDA", "TSLA", "AMD"],
    )
    return runner.run_chaos_training(config)


__all__ = [
    "RunnerConfig",
    "SimulatedTrade",
    "BatchSimulationResult",
    "SimulatedTradeRunner",
    "get_runner",
    "run_quick_chaos_training",
]
