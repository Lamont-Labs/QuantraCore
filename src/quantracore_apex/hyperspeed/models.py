"""
Hyperspeed Learning System - Data Models.

Defines all data structures for accelerated learning.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional
import uuid


class HyperspeedMode(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"
    BATTLE_SIMULATION = "battle_simulation"
    MULTI_SOURCE_FUSION = "multi_source_fusion"
    OVERNIGHT_INTENSIVE = "overnight_intensive"
    FULL_HYPERSPEED = "full_hyperspeed"


class ReplaySpeed(str, Enum):
    NORMAL = "1x"
    FAST = "10x"
    TURBO = "100x"
    HYPERSPEED = "1000x"
    MAXIMUM = "max"


class SimulationStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    CONTRARIAN = "contrarian"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


class DataSource(str, Enum):
    POLYGON = "polygon"
    ALPACA = "alpaca"
    BINANCE = "binance"
    OPTIONS_FLOW = "options_flow"
    DARK_POOL = "dark_pool"
    SENTIMENT = "sentiment"
    LEVEL2 = "level2"
    ECONOMIC = "economic"


@dataclass
class HyperspeedConfig:
    """Configuration for Hyperspeed Learning System."""
    
    mode: HyperspeedMode = HyperspeedMode.FULL_HYPERSPEED
    
    replay_speed: ReplaySpeed = ReplaySpeed.HYPERSPEED
    replay_years: int = 5
    replay_symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "NFLX",
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        "UBER", "LYFT", "ABNB", "DASH", "RBLX",
        "SNAP", "PINS", "TWLO", "DDOG", "NET",
        "CRWD", "ZS", "PANW", "FTNT", "OKTA",
        "SHOP", "SQ", "COIN", "HOOD", "SOFI",
        "PLTR", "SNOW", "MDB", "TEAM", "ZM",
        "ARM", "SMCI", "MRVL", "AVGO", "QCOM",
        "ENPH", "SEDG", "FSLR", "RUN", "NOVA",
    ])
    
    parallel_simulations: int = 100
    simulation_strategies: List[SimulationStrategy] = field(default_factory=lambda: [
        SimulationStrategy.CONSERVATIVE,
        SimulationStrategy.MODERATE,
        SimulationStrategy.AGGRESSIVE,
        SimulationStrategy.SCALPING,
        SimulationStrategy.SWING,
        SimulationStrategy.CONTRARIAN,
        SimulationStrategy.MOMENTUM,
        SimulationStrategy.MEAN_REVERSION,
    ])
    
    data_sources: List[DataSource] = field(default_factory=lambda: [
        DataSource.POLYGON,
        DataSource.ALPACA,
    ])
    
    overnight_start_hour: int = 16
    overnight_end_hour: int = 4
    overnight_training_enabled: bool = True
    
    max_samples_per_cycle: int = 100000
    min_samples_for_training: int = 5000
    training_batch_size: int = 10000
    
    save_to_database: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "replay_speed": self.replay_speed.value,
            "replay_years": self.replay_years,
            "replay_symbols_count": len(self.replay_symbols),
            "parallel_simulations": self.parallel_simulations,
            "simulation_strategies": [s.value for s in self.simulation_strategies],
            "data_sources": [d.value for d in self.data_sources],
            "overnight_training_enabled": self.overnight_training_enabled,
            "max_samples_per_cycle": self.max_samples_per_cycle,
        }


@dataclass
class ReplaySession:
    """A single historical replay session."""
    
    session_id: str = field(default_factory=lambda: f"REPLAY-{uuid.uuid4().hex[:8]}")
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    replay_start_date: date = field(default_factory=lambda: date.today() - timedelta(days=365*5))
    replay_end_date: date = field(default_factory=date.today)
    
    symbols_processed: int = 0
    bars_replayed: int = 0
    windows_generated: int = 0
    samples_created: int = 0
    
    predictions_made: int = 0
    outcomes_captured: int = 0
    
    speed_multiplier: float = 1000.0
    effective_days_per_minute: float = 0.0
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "replay_start_date": self.replay_start_date.isoformat(),
            "replay_end_date": self.replay_end_date.isoformat(),
            "symbols_processed": self.symbols_processed,
            "bars_replayed": self.bars_replayed,
            "windows_generated": self.windows_generated,
            "samples_created": self.samples_created,
            "predictions_made": self.predictions_made,
            "outcomes_captured": self.outcomes_captured,
            "speed_multiplier": self.speed_multiplier,
            "effective_days_per_minute": self.effective_days_per_minute,
            "error_count": len(self.errors),
        }


@dataclass
class BattleSimulation:
    """A parallel battle simulation result."""
    
    simulation_id: str = field(default_factory=lambda: f"SIM-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    strategy: SimulationStrategy = SimulationStrategy.MODERATE
    symbol: str = ""
    entry_date: date = field(default_factory=date.today)
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: str = "long"
    
    quantrascore: float = 0.0
    prediction_heads: Dict[str, float] = field(default_factory=dict)
    
    simulated_return_pct: float = 0.0
    actual_return_pct: float = 0.0
    prediction_accuracy: float = 0.0
    
    trade_duration_bars: int = 0
    stop_triggered: bool = False
    target_reached: bool = False
    
    lessons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
            "symbol": self.symbol,
            "entry_date": self.entry_date.isoformat(),
            "direction": self.direction,
            "quantrascore": self.quantrascore,
            "simulated_return_pct": self.simulated_return_pct,
            "actual_return_pct": self.actual_return_pct,
            "prediction_accuracy": self.prediction_accuracy,
            "trade_duration_bars": self.trade_duration_bars,
            "lessons_count": len(self.lessons),
        }


@dataclass
class AggregatedSample:
    """A training sample aggregated from multiple data sources."""
    
    sample_id: str = field(default_factory=lambda: f"SAMPLE-{uuid.uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    
    primary_features: Dict[str, float] = field(default_factory=dict)
    
    polygon_data: Dict[str, Any] = field(default_factory=dict)
    alpaca_data: Dict[str, Any] = field(default_factory=dict)
    options_flow_data: Dict[str, Any] = field(default_factory=dict)
    dark_pool_data: Dict[str, Any] = field(default_factory=dict)
    sentiment_data: Dict[str, Any] = field(default_factory=dict)
    level2_data: Dict[str, Any] = field(default_factory=dict)
    economic_data: Dict[str, Any] = field(default_factory=dict)
    
    data_sources_available: List[DataSource] = field(default_factory=list)
    data_completeness_score: float = 0.0
    
    labels: Dict[str, float] = field(default_factory=dict)
    
    def get_fused_features(self) -> Dict[str, float]:
        """Get all features fused from multiple sources."""
        fused = dict(self.primary_features)
        
        for key, value in self.polygon_data.items():
            if isinstance(value, (int, float)):
                fused[f"polygon_{key}"] = float(value)
        
        for key, value in self.alpaca_data.items():
            if isinstance(value, (int, float)):
                fused[f"alpaca_{key}"] = float(value)
        
        for key, value in self.options_flow_data.items():
            if isinstance(value, (int, float)):
                fused[f"options_{key}"] = float(value)
        
        for key, value in self.dark_pool_data.items():
            if isinstance(value, (int, float)):
                fused[f"darkpool_{key}"] = float(value)
        
        for key, value in self.sentiment_data.items():
            if isinstance(value, (int, float)):
                fused[f"sentiment_{key}"] = float(value)
        
        return fused


@dataclass
class TrainingCycle:
    """A complete training cycle in hyperspeed mode."""
    
    cycle_id: str = field(default_factory=lambda: f"CYCLE-{uuid.uuid4().hex[:8]}")
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    mode: HyperspeedMode = HyperspeedMode.FULL_HYPERSPEED
    
    replay_sessions: List[str] = field(default_factory=list)
    battle_simulations_count: int = 0
    aggregated_samples_count: int = 0
    
    total_bars_processed: int = 0
    total_predictions_made: int = 0
    total_outcomes_captured: int = 0
    
    training_samples_generated: int = 0
    training_triggered: bool = False
    model_updated: bool = False
    
    accuracy_before: Dict[str, float] = field(default_factory=dict)
    accuracy_after: Dict[str, float] = field(default_factory=dict)
    accuracy_improvement: Dict[str, float] = field(default_factory=dict)
    
    equivalent_real_time_days: float = 0.0
    actual_duration_seconds: float = 0.0
    acceleration_factor: float = 0.0
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        duration = self.actual_duration_seconds
        if self.completed_at and self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "mode": self.mode.value,
            "replay_sessions_count": len(self.replay_sessions),
            "battle_simulations_count": self.battle_simulations_count,
            "aggregated_samples_count": self.aggregated_samples_count,
            "total_bars_processed": self.total_bars_processed,
            "total_predictions_made": self.total_predictions_made,
            "total_outcomes_captured": self.total_outcomes_captured,
            "training_samples_generated": self.training_samples_generated,
            "training_triggered": self.training_triggered,
            "model_updated": self.model_updated,
            "accuracy_improvement": self.accuracy_improvement,
            "equivalent_real_time_days": self.equivalent_real_time_days,
            "actual_duration_seconds": duration,
            "acceleration_factor": self.acceleration_factor,
            "error_count": len(self.errors),
        }


@dataclass
class HyperspeedMetrics:
    """Aggregate metrics for Hyperspeed Learning System."""
    
    total_cycles_completed: int = 0
    total_samples_generated: int = 0
    total_bars_replayed: int = 0
    total_simulations_run: int = 0
    
    total_training_runs: int = 0
    total_model_updates: int = 0
    
    cumulative_real_time_equivalent_days: float = 0.0
    cumulative_actual_runtime_hours: float = 0.0
    
    average_acceleration_factor: float = 0.0
    peak_acceleration_factor: float = 0.0
    
    accuracy_trend: List[Dict[str, float]] = field(default_factory=list)
    
    last_cycle_at: Optional[datetime] = None
    system_active: bool = False
    overnight_mode_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cycles_completed": self.total_cycles_completed,
            "total_samples_generated": self.total_samples_generated,
            "total_bars_replayed": self.total_bars_replayed,
            "total_simulations_run": self.total_simulations_run,
            "total_training_runs": self.total_training_runs,
            "total_model_updates": self.total_model_updates,
            "cumulative_real_time_equivalent_days": self.cumulative_real_time_equivalent_days,
            "cumulative_actual_runtime_hours": self.cumulative_actual_runtime_hours,
            "average_acceleration_factor": self.average_acceleration_factor,
            "peak_acceleration_factor": self.peak_acceleration_factor,
            "accuracy_trend_length": len(self.accuracy_trend),
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "system_active": self.system_active,
            "overnight_mode_active": self.overnight_mode_active,
        }
