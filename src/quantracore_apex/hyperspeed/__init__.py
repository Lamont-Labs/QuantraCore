"""
Hyperspeed Learning System.

Accelerated ML training through:
- Historical data replay at 1000x speed
- Parallel battle simulations
- Multi-source data fusion
- Overnight intensive training cycles
"""

from .engine import HyperspeedEngine
from .replay import HistoricalReplayEngine
from .battle_cluster import ParallelBattleCluster
from .aggregator import MultiSourceAggregator
from .scheduler import OvernightScheduler
from .models import (
    HyperspeedConfig,
    ReplaySession,
    BattleSimulation,
    AggregatedSample,
    TrainingCycle,
    HyperspeedMetrics,
)

__all__ = [
    "HyperspeedEngine",
    "HistoricalReplayEngine",
    "ParallelBattleCluster",
    "MultiSourceAggregator",
    "OvernightScheduler",
    "HyperspeedConfig",
    "ReplaySession",
    "BattleSimulation",
    "AggregatedSample",
    "TrainingCycle",
    "HyperspeedMetrics",
]
