"""
QuantraCore Apex MonsterRunner Protocols (MR01-MR05)

MonsterRunner detects rare, explosive move precursors using
deterministic structural analysis. Stage 1 implementation.

Protocol Registry:
- MR01: Compression Explosion Detector
- MR02: Volume Anomaly Detector  
- MR03: Volatility Regime Shift Detector
- MR04: Institutional Footprint Detector
- MR05: Multi-Timeframe Alignment Detector

All protocols are deterministic and research-only.
"""

from .MR01 import run_MR01
from .MR02 import run_MR02
from .MR03 import run_MR03
from .MR04 import run_MR04
from .MR05 import run_MR05
from .monster_runner_loader import MonsterRunnerLoader, MonsterRunnerResult

__all__ = [
    "run_MR01",
    "run_MR02", 
    "run_MR03",
    "run_MR04",
    "run_MR05",
    "MonsterRunnerLoader",
    "MonsterRunnerResult",
]
