"""
QuantraCore Apex MonsterRunner Protocols (MR01-MR20)

MonsterRunner detects rare, explosive move precursors using
deterministic structural analysis. Stage 2 implementation.

Protocol Registry:
- MR01: Compression Explosion Detector
- MR02: Volume Anomaly Detector  
- MR03: Volatility Regime Shift Detector
- MR04: Institutional Footprint Detector
- MR05: Multi-Timeframe Alignment Detector
- MR06: Bollinger Breakout Detector
- MR07: Volume Explosion Detector
- MR08: Earnings Gap Runner Detector
- MR09: VWAP Breakout Detector
- MR10: NR7 Breakout Detector
- MR11: Short Squeeze Gamma Detector
- MR12: Crypto Pump Detector
- MR13: News Catalyst Detector
- MR14: Fractal Explosion Detector
- MR15: 100% Day Detector
- MR16: Parabolic Phase 3 Detector
- MR17: Meme Stock Frenzy Detector
- MR18: Options Gamma Ramp Detector
- MR19: FOMO Cascade Detector
- MR20: Nuclear Runner Detector

All protocols are deterministic and research-only.
"""

from .MR01 import run_MR01
from .MR02 import run_MR02
from .MR03 import run_MR03
from .MR04 import run_MR04
from .MR05 import run_MR05
from .MR06 import run_MR06
from .MR07 import run_MR07
from .MR08 import run_MR08
from .MR09 import run_MR09
from .MR10 import run_MR10
from .MR11 import run_MR11
from .MR12 import run_MR12
from .MR13 import run_MR13
from .MR14 import run_MR14
from .MR15 import run_MR15
from .MR16 import run_MR16
from .MR17 import run_MR17
from .MR18 import run_MR18
from .MR19 import run_MR19
from .MR20 import run_MR20
from .monster_runner_loader import MonsterRunnerLoader, MonsterRunnerResult

__all__ = [
    "run_MR01", "run_MR02", "run_MR03", "run_MR04", "run_MR05",
    "run_MR06", "run_MR07", "run_MR08", "run_MR09", "run_MR10",
    "run_MR11", "run_MR12", "run_MR13", "run_MR14", "run_MR15",
    "run_MR16", "run_MR17", "run_MR18", "run_MR19", "run_MR20",
    "MonsterRunnerLoader",
    "MonsterRunnerResult",
]
