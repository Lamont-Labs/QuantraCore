"""
Alpha Factory - 24/7 Live Trading Research Engine.

Combines live data feeds, ApexEngine analysis, and portfolio management
for continuous alpha generation research. Paper trading only.
"""

from .loop import AlphaFactoryLoop
from .dashboard import EquityCurvePlotter

__all__ = ["AlphaFactoryLoop", "EquityCurvePlotter"]
