"""
QuantraCore Apex v8.0

Institutional-Grade Deterministic AI Trading Intelligence Engine

Owner: Lamont Labs — Jesse J. Lamont
Status: Active — Core Engine

IMPORTANT: All outputs are framed as structural probabilities.
This is NOT trading advice.
"""

__version__ = "8.0.0"
__author__ = "Lamont Labs"
__status__ = "Active"

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import (
    OhlcvBar,
    OhlcvWindow,
    ApexContext,
    ApexResult,
)

__all__ = [
    "ApexEngine",
    "OhlcvBar",
    "OhlcvWindow",
    "ApexContext",
    "ApexResult",
    "__version__",
]
