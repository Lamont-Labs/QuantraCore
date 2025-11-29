"""
Estimated Move Module for QuantraCore Apex.

Provides deterministic + model-assisted "Estimated Move Range" for structural research.
NOT a prediction. NOT a financial guarantee. STRICTLY a statistical expansion window.

This module is RESEARCH-ONLY and never provides price targets or trading signals.
"""

from .engine import EstimatedMoveEngine
from .schemas import (
    EstimatedMoveInput,
    EstimatedMoveOutput,
    HorizonWindow,
    MoveRange,
)

__all__ = [
    "EstimatedMoveEngine",
    "EstimatedMoveInput",
    "EstimatedMoveOutput",
    "HorizonWindow",
    "MoveRange",
]
