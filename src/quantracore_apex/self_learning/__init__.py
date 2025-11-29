"""
Self-Learning Feedback Loop System for QuantraCore Apex.

This module implements a complete self-improving ecosystem where every
component feeds back into training quality:

1. MarketSimulator → Chaos samples → ApexLab training
2. Alpha Factory → Trade outcomes → Feedback loop → ApexLab training
3. Historical Backtest → Labeled samples → ApexLab training
4. ApexLab → Retrain ApexCore → Better predictions → Better signals
5. Validation → Quality metrics → Training weight adjustments

The loop never stops improving.
"""

from .unified_loop import UnifiedLearningLoop
from .chaos_generator import ChaosTrainingGenerator
from .backtest_generator import BacktestTrainingGenerator
from .quality_scorer import TrainingQualityScorer
from .auto_retrain import AutoRetrainTrigger

__all__ = [
    "UnifiedLearningLoop",
    "ChaosTrainingGenerator",
    "BacktestTrainingGenerator",
    "TrainingQualityScorer",
    "AutoRetrainTrigger",
]
