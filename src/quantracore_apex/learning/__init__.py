"""
AutoLearner - Automated Learning System for QuantraCore Apex

This module provides continuous self-improvement through:
1. Trade outcome tracking
2. Historical simulation backtesting
3. Automatic model retraining with feedback loops
4. Database persistence for learning history
5. Overnight training scheduler
"""

from .auto_learner import AutoLearner
from .simulation_engine import SimulationEngine
from .trade_tracker import TradeTracker
from .model_trainer import ModelTrainer

__all__ = ['AutoLearner', 'SimulationEngine', 'TradeTracker', 'ModelTrainer']
