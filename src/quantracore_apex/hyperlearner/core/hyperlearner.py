"""
HyperLearner - Hyper-Velocity Learning System.

Central orchestrator that captures everything the system does
and learns from it at an accelerated rate.

Enhanced with EnrichedDataFusion to incorporate all 7 data sources:
- Polygon.io: Market data
- Alpaca: Execution data  
- FRED: Economic indicators
- Finnhub: Social sentiment
- Alpha Vantage: News sentiment
- SEC EDGAR: Insider transactions
- Binance: Crypto correlations
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import threading
import logging
import os

from ..models import (
    LearningEvent,
    Outcome,
    EventOutcomePair,
    Pattern,
    LearningBatch,
    LearningMetrics,
    MetaLearningInsight,
    EventCategory,
    EventType,
    OutcomeType,
    LearningPriority,
)
from .event_bus import EventBus, get_event_bus, emit_event
from ..capture.outcome_tracker import OutcomeTracker
from ..patterns.pattern_miner import PatternMiner
from ..retraining.continuous_trainer import ContinuousTrainer
from ..meta.meta_learner import MetaLearner


logger = logging.getLogger(__name__)


class HyperLearner:
    """
    Hyper-Velocity Learning System.
    
    Captures EVERYTHING the system does and learns from it:
    - Wins → Reinforce patterns
    - Losses → Identify weaknesses
    - Passes → Validate quality control
    - Fails → Fix broken processes
    - Predictions → Track accuracy
    - Omega triggers → Evaluate safety
    - Battle results → Learn from competition
    
    Learning happens at accelerated velocity through:
    - Priority queuing (learn from high-value events first)
    - Pattern mining (discover what works/fails)
    - Continuous retraining (always improving)
    - Meta-learning (optimize the learning itself)
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        auto_retrain_threshold: int = 500,
        model_dir: str = "models/hyperlearner",
        enable_enrichment: bool = True,
    ):
        self._event_bus = get_event_bus()
        self._outcome_tracker = OutcomeTracker(event_bus=self._event_bus)
        self._pattern_miner = PatternMiner()
        self._continuous_trainer = ContinuousTrainer(
            batch_size=batch_size,
            auto_retrain_threshold=auto_retrain_threshold,
            model_save_dir=model_dir,
        )
        self._meta_learner = MetaLearner()
        
        self._started_at: Optional[datetime] = None
        self._running = False
        self._cleanup_thread: Optional[threading.Thread] = None
        
        self._total_events = 0
        self._total_learnings = 0
        
        self._data_fusion = None
        self._enable_enrichment = enable_enrichment
        if enable_enrichment:
            try:
                from src.quantracore_apex.data_layer.enriched_data_fusion import get_enriched_data_fusion
                self._data_fusion = get_enriched_data_fusion()
                logger.info("[HyperLearner] EnrichedDataFusion enabled - all 7 data sources active")
            except Exception as e:
                logger.warning(f"[HyperLearner] EnrichedDataFusion disabled: {e}")
        
        self._setup_internal_subscriptions()
        
        os.makedirs(model_dir, exist_ok=True)
        
    def _setup_internal_subscriptions(self):
        """Setup internal event processing."""
        self._event_bus.subscribe(self._on_any_event)
        
        self._continuous_trainer.register_training_callback(self._on_batch_trained)
        
    def start(self):
        """Start the hyper-learning system."""
        if self._running:
            return
            
        self._running = True
        self._started_at = datetime.utcnow()
        
        self._event_bus.start()
        
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("[HyperLearner] Started - capturing ALL events for accelerated learning")
        
    def stop(self):
        """Stop the hyper-learning system."""
        self._running = False
        self._event_bus.stop()
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            
        logger.info("[HyperLearner] Stopped")
        
    def _on_any_event(self, event: LearningEvent):
        """Process any incoming event."""
        self._total_events += 1
        
        matched_patterns = self._pattern_miner.match_pattern(
            EventOutcomePair(
                event=event,
                outcome=Outcome(
                    outcome_id="pending",
                    event_id=event.event_id,
                    timestamp=datetime.utcnow(),
                    outcome_type=OutcomeType.PENDING,
                ),
                learning_priority=LearningPriority.MEDIUM,
            )
        )
        
        if matched_patterns:
            win_patterns = [p for p in matched_patterns if p.pattern_type == "WIN"]
            loss_patterns = [p for p in matched_patterns if p.pattern_type == "LOSS"]
            
            if loss_patterns and not win_patterns:
                logger.warning(f"[HyperLearner] Loss pattern detected for {event.event_type.value}")
                
        self._meta_learner.record_category_outcome(
            event.category,
            was_valuable=event.confidence > 0.7,
            learning_contribution=event.confidence,
        )
        
    def _on_batch_trained(self, batch: LearningBatch):
        """Handle completed training batch."""
        result = self._continuous_trainer.train_models(batch)
        
        self._total_learnings += batch.total_samples
        
        for pair in batch.samples:
            self._pattern_miner.ingest_pair(pair)
            
        metrics = self._continuous_trainer.get_metrics()
        self._meta_learner.record_metrics(metrics)
        
        logger.info(f"[HyperLearner] Batch trained: {batch.total_samples} samples, total learnings: {self._total_learnings}")
        
    def _periodic_cleanup(self):
        """Periodic maintenance tasks."""
        import time
        
        while self._running:
            time.sleep(300)
            
            self._outcome_tracker.cleanup_expired()
            
            pairs = self._outcome_tracker.get_learning_pairs(limit=1000)
            for pair in pairs:
                self._continuous_trainer.add_sample(pair)
                
            insights = self._meta_learner.generate_optimization_insights()
            if insights:
                logger.info(f"[HyperLearner] Generated {len(insights)} optimization insights")
                
    def emit(
        self,
        category: EventCategory,
        event_type: EventType,
        source: str,
        context: Dict[str, Any],
        symbol: Optional[str] = None,
        decision: Optional[str] = None,
        confidence: float = 0.0,
        priority: LearningPriority = LearningPriority.MEDIUM,
    ) -> str:
        """
        Emit an event for learning.
        
        This is the main API for components to report what they're doing.
        """
        return emit_event(
            category=category,
            event_type=event_type,
            source=source,
            context=context,
            symbol=symbol,
            decision_made=decision,
            confidence=confidence,
            priority=priority,
        )
        
    def record_outcome(
        self,
        event_id: str,
        outcome_type: OutcomeType,
        return_pct: Optional[float] = None,
        was_correct: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[EventOutcomePair]:
        """Record the outcome of a previous event."""
        return self._outcome_tracker.record_outcome(
            event_id=event_id,
            outcome_type=outcome_type,
            return_pct=return_pct,
            was_correct=was_correct,
            details=details,
        )
        
    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        direction: str,
        quantrascore: float,
        regime: str,
        protocols_fired: List[str],
    ) -> str:
        """Record a complete trade for learning."""
        return_pct = ((exit_price - entry_price) / entry_price) * 100
        if direction.upper() == "SHORT":
            return_pct = -return_pct
            
        outcome_type = OutcomeType.WIN if return_pct > 0 else OutcomeType.LOSS
        
        priority = LearningPriority.HIGH
        if abs(return_pct) > 5:
            priority = LearningPriority.CRITICAL
            
        event_id = self.emit(
            category=EventCategory.EXECUTION,
            event_type=EventType.TRADE_EXITED,
            source="trade_recorder",
            context={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "quantrascore": quantrascore,
                "regime": regime,
                "protocols_fired": protocols_fired,
                "return_pct": return_pct,
            },
            symbol=symbol,
            confidence=quantrascore / 100,
            priority=priority,
        )
        
        self._outcome_tracker.record_outcome(
            event_id=event_id,
            outcome_type=outcome_type,
            return_pct=return_pct,
            was_correct=outcome_type == OutcomeType.WIN,
            details={"immediate_trade": True},
        )
        
        return event_id
        
    def record_signal_outcome(
        self,
        symbol: str,
        quantrascore: float,
        was_taken: bool,
        outcome: str,
        return_pct: Optional[float] = None,
    ) -> str:
        """Record signal generation and outcome."""
        event_type = EventType.SIGNAL_PASSED if was_taken else EventType.SIGNAL_REJECTED
        
        if was_taken:
            if return_pct and return_pct > 0:
                outcome_type = OutcomeType.WIN
            elif return_pct and return_pct < 0:
                outcome_type = OutcomeType.LOSS
            else:
                outcome_type = OutcomeType.NEUTRAL
        else:
            if outcome == "correct_rejection":
                outcome_type = OutcomeType.PASS
            else:
                outcome_type = OutcomeType.FAIL
                
        event_id = self.emit(
            category=EventCategory.SIGNAL,
            event_type=event_type,
            source="signal_tracker",
            context={
                "quantrascore": quantrascore,
                "was_taken": was_taken,
                "outcome": outcome,
                "return_pct": return_pct,
            },
            symbol=symbol,
            confidence=quantrascore / 100,
            priority=LearningPriority.HIGH,
        )
        
        self._outcome_tracker.record_outcome(
            event_id=event_id,
            outcome_type=outcome_type,
            return_pct=return_pct,
            was_correct=(outcome_type in {OutcomeType.WIN, OutcomeType.PASS}),
        )
        
        return event_id
        
    def record_omega_trigger(
        self,
        omega_id: str,
        trigger_reason: str,
        symbol: Optional[str] = None,
        was_correct: Optional[bool] = None,
    ) -> str:
        """Record an Omega directive trigger."""
        return self.emit(
            category=EventCategory.OMEGA,
            event_type=EventType.OMEGA_TRIGGERED,
            source="omega_directives",
            context={
                "omega_id": omega_id,
                "reason": trigger_reason,
            },
            symbol=symbol,
            priority=LearningPriority.CRITICAL,
        )
        
    def record_battle_result(
        self,
        symbol: str,
        institution: str,
        outcome: str,
        our_return: float,
        their_return: float,
    ) -> str:
        """Record battle simulator result."""
        if outcome == "WIN":
            event_type = EventType.BATTLE_WON
            outcome_type = OutcomeType.WIN
        elif outcome == "LOSS":
            event_type = EventType.BATTLE_LOST
            outcome_type = OutcomeType.LOSS
        else:
            event_type = EventType.BATTLE_TIE
            outcome_type = OutcomeType.NEUTRAL
            
        event_id = self.emit(
            category=EventCategory.BATTLE,
            event_type=event_type,
            source="battle_simulator",
            context={
                "institution": institution,
                "our_return": our_return,
                "their_return": their_return,
                "alpha": our_return - their_return,
            },
            symbol=symbol,
            priority=LearningPriority.HIGH,
        )
        
        self._outcome_tracker.record_outcome(
            event_id=event_id,
            outcome_type=outcome_type,
            return_pct=our_return,
            was_correct=(outcome_type == OutcomeType.WIN),
        )
        
        return event_id
        
    def record_prediction(
        self,
        prediction_type: str,
        predicted: Any,
        actual: Any,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a prediction and its actual outcome."""
        was_correct = (predicted == actual)
        
        event_type = EventType.PREDICTION_CORRECT if was_correct else EventType.PREDICTION_WRONG
        outcome_type = OutcomeType.WIN if was_correct else OutcomeType.LOSS
        
        event_id = self.emit(
            category=EventCategory.PREDICTION,
            event_type=event_type,
            source="prediction_tracker",
            context={
                "prediction_type": prediction_type,
                "predicted": str(predicted),
                "actual": str(actual),
                **(context or {}),
            },
            symbol=symbol,
            priority=LearningPriority.MEDIUM,
        )
        
        self._outcome_tracker.record_outcome(
            event_id=event_id,
            outcome_type=outcome_type,
            was_correct=was_correct,
        )
        
        return event_id
        
    def force_learning_cycle(self) -> Dict[str, Any]:
        """Force an immediate learning cycle."""
        pairs = self._outcome_tracker.get_learning_pairs(limit=1000)
        
        for pair in pairs:
            self._continuous_trainer.add_sample(pair)
            
        batch = self._continuous_trainer.force_batch()
        
        insights = self._meta_learner.generate_optimization_insights()
        
        return {
            "pairs_processed": len(pairs),
            "batch_created": batch is not None,
            "batch_id": batch.batch_id if batch else None,
            "insights_generated": len(insights),
        }
        
    def get_win_patterns(self, limit: int = 10) -> List[Pattern]:
        """Get top winning patterns."""
        return self._pattern_miner.get_win_patterns(limit=limit)
        
    def get_loss_patterns(self, limit: int = 10) -> List[Pattern]:
        """Get patterns to avoid."""
        return self._pattern_miner.get_loss_patterns(limit=limit)
        
    def get_optimization_insights(self) -> List[MetaLearningInsight]:
        """Get meta-learning optimization insights."""
        return self._meta_learner.get_insights()
        
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """Get optimized learning parameters."""
        return self._meta_learner.get_optimal_parameters()
        
    def get_learning_health(self) -> Dict[str, Any]:
        """Get overall learning system health."""
        return self._meta_learner.get_learning_health()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        event_bus_stats = self._event_bus.get_stats()
        outcome_stats = self._outcome_tracker.get_stats()
        pattern_stats = self._pattern_miner.get_stats()
        trainer_stats = self._continuous_trainer.get_stats()
        health = self._meta_learner.get_learning_health()
        
        uptime = (datetime.utcnow() - self._started_at).total_seconds() if self._started_at else 0
        
        return {
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "uptime_hours": round(uptime / 3600, 2),
            "total_events": self._total_events,
            "total_learnings": self._total_learnings,
            "learning_velocity": self._total_learnings / max(uptime / 3600, 1),
            "event_bus": event_bus_stats,
            "outcomes": outcome_stats,
            "patterns": pattern_stats,
            "training": trainer_stats,
            "health": health,
        }
        
    def export_training_data(self) -> List[Dict[str, Any]]:
        """Export all training data for ApexLab integration."""
        return self._continuous_trainer.get_training_data_for_apexlab()
    
    def get_data_fusion_status(self) -> Dict[str, Any]:
        """
        Get status of all 7 data sources feeding into learning cycles.
        
        Returns:
            Dict with status of each data source and total active count
        """
        if not self._data_fusion:
            return {
                "enabled": False,
                "message": "EnrichedDataFusion not initialized",
                "sources": {}
            }
        
        try:
            status = self._data_fusion.get_status()
            return {
                "enabled": True,
                "total_active": status.get("total_active", 0),
                "learning_ready": status.get("learning_ready", False),
                "sources": {
                    "polygon": status.get("polygon", {}),
                    "alpaca": status.get("alpaca", {}),
                    "fred": status.get("fred", {}),
                    "finnhub": status.get("finnhub", {}),
                    "alpha_vantage": status.get("alpha_vantage", {}),
                    "sec_edgar": status.get("sec_edgar", {}),
                    "binance": status.get("binance", {}),
                }
            }
        except Exception as e:
            return {
                "enabled": False,
                "message": f"Error: {e}",
                "sources": {}
            }
    
    def enrich_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Get enriched features for a symbol from all 7 data sources.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict with enriched features ready for ML
        """
        if not self._data_fusion:
            return {"error": "EnrichedDataFusion not enabled"}
        
        try:
            sample = self._data_fusion.enrich_sample(symbol)
            return {
                "symbol": symbol,
                "timestamp": sample.timestamp.isoformat(),
                "feature_count": sample.feature_count,
                "sources_used": sample.sources_used,
                "sentiment_features": sample.sentiment_features,
                "economic_features": sample.economic_features,
                "insider_features": sample.insider_features,
                "crypto_features": sample.crypto_features,
            }
        except Exception as e:
            return {"error": str(e)}


_global_hyperlearner: Optional[HyperLearner] = None


def get_hyperlearner() -> HyperLearner:
    """Get the global HyperLearner instance."""
    global _global_hyperlearner
    if _global_hyperlearner is None:
        _global_hyperlearner = HyperLearner()
        _global_hyperlearner.start()
    return _global_hyperlearner
