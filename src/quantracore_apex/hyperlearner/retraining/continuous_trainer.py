"""
HyperLearner Continuous Trainer.

Continuously retrains models with prioritized learning samples.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict
import threading
import queue
import logging
import hashlib
import json
import os

from ..models import (
    EventOutcomePair,
    LearningBatch,
    LearningPriority,
    LearningMetrics,
    Pattern,
)
from ..patterns.pattern_miner import PatternMiner


logger = logging.getLogger(__name__)


class ContinuousTrainer:
    """
    Hyper-velocity continuous training system.
    
    Features:
    - Priority queue for high-value learning samples
    - Batched training for efficiency
    - Model checkpointing
    - Performance tracking
    - Automatic retraining triggers
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        min_batch_interval_seconds: int = 60,
        auto_retrain_threshold: int = 500,
        model_save_dir: str = "models/hyperlearner",
    ):
        self._batch_size = batch_size
        self._min_batch_interval = timedelta(seconds=min_batch_interval_seconds)
        self._auto_retrain_threshold = auto_retrain_threshold
        self._model_save_dir = model_save_dir
        
        self._priority_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._training_samples: List[EventOutcomePair] = []
        self._sample_buffer: List[EventOutcomePair] = []
        
        self._batches_created: List[LearningBatch] = []
        self._last_batch_time: Optional[datetime] = None
        self._last_retrain_time: Optional[datetime] = None
        
        self._metrics = LearningMetrics()
        self._pattern_miner = PatternMiner()
        
        self._training_callbacks: List[Callable[[LearningBatch], None]] = []
        self._lock = threading.Lock()
        
        self._model_versions: List[Dict[str, Any]] = []
        
        os.makedirs(model_save_dir, exist_ok=True)
        
    def add_sample(self, pair: EventOutcomePair):
        """Add a learning sample to the training queue."""
        priority = -pair.learning_priority.value
        
        self._priority_queue.put((priority, datetime.utcnow(), pair))
        
        with self._lock:
            self._sample_buffer.append(pair)
            self._metrics.total_events_captured += 1
            
        self._pattern_miner.ingest_pair(pair)
        
        if len(self._sample_buffer) >= self._auto_retrain_threshold:
            self._trigger_batch_creation()
            
    def add_samples(self, pairs: List[EventOutcomePair]):
        """Add multiple samples."""
        for pair in pairs:
            self.add_sample(pair)
            
    def register_training_callback(self, callback: Callable[[LearningBatch], None]):
        """Register callback for when a training batch is ready."""
        self._training_callbacks.append(callback)
        
    def _trigger_batch_creation(self):
        """Create a new training batch."""
        now = datetime.utcnow()
        
        if self._last_batch_time and (now - self._last_batch_time) < self._min_batch_interval:
            return
            
        batch = self._create_batch()
        
        if batch and batch.ready_for_training:
            for callback in self._training_callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"[ContinuousTrainer] Training callback error: {e}")
                    
    def _create_batch(self) -> Optional[LearningBatch]:
        """Create a prioritized training batch."""
        samples = []
        
        while not self._priority_queue.empty() and len(samples) < self._batch_size:
            try:
                priority, timestamp, pair = self._priority_queue.get_nowait()
                samples.append(pair)
            except queue.Empty:
                break
                
        if len(samples) < 10:
            for pair in samples:
                self._priority_queue.put((-pair.learning_priority.value, datetime.utcnow(), pair))
            return None
            
        priority_breakdown = defaultdict(int)
        categories = set()
        
        for pair in samples:
            priority_breakdown[pair.learning_priority.name] += 1
            categories.add(pair.event.category.value)
            
        batch_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}-{len(samples)}".encode()
        ).hexdigest()[:12]
        
        batch = LearningBatch(
            batch_id=batch_id,
            created_at=datetime.utcnow(),
            samples=samples,
            total_samples=len(samples),
            priority_breakdown=dict(priority_breakdown),
            categories_included=list(categories),
            ready_for_training=True,
        )
        
        with self._lock:
            self._batches_created.append(batch)
            self._training_samples.extend(samples)
            self._sample_buffer = [s for s in self._sample_buffer if s not in samples]
            self._last_batch_time = datetime.utcnow()
            self._metrics.total_retraining_cycles += 1
            self._metrics.last_retrain_at = datetime.utcnow()
            
        logger.info(f"[ContinuousTrainer] Created batch {batch_id} with {len(samples)} samples")
        
        return batch
        
    def train_models(self, batch: LearningBatch) -> Dict[str, Any]:
        """
        Train models with the batch.
        
        This creates training data compatible with ApexLab.
        """
        training_data = []
        
        for pair in batch.samples:
            sample = pair.to_training_sample()
            sample["event_id"] = pair.event.event_id
            sample["category"] = pair.event.category.value
            sample["event_type"] = pair.event.event_type.value
            sample["timestamp"] = pair.event.timestamp.isoformat()
            training_data.append(sample)
            
        training_file = os.path.join(
            self._model_save_dir,
            f"training_batch_{batch.batch_id}.json"
        )
        
        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=2)
            
        wins = sum(1 for s in training_data if s.get("is_win", 0) == 1.0)
        losses = sum(1 for s in training_data if s.get("is_loss", 0) == 1.0)
        
        avg_return = sum(s.get("return_pct", 0) for s in training_data) / len(training_data)
        
        with self._lock:
            self._metrics.win_rate_trend.append(wins / max(wins + losses, 1))
            if len(self._metrics.win_rate_trend) > 100:
                self._metrics.win_rate_trend = self._metrics.win_rate_trend[-100:]
                
        version_info = {
            "batch_id": batch.batch_id,
            "trained_at": datetime.utcnow().isoformat(),
            "samples": batch.total_samples,
            "win_rate": wins / max(wins + losses, 1),
            "avg_return": avg_return,
            "training_file": training_file,
        }
        
        with self._lock:
            self._model_versions.append(version_info)
            
        logger.info(f"[ContinuousTrainer] Trained on batch {batch.batch_id}: {wins}W/{losses}L, avg_return={avg_return:.2f}%")
        
        return version_info
        
    def get_training_data_for_apexlab(self) -> List[Dict[str, Any]]:
        """Export all training data in ApexLab format."""
        samples = []
        
        with self._lock:
            for pair in self._training_samples:
                sample = pair.to_training_sample()
                
                if pair.event.symbol:
                    sample["symbol"] = pair.event.symbol
                if "regime" in pair.event.context:
                    sample["regime"] = pair.event.context["regime"]
                    
                samples.append(sample)
                
        return samples
        
    def get_patterns(self) -> Dict[str, List[Pattern]]:
        """Get discovered patterns."""
        return {
            "win_patterns": self._pattern_miner.get_win_patterns(),
            "loss_patterns": self._pattern_miner.get_loss_patterns(),
            "all_patterns": self._pattern_miner.get_all_patterns(),
        }
        
    def get_metrics(self) -> LearningMetrics:
        """Get training metrics."""
        with self._lock:
            pattern_stats = self._pattern_miner.get_stats()
            self._metrics.total_patterns_discovered = pattern_stats["total_patterns"]
            
            if len(self._batches_created) >= 2:
                recent = self._batches_created[-5:]
                returns = []
                for batch in recent:
                    for pair in batch.samples:
                        if pair.outcome.return_pct is not None:
                            returns.append(pair.outcome.return_pct)
                if returns:
                    self._metrics.model_improvement_rate = sum(returns) / len(returns)
                    
            return self._metrics
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        with self._lock:
            queue_size = self._priority_queue.qsize()
            buffer_size = len(self._sample_buffer)
            total_samples = len(self._training_samples)
            batches = len(self._batches_created)
            
        pattern_stats = self._pattern_miner.get_stats()
        
        return {
            "queue_size": queue_size,
            "buffer_size": buffer_size,
            "total_training_samples": total_samples,
            "batches_created": batches,
            "model_versions": len(self._model_versions),
            "last_batch": self._last_batch_time.isoformat() if self._last_batch_time else None,
            "patterns": pattern_stats,
            "metrics": self._metrics.to_dict(),
        }
        
    def force_batch(self) -> Optional[LearningBatch]:
        """Force creation of a batch regardless of thresholds."""
        with self._lock:
            for pair in self._sample_buffer:
                self._priority_queue.put((-pair.learning_priority.value, datetime.utcnow(), pair))
                
        batch = self._create_batch()
        if batch:
            self.train_models(batch)
        return batch
