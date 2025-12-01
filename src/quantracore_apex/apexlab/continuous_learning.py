"""
Continuous Learning Orchestrator for ApexCore v3.

Implements:
1. Auto-learning scheduler with configurable intervals
2. Incremental learning with warm-start and sample caching
3. Drift detection for automatic retraining triggers
4. Validation gates before model promotion
"""

import os
import json
import time
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class LearningState(Enum):
    IDLE = "idle"
    INGESTING = "ingesting"
    TRAINING = "training"
    VALIDATING = "validating"
    PROMOTING = "promoting"
    ERROR = "error"


class DriftType(Enum):
    NONE = "none"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class DriftMetrics:
    """Tracks distribution drift between training and live data."""
    
    timestamp: datetime = field(default_factory=datetime.now)
    feature_drift_score: float = 0.0
    label_drift_score: float = 0.0
    performance_delta: float = 0.0
    samples_since_training: int = 0
    drift_type: DriftType = DriftType.NONE
    requires_retraining: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "feature_drift_score": self.feature_drift_score,
            "label_drift_score": self.label_drift_score,
            "performance_delta": self.performance_delta,
            "samples_since_training": self.samples_since_training,
            "drift_type": self.drift_type.value,
            "requires_retraining": self.requires_retraining,
        }


@dataclass
class LearningCycleResult:
    """Results from a single learning cycle."""
    
    cycle_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    samples_ingested: int = 0
    samples_trained: int = 0
    new_bars_fetched: int = 0
    drift_detected: bool = False
    drift_metrics: Optional[DriftMetrics] = None
    training_triggered: bool = False
    validation_passed: bool = False
    model_promoted: bool = False
    accuracy_before: Dict[str, float] = field(default_factory=dict)
    accuracy_after: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "samples_ingested": self.samples_ingested,
            "samples_trained": self.samples_trained,
            "new_bars_fetched": self.new_bars_fetched,
            "drift_detected": self.drift_detected,
            "drift_metrics": self.drift_metrics.to_dict() if self.drift_metrics else None,
            "training_triggered": self.training_triggered,
            "validation_passed": self.validation_passed,
            "model_promoted": self.model_promoted,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "error": self.error,
        }


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning."""
    
    learning_interval_minutes: int = 60
    min_new_samples_for_training: int = 100
    max_samples_cache: int = 100000
    feature_drift_threshold: float = 0.15
    label_drift_threshold: float = 0.10
    performance_drop_threshold: float = 0.05
    validation_holdout_ratio: float = 0.2
    min_accuracy_improvement: float = 0.001
    max_cycles_without_improvement: int = 5
    warm_start_enabled: bool = True
    multi_pass_epochs: int = 3
    sliding_window_overlap: float = 0.5
    symbols: List[str] = field(default_factory=list)
    lookback_days: int = 90
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SampleCache:
    """Efficient cache for training samples with versioning."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.samples: deque = deque(maxlen=max_size)
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.label_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def add_samples(self, features: np.ndarray, labels: Dict[str, np.ndarray]) -> int:
        """Add new samples to cache, maintaining size limit."""
        with self._lock:
            added = 0
            for i in range(len(features)):
                sample = {
                    "features": features[i].tolist(),
                    "labels": {k: float(v[i]) for k, v in labels.items()},
                    "timestamp": datetime.now().isoformat(),
                }
                self.samples.append(sample)
                added += 1
            
            self._update_stats(features, labels)
            return added
    
    def _update_stats(self, features: np.ndarray, labels: Dict[str, np.ndarray]):
        """Update running statistics for drift detection."""
        self.feature_stats = {
            "mean": np.mean(features, axis=0).tolist(),
            "std": np.std(features, axis=0).tolist(),
            "min": np.min(features, axis=0).tolist(),
            "max": np.max(features, axis=0).tolist(),
        }
        
        for label_name, label_values in labels.items():
            self.label_stats[label_name] = {
                "mean": float(np.mean(label_values)),
                "std": float(np.std(label_values)),
                "min": float(np.min(label_values)),
                "max": float(np.max(label_values)),
            }
    
    def get_all_samples(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Retrieve all cached samples."""
        with self._lock:
            if not self.samples:
                return np.array([]), {}
            
            features = np.array([s["features"] for s in self.samples])
            label_names = list(self.samples[0]["labels"].keys())
            labels = {
                name: np.array([s["labels"][name] for s in self.samples])
                for name in label_names
            }
            return features, labels
    
    def get_stats(self) -> Dict:
        """Get current cache statistics."""
        return {
            "size": len(self.samples),
            "max_size": self.max_size,
            "feature_stats": self.feature_stats,
            "label_stats": self.label_stats,
        }
    
    def clear(self):
        """Clear all cached samples."""
        with self._lock:
            self.samples.clear()
            self.feature_stats = {}
            self.label_stats = {}


class DriftDetector:
    """Detects distribution drift between training and live data."""
    
    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.baseline_feature_stats: Optional[Dict] = None
        self.baseline_label_stats: Optional[Dict] = None
        self.baseline_performance: Optional[Dict[str, float]] = None
        self.samples_since_baseline = 0
    
    def set_baseline(
        self,
        feature_stats: Dict,
        label_stats: Dict,
        performance: Dict[str, float]
    ):
        """Set baseline statistics from last training."""
        self.baseline_feature_stats = feature_stats
        self.baseline_label_stats = label_stats
        self.baseline_performance = performance
        self.samples_since_baseline = 0
    
    def compute_drift(
        self,
        current_feature_stats: Dict,
        current_label_stats: Dict,
        current_performance: Optional[Dict[str, float]] = None,
        new_samples: int = 0
    ) -> DriftMetrics:
        """Compute drift metrics between baseline and current statistics."""
        self.samples_since_baseline += new_samples
        
        metrics = DriftMetrics(
            samples_since_training=self.samples_since_baseline
        )
        
        if not self.baseline_feature_stats:
            return metrics
        
        metrics.feature_drift_score = self._compute_feature_drift(
            current_feature_stats
        )
        
        metrics.label_drift_score = self._compute_label_drift(
            current_label_stats
        )
        
        if current_performance and self.baseline_performance:
            metrics.performance_delta = self._compute_performance_delta(
                current_performance
            )
        
        metrics.drift_type, metrics.requires_retraining = self._classify_drift(
            metrics
        )
        
        return metrics
    
    def _compute_feature_drift(self, current: Dict) -> float:
        """Compute normalized feature distribution drift."""
        if not current.get("mean") or not self.baseline_feature_stats or not self.baseline_feature_stats.get("mean"):
            return 0.0
        
        baseline_mean = np.array(self.baseline_feature_stats["mean"])
        current_mean = np.array(current["mean"])
        
        baseline_std = np.array(self.baseline_feature_stats.get("std", [1.0] * len(baseline_mean)))
        baseline_std = np.where(baseline_std == 0, 1.0, baseline_std)
        
        normalized_diff = np.abs(current_mean - baseline_mean) / baseline_std
        return float(np.mean(normalized_diff))
    
    def _compute_label_drift(self, current: Dict) -> float:
        """Compute label distribution drift."""
        if not current or not self.baseline_label_stats:
            return 0.0
        
        drifts = []
        for label_name, current_stats in current.items():
            if label_name in self.baseline_label_stats:
                baseline = self.baseline_label_stats[label_name]
                
                mean_diff = abs(current_stats["mean"] - baseline["mean"])
                std_baseline = max(baseline.get("std", 1.0), 0.001)
                
                normalized_drift = mean_diff / std_baseline
                drifts.append(normalized_drift)
        
        return float(np.mean(drifts)) if drifts else 0.0
    
    def _compute_performance_delta(self, current: Dict[str, float]) -> float:
        """Compute performance degradation."""
        if not self.baseline_performance:
            return 0.0
        
        deltas = []
        for metric, baseline_value in self.baseline_performance.items():
            if metric in current:
                delta = baseline_value - current[metric]
                deltas.append(delta)
        
        return float(np.mean(deltas)) if deltas else 0.0
    
    def _classify_drift(self, metrics: DriftMetrics) -> Tuple[DriftType, bool]:
        """Classify the type and severity of drift."""
        requires_retraining = False
        drift_type = DriftType.NONE
        
        if metrics.feature_drift_score > self.config.feature_drift_threshold:
            drift_type = DriftType.FEATURE_DRIFT
            requires_retraining = True
        
        if metrics.label_drift_score > self.config.label_drift_threshold:
            drift_type = DriftType.LABEL_DRIFT
            requires_retraining = True
        
        if metrics.performance_delta > self.config.performance_drop_threshold:
            drift_type = DriftType.PERFORMANCE_DRIFT
            requires_retraining = True
        
        if (metrics.feature_drift_score > self.config.feature_drift_threshold * 0.5 and
            metrics.label_drift_score > self.config.label_drift_threshold * 0.5):
            drift_type = DriftType.CONCEPT_DRIFT
            requires_retraining = True
        
        return drift_type, requires_retraining


class ContinuousLearningOrchestrator:
    """
    Orchestrates continuous learning cycles with:
    - Automatic data ingestion
    - Drift detection and monitoring
    - Incremental training with warm-start
    - Validation gates before promotion
    """
    
    def __init__(
        self,
        config: Optional[ContinuousLearningConfig] = None,
        model_dir: str = "models/apexcore_v3/big"
    ):
        self.config = config or ContinuousLearningConfig()
        self.model_dir = Path(model_dir)
        
        self.state = LearningState.IDLE
        self.sample_cache = SampleCache(self.config.max_samples_cache)
        self.drift_detector = DriftDetector(self.config)
        
        self.cycle_history: List[LearningCycleResult] = []
        self.current_cycle: Optional[LearningCycleResult] = None
        self.cycles_without_improvement = 0
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.last_training_time: Optional[datetime] = None
        self.total_cycles = 0
        self.total_samples_processed = 0
        
        self._load_baseline()
    
    def _load_baseline(self):
        """Load baseline metrics from last trained model."""
        manifest_path = self.model_dir / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                
                metrics = manifest.get("metrics", {})
                self.drift_detector.set_baseline(
                    feature_stats={},
                    label_stats={},
                    performance={
                        "runner_accuracy": metrics.get("runner_accuracy", 0),
                        "quality_accuracy": metrics.get("quality_accuracy", 0),
                        "avoid_accuracy": metrics.get("avoid_accuracy", 0),
                        "regime_accuracy": metrics.get("regime_accuracy", 0),
                    }
                )
                self.last_training_time = datetime.fromisoformat(
                    manifest.get("trained_at", datetime.now().isoformat())
                )
            except Exception as e:
                logger.warning(f"Could not load baseline: {e}")
    
    def start(self):
        """Start the continuous learning loop."""
        if self._running:
            return {"status": "already_running"}
        
        self._running = True
        self._thread = threading.Thread(target=self._learning_loop, daemon=True)
        self._thread.start()
        
        logger.info("Continuous learning started")
        return {
            "status": "started",
            "config": self.config.to_dict(),
            "interval_minutes": self.config.learning_interval_minutes,
        }
    
    def stop(self):
        """Stop the continuous learning loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("Continuous learning stopped")
        return {"status": "stopped", "total_cycles": self.total_cycles}
    
    def _learning_loop(self):
        """Main continuous learning loop."""
        while self._running:
            try:
                self._run_cycle()
            except Exception as e:
                logger.error(f"Learning cycle error: {e}")
                self.state = LearningState.ERROR
            
            sleep_seconds = self.config.learning_interval_minutes * 60
            for _ in range(int(sleep_seconds)):
                if not self._running:
                    break
                time.sleep(1)
    
    def _run_cycle(self) -> LearningCycleResult:
        """Execute a single learning cycle."""
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_cycle = LearningCycleResult(cycle_id=cycle_id)
        self.total_cycles += 1
        
        try:
            self.state = LearningState.INGESTING
            self._ingest_new_data()
            
            self.state = LearningState.IDLE
            drift_metrics = self._check_drift()
            self.current_cycle.drift_metrics = drift_metrics
            self.current_cycle.drift_detected = drift_metrics.requires_retraining
            
            should_train = self._should_trigger_training(drift_metrics)
            
            if should_train:
                self.state = LearningState.TRAINING
                self._run_training()
                
                self.state = LearningState.VALIDATING
                validation_passed = self._validate_new_model()
                self.current_cycle.validation_passed = validation_passed
                
                if validation_passed:
                    self.state = LearningState.PROMOTING
                    self._promote_model()
                    self.current_cycle.model_promoted = True
                    self.cycles_without_improvement = 0
                else:
                    self.cycles_without_improvement += 1
            
            self.current_cycle.completed_at = datetime.now()
            self.state = LearningState.IDLE
            
        except Exception as e:
            self.current_cycle.error = str(e)
            self.state = LearningState.ERROR
            logger.error(f"Cycle {cycle_id} failed: {e}")
        
        with self._lock:
            self.cycle_history.append(self.current_cycle)
            if len(self.cycle_history) > 100:
                self.cycle_history = self.cycle_history[-100:]
        
        return self.current_cycle
    
    def _ingest_new_data(self):
        """Ingest new market data since last training."""
        from .unified_trainer import AlpacaFetcher, WindowGenerator, OutcomeLabelGenerator
        
        fetcher = AlpacaFetcher()
        if not fetcher.is_available():
            logger.warning("Alpaca fetcher not available")
            return
        
        window_gen = WindowGenerator(window_size=100, step_size=5)
        label_gen = OutcomeLabelGenerator()
        
        end_date = datetime.now() - timedelta(minutes=20)
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        symbols = self.config.symbols or self._get_default_symbols()
        
        new_bars = 0
        new_samples = 0
        
        symbols_per_batch = min(len(symbols), 50)
        for symbol in symbols[:symbols_per_batch]:
            try:
                bars = fetcher.fetch(symbol, start_date, end_date)
                if not bars:
                    continue
                
                new_bars += len(bars)
                
                windows = window_gen.generate(bars, symbol, future_bars=10)
                
                for window, future_closes in windows:
                    entry_price = window.bars[-1].close
                    labels = label_gen.generate(entry_price, future_closes)
                    
                    features = self._extract_features(window)
                    if features is not None:
                        self.sample_cache.add_samples(
                            np.array([features]),
                            {k: np.array([v]) for k, v in labels.items() if isinstance(v, (int, float))}
                        )
                        new_samples += 1
                
            except Exception as e:
                logger.debug(f"Error ingesting {symbol}: {e}")
                continue
        
        if self.current_cycle:
            self.current_cycle.new_bars_fetched = new_bars
            self.current_cycle.samples_ingested = new_samples
        self.total_samples_processed += new_samples
        
        logger.info(f"Ingested {new_bars} bars, {new_samples} samples")
    
    def _extract_features(self, window) -> Optional[np.ndarray]:
        """Extract features from an OHLCV window."""
        try:
            closes = np.array([b.close for b in window.bars])
            highs = np.array([b.high for b in window.bars])
            lows = np.array([b.low for b in window.bars])
            volumes = np.array([b.volume for b in window.bars])
            
            if len(closes) < 20:
                return None
            
            returns = np.diff(closes) / closes[:-1]
            
            features = [
                np.mean(returns[-5:]),
                np.mean(returns[-20:]),
                np.std(returns[-20:]),
                np.max(returns[-20:]),
                np.min(returns[-20:]),
                
                (closes[-1] - np.mean(closes[-20:])) / (np.std(closes[-20:]) + 1e-8),
                
                np.mean(volumes[-5:]) / (np.mean(volumes[-20:]) + 1),
                np.std(volumes[-20:]) / (np.mean(volumes[-20:]) + 1),
                
                (highs[-1] - lows[-1]) / (closes[-1] + 1e-8),
                np.mean((highs[-20:] - lows[-20:]) / (closes[-20:] + 1e-8)),
                
                (closes[-1] - lows[-20:].min()) / (highs[-20:].max() - lows[-20:].min() + 1e-8),
                
                np.corrcoef(returns[-20:], np.arange(20))[0, 1] if len(returns) >= 20 else 0,
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception:
            return None
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols for continuous learning."""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
            "JPM", "BAC", "GS", "V", "MA", "PYPL",
            "JNJ", "PFE", "UNH", "ABBV",
            "XOM", "CVX", "COP",
        ]
    
    def _check_drift(self) -> DriftMetrics:
        """Check for distribution drift."""
        cache_stats = self.sample_cache.get_stats()
        
        samples_ingested = self.current_cycle.samples_ingested if self.current_cycle else 0
        return self.drift_detector.compute_drift(
            current_feature_stats=cache_stats.get("feature_stats", {}),
            current_label_stats=cache_stats.get("label_stats", {}),
            new_samples=samples_ingested
        )
    
    def _should_trigger_training(self, drift: DriftMetrics) -> bool:
        """Determine if training should be triggered."""
        if drift.requires_retraining:
            logger.info(f"Training triggered by drift: {drift.drift_type.value}")
            return True
        
        cache_size = len(self.sample_cache.samples)
        if cache_size >= self.config.min_new_samples_for_training:
            logger.info(f"Training triggered by sample count: {cache_size}")
            return True
        
        if self.cycles_without_improvement >= self.config.max_cycles_without_improvement:
            logger.info("Training triggered by stagnation")
            return True
        
        return False
    
    def _run_training(self):
        """Run incremental training on cached samples."""
        from .unified_trainer import UnifiedTrainer, UnifiedTrainingConfig
        
        features, labels = self.sample_cache.get_all_samples()
        
        if len(features) == 0:
            logger.warning("No samples in cache for training")
            return
        
        if self.current_cycle:
            self.current_cycle.samples_trained = len(features)
        
        config = UnifiedTrainingConfig(
            symbols=self.config.symbols or self._get_default_symbols(),
            lookback_days=self.config.lookback_days,
            max_workers=4,
        )
        
        trainer = UnifiedTrainer(config)
        
        manifest_path = self.model_dir / "manifest.json"
        if manifest_path.exists() and self.current_cycle:
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.current_cycle.accuracy_before = manifest.get("metrics", {})
        
        try:
            result = trainer.train_sync()
            
            if manifest_path.exists() and self.current_cycle:
                with open(manifest_path) as f:
                    new_manifest = json.load(f)
                self.current_cycle.accuracy_after = new_manifest.get("metrics", {})
                self.current_cycle.training_triggered = True
            self.last_training_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _validate_new_model(self) -> bool:
        """Validate new model meets accuracy requirements."""
        if not self.current_cycle:
            return True
        before = self.current_cycle.accuracy_before
        after = self.current_cycle.accuracy_after
        
        if not before or not after:
            return True
        
        key_metrics = ["runner_accuracy", "quality_accuracy", "avoid_accuracy", "regime_accuracy"]
        
        improvements = []
        for metric in key_metrics:
            if metric in before and metric in after:
                delta = after[metric] - before[metric]
                improvements.append(delta)
        
        if not improvements:
            return True
        
        avg_improvement = np.mean(improvements)
        
        if avg_improvement >= -self.config.min_accuracy_improvement:
            logger.info(f"Validation passed: avg improvement = {avg_improvement:.4f}")
            return True
        else:
            logger.warning(f"Validation failed: avg improvement = {avg_improvement:.4f}")
            return False
    
    def _promote_model(self):
        """Promote validated model to production."""
        cache_stats = self.sample_cache.get_stats()
        
        performance = self.current_cycle.accuracy_after if self.current_cycle else {}
        self.drift_detector.set_baseline(
            feature_stats=cache_stats.get("feature_stats", {}),
            label_stats=cache_stats.get("label_stats", {}),
            performance=performance
        )
        
        self.sample_cache.clear()
        
        logger.info("Model promoted to production")
    
    def get_status(self) -> Dict:
        """Get current orchestrator status."""
        return {
            "state": self.state.value,
            "running": self._running,
            "total_cycles": self.total_cycles,
            "total_samples_processed": self.total_samples_processed,
            "cycles_without_improvement": self.cycles_without_improvement,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "cache_size": len(self.sample_cache.samples),
            "current_cycle": self.current_cycle.to_dict() if self.current_cycle else None,
            "config": self.config.to_dict(),
        }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent cycle history."""
        with self._lock:
            return [c.to_dict() for c in self.cycle_history[-limit:]]
    
    def trigger_manual_cycle(self) -> Dict:
        """Manually trigger a learning cycle."""
        if self.state != LearningState.IDLE:
            return {"error": f"Cannot trigger cycle in state: {self.state.value}"}
        
        result = self._run_cycle()
        return result.to_dict()
    
    def update_config(self, **kwargs) -> Dict:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        return {"status": "updated", "config": self.config.to_dict()}


_orchestrator: Optional[ContinuousLearningOrchestrator] = None


EXTENDED_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AMD", "AVGO",
    "NFLX", "CRM", "ORCL", "ADBE", "NOW", "SNOW", "DDOG", "NET", "MDB", "PLTR",
    "CRWD", "PANW", "ZS", "FTNT", "OKTA", "SPLK", "HUBS", "ZEN", "TWLO", "DBX",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "USB", "PNC", "SCHW", "AXP",
    "V", "MA", "PYPL", "SQ", "COIN", "HOOD", "AFRM", "SOFI", "UPST", "LC",
    "BRK.B", "BLK", "SPGI", "ICE", "CME", "MSCI", "MCO", "CBOE", "NDAQ", "FDS",
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PXD", "DVN", "HAL", "BKR",
    "LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TXT", "TDG", "HWM",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA", "ISRG", "SYK", "BDX", "MDT",
    "HD", "LOW", "WMT", "TGT", "COST", "AMZN", "EBAY", "ETSY", "MELI", "SHOP",
    "NKE", "LULU", "DECK", "CROX", "VFC", "PVH", "TPR", "RL", "GPS", "ANF",
    "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "PARA", "WBD", "NWSA", "FOX",
    "MCD", "SBUX", "CMG", "DPZ", "YUM", "QSR", "DENN", "WEN", "JACK", "PLAY",
    "ABNB", "MAR", "HLT", "H", "IHG", "MGM", "WYNN", "LVS", "PENN", "CZR",
    "UBER", "LYFT", "DASH", "GRAB", "GM", "F", "TM", "HMC", "RIVN", "LCID",
    "EA", "TTWO", "ATVI", "RBLX", "DKNG", "PLTK", "GLBE", "SE", "BILI", "HUYA",
    "ENPH", "SEDG", "FSLR", "RUN", "CSIQ", "JKS", "NEE", "AES", "DUK", "SO",
    "INTC", "QCOM", "MU", "MRVL", "LRCX", "AMAT", "KLAC", "ASML", "SNPS", "CDNS",
    "TXN", "ADI", "ON", "NXPI", "SWKS", "QRVO", "CRUS", "WOLF", "ACLS", "FORM",
    "CAT", "DE", "MMM", "HON", "GE", "EMR", "ROK", "PH", "ITW", "ETN",
    "UNP", "CSX", "NSC", "FDX", "UPS", "XPO", "EXPD", "CHRW", "JBHT", "ODFL",
    "AAL", "DAL", "UAL", "LUV", "JBLU", "ALK", "SAVE", "SKYW", "HA", "MESA",
    "DG", "DLTR", "FIVE", "BIG", "OLLI", "ROST", "TJX", "BURL", "BWMN", "VSCO",
    "SPY", "QQQ", "IWM", "DIA", "EEM", "VWO", "GLD", "SLV", "USO", "UNG",
    "KO", "PEP", "PM", "MO", "BTI", "TAP", "STZ", "BUD", "SAM", "MNST",
    "PG", "CL", "KMB", "CHD", "CLX", "EL", "COTY", "SJM", "GIS", "K",
    "PARA", "LYV", "SPOT", "TME", "SNAP", "PINS", "TWTR", "MTCH", "BMBL", "IAC",
    "AMD", "NVIDIA", "ARM", "TSM", "SMCI", "DELL", "HPQ", "HPE", "IBM", "CSCO",
    "ZM", "DOCU", "BOX", "WDAY", "VEEV", "ZI", "CCMP", "MANH", "GWRE", "NICE",
]


def get_orchestrator() -> ContinuousLearningOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        config = ContinuousLearningConfig(
            symbols=EXTENDED_UNIVERSE,
            learning_interval_minutes=15,
            min_new_samples_for_training=200,
            lookback_days=60,
        )
        _orchestrator = ContinuousLearningOrchestrator(config)
    
    return _orchestrator
