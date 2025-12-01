"""
Incremental Learning Module for ApexCore V3.

Provides efficient continuous learning with:
- LightGBM warm-start (builds on previous model)
- Time-decay sample weighting (recent data weighted higher)
- Dual-buffer system (anchor reservoir + recency buffer)
- Automatic rare pattern preservation

This enables the model to learn efficiently while retaining knowledge.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
import threading
import hashlib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import joblib

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LIGHTGBM_AVAILABLE = False
    lgb = None

logger = logging.getLogger(__name__)

if not LIGHTGBM_AVAILABLE:
    logger.warning("[IncrementalLearning] LightGBM not available, using scikit-learn with warm_start")


@dataclass
class SampleRecord:
    """A training sample with metadata for buffer management."""
    features: List[float]
    labels: Dict[str, float]
    timestamp: str
    symbol: str = ""
    is_rare: bool = False
    sample_hash: str = ""
    
    def __post_init__(self):
        if not self.sample_hash:
            content = f"{self.features}{self.labels}"
            self.sample_hash = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class BufferStats:
    """Statistics for the sample buffers."""
    anchor_size: int
    recency_size: int
    total_samples: int
    rare_pattern_count: int
    oldest_sample: Optional[str]
    newest_sample: Optional[str]
    runner_ratio: float
    crash_ratio: float


class DualBufferSystem:
    """
    Dual-buffer system for optimal knowledge retention.
    
    - Anchor Buffer: Preserves rare patterns (runners, crashes, regime changes)
    - Recency Buffer: Rolling window of recent samples with time decay
    
    This ensures the model never forgets rare but important patterns
    while continuously learning from new market data.
    """
    
    def __init__(
        self,
        anchor_size: int = 20000,
        recency_size: int = 80000,
        runner_threshold: float = 0.05,
        crash_threshold: float = -0.05,
    ):
        self.anchor_size = anchor_size
        self.recency_size = recency_size
        self.runner_threshold = runner_threshold
        self.crash_threshold = crash_threshold
        
        self.anchor_buffer: List[SampleRecord] = []
        self.recency_buffer: deque = deque(maxlen=recency_size)
        
        self._lock = threading.RLock()
        self._rare_hashes: set = set()
    
    def _is_rare_pattern(self, labels: Dict[str, float]) -> bool:
        """Determine if sample represents a rare pattern worth preserving."""
        max_runup = labels.get("max_runup_5d", 0)
        max_drawdown = labels.get("max_drawdown_5d", 0)
        hit_runner = labels.get("hit_runner_threshold", 0)
        
        is_runner = max_runup >= self.runner_threshold or hit_runner > 0
        is_crash = max_drawdown <= self.crash_threshold
        
        return is_runner or is_crash
    
    def add_sample(
        self,
        features: List[float],
        labels: Dict[str, float],
        symbol: str = "",
        timestamp: Optional[str] = None,
    ) -> str:
        """Add a sample to the appropriate buffer."""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        is_rare = self._is_rare_pattern(labels)
        
        sample = SampleRecord(
            features=features,
            labels=labels,
            timestamp=timestamp,
            symbol=symbol,
            is_rare=is_rare,
        )
        
        with self._lock:
            if is_rare and sample.sample_hash not in self._rare_hashes:
                if len(self.anchor_buffer) >= self.anchor_size:
                    removed = self.anchor_buffer.pop(0)
                    self._rare_hashes.discard(removed.sample_hash)
                
                self.anchor_buffer.append(sample)
                self._rare_hashes.add(sample.sample_hash)
                return "anchor"
            else:
                self.recency_buffer.append(sample)
                return "recency"
    
    def add_batch(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        symbols: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Add a batch of samples efficiently."""
        n_samples = len(features)
        
        if symbols is None:
            symbols = [""] * n_samples
        if timestamps is None:
            timestamps = [datetime.now().isoformat()] * n_samples
        
        counts = {"anchor": 0, "recency": 0}
        
        for i in range(n_samples):
            sample_labels = {k: float(v[i]) for k, v in labels.items()}
            buffer = self.add_sample(
                features=features[i].tolist(),
                labels=sample_labels,
                symbol=symbols[i] if i < len(symbols) else "",
                timestamp=timestamps[i] if i < len(timestamps) else None,
            )
            counts[buffer] += 1
        
        return counts
    
    def get_training_data(
        self,
        decay_halflife_days: float = 30.0,
        include_anchor_ratio: float = 0.3,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Get all samples with time-decay weights.
        
        Returns:
            features: Feature matrix
            labels: Dictionary of label arrays
            weights: Sample weights (time-decayed)
        """
        with self._lock:
            all_samples = []
            
            n_anchor = int(len(self.anchor_buffer) * include_anchor_ratio)
            if n_anchor > 0 and len(self.anchor_buffer) > 0:
                import random
                anchor_samples = random.sample(
                    self.anchor_buffer, 
                    min(n_anchor, len(self.anchor_buffer))
                )
                all_samples.extend(anchor_samples)
            
            all_samples.extend(list(self.recency_buffer))
        
        if not all_samples:
            return np.array([]), {}, np.array([])
        
        features = np.array([s.features for s in all_samples])
        
        label_names = list(all_samples[0].labels.keys())
        labels = {
            name: np.array([s.labels.get(name, 0) for s in all_samples])
            for name in label_names
        }
        
        now = datetime.now()
        weights = []
        decay_rate = np.log(2) / (decay_halflife_days * 24 * 3600)
        
        for sample in all_samples:
            try:
                sample_time = datetime.fromisoformat(sample.timestamp.replace("Z", "+00:00").replace("+00:00", ""))
                age_seconds = (now - sample_time).total_seconds()
            except:
                age_seconds = 0
            
            base_weight = np.exp(-decay_rate * age_seconds)
            
            if sample.is_rare:
                base_weight *= 2.0
            
            weights.append(max(base_weight, 0.1))
        
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        return features, labels, weights
    
    def get_stats(self) -> BufferStats:
        """Get buffer statistics."""
        with self._lock:
            all_samples = list(self.anchor_buffer) + list(self.recency_buffer)
            
            if not all_samples:
                return BufferStats(
                    anchor_size=0,
                    recency_size=0,
                    total_samples=0,
                    rare_pattern_count=0,
                    oldest_sample=None,
                    newest_sample=None,
                    runner_ratio=0.0,
                    crash_ratio=0.0,
                )
            
            timestamps = [s.timestamp for s in all_samples]
            timestamps.sort()
            
            runners = sum(1 for s in all_samples if s.labels.get("hit_runner_threshold", 0) > 0)
            crashes = sum(1 for s in all_samples if s.labels.get("max_drawdown_5d", 0) <= self.crash_threshold)
            
            return BufferStats(
                anchor_size=len(self.anchor_buffer),
                recency_size=len(self.recency_buffer),
                total_samples=len(all_samples),
                rare_pattern_count=len(self._rare_hashes),
                oldest_sample=timestamps[0] if timestamps else None,
                newest_sample=timestamps[-1] if timestamps else None,
                runner_ratio=runners / len(all_samples) if all_samples else 0,
                crash_ratio=crashes / len(all_samples) if all_samples else 0,
            )
    
    def save(self, path: str) -> None:
        """Save buffers to disk."""
        with self._lock:
            data = {
                "anchor_buffer": [asdict(s) for s in self.anchor_buffer],
                "recency_buffer": [asdict(s) for s in self.recency_buffer],
                "rare_hashes": list(self._rare_hashes),
                "config": {
                    "anchor_size": self.anchor_size,
                    "recency_size": self.recency_size,
                    "runner_threshold": self.runner_threshold,
                    "crash_threshold": self.crash_threshold,
                },
            }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        
        logger.info(f"[DualBuffer] Saved {len(self.anchor_buffer)} anchor + {len(self.recency_buffer)} recency samples")
    
    def load(self, path: str) -> bool:
        """Load buffers from disk."""
        if not Path(path).exists():
            return False
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            with self._lock:
                self.anchor_buffer = [
                    SampleRecord(**s) for s in data.get("anchor_buffer", [])
                ]
                self.recency_buffer = deque(
                    [SampleRecord(**s) for s in data.get("recency_buffer", [])],
                    maxlen=self.recency_size,
                )
                self._rare_hashes = set(data.get("rare_hashes", []))
            
            logger.info(f"[DualBuffer] Loaded {len(self.anchor_buffer)} anchor + {len(self.recency_buffer)} recency samples")
            return True
        except Exception as e:
            logger.error(f"[DualBuffer] Failed to load: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            self.anchor_buffer.clear()
            self.recency_buffer.clear()
            self._rare_hashes.clear()


class GBMHead:
    """
    A gradient boosting prediction head with warm-start support.
    
    Uses LightGBM if available, otherwise falls back to scikit-learn.
    Both support warm_start for efficient incremental learning.
    """
    
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 250,
        max_depth: int = 7,
        learning_rate: float = 0.08,
        num_leaves: int = 31,
    ):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.use_lightgbm = LIGHTGBM_AVAILABLE
        
        self.model = None
        self._is_fitted = False
        self._n_trees = 0
        
        if not self.use_lightgbm:
            self._init_sklearn_model()
    
    def _init_sklearn_model(self):
        """Initialize scikit-learn model with warm_start=True."""
        if self.task == "regression":
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                random_state=42,
                warm_start=True,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                random_state=42,
                warm_start=True,
            )
    
    def _get_lgb_params(self) -> Dict:
        """Get LightGBM parameters."""
        params = {
            "objective": "regression" if self.task == "regression" else "multiclass",
            "metric": "rmse" if self.task == "regression" else "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }
        
        if self.task == "binary":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        
        return params
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        warm_start: bool = False,
        n_new_trees: int = 50,
    ) -> Dict[str, float]:
        """
        Train the head with optional warm-start.
        
        Args:
            X: Features
            y: Labels
            sample_weight: Sample weights
            warm_start: If True, add trees to existing model
            n_new_trees: Number of trees to add in warm-start mode
        
        Returns:
            Training metrics
        """
        if self.use_lightgbm:
            return self._fit_lightgbm(X, y, sample_weight, warm_start, n_new_trees)
        else:
            return self._fit_sklearn(X, y, sample_weight, warm_start, n_new_trees)
    
    def _fit_sklearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        warm_start: bool = False,
        n_new_trees: int = 50,
    ) -> Dict[str, float]:
        """Fit using scikit-learn with warm_start."""
        if warm_start and self._is_fitted:
            current_trees = self.model.n_estimators
            self.model.n_estimators = current_trees + n_new_trees
            self.model.fit(X, y, sample_weight=sample_weight)
            self._n_trees = self.model.n_estimators
            logger.info(f"[GBMHead-sklearn] Warm-start: added {n_new_trees} trees, total: {self._n_trees}")
        else:
            if self.model is None:
                self._init_sklearn_model()
            self.model.fit(X, y, sample_weight=sample_weight)
            self._n_trees = self.model.n_estimators
            logger.info(f"[GBMHead-sklearn] Fresh training: {self._n_trees} trees")
        
        self._is_fitted = True
        
        pred = self.predict(X)
        if self.task == "regression":
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            return {"rmse": rmse}
        else:
            if self.task == "binary":
                acc = float(np.mean((pred > 0.5).astype(int) == y))
            else:
                acc = float(np.mean(pred == y))
            return {"accuracy": acc}
    
    def _fit_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        warm_start: bool = False,
        n_new_trees: int = 50,
    ) -> Dict[str, float]:
        """Fit using LightGBM with warm-start."""
        if self.task == "multiclass":
            num_class = len(np.unique(y))
            params = self._get_lgb_params()
            params["num_class"] = num_class
        else:
            params = self._get_lgb_params()
        
        dataset = lgb.Dataset(
            X, 
            label=y, 
            weight=sample_weight,
            free_raw_data=False,
        )
        
        if warm_start and self.model is not None:
            new_model = lgb.train(
                params,
                dataset,
                num_boost_round=n_new_trees,
                init_model=self.model,
            )
            self.model = new_model
            self._n_trees += n_new_trees
            logger.info(f"[GBMHead-lgb] Warm-start: added {n_new_trees} trees, total: {self._n_trees}")
        else:
            self.model = lgb.train(
                params,
                dataset,
                num_boost_round=self.n_estimators,
            )
            self._n_trees = self.n_estimators
            logger.info(f"[GBMHead-lgb] Fresh training: {self._n_trees} trees")
        
        self._is_fitted = True
        
        pred = self.predict(X)
        if self.task == "regression":
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            return {"rmse": rmse}
        else:
            if self.task == "binary":
                acc = float(np.mean((pred > 0.5).astype(int) == y))
            else:
                acc = float(np.mean(pred == y))
            return {"accuracy": acc}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        
        if self.use_lightgbm:
            raw_pred = self.model.predict(X)
            if self.task == "multiclass":
                return np.argmax(raw_pred, axis=1)
            elif self.task == "binary":
                return raw_pred
            else:
                return raw_pred
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions (for classifiers)."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model not fitted")
        
        if self.use_lightgbm:
            return self.model.predict(X)
        else:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is not None:
            if self.use_lightgbm:
                self.model.save_model(path)
            else:
                joblib.dump(self.model, path)
    
    def load(self, path: str) -> bool:
        """Load model from disk."""
        if not Path(path).exists():
            return False
        
        try:
            if self.use_lightgbm:
                self.model = lgb.Booster(model_file=path)
                self._n_trees = self.model.num_trees()
            else:
                self.model = joblib.load(path)
                self._n_trees = self.model.n_estimators
            self._is_fitted = True
            return True
        except Exception as e:
            logger.error(f"[GBMHead] Failed to load: {e}")
            return False


LightGBMHead = GBMHead


@dataclass
class IncrementalManifest:
    """Manifest for incrementally trained model."""
    version: str
    base_version: str
    trained_at: str
    training_mode: str
    total_samples: int
    anchor_samples: int
    recency_samples: int
    trees_per_head: Dict[str, int]
    metrics: Dict[str, float]
    decay_halflife_days: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IncrementalApexCore:
    """
    Incremental learning version of ApexCore V3.
    
    Uses LightGBM with warm-start for efficient continuous learning
    while preserving knowledge of rare patterns.
    
    Features:
    - Warm-start training (adds trees without forgetting)
    - Dual-buffer for pattern preservation
    - Time-decay sample weighting
    - Automatic rare event detection
    """
    
    FEATURE_NAMES = [
        "quantra_score", "entropy_band_encoded", "suppression_state_encoded",
        "regime_type_encoded", "volatility_band_encoded", "liquidity_band_encoded",
        "risk_tier_encoded", "protocol_active_count", "protocol_weighted_score",
        "ret_1d", "ret_3d", "ret_5d", "max_runup_5d", "max_drawdown_5d",
        "vix_level", "vix_percentile", "sector_momentum", "market_breadth",
    ]
    
    def __init__(
        self,
        model_size: str = "big",
        decay_halflife_days: float = 30.0,
    ):
        self.model_size = model_size
        self.version = "3.1.0"
        self.decay_halflife_days = decay_halflife_days
        
        n_estimators = 100 if model_size == "mini" else 250
        max_depth = 4 if model_size == "mini" else 7
        
        self.quantrascore_head = LightGBMHead("regression", n_estimators, max_depth)
        self.runner_head = LightGBMHead("binary", n_estimators, max_depth)
        self.quality_head = LightGBMHead("multiclass", n_estimators, max_depth)
        self.avoid_head = LightGBMHead("binary", n_estimators, max_depth)
        self.regime_head = LightGBMHead("multiclass", n_estimators, max_depth)
        self.timing_head = LightGBMHead("multiclass", n_estimators, max_depth)
        self.runup_head = LightGBMHead("regression", n_estimators, max_depth)
        
        self.scaler = StandardScaler()
        self.quality_encoder = LabelEncoder()
        self.regime_encoder = LabelEncoder()
        self.timing_encoder = LabelEncoder()
        
        self.buffer = DualBufferSystem()
        
        self._is_fitted = False
        self._base_version: Optional[str] = None
        self._manifest: Optional[IncrementalManifest] = None
    
    def add_samples(
        self,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Add training samples to the buffer."""
        features = self._prepare_features(rows)
        
        labels = {
            "quantra_score": np.array([r.get("quantra_score", 50) for r in rows]),
            "hit_runner_threshold": np.array([r.get("hit_runner_threshold", 0) for r in rows]),
            "future_quality_tier": np.array([r.get("future_quality_tier", "C") for r in rows]),
            "avoid_trade": np.array([r.get("avoid_trade", 0) for r in rows]),
            "regime_label": np.array([r.get("regime_label", "chop") for r in rows]),
            "timing_bucket": np.array([r.get("timing_bucket", "none") for r in rows]),
            "max_runup_5d": np.array([r.get("max_runup_5d", 0) for r in rows]),
            "max_drawdown_5d": np.array([r.get("max_drawdown_5d", 0) for r in rows]),
        }
        
        symbols = [r.get("symbol", "") for r in rows]
        timestamps = [r.get("timestamp", datetime.now().isoformat()) for r in rows]
        
        str_labels = {}
        for k, v in labels.items():
            if v.dtype == object or k in ["future_quality_tier", "regime_label", "timing_bucket"]:
                str_labels[k] = v.astype(str)
            else:
                str_labels[k] = v.astype(float)
        
        float_labels = {}
        for k, v in str_labels.items():
            if k in ["future_quality_tier", "regime_label", "timing_bucket"]:
                float_labels[k] = np.zeros(len(v))
            else:
                float_labels[k] = v.astype(float)
        
        return self.buffer.add_batch(features, float_labels, symbols, timestamps)
    
    def _prepare_features(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare feature matrix."""
        entropy_map = {"low": 0, "mid": 1, "high": 2}
        suppression_map = {"none": 0, "suppressed": 1, "blocked": 2}
        regime_map = {"trend_up": 0, "trend_down": 1, "chop": 2, "squeeze": 3, "crash": 4}
        volatility_map = {"low": 0, "mid": 1, "high": 2}
        liquidity_map = {"low": 0, "mid": 1, "high": 2}
        risk_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        
        features = []
        for row in rows:
            protocol_ids = row.get("protocol_ids", [])
            protocol_count = len(protocol_ids) if isinstance(protocol_ids, list) else 0
            
            feature_vec = [
                float(row.get("quantra_score", 50)),
                entropy_map.get(str(row.get("entropy_band", "mid")).lower(), 1),
                suppression_map.get(str(row.get("suppression_state", "none")).lower(), 0),
                regime_map.get(str(row.get("regime_type", "chop")).lower(), 2),
                volatility_map.get(str(row.get("volatility_band", "mid")).lower(), 1),
                liquidity_map.get(str(row.get("liquidity_band", "mid")).lower(), 1),
                risk_map.get(str(row.get("risk_tier", "medium")).lower(), 1),
                protocol_count,
                float(row.get("protocol_weighted_score", 1.0)),
                float(row.get("ret_1d", 0)),
                float(row.get("ret_3d", 0)),
                float(row.get("ret_5d", 0)),
                float(row.get("max_runup_5d", 0)),
                float(row.get("max_drawdown_5d", 0)),
                float(row.get("vix_level", 20.0)),
                float(row.get("vix_percentile", 50.0)),
                float(row.get("sector_momentum", 0.0)),
                float(row.get("market_breadth", 0.5)),
            ]
            features.append(feature_vec)
        
        return np.array(features)
    
    def train(
        self,
        warm_start: bool = True,
        n_new_trees: int = 50,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train all heads with optional warm-start.
        
        Args:
            warm_start: If True, add trees to existing model
            n_new_trees: Number of trees to add in warm-start mode
            validation_split: Fraction of data for validation
        
        Returns:
            Training metrics
        """
        X, raw_labels, weights = self.buffer.get_training_data(
            decay_halflife_days=self.decay_halflife_days,
        )
        
        if len(X) < 30:
            raise ValueError(f"Insufficient training data: {len(X)} samples (need 30+)")
        
        logger.info(f"[IncrementalApexCore] Training on {len(X)} samples (warm_start={warm_start})")
        
        y_quantrascore = raw_labels.get("quantra_score", np.full(len(X), 50))
        y_runner = raw_labels.get("hit_runner_threshold", np.zeros(len(X)))
        y_avoid = raw_labels.get("avoid_trade", np.zeros(len(X)))
        y_runup = raw_labels.get("max_runup_5d", np.zeros(len(X)))
        
        y_quality = np.array(["C"] * len(X))
        y_regime = np.array(["chop"] * len(X))
        y_timing = np.array(["none"] * len(X))
        
        if not warm_start or not self._is_fitted:
            self.quality_encoder.fit(["A", "B", "C", "D", "F"])
            self.regime_encoder.fit(["trend_up", "trend_down", "chop", "squeeze", "crash"])
            self.timing_encoder.fit(["immediate", "very_soon", "soon", "late", "none"])
        
        y_quality_enc = self.quality_encoder.transform(y_quality)
        y_regime_enc = self.regime_encoder.transform(y_regime)
        y_timing_enc = self.timing_encoder.transform(y_timing)
        
        n_val = int(len(X) * validation_split)
        indices = np.random.RandomState(42).permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        train_weights = weights[train_idx]
        
        if not warm_start or not self._is_fitted:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        metrics = {}
        
        logger.info("Training QuantraScore head...")
        m = self.quantrascore_head.fit(
            X_train_scaled, y_quantrascore[train_idx], 
            sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
        )
        qs_pred = self.quantrascore_head.predict(X_val_scaled)
        metrics["quantrascore_rmse"] = float(np.sqrt(np.mean((qs_pred - y_quantrascore[val_idx]) ** 2)))
        
        logger.info("Training Runner head...")
        if len(np.unique(y_runner[train_idx])) > 1:
            self.runner_head.fit(
                X_train_scaled, y_runner[train_idx],
                sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
            )
            runner_pred = self.runner_head.predict(X_val_scaled)
            metrics["runner_accuracy"] = float(np.mean((runner_pred > 0.5) == y_runner[val_idx]))
        else:
            metrics["runner_accuracy"] = 0.5
        
        logger.info("Training Quality head...")
        self.quality_head.fit(
            X_train_scaled, y_quality_enc[train_idx],
            sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
        )
        quality_pred = self.quality_head.predict(X_val_scaled)
        metrics["quality_accuracy"] = float(np.mean(quality_pred == y_quality_enc[val_idx]))
        
        logger.info("Training Avoid head...")
        if len(np.unique(y_avoid[train_idx])) > 1:
            self.avoid_head.fit(
                X_train_scaled, y_avoid[train_idx],
                sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
            )
            avoid_pred = self.avoid_head.predict(X_val_scaled)
            metrics["avoid_accuracy"] = float(np.mean((avoid_pred > 0.5) == y_avoid[val_idx]))
        else:
            metrics["avoid_accuracy"] = 0.5
        
        logger.info("Training Regime head...")
        self.regime_head.fit(
            X_train_scaled, y_regime_enc[train_idx],
            sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
        )
        regime_pred = self.regime_head.predict(X_val_scaled)
        metrics["regime_accuracy"] = float(np.mean(regime_pred == y_regime_enc[val_idx]))
        
        logger.info("Training Timing head...")
        self.timing_head.fit(
            X_train_scaled, y_timing_enc[train_idx],
            sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
        )
        timing_pred = self.timing_head.predict(X_val_scaled)
        metrics["timing_accuracy"] = float(np.mean(timing_pred == y_timing_enc[val_idx]))
        
        logger.info("Training Runup head...")
        self.runup_head.fit(
            X_train_scaled, y_runup[train_idx],
            sample_weight=train_weights, warm_start=warm_start, n_new_trees=n_new_trees
        )
        runup_pred = self.runup_head.predict(X_val_scaled)
        metrics["runup_rmse"] = float(np.sqrt(np.mean((runup_pred - y_runup[val_idx]) ** 2)))
        
        self._is_fitted = True
        
        buffer_stats = self.buffer.get_stats()
        self._manifest = IncrementalManifest(
            version=self.version,
            base_version=self._base_version or "none",
            trained_at=datetime.utcnow().isoformat(),
            training_mode="warm_start" if warm_start else "full",
            total_samples=buffer_stats.total_samples,
            anchor_samples=buffer_stats.anchor_size,
            recency_samples=buffer_stats.recency_size,
            trees_per_head={
                "quantrascore": self.quantrascore_head._n_trees,
                "runner": self.runner_head._n_trees,
                "quality": self.quality_head._n_trees,
                "avoid": self.avoid_head._n_trees,
                "regime": self.regime_head._n_trees,
                "timing": self.timing_head._n_trees,
                "runup": self.runup_head._n_trees,
            },
            metrics=metrics,
            decay_halflife_days=self.decay_halflife_days,
        )
        
        logger.info(f"[IncrementalApexCore] Training complete: {metrics}")
        return metrics
    
    def save(self, path: str) -> None:
        """Save model and buffers to disk."""
        model_dir = Path(path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.quantrascore_head.save(str(model_dir / "quantrascore.lgb"))
        self.runner_head.save(str(model_dir / "runner.lgb"))
        self.quality_head.save(str(model_dir / "quality.lgb"))
        self.avoid_head.save(str(model_dir / "avoid.lgb"))
        self.regime_head.save(str(model_dir / "regime.lgb"))
        self.timing_head.save(str(model_dir / "timing.lgb"))
        self.runup_head.save(str(model_dir / "runup.lgb"))
        
        joblib.dump(self.scaler, model_dir / "scaler.joblib")
        joblib.dump(self.quality_encoder, model_dir / "quality_encoder.joblib")
        joblib.dump(self.regime_encoder, model_dir / "regime_encoder.joblib")
        joblib.dump(self.timing_encoder, model_dir / "timing_encoder.joblib")
        
        self.buffer.save(str(model_dir / "buffer.json"))
        
        if self._manifest:
            with open(model_dir / "manifest.json", "w") as f:
                json.dump(self._manifest.to_dict(), f, indent=2)
        
        logger.info(f"[IncrementalApexCore] Model saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load model and buffers from disk."""
        model_dir = Path(path)
        
        if not model_dir.exists():
            return False
        
        try:
            self.quantrascore_head.load(str(model_dir / "quantrascore.lgb"))
            self.runner_head.load(str(model_dir / "runner.lgb"))
            self.quality_head.load(str(model_dir / "quality.lgb"))
            self.avoid_head.load(str(model_dir / "avoid.lgb"))
            self.regime_head.load(str(model_dir / "regime.lgb"))
            self.timing_head.load(str(model_dir / "timing.lgb"))
            self.runup_head.load(str(model_dir / "runup.lgb"))
            
            self.scaler = joblib.load(model_dir / "scaler.joblib")
            self.quality_encoder = joblib.load(model_dir / "quality_encoder.joblib")
            self.regime_encoder = joblib.load(model_dir / "regime_encoder.joblib")
            self.timing_encoder = joblib.load(model_dir / "timing_encoder.joblib")
            
            self.buffer.load(str(model_dir / "buffer.json"))
            
            manifest_path = model_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    data = json.load(f)
                self._manifest = IncrementalManifest(**data)
                self._base_version = self._manifest.base_version
            
            self._is_fitted = True
            logger.info(f"[IncrementalApexCore] Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"[IncrementalApexCore] Failed to load: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status."""
        buffer_stats = self.buffer.get_stats()
        
        return {
            "version": self.version,
            "model_size": self.model_size,
            "is_fitted": self._is_fitted,
            "decay_halflife_days": self.decay_halflife_days,
            "buffer": {
                "anchor_samples": buffer_stats.anchor_size,
                "recency_samples": buffer_stats.recency_size,
                "total_samples": buffer_stats.total_samples,
                "rare_patterns": buffer_stats.rare_pattern_count,
                "runner_ratio": buffer_stats.runner_ratio,
            },
            "heads": {
                "quantrascore_trees": self.quantrascore_head._n_trees,
                "runner_trees": self.runner_head._n_trees,
                "quality_trees": self.quality_head._n_trees,
                "avoid_trees": self.avoid_head._n_trees,
                "regime_trees": self.regime_head._n_trees,
                "timing_trees": self.timing_head._n_trees,
                "runup_trees": self.runup_head._n_trees,
            },
            "manifest": self._manifest.to_dict() if self._manifest else None,
        }


_incremental_model: Optional[IncrementalApexCore] = None

def get_incremental_model(model_size: str = "big") -> IncrementalApexCore:
    """Get or create the incremental model singleton."""
    global _incremental_model
    
    if _incremental_model is None:
        _incremental_model = IncrementalApexCore(model_size=model_size)
        
        model_path = f"models/apexcore_v3_incremental/{model_size}"
        if Path(model_path).exists():
            _incremental_model.load(model_path)
    
    return _incremental_model
