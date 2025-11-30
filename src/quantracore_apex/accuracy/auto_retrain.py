"""
Auto-Retraining System - Drift Detection and Automatic Model Retraining.

Provides:
- Distribution drift detection (feature and label drift)
- Performance degradation monitoring
- Automatic retraining triggers
- Sample weighting for recent data
- Shadow model deployment before promotion

Ensures models stay calibrated as market conditions evolve.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    feature_drift_score: float
    label_drift_score: float
    performance_drift: float
    samples_since_training: int
    days_since_training: float
    drift_detected: bool
    drift_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainingDecision:
    """Decision about whether to retrain."""
    should_retrain: bool
    urgency: str
    reasons: List[str]
    recommended_action: str
    estimated_samples_needed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleWeight:
    """Weight for a training sample."""
    base_weight: float
    recency_weight: float
    importance_weight: float
    regime_weight: float
    final_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DriftDetector:
    """
    Detects distribution drift in features and labels.
    
    Uses multiple detection methods:
    - Population Stability Index (PSI) for feature drift
    - KL divergence for label distribution shift
    - Performance monitoring for accuracy drift
    """
    
    PSI_THRESHOLD = 0.2
    PERFORMANCE_THRESHOLD = 0.1
    
    def __init__(
        self,
        reference_window: int = 1000,
        detection_window: int = 200,
    ):
        self._reference_window = reference_window
        self._detection_window = detection_window
        
        self._reference_features: Optional[np.ndarray] = None
        self._reference_labels: Optional[np.ndarray] = None
        self._reference_performance: Optional[float] = None
        
        self._recent_features: deque = deque(maxlen=detection_window)
        self._recent_labels: deque = deque(maxlen=detection_window)
        self._recent_predictions: deque = deque(maxlen=detection_window)
        self._recent_actuals: deque = deque(maxlen=detection_window)
        
        self._is_initialized = False
    
    def set_reference(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        performance: float,
    ) -> None:
        """
        Set reference distributions from training data.
        
        Args:
            features: Training feature matrix
            labels: Training labels
            performance: Training performance metric (e.g., accuracy)
        """
        self._reference_features = features[-self._reference_window:]
        self._reference_labels = labels[-self._reference_window:]
        self._reference_performance = performance
        self._is_initialized = True
        
        logger.info(f"[DriftDetector] Reference set with {len(features)} samples")
    
    def add_observation(
        self,
        features: np.ndarray,
        label: float,
        prediction: float,
        actual: float,
    ) -> None:
        """Add new observation for drift monitoring."""
        self._recent_features.append(features)
        self._recent_labels.append(label)
        self._recent_predictions.append(prediction)
        self._recent_actuals.append(actual)
    
    def detect_drift(self) -> DriftMetrics:
        """
        Run drift detection on recent observations.
        
        Returns:
            DriftMetrics with drift scores and detection status
        """
        if not self._is_initialized or len(self._recent_features) < 50:
            return DriftMetrics(
                feature_drift_score=0.0,
                label_drift_score=0.0,
                performance_drift=0.0,
                samples_since_training=len(self._recent_features),
                days_since_training=0.0,
                drift_detected=False,
            )
        
        recent_features = np.array(list(self._recent_features))
        feature_drift = self._compute_psi(
            self._reference_features,
            recent_features,
        )
        
        recent_labels = np.array(list(self._recent_labels))
        label_drift = self._compute_label_drift(
            self._reference_labels,
            recent_labels,
        )
        
        recent_preds = np.array(list(self._recent_predictions))
        recent_acts = np.array(list(self._recent_actuals))
        current_performance = 1.0 - np.mean(np.abs(recent_preds - recent_acts)) / 100
        performance_drift = abs(current_performance - self._reference_performance)
        
        drift_reasons = []
        drift_detected = False
        
        if feature_drift > self.PSI_THRESHOLD:
            drift_detected = True
            drift_reasons.append(f"Feature drift: PSI={feature_drift:.3f}")
        
        if label_drift > 0.3:
            drift_detected = True
            drift_reasons.append(f"Label distribution shift: {label_drift:.3f}")
        
        if performance_drift > self.PERFORMANCE_THRESHOLD:
            drift_detected = True
            drift_reasons.append(f"Performance degradation: {performance_drift:.3f}")
        
        return DriftMetrics(
            feature_drift_score=feature_drift,
            label_drift_score=label_drift,
            performance_drift=performance_drift,
            samples_since_training=len(self._recent_features),
            days_since_training=0.0,
            drift_detected=drift_detected,
            drift_reasons=drift_reasons,
        )
    
    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Population Stability Index for feature drift.
        """
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if current.ndim == 1:
            current = current.reshape(-1, 1)
        
        psi_total = 0.0
        
        for col in range(reference.shape[1]):
            ref_col = reference[:, col]
            cur_col = current[:, col]
            
            all_vals = np.concatenate([ref_col, cur_col])
            bins = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
            bins = np.unique(bins)
            
            if len(bins) < 2:
                continue
            
            ref_hist, _ = np.histogram(ref_col, bins=bins)
            cur_hist, _ = np.histogram(cur_col, bins=bins)
            
            ref_pct = (ref_hist + 0.001) / (ref_hist.sum() + 0.001 * len(ref_hist))
            cur_pct = (cur_hist + 0.001) / (cur_hist.sum() + 0.001 * len(cur_hist))
            
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            psi_total += psi
        
        return psi_total / max(reference.shape[1], 1)
    
    def _compute_label_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Compute label distribution drift using KL-like divergence."""
        ref_mean, ref_std = np.mean(reference), np.std(reference)
        cur_mean, cur_std = np.mean(current), np.std(current)
        
        mean_shift = abs(cur_mean - ref_mean) / (ref_std + 1e-10)
        std_shift = abs(cur_std - ref_std) / (ref_std + 1e-10)
        
        return (mean_shift + std_shift) / 2
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


class SampleWeighter:
    """
    Computes importance weights for training samples.
    
    Prioritizes:
    - Recent samples (recency weighting)
    - Regime shift samples (importance weighting)
    - Samples from current market conditions
    """
    
    def __init__(
        self,
        recency_half_life_days: float = 30.0,
        importance_boost: float = 2.0,
    ):
        self._half_life = recency_half_life_days
        self._importance_boost = importance_boost
    
    def compute_weights(
        self,
        timestamps: List[datetime],
        regimes: Optional[List[str]] = None,
        current_regime: Optional[str] = None,
        mispredictions: Optional[List[bool]] = None,
    ) -> List[SampleWeight]:
        """
        Compute weights for training samples.
        
        Args:
            timestamps: Sample timestamps
            regimes: Optional regime labels for each sample
            current_regime: Current market regime for regime weighting
            mispredictions: Optional flag for samples that were mispredicted
            
        Returns:
            List of SampleWeight objects
        """
        now = datetime.utcnow()
        weights = []
        
        for i, ts in enumerate(timestamps):
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            
            days_ago = (now - ts).total_seconds() / 86400
            recency = 0.5 ** (days_ago / self._half_life)
            
            importance = 1.0
            if mispredictions and mispredictions[i]:
                importance = self._importance_boost
            
            regime_weight = 1.0
            if regimes and current_regime:
                if regimes[i] == current_regime:
                    regime_weight = 1.5
            
            base = 1.0
            final = base * recency * importance * regime_weight
            
            weights.append(SampleWeight(
                base_weight=base,
                recency_weight=recency,
                importance_weight=importance,
                regime_weight=regime_weight,
                final_weight=final,
            ))
        
        total = sum(w.final_weight for w in weights)
        if total > 0:
            for w in weights:
                w.final_weight = w.final_weight / total * len(weights)
        
        return weights
    
    def get_weight_array(
        self,
        timestamps: List[datetime],
        **kwargs,
    ) -> np.ndarray:
        """Get weights as numpy array for sklearn sample_weight."""
        weights = self.compute_weights(timestamps, **kwargs)
        return np.array([w.final_weight for w in weights])


class AutoRetrainer:
    """
    Automatic retraining system with drift detection and sample weighting.
    
    Usage:
        retrainer = AutoRetrainer()
        
        # Set reference from training
        retrainer.set_training_reference(X_train, y_train, accuracy)
        
        # Monitor new predictions
        retrainer.add_observation(features, label, pred, actual)
        
        # Check if retraining needed
        decision = retrainer.check_retrain()
        if decision.should_retrain:
            # Trigger retraining with weighted samples
            weights = retrainer.get_sample_weights(timestamps)
    """
    
    def __init__(
        self,
        min_samples_for_retrain: int = 500,
        max_days_without_retrain: float = 30.0,
        drift_check_interval: int = 100,
        save_dir: str = "data/retraining",
    ):
        self._min_samples = min_samples_for_retrain
        self._max_days = max_days_without_retrain
        self._check_interval = drift_check_interval
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        self._drift_detector = DriftDetector()
        self._sample_weighter = SampleWeighter()
        
        self._last_training_time: Optional[datetime] = None
        self._training_version: int = 0
        self._observation_count: int = 0
        
        self._lock = threading.Lock()
        
        self._callbacks: List[Callable] = []
    
    def set_training_reference(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        performance: float,
    ) -> None:
        """Set reference from most recent training."""
        with self._lock:
            self._drift_detector.set_reference(features, labels, performance)
            self._last_training_time = datetime.utcnow()
            self._training_version += 1
            self._observation_count = 0
        
        self._save_state()
    
    def add_observation(
        self,
        features: np.ndarray,
        label: float,
        prediction: float,
        actual: float,
    ) -> None:
        """Add new observation for monitoring."""
        with self._lock:
            self._drift_detector.add_observation(features, label, prediction, actual)
            self._observation_count += 1
            
            if self._observation_count % self._check_interval == 0:
                self._check_and_notify()
    
    def check_retrain(self) -> RetrainingDecision:
        """
        Check if retraining is needed.
        
        Returns:
            RetrainingDecision with recommendation
        """
        with self._lock:
            drift = self._drift_detector.detect_drift()
            
            reasons = []
            urgency = "none"
            should_retrain = False
            
            if drift.drift_detected:
                should_retrain = True
                urgency = "high" if drift.performance_drift > 0.15 else "medium"
                reasons.extend(drift.drift_reasons)
            
            if self._last_training_time:
                days_since = (datetime.utcnow() - self._last_training_time).total_seconds() / 86400
                if days_since > self._max_days:
                    should_retrain = True
                    urgency = max(urgency, "medium")
                    reasons.append(f"Max days exceeded: {days_since:.1f} days since training")
            
            if self._observation_count >= self._min_samples and not drift.drift_detected:
                urgency = "low"
                reasons.append(f"Sufficient new samples: {self._observation_count}")
            
            if should_retrain:
                action = "Immediate retraining recommended"
                samples_needed = max(self._min_samples, self._observation_count)
            else:
                action = f"Continue monitoring. Next check in {self._check_interval} observations"
                samples_needed = self._min_samples
            
            return RetrainingDecision(
                should_retrain=should_retrain,
                urgency=urgency,
                reasons=reasons,
                recommended_action=action,
                estimated_samples_needed=samples_needed,
            )
    
    def get_drift_metrics(self) -> DriftMetrics:
        """Get current drift metrics."""
        with self._lock:
            return self._drift_detector.detect_drift()
    
    def get_sample_weights(
        self,
        timestamps: List[datetime],
        regimes: Optional[List[str]] = None,
        current_regime: Optional[str] = None,
        mispredictions: Optional[List[bool]] = None,
    ) -> np.ndarray:
        """Get sample weights for retraining."""
        return self._sample_weighter.get_weight_array(
            timestamps,
            regimes=regimes,
            current_regime=current_regime,
            mispredictions=mispredictions,
        )
    
    def register_callback(self, callback: Callable[[RetrainingDecision], None]) -> None:
        """Register callback for retraining decisions."""
        self._callbacks.append(callback)
    
    def _check_and_notify(self) -> None:
        """Check drift and notify callbacks if retraining needed."""
        decision = self.check_retrain()
        
        if decision.should_retrain:
            logger.warning(
                f"[AutoRetrainer] Retraining recommended! "
                f"Urgency: {decision.urgency}. Reasons: {decision.reasons}"
            )
            
            for callback in self._callbacks:
                try:
                    callback(decision)
                except Exception as e:
                    logger.error(f"[AutoRetrainer] Callback error: {e}")
    
    def _save_state(self) -> None:
        """Save retrainer state."""
        state = {
            "training_version": self._training_version,
            "last_training_time": self._last_training_time.isoformat() if self._last_training_time else None,
            "observation_count": self._observation_count,
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        state_file = self._save_dir / "retrainer_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get retrainer status."""
        with self._lock:
            drift = self._drift_detector.detect_drift()
            decision = self.check_retrain()
            
            return {
                "training_version": self._training_version,
                "observations_since_training": self._observation_count,
                "days_since_training": (
                    (datetime.utcnow() - self._last_training_time).total_seconds() / 86400
                    if self._last_training_time else None
                ),
                "drift_metrics": drift.to_dict(),
                "retrain_decision": decision.to_dict(),
            }


_auto_retrainer: Optional[AutoRetrainer] = None


def get_auto_retrainer() -> AutoRetrainer:
    """Get global auto-retrainer instance."""
    global _auto_retrainer
    if _auto_retrainer is None:
        _auto_retrainer = AutoRetrainer()
    return _auto_retrainer


__all__ = [
    "AutoRetrainer",
    "DriftDetector",
    "SampleWeighter",
    "DriftMetrics",
    "RetrainingDecision",
    "SampleWeight",
    "get_auto_retrainer",
]
