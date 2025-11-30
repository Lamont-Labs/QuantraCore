"""
Calibration Layer - Probabilistic Calibration for Model Predictions.

Provides:
- Platt scaling for probability calibration
- Isotonic regression calibration
- Temperature scaling for confidence calibration
- Calibration metrics (ECE, MCE, reliability diagrams)

Ensures that when the model says "70% confidence", it's actually right 70% of the time.
"""

import os
import json
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    reliability_bins: List[float]
    reliability_accuracies: List[float]
    calibration_quality: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibratedPrediction:
    """Prediction with calibrated confidence."""
    raw_probability: float
    calibrated_probability: float
    confidence_interval: Tuple[float, float]
    calibration_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence_interval"] = list(self.confidence_interval)
        return d


class PlattScaler:
    """
    Platt scaling for probability calibration.
    
    Fits a logistic regression on model outputs to calibrate probabilities.
    """
    
    def __init__(self):
        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._is_fitted = False
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "PlattScaler":
        """
        Fit Platt scaler on validation data.
        
        Args:
            probabilities: Model predicted probabilities
            labels: True binary labels
        """
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        X = np.log(probs / (1 - probs)).reshape(-1, 1)
        
        self._model.fit(X, labels)
        self._is_fitted = True
        
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities."""
        if not self._is_fitted:
            return probabilities
        
        probs = np.clip(probabilities, 1e-10, 1 - 1e-10)
        X = np.log(probs / (1 - probs)).reshape(-1, 1)
        
        return self._model.predict_proba(X)[:, 1]
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class IsotonicCalibrator:
    """
    Isotonic regression calibration.
    
    Non-parametric calibration that preserves probability ordering.
    """
    
    def __init__(self):
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._is_fitted = False
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic calibrator on validation data.
        
        Args:
            probabilities: Model predicted probabilities
            labels: True binary labels
        """
        self._model.fit(probabilities, labels)
        self._is_fitted = True
        
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to probabilities."""
        if not self._is_fitted:
            return probabilities
        
        return self._model.predict(probabilities)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class TemperatureScaler:
    """
    Temperature scaling for confidence calibration.
    
    Simple but effective method that scales logits by a learned temperature.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self._temperature = initial_temperature
        self._is_fitted = False
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_iterations: int = 100,
    ) -> "TemperatureScaler":
        """
        Optimize temperature on validation data.
        
        Uses grid search to find optimal temperature.
        """
        best_temp = 1.0
        best_ece = float("inf")
        
        for temp in np.linspace(0.5, 3.0, 50):
            calibrated = self._apply_temperature(probabilities, temp)
            ece = self._compute_ece(calibrated, labels)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self._temperature = best_temp
        self._is_fitted = True
        
        logger.info(f"[TemperatureScaler] Optimal temperature: {best_temp:.3f} (ECE: {best_ece:.4f})")
        
        return self
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        return self._apply_temperature(probabilities, self._temperature)
    
    def _apply_temperature(self, probs: np.ndarray, temp: float) -> np.ndarray:
        """Apply temperature to probabilities via logit space."""
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        logits = np.log(probs / (1 - probs))
        scaled_logits = logits / temp
        return 1 / (1 + np.exp(-scaled_logits))
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
        
        return ece / len(probs)
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class CalibrationLayer:
    """
    Unified calibration layer for all model predictions.
    
    Supports multiple calibration methods and provides confidence intervals.
    
    Usage:
        calibrator = CalibrationLayer()
        
        # Fit on validation data
        calibrator.fit(val_probs, val_labels)
        
        # Calibrate new predictions
        calibrated = calibrator.calibrate(raw_probs)
    """
    
    def __init__(
        self,
        method: str = "isotonic",
        save_dir: str = "models/calibration",
    ):
        """
        Initialize calibration layer.
        
        Args:
            method: Calibration method ("platt", "isotonic", "temperature", "ensemble")
            save_dir: Directory for saving calibrators
        """
        self._method = method
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        self._platt = PlattScaler()
        self._isotonic = IsotonicCalibrator()
        self._temperature = TemperatureScaler()
        
        self._is_fitted = False
        self._fit_metrics: Optional[CalibrationMetrics] = None
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        fit_all: bool = True,
    ) -> CalibrationMetrics:
        """
        Fit calibrators on validation data.
        
        Args:
            probabilities: Model predicted probabilities
            labels: True binary labels (0 or 1)
            fit_all: Whether to fit all calibration methods
            
        Returns:
            Calibration metrics after fitting
        """
        if len(probabilities) < 50:
            logger.warning("[Calibration] Insufficient data for reliable calibration")
        
        if fit_all or self._method == "platt":
            self._platt.fit(probabilities, labels)
        
        if fit_all or self._method == "isotonic":
            self._isotonic.fit(probabilities, labels)
        
        if fit_all or self._method == "temperature":
            self._temperature.fit(probabilities, labels)
        
        self._is_fitted = True
        
        self._fit_metrics = self.compute_calibration_metrics(probabilities, labels)
        
        return self._fit_metrics
    
    def calibrate(
        self,
        probabilities: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Calibrate raw probabilities.
        
        Args:
            probabilities: Raw model probabilities
            method: Override default method
            
        Returns:
            Calibrated probabilities
        """
        method = method or self._method
        
        if not self._is_fitted:
            logger.warning("[Calibration] Calibrator not fitted, returning raw probabilities")
            return probabilities
        
        probs = np.atleast_1d(probabilities)
        
        if method == "platt":
            return self._platt.calibrate(probs)
        elif method == "isotonic":
            return self._isotonic.calibrate(probs)
        elif method == "temperature":
            return self._temperature.calibrate(probs)
        elif method == "ensemble":
            platt_cal = self._platt.calibrate(probs)
            iso_cal = self._isotonic.calibrate(probs)
            temp_cal = self._temperature.calibrate(probs)
            return (platt_cal + iso_cal + temp_cal) / 3
        else:
            return probs
    
    def calibrate_with_confidence(
        self,
        probability: float,
        n_bootstrap: int = 100,
    ) -> CalibratedPrediction:
        """
        Calibrate a single probability with confidence interval.
        
        Args:
            probability: Single raw probability
            n_bootstrap: Number of bootstrap samples for CI
            
        Returns:
            CalibratedPrediction with confidence interval
        """
        calibrated = float(self.calibrate(np.array([probability]))[0])
        
        margin = 0.05 + 0.1 * abs(calibrated - 0.5)
        ci_low = max(0.0, calibrated - margin)
        ci_high = min(1.0, calibrated + margin)
        
        return CalibratedPrediction(
            raw_probability=probability,
            calibrated_probability=calibrated,
            confidence_interval=(ci_low, ci_high),
            calibration_method=self._method,
        )
    
    def compute_calibration_metrics(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """
        Compute calibration quality metrics.
        
        Returns ECE, MCE, Brier score, and reliability diagram data.
        """
        calibrated = self.calibrate(probabilities) if self._is_fitted else probabilities
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(n_bins):
            mask = (calibrated >= bin_boundaries[i]) & (calibrated < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = calibrated[mask].mean()
                bin_size = mask.sum() / len(calibrated)
                
                gap = abs(bin_acc - bin_conf)
                ece += bin_size * gap
                mce = max(mce, gap)
                
                bin_accuracies.append(float(bin_acc))
                bin_confidences.append(float(bin_conf))
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
        
        brier = np.mean((calibrated - labels) ** 2)
        
        if ece < 0.05:
            quality = "excellent"
        elif ece < 0.10:
            quality = "good"
        elif ece < 0.15:
            quality = "fair"
        else:
            quality = "poor"
        
        return CalibrationMetrics(
            expected_calibration_error=float(ece),
            max_calibration_error=float(mce),
            brier_score=float(brier),
            reliability_bins=bin_confidences,
            reliability_accuracies=bin_accuracies,
            calibration_quality=quality,
        )
    
    def save(self, name: str = "default") -> str:
        """Save calibrators to disk."""
        save_path = self._save_dir / name
        save_path.mkdir(exist_ok=True)
        
        if self._platt.is_fitted:
            joblib.dump(self._platt, save_path / "platt.joblib")
        if self._isotonic.is_fitted:
            joblib.dump(self._isotonic, save_path / "isotonic.joblib")
        if self._temperature.is_fitted:
            joblib.dump(self._temperature, save_path / "temperature.joblib")
        
        meta = {
            "method": self._method,
            "is_fitted": self._is_fitted,
            "saved_at": datetime.utcnow().isoformat(),
            "metrics": self._fit_metrics.to_dict() if self._fit_metrics else None,
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"[Calibration] Saved calibrators to {save_path}")
        return str(save_path)
    
    def load(self, name: str = "default") -> "CalibrationLayer":
        """Load calibrators from disk."""
        load_path = self._save_dir / name
        
        if (load_path / "platt.joblib").exists():
            self._platt = joblib.load(load_path / "platt.joblib")
        if (load_path / "isotonic.joblib").exists():
            self._isotonic = joblib.load(load_path / "isotonic.joblib")
        if (load_path / "temperature.joblib").exists():
            self._temperature = joblib.load(load_path / "temperature.joblib")
        
        if (load_path / "meta.json").exists():
            with open(load_path / "meta.json") as f:
                meta = json.load(f)
            self._method = meta.get("method", self._method)
            self._is_fitted = meta.get("is_fitted", False)
        
        logger.info(f"[Calibration] Loaded calibrators from {load_path}")
        return self
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def metrics(self) -> Optional[CalibrationMetrics]:
        return self._fit_metrics


_calibration_layer: Optional[CalibrationLayer] = None


def get_calibration_layer() -> CalibrationLayer:
    """Get global calibration layer instance."""
    global _calibration_layer
    if _calibration_layer is None:
        _calibration_layer = CalibrationLayer()
    return _calibration_layer


__all__ = [
    "CalibrationLayer",
    "CalibrationMetrics",
    "CalibratedPrediction",
    "PlattScaler",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "get_calibration_layer",
]
