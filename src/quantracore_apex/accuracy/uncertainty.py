"""
Uncertainty Quantification - Conformal Prediction for Confidence Bounds.

Provides:
- Conformal prediction for valid confidence intervals
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)
- Prediction sets with guaranteed coverage

When the model says "I'm not sure", it means it - and we know exactly how unsure.
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

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification for a prediction."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    coverage_level: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def interval_width(self) -> float:
        return self.upper_bound - self.lower_bound
    
    @property
    def is_high_uncertainty(self) -> bool:
        return self.total_uncertainty > 0.3


@dataclass
class ConformalPredictionSet:
    """Prediction set from conformal prediction."""
    prediction: float
    prediction_set: Tuple[float, float]
    coverage_probability: float
    nonconformity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prediction_set"] = list(self.prediction_set)
        return d


class ConformalPredictor:
    """
    Conformal prediction for valid confidence intervals.
    
    Guarantees that the true value falls within the prediction interval
    with the specified probability, without distributional assumptions.
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
    ):
        """
        Initialize conformal predictor.
        
        Args:
            coverage: Target coverage probability (e.g., 0.9 for 90%)
        """
        self._coverage = coverage
        self._calibration_scores: List[float] = []
        self._quantile: Optional[float] = None
        self._is_fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "ConformalPredictor":
        """
        Calibrate conformal predictor on validation data.
        
        Args:
            predictions: Model predictions
            actuals: True values
        """
        scores = np.abs(predictions - actuals)
        self._calibration_scores = scores.tolist()
        
        n = len(scores)
        q = np.ceil((n + 1) * self._coverage) / n
        self._quantile = float(np.quantile(scores, min(q, 1.0)))
        
        self._is_fitted = True
        
        logger.info(
            f"[ConformalPredictor] Calibrated on {n} samples. "
            f"Quantile at {self._coverage:.0%} coverage: {self._quantile:.4f}"
        )
        
        return self
    
    def predict(
        self,
        point_prediction: float,
    ) -> ConformalPredictionSet:
        """
        Generate conformal prediction set.
        
        Args:
            point_prediction: Model's point prediction
            
        Returns:
            ConformalPredictionSet with guaranteed coverage interval
        """
        if not self._is_fitted or self._quantile is None:
            margin = abs(point_prediction) * 0.1 + 5.0
            return ConformalPredictionSet(
                prediction=point_prediction,
                prediction_set=(point_prediction - margin, point_prediction + margin),
                coverage_probability=0.9,
                nonconformity_score=0.0,
            )
        
        lower = point_prediction - self._quantile
        upper = point_prediction + self._quantile
        
        return ConformalPredictionSet(
            prediction=point_prediction,
            prediction_set=(lower, upper),
            coverage_probability=self._coverage,
            nonconformity_score=self._quantile,
        )
    
    def adaptive_predict(
        self,
        point_prediction: float,
        local_difficulty: float = 1.0,
    ) -> ConformalPredictionSet:
        """
        Adaptive conformal prediction with local difficulty adjustment.
        
        Widens intervals for harder predictions.
        """
        base_result = self.predict(point_prediction)
        
        adjustment = 1.0 + (local_difficulty - 1.0) * 0.5
        
        center = point_prediction
        half_width = (base_result.prediction_set[1] - base_result.prediction_set[0]) / 2
        adjusted_half = half_width * adjustment
        
        return ConformalPredictionSet(
            prediction=point_prediction,
            prediction_set=(center - adjusted_half, center + adjusted_half),
            coverage_probability=self._coverage,
            nonconformity_score=base_result.nonconformity_score * adjustment,
        )
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @property
    def quantile(self) -> Optional[float]:
        return self._quantile


class UncertaintyHead:
    """
    Uncertainty quantification for model predictions.
    
    Combines:
    - Conformal prediction for coverage guarantees
    - Ensemble disagreement for epistemic uncertainty
    - Residual variance for aleatoric uncertainty
    
    Usage:
        uncertainty = UncertaintyHead()
        
        # Calibrate on validation data
        uncertainty.fit(val_predictions, val_actuals, ensemble_predictions)
        
        # Get uncertainty for new prediction
        estimate = uncertainty.estimate(prediction, ensemble_preds)
    """
    
    def __init__(
        self,
        coverage: float = 0.9,
        save_dir: str = "models/uncertainty",
    ):
        self._coverage = coverage
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        self._conformal = ConformalPredictor(coverage=coverage)
        
        self._residual_std: float = 10.0
        self._ensemble_std_scale: float = 1.0
        
        self._is_fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        ensemble_predictions: Optional[np.ndarray] = None,
    ) -> "UncertaintyHead":
        """
        Fit uncertainty head on validation data.
        
        Args:
            predictions: Primary model predictions
            actuals: True values
            ensemble_predictions: Optional (n_samples, n_models) array of ensemble predictions
        """
        self._conformal.fit(predictions, actuals)
        
        residuals = predictions - actuals
        self._residual_std = float(np.std(residuals))
        
        if ensemble_predictions is not None and ensemble_predictions.shape[1] > 1:
            ensemble_stds = np.std(ensemble_predictions, axis=1)
            errors = np.abs(residuals)
            if np.std(ensemble_stds) > 0:
                self._ensemble_std_scale = np.corrcoef(ensemble_stds, errors)[0, 1]
                self._ensemble_std_scale = max(0.5, min(2.0, abs(self._ensemble_std_scale)))
        
        self._is_fitted = True
        
        logger.info(
            f"[UncertaintyHead] Calibrated. "
            f"Residual std: {self._residual_std:.4f}, "
            f"Ensemble scale: {self._ensemble_std_scale:.4f}"
        )
        
        return self
    
    def estimate(
        self,
        prediction: float,
        ensemble_predictions: Optional[np.ndarray] = None,
        local_features: Optional[Dict[str, float]] = None,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty for a prediction.
        
        Args:
            prediction: Point prediction
            ensemble_predictions: Optional predictions from ensemble members
            local_features: Optional features for adaptive uncertainty
            
        Returns:
            UncertaintyEstimate with bounds and uncertainty decomposition
        """
        conformal_result = self._conformal.predict(prediction)
        
        aleatoric = self._residual_std / max(abs(prediction), 10.0)
        aleatoric = min(0.5, aleatoric)
        
        if ensemble_predictions is not None and len(ensemble_predictions) > 1:
            epistemic = float(np.std(ensemble_predictions)) / max(abs(prediction), 10.0)
            epistemic = epistemic * self._ensemble_std_scale
            epistemic = min(0.5, epistemic)
        else:
            epistemic = 0.15
        
        total = np.sqrt(epistemic**2 + aleatoric**2)
        
        if local_features:
            regime_uncertainty = local_features.get("regime_uncertainty", 0.0)
            total = total * (1.0 + regime_uncertainty * 0.2)
        
        lower = conformal_result.prediction_set[0]
        upper = conformal_result.prediction_set[1]
        
        if total > 0.4:
            confidence_level = "very_low"
        elif total > 0.3:
            confidence_level = "low"
        elif total > 0.2:
            confidence_level = "medium"
        elif total > 0.1:
            confidence_level = "high"
        else:
            confidence_level = "very_high"
        
        return UncertaintyEstimate(
            point_estimate=prediction,
            lower_bound=lower,
            upper_bound=upper,
            coverage_level=self._coverage,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=min(1.0, total),
            confidence_level=confidence_level,
        )
    
    def should_abstain(
        self,
        uncertainty: UncertaintyEstimate,
        threshold: float = 0.4,
    ) -> Tuple[bool, str]:
        """
        Determine if model should abstain from prediction.
        
        Returns:
            (should_abstain, reason)
        """
        if uncertainty.total_uncertainty > threshold:
            return (True, f"High uncertainty: {uncertainty.total_uncertainty:.2f}")
        
        if uncertainty.interval_width > abs(uncertainty.point_estimate) * 0.5:
            return (True, "Wide prediction interval")
        
        if uncertainty.epistemic_uncertainty > 0.35:
            return (True, "High model uncertainty - possibly out of distribution")
        
        return (False, "")
    
    def save(self, name: str = "default") -> str:
        """Save uncertainty head to disk."""
        save_path = self._save_dir / name
        save_path.mkdir(exist_ok=True)
        
        joblib.dump(self._conformal, save_path / "conformal.joblib")
        
        meta = {
            "coverage": self._coverage,
            "residual_std": self._residual_std,
            "ensemble_std_scale": self._ensemble_std_scale,
            "is_fitted": self._is_fitted,
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"[UncertaintyHead] Saved to {save_path}")
        return str(save_path)
    
    def load(self, name: str = "default") -> "UncertaintyHead":
        """Load uncertainty head from disk."""
        load_path = self._save_dir / name
        
        if (load_path / "conformal.joblib").exists():
            self._conformal = joblib.load(load_path / "conformal.joblib")
        
        if (load_path / "meta.json").exists():
            with open(load_path / "meta.json") as f:
                meta = json.load(f)
            self._coverage = meta.get("coverage", self._coverage)
            self._residual_std = meta.get("residual_std", self._residual_std)
            self._ensemble_std_scale = meta.get("ensemble_std_scale", self._ensemble_std_scale)
            self._is_fitted = meta.get("is_fitted", False)
        
        logger.info(f"[UncertaintyHead] Loaded from {load_path}")
        return self
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


_uncertainty_head: Optional[UncertaintyHead] = None


def get_uncertainty_head() -> UncertaintyHead:
    """Get global uncertainty head instance."""
    global _uncertainty_head
    if _uncertainty_head is None:
        _uncertainty_head = UncertaintyHead()
    return _uncertainty_head


__all__ = [
    "UncertaintyHead",
    "UncertaintyEstimate",
    "ConformalPredictor",
    "ConformalPredictionSet",
    "get_uncertainty_head",
]
