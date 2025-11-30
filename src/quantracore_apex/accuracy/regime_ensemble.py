"""
Regime-Gated Ensemble - Different Models for Different Market Conditions.

Provides:
- Regime detection and classification
- Specialized models per regime (trending, choppy, volatile, squeeze)
- Dynamic model routing based on market conditions
- Ensemble combining with regime-weighted averaging

The key insight: a model trained on trending markets performs poorly in choppy
conditions, and vice versa. This ensemble routes predictions to the right specialist.
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
from enum import Enum
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    SQUEEZE = "squeeze"
    VOLATILE = "volatile"
    CRASH = "crash"


@dataclass
class RegimeDetection:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float
    regime_probabilities: Dict[str, float]
    features_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


@dataclass
class EnsemblePrediction:
    """Prediction from regime-gated ensemble."""
    primary_prediction: float
    regime: MarketRegime
    regime_confidence: float
    specialist_predictions: Dict[str, float]
    weights_used: Dict[str, float]
    uncertainty: float
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


class RegimeDetector:
    """
    Detects current market regime from features.
    
    Uses a combination of:
    - Trend strength (ADX-like)
    - Volatility regime
    - Compression detection
    - Directional consistency
    """
    
    REGIME_FEATURES = [
        "trend_strength",
        "volatility_ratio",
        "compression_score",
        "directional_consistency",
        "range_expansion",
    ]
    
    def __init__(self):
        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._encoder = LabelEncoder()
        self._is_fitted = False
    
    def fit(
        self,
        features: np.ndarray,
        regimes: np.ndarray,
    ) -> "RegimeDetector":
        """
        Train regime detector.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            regimes: Regime labels
        """
        X_scaled = self._scaler.fit_transform(features)
        y_encoded = self._encoder.fit_transform(regimes)
        
        self._model.fit(X_scaled, y_encoded)
        self._is_fitted = True
        
        logger.info(f"[RegimeDetector] Trained on {len(features)} samples")
        
        return self
    
    def detect(self, features: np.ndarray) -> RegimeDetection:
        """
        Detect regime from features.
        
        Args:
            features: Feature vector (1, n_features) or (n_features,)
            
        Returns:
            RegimeDetection with predicted regime and confidence
        """
        if not self._is_fitted:
            return RegimeDetection(
                regime=MarketRegime.CHOP,
                confidence=0.5,
                regime_probabilities={r.value: 0.2 for r in MarketRegime},
                features_used=self.REGIME_FEATURES,
            )
        
        X = np.atleast_2d(features)
        X_scaled = self._scaler.transform(X)
        
        pred_encoded = self._model.predict(X_scaled)[0]
        pred_label = self._encoder.inverse_transform([pred_encoded])[0]
        
        proba = self._model.predict_proba(X_scaled)[0]
        confidence = float(np.max(proba))
        
        regime_probs = {
            self._encoder.classes_[i]: float(proba[i])
            for i in range(len(proba))
        }
        
        return RegimeDetection(
            regime=MarketRegime(pred_label),
            confidence=confidence,
            regime_probabilities=regime_probs,
            features_used=self.REGIME_FEATURES,
        )
    
    def detect_from_price_action(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> RegimeDetection:
        """
        Detect regime directly from price data.
        
        Computes regime features internally.
        """
        if len(closes) < 20:
            return RegimeDetection(
                regime=MarketRegime.CHOP,
                confidence=0.3,
                regime_probabilities={r.value: 0.17 for r in MarketRegime},
                features_used=self.REGIME_FEATURES,
            )
        
        returns = np.diff(closes) / closes[:-1]
        
        up_moves = sum(1 for r in returns[-20:] if r > 0)
        directional_consistency = up_moves / 20
        
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0
        
        recent_vol = np.std(returns[-5:])
        longer_vol = np.std(returns[-20:])
        volatility_ratio = recent_vol / longer_vol if longer_vol > 0 else 1.0
        
        ranges = highs - lows
        avg_range = np.mean(ranges[-20:])
        recent_range = np.mean(ranges[-5:])
        compression_score = 1.0 - (recent_range / avg_range) if avg_range > 0 else 0.0
        
        range_expansion = recent_range / avg_range if avg_range > 0 else 1.0
        
        if trend_strength > 0.02 and directional_consistency > 0.6:
            regime = MarketRegime.TREND_UP
            confidence = 0.7 + 0.3 * directional_consistency
        elif trend_strength < -0.02 and directional_consistency < 0.4:
            regime = MarketRegime.TREND_DOWN
            confidence = 0.7 + 0.3 * (1 - directional_consistency)
        elif compression_score > 0.5:
            regime = MarketRegime.SQUEEZE
            confidence = 0.6 + 0.3 * compression_score
        elif volatility_ratio > 1.5:
            regime = MarketRegime.VOLATILE
            confidence = 0.6 + 0.2 * min(volatility_ratio - 1, 1)
        else:
            regime = MarketRegime.CHOP
            confidence = 0.5
        
        regime_probs = {r.value: 0.1 for r in MarketRegime}
        regime_probs[regime.value] = confidence
        
        return RegimeDetection(
            regime=regime,
            confidence=min(0.95, confidence),
            regime_probabilities=regime_probs,
            features_used=self.REGIME_FEATURES,
        )
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class RegimeSpecialist:
    """
    Specialist model for a specific market regime.
    
    Each specialist is optimized for its regime conditions.
    """
    
    def __init__(
        self,
        regime: MarketRegime,
        n_estimators: int = 150,
        max_depth: int = 5,
    ):
        self.regime = regime
        
        self._regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self._classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._training_samples = 0
    
    def fit(
        self,
        X: np.ndarray,
        y_score: np.ndarray,
        y_direction: np.ndarray,
    ) -> "RegimeSpecialist":
        """
        Train specialist on regime-specific data.
        
        Args:
            X: Feature matrix
            y_score: Target scores (for regression)
            y_direction: Target directions (for classification)
        """
        X_scaled = self._scaler.fit_transform(X)
        
        self._regressor.fit(X_scaled, y_score)
        
        if len(np.unique(y_direction)) > 1:
            self._classifier.fit(X_scaled, y_direction)
        
        self._is_fitted = True
        self._training_samples = len(X)
        
        logger.info(f"[Specialist:{self.regime.value}] Trained on {len(X)} samples")
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[float, float, float]:
        """
        Make predictions.
        
        Returns:
            (score_prediction, direction_probability, confidence)
        """
        if not self._is_fitted:
            return (50.0, 0.5, 0.3)
        
        X_scaled = self._scaler.transform(np.atleast_2d(X))
        
        score_pred = float(self._regressor.predict(X_scaled)[0])
        
        try:
            dir_proba = float(self._classifier.predict_proba(X_scaled)[0, 1])
        except:
            dir_proba = 0.5
        
        confidence = min(0.9, 0.5 + self._training_samples / 1000)
        
        return (score_pred, dir_proba, confidence)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class RegimeGatedEnsemble:
    """
    Ensemble that routes predictions through regime-specific specialists.
    
    Usage:
        ensemble = RegimeGatedEnsemble()
        
        # Train on labeled data
        ensemble.fit(X_train, y_train, regimes_train)
        
        # Predict with regime routing
        prediction = ensemble.predict(X_test, regime_features)
    """
    
    def __init__(
        self,
        save_dir: str = "models/ensemble",
    ):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        self._detector = RegimeDetector()
        
        self._specialists: Dict[MarketRegime, RegimeSpecialist] = {
            regime: RegimeSpecialist(regime)
            for regime in MarketRegime
        }
        
        self._global_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
        )
        self._global_scaler = StandardScaler()
        
        self._is_fitted = False
        self._regime_weights = {r: 1.0 for r in MarketRegime}
    
    def fit(
        self,
        X: np.ndarray,
        y_score: np.ndarray,
        y_direction: np.ndarray,
        regimes: np.ndarray,
    ) -> "RegimeGatedEnsemble":
        """
        Train ensemble on labeled data.
        
        Args:
            X: Feature matrix
            y_score: Target scores
            y_direction: Target directions (0 or 1)
            regimes: Regime labels for each sample
        """
        X_scaled = self._global_scaler.fit_transform(X)
        self._global_model.fit(X_scaled, y_score)
        
        regime_features = self._extract_regime_features(X)
        if len(regime_features) >= 50:
            self._detector.fit(regime_features, regimes)
        
        for regime in MarketRegime:
            mask = regimes == regime.value
            if mask.sum() >= 20:
                self._specialists[regime].fit(
                    X[mask],
                    y_score[mask],
                    y_direction[mask],
                )
        
        self._compute_regime_weights(X, y_score, regimes)
        
        self._is_fitted = True
        logger.info("[RegimeGatedEnsemble] Training complete")
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        regime_features: Optional[np.ndarray] = None,
    ) -> EnsemblePrediction:
        """
        Make prediction with regime routing.
        
        Args:
            X: Feature vector
            regime_features: Optional explicit regime features
            
        Returns:
            EnsemblePrediction with ensemble result and uncertainty
        """
        if not self._is_fitted:
            return EnsemblePrediction(
                primary_prediction=50.0,
                regime=MarketRegime.CHOP,
                regime_confidence=0.5,
                specialist_predictions={},
                weights_used={},
                uncertainty=0.3,
            )
        
        X_2d = np.atleast_2d(X)
        
        if regime_features is not None:
            detection = self._detector.detect(regime_features)
        else:
            detection = RegimeDetection(
                regime=MarketRegime.CHOP,
                confidence=0.5,
                regime_probabilities={r.value: 0.17 for r in MarketRegime},
                features_used=[],
            )
        
        X_scaled = self._global_scaler.transform(X_2d)
        global_pred = float(self._global_model.predict(X_scaled)[0])
        
        specialist_preds = {}
        for regime, specialist in self._specialists.items():
            if specialist.is_fitted:
                score, _, _ = specialist.predict(X_2d[0])
                specialist_preds[regime.value] = score
        
        current_regime = detection.regime
        regime_conf = detection.regime_confidence
        
        weights = {}
        weighted_sum = 0.0
        weight_total = 0.0
        
        weights["global"] = 0.3
        weighted_sum += global_pred * 0.3
        weight_total += 0.3
        
        if current_regime.value in specialist_preds:
            specialist_weight = 0.5 * regime_conf
            weights[current_regime.value] = specialist_weight
            weighted_sum += specialist_preds[current_regime.value] * specialist_weight
            weight_total += specialist_weight
        
        for regime_val, pred in specialist_preds.items():
            if regime_val != current_regime.value:
                regime_prob = detection.regime_probabilities.get(regime_val, 0.1)
                if regime_prob > 0.15:
                    w = 0.2 * regime_prob
                    weights[regime_val] = w
                    weighted_sum += pred * w
                    weight_total += w
        
        final_pred = weighted_sum / weight_total if weight_total > 0 else global_pred
        
        if len(specialist_preds) > 1:
            pred_std = np.std(list(specialist_preds.values()))
            uncertainty = min(0.5, pred_std / 20)
        else:
            uncertainty = 0.3
        
        return EnsemblePrediction(
            primary_prediction=final_pred,
            regime=current_regime,
            regime_confidence=regime_conf,
            specialist_predictions=specialist_preds,
            weights_used=weights,
            uncertainty=uncertainty,
        )
    
    def _extract_regime_features(self, X: np.ndarray) -> np.ndarray:
        """Extract regime-specific features from general features."""
        if X.shape[1] >= 5:
            return X[:, :5]
        return X
    
    def _compute_regime_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regimes: np.ndarray,
    ) -> None:
        """Compute optimal weights for each regime based on performance."""
        for regime in MarketRegime:
            mask = regimes == regime.value
            if mask.sum() < 10:
                continue
            
            specialist = self._specialists[regime]
            if not specialist.is_fitted:
                continue
            
            errors = []
            for i in np.where(mask)[0]:
                pred, _, _ = specialist.predict(X[i:i+1])
                errors.append(abs(pred - y[i]))
            
            avg_error = np.mean(errors)
            self._regime_weights[regime] = 1.0 / (1.0 + avg_error / 10)
    
    def save(self, name: str = "default") -> str:
        """Save ensemble to disk."""
        save_path = self._save_dir / name
        save_path.mkdir(exist_ok=True)
        
        joblib.dump(self._detector, save_path / "detector.joblib")
        joblib.dump(self._global_model, save_path / "global_model.joblib")
        joblib.dump(self._global_scaler, save_path / "global_scaler.joblib")
        
        for regime, specialist in self._specialists.items():
            if specialist.is_fitted:
                joblib.dump(specialist, save_path / f"specialist_{regime.value}.joblib")
        
        meta = {
            "is_fitted": self._is_fitted,
            "regime_weights": {r.value: w for r, w in self._regime_weights.items()},
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"[RegimeGatedEnsemble] Saved to {save_path}")
        return str(save_path)
    
    def load(self, name: str = "default") -> "RegimeGatedEnsemble":
        """Load ensemble from disk."""
        load_path = self._save_dir / name
        
        if (load_path / "detector.joblib").exists():
            self._detector = joblib.load(load_path / "detector.joblib")
        if (load_path / "global_model.joblib").exists():
            self._global_model = joblib.load(load_path / "global_model.joblib")
        if (load_path / "global_scaler.joblib").exists():
            self._global_scaler = joblib.load(load_path / "global_scaler.joblib")
        
        for regime in MarketRegime:
            specialist_path = load_path / f"specialist_{regime.value}.joblib"
            if specialist_path.exists():
                self._specialists[regime] = joblib.load(specialist_path)
        
        if (load_path / "meta.json").exists():
            with open(load_path / "meta.json") as f:
                meta = json.load(f)
            self._is_fitted = meta.get("is_fitted", False)
        
        logger.info(f"[RegimeGatedEnsemble] Loaded from {load_path}")
        return self
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


_ensemble_instance: Optional[RegimeGatedEnsemble] = None


def get_regime_ensemble() -> RegimeGatedEnsemble:
    """Get global ensemble instance."""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = RegimeGatedEnsemble()
    return _ensemble_instance


__all__ = [
    "RegimeGatedEnsemble",
    "RegimeDetector",
    "RegimeSpecialist",
    "MarketRegime",
    "RegimeDetection",
    "EnsemblePrediction",
    "get_regime_ensemble",
]
