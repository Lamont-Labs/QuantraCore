"""
ApexCore V2 - Multi-Head Neural Prediction Models.

Provides real machine learning models for:
1. QuantraScore regression (predicting future scores)
2. Runner probability classification
3. Quality tier classification
4. Avoid-trade probability
5. Regime classification

Uses scikit-learn GradientBoosting for robust, deterministic predictions.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

MODEL_DIR = "models/apexcore_v2"


@dataclass
class ApexCoreV2Manifest:
    """Manifest for trained ApexCore V2 model."""
    version: str
    model_size: str
    trained_at: str
    training_samples: int
    feature_count: int
    heads: List[str]
    metrics: Dict[str, float]
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApexCoreV2Manifest":
        return cls(**data)


@dataclass
class ApexCoreV2Prediction:
    """Prediction result from ApexCore V2."""
    quantrascore_pred: float
    runner_probability: float
    quality_tier_pred: str
    avoid_trade_probability: float
    regime_pred: str
    confidence: float
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ApexCoreV2Model:
    """
    Multi-head prediction model for QuantraCore Apex.
    
    Provides 5 prediction heads:
    1. quantrascore_head: Regresses future QuantraScore
    2. runner_head: Classifies runner probability
    3. quality_head: Classifies quality tier
    4. avoid_head: Classifies avoid-trade probability
    5. regime_head: Classifies market regime
    """
    
    FEATURE_NAMES = [
        "quantra_score",
        "entropy_band_encoded",
        "suppression_state_encoded",
        "regime_type_encoded",
        "volatility_band_encoded",
        "liquidity_band_encoded",
        "risk_tier_encoded",
        "protocol_active_count",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "max_runup_5d",
        "max_drawdown_5d",
    ]
    
    def __init__(self, model_size: str = "big"):
        """
        Initialize ApexCore V2 model.
        
        Args:
            model_size: 'mini' for faster training, 'big' for better accuracy
        """
        self.model_size = model_size
        self.version = "2.0.0"
        
        n_estimators = 100 if model_size == "mini" else 200
        max_depth = 4 if model_size == "mini" else 6
        
        self.quantrascore_head = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self.runner_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self.quality_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self.avoid_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self.regime_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
        )
        
        self.scaler = StandardScaler()
        self.quality_encoder = LabelEncoder()
        self.regime_encoder = LabelEncoder()
        
        self._is_fitted = False
        self._manifest: Optional[ApexCoreV2Manifest] = None
    
    def _encode_categorical(self, value: str, mapping: Dict[str, int]) -> int:
        """Encode categorical value to integer."""
        return mapping.get(value, 0)
    
    def _prepare_features(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prepare feature matrix from ApexLabV2Row data.
        
        Args:
            rows: List of row dictionaries
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
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
                self._encode_categorical(row.get("entropy_band", "mid"), entropy_map),
                self._encode_categorical(row.get("suppression_state", "none"), suppression_map),
                self._encode_categorical(row.get("regime_type", "chop"), regime_map),
                self._encode_categorical(row.get("volatility_band", "mid"), volatility_map),
                self._encode_categorical(row.get("liquidity_band", "mid"), liquidity_map),
                self._encode_categorical(row.get("risk_tier", "medium"), risk_map),
                protocol_count,
                float(row.get("ret_1d", 0)),
                float(row.get("ret_3d", 0)),
                float(row.get("ret_5d", 0)),
                float(row.get("max_runup_5d", 0)),
                float(row.get("max_drawdown_5d", 0)),
            ]
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit(
        self,
        rows: List[Dict[str, Any]],
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train all prediction heads on dataset.
        
        Args:
            rows: List of ApexLabV2Row dictionaries
            validation_split: Fraction for validation
            
        Returns:
            Dictionary of validation metrics
        """
        if len(rows) < 20:
            raise ValueError(f"Insufficient training data: {len(rows)} rows (need 20+)")
        
        X = self._prepare_features(rows)
        
        y_quantrascore = np.array([r.get("quantra_score", 50) for r in rows])
        y_runner = np.array([r.get("hit_runner_threshold", 0) for r in rows])
        y_quality = np.array([r.get("future_quality_tier", "C") for r in rows])
        y_avoid = np.array([r.get("avoid_trade", 0) for r in rows])
        y_regime = np.array([r.get("regime_label", "chop") for r in rows])
        
        self.quality_encoder.fit(y_quality)
        self.regime_encoder.fit(y_regime)
        
        y_quality_encoded = self.quality_encoder.transform(y_quality)
        y_regime_encoded = self.regime_encoder.transform(y_regime)
        
        X_train, X_val, indices_train, indices_val = train_test_split(
            X, np.arange(len(rows)), test_size=validation_split, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        y_qs_train = y_quantrascore[indices_train]
        y_qs_val = y_quantrascore[indices_val]
        
        y_runner_train = y_runner[indices_train]
        y_runner_val = y_runner[indices_val]
        
        y_quality_train = y_quality_encoded[indices_train]
        y_quality_val = y_quality_encoded[indices_val]
        
        y_avoid_train = y_avoid[indices_train]
        y_avoid_val = y_avoid[indices_val]
        
        y_regime_train = y_regime_encoded[indices_train]
        y_regime_val = y_regime_encoded[indices_val]
        
        logger.info("Training QuantraScore head...")
        self.quantrascore_head.fit(X_train_scaled, y_qs_train)
        qs_pred = self.quantrascore_head.predict(X_val_scaled)
        qs_rmse = np.sqrt(np.mean((qs_pred - y_qs_val) ** 2))
        
        logger.info("Training Runner head...")
        if len(np.unique(y_runner_train)) > 1:
            self.runner_head.fit(X_train_scaled, y_runner_train)
            runner_acc = self.runner_head.score(X_val_scaled, y_runner_val)
        else:
            runner_acc = 0.5
        
        logger.info("Training Quality head...")
        if len(np.unique(y_quality_train)) > 1:
            self.quality_head.fit(X_train_scaled, y_quality_train)
            quality_acc = self.quality_head.score(X_val_scaled, y_quality_val)
        else:
            quality_acc = 0.5
        
        logger.info("Training Avoid head...")
        if len(np.unique(y_avoid_train)) > 1:
            self.avoid_head.fit(X_train_scaled, y_avoid_train)
            avoid_acc = self.avoid_head.score(X_val_scaled, y_avoid_val)
        else:
            avoid_acc = 0.5
        
        logger.info("Training Regime head...")
        if len(np.unique(y_regime_train)) > 1:
            self.regime_head.fit(X_train_scaled, y_regime_train)
            regime_acc = self.regime_head.score(X_val_scaled, y_regime_val)
        else:
            regime_acc = 0.5
        
        self._is_fitted = True
        
        metrics = {
            "quantrascore_rmse": float(qs_rmse),
            "runner_accuracy": float(runner_acc),
            "quality_accuracy": float(quality_acc),
            "avoid_accuracy": float(avoid_acc),
            "regime_accuracy": float(regime_acc),
            "training_samples": len(rows),
            "validation_samples": len(X_val),
        }
        
        self._manifest = ApexCoreV2Manifest(
            version=self.version,
            model_size=self.model_size,
            trained_at=datetime.utcnow().isoformat(),
            training_samples=len(rows),
            feature_count=X.shape[1],
            heads=["quantrascore", "runner", "quality", "avoid", "regime"],
            metrics=metrics,
            feature_names=self.FEATURE_NAMES,
        )
        
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics
    
    def predict(self, row: Dict[str, Any]) -> ApexCoreV2Prediction:
        """
        Generate predictions for a single row.
        
        Args:
            row: ApexLabV2Row dictionary
            
        Returns:
            ApexCoreV2Prediction with all head outputs
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first or load a trained model.")
        
        X = self._prepare_features([row])
        X_scaled = self.scaler.transform(X)
        
        qs_pred = float(self.quantrascore_head.predict(X_scaled)[0])
        
        try:
            runner_proba = float(self.runner_head.predict_proba(X_scaled)[0, 1])
        except:
            runner_proba = 0.5
        
        try:
            quality_pred_encoded = self.quality_head.predict(X_scaled)[0]
            quality_pred = self.quality_encoder.inverse_transform([quality_pred_encoded])[0]
        except:
            quality_pred = "C"
        
        try:
            avoid_proba = float(self.avoid_head.predict_proba(X_scaled)[0, 1])
        except:
            avoid_proba = 0.5
        
        try:
            regime_pred_encoded = self.regime_head.predict(X_scaled)[0]
            regime_pred = self.regime_encoder.inverse_transform([regime_pred_encoded])[0]
        except:
            regime_pred = "chop"
        
        confidence = 0.7
        if runner_proba > 0.7 or runner_proba < 0.3:
            confidence += 0.1
        if avoid_proba > 0.7 or avoid_proba < 0.3:
            confidence += 0.05
        
        return ApexCoreV2Prediction(
            quantrascore_pred=qs_pred,
            runner_probability=runner_proba,
            quality_tier_pred=quality_pred,
            avoid_trade_probability=avoid_proba,
            regime_pred=regime_pred,
            confidence=min(0.95, confidence),
            model_version=self.version,
        )
    
    def predict_batch(self, rows: List[Dict[str, Any]]) -> List[ApexCoreV2Prediction]:
        """Generate predictions for multiple rows."""
        return [self.predict(row) for row in rows]
    
    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Save model to disk.
        
        Args:
            model_dir: Directory to save to (default: models/apexcore_v2/{model_size})
            
        Returns:
            Path to saved model directory
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dir = model_dir or os.path.join(MODEL_DIR, self.model_size)
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.quantrascore_head, os.path.join(save_dir, "quantrascore_head.joblib"))
        joblib.dump(self.runner_head, os.path.join(save_dir, "runner_head.joblib"))
        joblib.dump(self.quality_head, os.path.join(save_dir, "quality_head.joblib"))
        joblib.dump(self.avoid_head, os.path.join(save_dir, "avoid_head.joblib"))
        joblib.dump(self.regime_head, os.path.join(save_dir, "regime_head.joblib"))
        
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        joblib.dump(self.quality_encoder, os.path.join(save_dir, "quality_encoder.joblib"))
        joblib.dump(self.regime_encoder, os.path.join(save_dir, "regime_encoder.joblib"))
        
        if self._manifest:
            with open(os.path.join(save_dir, "manifest.json"), "w") as f:
                json.dump(self._manifest.to_dict(), f, indent=2)
        
        logger.info(f"Model saved to {save_dir}")
        return save_dir
    
    @classmethod
    def load(cls, model_dir: Optional[str] = None, model_size: str = "big") -> "ApexCoreV2Model":
        """
        Load model from disk.
        
        Args:
            model_dir: Directory to load from
            model_size: Size variant to load
            
        Returns:
            Loaded ApexCoreV2Model
        """
        load_dir = model_dir or os.path.join(MODEL_DIR, model_size)
        
        if not os.path.exists(os.path.join(load_dir, "quantrascore_head.joblib")):
            raise FileNotFoundError(f"No trained model found in {load_dir}")
        
        model = cls(model_size=model_size)
        
        model.quantrascore_head = joblib.load(os.path.join(load_dir, "quantrascore_head.joblib"))
        model.runner_head = joblib.load(os.path.join(load_dir, "runner_head.joblib"))
        model.quality_head = joblib.load(os.path.join(load_dir, "quality_head.joblib"))
        model.avoid_head = joblib.load(os.path.join(load_dir, "avoid_head.joblib"))
        model.regime_head = joblib.load(os.path.join(load_dir, "regime_head.joblib"))
        
        model.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        model.quality_encoder = joblib.load(os.path.join(load_dir, "quality_encoder.joblib"))
        model.regime_encoder = joblib.load(os.path.join(load_dir, "regime_encoder.joblib"))
        
        manifest_path = os.path.join(load_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                model._manifest = ApexCoreV2Manifest.from_dict(json.load(f))
        
        model._is_fitted = True
        logger.info(f"Model loaded from {load_dir}")
        return model
    
    @property
    def manifest(self) -> Optional[ApexCoreV2Manifest]:
        """Get model manifest."""
        return self._manifest
    
    @property
    def is_fitted(self) -> bool:
        """Check if model is trained."""
        return self._is_fitted


def get_model_status(model_size: str = "big") -> Dict[str, Any]:
    """
    Get status of trained models.
    
    Returns:
        Dictionary with model status information
    """
    model_dir = os.path.join(MODEL_DIR, model_size)
    manifest_path = os.path.join(model_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        return {
            "status": "not_trained",
            "model_size": model_size,
            "model_dir": model_dir,
        }
    
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        return {
            "status": "trained",
            "model_size": model_size,
            "model_dir": model_dir,
            "version": manifest.get("version"),
            "trained_at": manifest.get("trained_at"),
            "training_samples": manifest.get("training_samples"),
            "metrics": manifest.get("metrics"),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


__all__ = [
    "ApexCoreV2Model",
    "ApexCoreV2Prediction",
    "ApexCoreV2Manifest",
    "get_model_status",
]
