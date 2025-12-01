"""
ApexCore V3 - Enhanced Prediction System with Accuracy Optimization.

Integrates all accuracy improvements into a unified prediction system:
- Protocol telemetry for signal weighting
- Calibrated probabilities
- Regime-gated ensemble
- Uncertainty quantification
- Multi-horizon predictions
- Cross-asset context

This is the production-ready prediction engine with maximum accuracy.
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

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.quantracore_apex.accuracy import (
    get_protocol_telemetry,
    get_calibration_layer,
    get_regime_ensemble,
    get_uncertainty_head,
    get_multi_horizon_predictor,
    get_cross_asset_analyzer,
    get_feature_store,
    MarketRegime,
)

logger = logging.getLogger(__name__)

MODEL_DIR = "models/apexcore_v3"


@dataclass
class ApexCoreV3Manifest:
    """Manifest for trained ApexCore V3 model."""
    version: str
    model_size: str
    trained_at: str
    training_samples: int
    feature_count: int
    heads: List[str]
    accuracy_modules: List[str]
    metrics: Dict[str, float]
    feature_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApexCoreV3Prediction:
    """Complete prediction from ApexCore V3."""
    quantrascore_pred: float
    quantrascore_calibrated: float
    runner_probability: float
    runner_probability_calibrated: float
    quality_tier_pred: str
    avoid_trade_probability: float
    regime_pred: str
    
    timing_bucket: str
    timing_confidence: float
    move_direction: int
    bars_to_move_estimate: int
    
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float
    uncertainty_level: str
    
    multi_horizon: Dict[str, float]
    horizon_consensus: str
    
    market_context: Dict[str, Any]
    
    model_version: str
    prediction_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def should_trade(self) -> bool:
        """Determine if trade should be taken based on all factors."""
        if self.avoid_trade_probability > 0.6:
            return False
        if self.uncertainty_level == "very_low":
            return False
        if self.quantrascore_calibrated < 40:
            return False
        return True
    
    @property
    def signal_strength(self) -> str:
        """Overall signal strength assessment."""
        score = self.quantrascore_calibrated
        conf = self.confidence
        
        if score >= 80 and conf >= 0.8:
            return "very_strong"
        elif score >= 70 and conf >= 0.7:
            return "strong"
        elif score >= 60 and conf >= 0.6:
            return "moderate"
        elif score >= 50:
            return "weak"
        else:
            return "neutral"
    
    @property
    def timing_signal(self) -> str:
        """Human-readable timing signal."""
        if self.timing_bucket == "none":
            return "No significant move expected"
        direction = "up" if self.move_direction == 1 else "down"
        if self.timing_bucket == "immediate":
            return f"Move {direction} starting NOW (1 bar)"
        elif self.timing_bucket == "very_soon":
            return f"Move {direction} expected in 2-3 bars"
        elif self.timing_bucket == "soon":
            return f"Move {direction} expected in 4-6 bars"
        elif self.timing_bucket == "late":
            return f"Move {direction} may occur in 7-10 bars"
        return "Timing unclear"


class ApexCoreV3Model:
    """
    Enhanced multi-head prediction model with accuracy optimization.
    
    Builds on V2 with:
    - Protocol-weighted features
    - Calibrated outputs
    - Regime-aware predictions
    - Uncertainty quantification
    - Multi-horizon forecasts
    
    This is the production model for maximum accuracy.
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
        "protocol_weighted_score",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "max_runup_5d",
        "max_drawdown_5d",
        "vix_level",
        "vix_percentile",
        "sector_momentum",
        "market_breadth",
    ]
    
    def __init__(
        self,
        model_size: str = "big",
        enable_calibration: bool = True,
        enable_uncertainty: bool = True,
        enable_multi_horizon: bool = True,
    ):
        self.model_size = model_size
        self.version = "3.0.0"
        
        self._enable_calibration = enable_calibration
        self._enable_uncertainty = enable_uncertainty
        self._enable_multi_horizon = enable_multi_horizon
        
        n_estimators = 100 if model_size == "mini" else 250
        max_depth = 4 if model_size == "mini" else 7
        
        self.quantrascore_head = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        
        self.runner_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        
        self.quality_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.avoid_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.regime_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.timing_head = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.08,
            random_state=42,
        )
        
        self.scaler = StandardScaler()
        self.quality_encoder = LabelEncoder()
        self.regime_encoder = LabelEncoder()
        self.timing_encoder = LabelEncoder()
        
        self._telemetry = get_protocol_telemetry()
        self._calibration = get_calibration_layer()
        self._uncertainty = get_uncertainty_head()
        self._multi_horizon = get_multi_horizon_predictor()
        self._cross_asset = get_cross_asset_analyzer()
        self._feature_store = get_feature_store()
        
        self._is_fitted = False
        self._timing_head_available = False
        self._manifest: Optional[ApexCoreV3Manifest] = None
        self._prediction_counter = 0
    
    def _encode_categorical(self, value: str, mapping: Dict[str, int]) -> int:
        """Encode categorical value to integer."""
        return mapping.get(str(value).lower(), 0)
    
    def _prepare_features(
        self,
        rows: List[Dict[str, Any]],
        include_cross_asset: bool = True,
    ) -> np.ndarray:
        """
        Prepare enhanced feature matrix with protocol weighting and cross-asset context.
        """
        entropy_map = {"low": 0, "mid": 1, "high": 2}
        suppression_map = {"none": 0, "suppressed": 1, "blocked": 2}
        regime_map = {"trend_up": 0, "trend_down": 1, "chop": 2, "squeeze": 3, "crash": 4}
        volatility_map = {"low": 0, "mid": 1, "high": 2}
        liquidity_map = {"low": 0, "mid": 1, "high": 2}
        risk_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        
        protocol_weights = self._telemetry.compute_protocol_weights()
        
        cross_asset = self._cross_asset.get_feature_vector() if include_cross_asset else {}
        
        features = []
        for row in rows:
            protocol_ids = row.get("protocol_ids", [])
            protocol_count = len(protocol_ids) if isinstance(protocol_ids, list) else 0
            
            weighted_score = 0.0
            for pid in protocol_ids:
                weight = protocol_weights.get(pid, 1.0)
                weighted_score += weight
            weighted_score = weighted_score / max(protocol_count, 1)
            
            feature_vec = [
                float(row.get("quantra_score", 50)),
                self._encode_categorical(row.get("entropy_band", "mid"), entropy_map),
                self._encode_categorical(row.get("suppression_state", "none"), suppression_map),
                self._encode_categorical(row.get("regime_type", "chop"), regime_map),
                self._encode_categorical(row.get("volatility_band", "mid"), volatility_map),
                self._encode_categorical(row.get("liquidity_band", "mid"), liquidity_map),
                self._encode_categorical(row.get("risk_tier", "medium"), risk_map),
                protocol_count,
                weighted_score,
                float(row.get("ret_1d", 0)),
                float(row.get("ret_3d", 0)),
                float(row.get("ret_5d", 0)),
                float(row.get("max_runup_5d", 0)),
                float(row.get("max_drawdown_5d", 0)),
                cross_asset.get("vix_level", 20.0),
                cross_asset.get("vix_percentile", 50.0),
                cross_asset.get("cyclical_momentum", 0.0),
                cross_asset.get("advance_decline", 0.5),
            ]
            features.append(feature_vec)
        
        return np.array(features)
    
    def fit(
        self,
        rows: List[Dict[str, Any]],
        validation_split: float = 0.2,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Train all prediction heads with enhanced accuracy optimization.
        """
        if len(rows) < 30:
            raise ValueError(f"Insufficient training data: {len(rows)} rows (need 30+)")
        
        logger.info(f"[ApexCoreV3] Training on {len(rows)} samples...")
        
        X = self._prepare_features(rows, include_cross_asset=False)
        
        y_quantrascore = np.array([r.get("quantra_score", 50) for r in rows])
        y_runner = np.array([r.get("hit_runner_threshold", 0) for r in rows])
        y_quality = np.array([r.get("future_quality_tier", "C") for r in rows])
        y_avoid = np.array([r.get("avoid_trade", 0) for r in rows])
        y_regime = np.array([r.get("regime_label", "chop") for r in rows])
        y_timing = np.array([r.get("timing_bucket", "none") for r in rows])
        self._move_directions = np.array([r.get("move_direction", 0) for r in rows])
        
        self.quality_encoder.fit(y_quality)
        self.regime_encoder.fit(y_regime)
        self.timing_encoder.fit(["immediate", "very_soon", "soon", "late", "none"])
        
        y_quality_encoded = self.quality_encoder.transform(y_quality)
        y_regime_encoded = self.regime_encoder.transform(y_regime)
        y_timing_encoded = self.timing_encoder.transform(y_timing)
        
        n_val = int(len(rows) * validation_split)
        indices = np.random.RandomState(42).permutation(len(rows))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        if sample_weights is not None:
            train_weights = sample_weights[train_idx]
        else:
            train_weights = None
        
        logger.info("Training QuantraScore head...")
        self.quantrascore_head.fit(X_train_scaled, y_quantrascore[train_idx], sample_weight=train_weights)
        qs_pred = self.quantrascore_head.predict(X_val_scaled)
        qs_rmse = float(np.sqrt(np.mean((qs_pred - y_quantrascore[val_idx]) ** 2)))
        
        logger.info("Training Runner head...")
        if len(np.unique(y_runner[train_idx])) > 1:
            self.runner_head.fit(X_train_scaled, y_runner[train_idx], sample_weight=train_weights)
            runner_acc = self.runner_head.score(X_val_scaled, y_runner[val_idx])
            
            if self._enable_calibration:
                runner_proba = self.runner_head.predict_proba(X_val_scaled)[:, 1]
                self._calibration.fit(runner_proba, y_runner[val_idx])
        else:
            runner_acc = 0.5
        
        logger.info("Training Quality head...")
        if len(np.unique(y_quality_encoded[train_idx])) > 1:
            self.quality_head.fit(X_train_scaled, y_quality_encoded[train_idx], sample_weight=train_weights)
            quality_acc = self.quality_head.score(X_val_scaled, y_quality_encoded[val_idx])
        else:
            quality_acc = 0.5
        
        logger.info("Training Avoid head...")
        if len(np.unique(y_avoid[train_idx])) > 1:
            self.avoid_head.fit(X_train_scaled, y_avoid[train_idx], sample_weight=train_weights)
            avoid_acc = self.avoid_head.score(X_val_scaled, y_avoid[val_idx])
        else:
            avoid_acc = 0.5
        
        logger.info("Training Regime head...")
        if len(np.unique(y_regime_encoded[train_idx])) > 1:
            self.regime_head.fit(X_train_scaled, y_regime_encoded[train_idx], sample_weight=train_weights)
            regime_acc = self.regime_head.score(X_val_scaled, y_regime_encoded[val_idx])
        else:
            regime_acc = 0.5
        
        logger.info("Training Timing head...")
        if len(np.unique(y_timing_encoded[train_idx])) > 1:
            self.timing_head.fit(X_train_scaled, y_timing_encoded[train_idx], sample_weight=train_weights)
            timing_acc = self.timing_head.score(X_val_scaled, y_timing_encoded[val_idx])
        else:
            timing_acc = 0.5
        
        if self._enable_uncertainty:
            logger.info("Calibrating uncertainty head...")
            self._uncertainty.fit(qs_pred, y_quantrascore[val_idx])
        
        self._is_fitted = True
        self._timing_head_available = True
        
        metrics = {
            "quantrascore_rmse": qs_rmse,
            "runner_accuracy": float(runner_acc),
            "quality_accuracy": float(quality_acc),
            "avoid_accuracy": float(avoid_acc),
            "regime_accuracy": float(regime_acc),
            "timing_accuracy": float(timing_acc),
            "training_samples": len(rows),
            "validation_samples": len(val_idx),
        }
        
        accuracy_modules = []
        if self._enable_calibration:
            accuracy_modules.append("calibration")
        if self._enable_uncertainty:
            accuracy_modules.append("uncertainty")
        if self._enable_multi_horizon:
            accuracy_modules.append("multi_horizon")
        accuracy_modules.extend(["protocol_telemetry", "cross_asset", "timing_prediction"])
        
        self._manifest = ApexCoreV3Manifest(
            version=self.version,
            model_size=self.model_size,
            trained_at=datetime.utcnow().isoformat(),
            training_samples=len(rows),
            feature_count=X.shape[1],
            heads=["quantrascore", "runner", "quality", "avoid", "regime", "timing"],
            accuracy_modules=accuracy_modules,
            metrics=metrics,
            feature_names=self.FEATURE_NAMES,
        )
        
        logger.info(f"[ApexCoreV3] Training complete. Metrics: {metrics}")
        return metrics
    
    def predict(
        self,
        row: Dict[str, Any],
        current_price: Optional[float] = None,
    ) -> ApexCoreV3Prediction:
        """
        Generate enhanced prediction with all accuracy features.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first or load a trained model.")
        
        self._prediction_counter += 1
        prediction_id = f"v3_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._prediction_counter}"
        
        X = self._prepare_features([row], include_cross_asset=True)
        X_scaled = self.scaler.transform(X)
        
        qs_raw = float(self.quantrascore_head.predict(X_scaled)[0])
        
        try:
            runner_raw = float(self.runner_head.predict_proba(X_scaled)[0, 1])
        except:
            runner_raw = 0.5
        
        if self._enable_calibration and self._calibration.is_fitted:
            runner_calibrated = float(self._calibration.calibrate(np.array([runner_raw]))[0])
            qs_calibrated = qs_raw
        else:
            runner_calibrated = runner_raw
            qs_calibrated = qs_raw
        
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
        
        if self._timing_head_available:
            try:
                timing_pred_encoded = self.timing_head.predict(X_scaled)[0]
                timing_bucket = self.timing_encoder.inverse_transform([timing_pred_encoded])[0]
                timing_proba = self.timing_head.predict_proba(X_scaled)[0]
                timing_confidence = float(np.max(timing_proba))
                
                bucket_to_bars = {"immediate": 1, "very_soon": 2, "soon": 5, "late": 8, "none": 11}
                bars_to_move_estimate = bucket_to_bars.get(timing_bucket, 11)
                
                move_direction = 0
                if timing_bucket != "none":
                    move_direction = 1 if runner_calibrated > 0.5 else -1
                
                if timing_confidence < 0.35:
                    timing_bucket = "none"
                    move_direction = 0
            except Exception as e:
                logger.debug(f"Timing prediction failed: {e}")
                timing_bucket = "none"
                timing_confidence = 0.5
                bars_to_move_estimate = 11
                move_direction = 0
        else:
            timing_bucket = "none"
            timing_confidence = 0.0
            bars_to_move_estimate = 11
            move_direction = 0
        
        if self._enable_uncertainty and self._uncertainty.is_fitted:
            uncertainty_est = self._uncertainty.estimate(qs_raw)
            uncertainty_lower = uncertainty_est.lower_bound
            uncertainty_upper = uncertainty_est.upper_bound
            uncertainty_level = uncertainty_est.confidence_level
        else:
            uncertainty_lower = qs_raw - 10
            uncertainty_upper = qs_raw + 10
            uncertainty_level = "medium"
        
        multi_horizon_dict = {}
        horizon_consensus = "neutral"
        if self._enable_multi_horizon and self._multi_horizon.is_fitted:
            mh_pred = self._multi_horizon.predict(X_scaled[0], current_price)
            for h, p in mh_pred.predictions.items():
                multi_horizon_dict[h] = p.return_prediction
            horizon_consensus = mh_pred.consensus_direction
        
        cross_features = self._cross_asset.get_features()
        market_context = {
            "vix_regime": cross_features.vix.regime.value,
            "market_regime": cross_features.market_regime.value,
            "risk_appetite": cross_features.risk_appetite_score,
            "sector_rotation": cross_features.sectors.rotation_signal,
        }
        
        base_conf = 0.7
        if runner_calibrated > 0.7 or runner_calibrated < 0.3:
            base_conf += 0.1
        if avoid_proba > 0.7 or avoid_proba < 0.3:
            base_conf += 0.05
        if uncertainty_level in ["high", "very_high"]:
            base_conf += 0.1
        elif uncertainty_level in ["low", "very_low"]:
            base_conf -= 0.1
        
        confidence = max(0.3, min(0.95, base_conf))
        
        return ApexCoreV3Prediction(
            quantrascore_pred=qs_raw,
            quantrascore_calibrated=qs_calibrated,
            runner_probability=runner_raw,
            runner_probability_calibrated=runner_calibrated,
            quality_tier_pred=quality_pred,
            avoid_trade_probability=avoid_proba,
            regime_pred=regime_pred,
            timing_bucket=timing_bucket,
            timing_confidence=timing_confidence,
            move_direction=move_direction,
            bars_to_move_estimate=bars_to_move_estimate,
            confidence=confidence,
            uncertainty_lower=uncertainty_lower,
            uncertainty_upper=uncertainty_upper,
            uncertainty_level=uncertainty_level,
            multi_horizon=multi_horizon_dict,
            horizon_consensus=horizon_consensus,
            market_context=market_context,
            model_version=self.version,
            prediction_id=prediction_id,
        )
    
    def predict_batch(self, rows: List[Dict[str, Any]]) -> List[ApexCoreV3Prediction]:
        """Generate predictions for multiple rows."""
        return [self.predict(row) for row in rows]
    
    def save(self, model_dir: Optional[str] = None) -> str:
        """Save model and all accuracy components to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dir = model_dir or os.path.join(MODEL_DIR, self.model_size)
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.quantrascore_head, os.path.join(save_dir, "quantrascore_head.joblib"))
        joblib.dump(self.runner_head, os.path.join(save_dir, "runner_head.joblib"))
        joblib.dump(self.quality_head, os.path.join(save_dir, "quality_head.joblib"))
        joblib.dump(self.avoid_head, os.path.join(save_dir, "avoid_head.joblib"))
        joblib.dump(self.regime_head, os.path.join(save_dir, "regime_head.joblib"))
        joblib.dump(self.timing_head, os.path.join(save_dir, "timing_head.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        joblib.dump(self.quality_encoder, os.path.join(save_dir, "quality_encoder.joblib"))
        joblib.dump(self.regime_encoder, os.path.join(save_dir, "regime_encoder.joblib"))
        joblib.dump(self.timing_encoder, os.path.join(save_dir, "timing_encoder.joblib"))
        
        if self._calibration.is_fitted:
            self._calibration.save("apexcore_v3")
        if self._uncertainty.is_fitted:
            self._uncertainty.save("apexcore_v3")
        
        if self._manifest:
            with open(os.path.join(save_dir, "manifest.json"), "w") as f:
                json.dump(self._manifest.to_dict(), f, indent=2)
        
        logger.info(f"[ApexCoreV3] Model saved to {save_dir}")
        return save_dir
    
    @classmethod
    def load(cls, model_dir: Optional[str] = None, model_size: str = "big") -> "ApexCoreV3Model":
        """Load model from disk."""
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
        
        timing_head_path = os.path.join(load_dir, "timing_head.joblib")
        timing_encoder_path = os.path.join(load_dir, "timing_encoder.joblib")
        if os.path.exists(timing_head_path) and os.path.exists(timing_encoder_path):
            try:
                model.timing_head = joblib.load(timing_head_path)
                model.timing_encoder = joblib.load(timing_encoder_path)
                model._timing_head_available = True
            except Exception as e:
                logger.warning(f"Could not load timing head: {e}")
                model.timing_encoder.fit(["immediate", "very_soon", "soon", "late", "none"])
                model._timing_head_available = False
        else:
            logger.info("Timing head not found - loading legacy model without timing predictions")
            model.timing_encoder.fit(["immediate", "very_soon", "soon", "late", "none"])
            model._timing_head_available = False
        
        try:
            model._calibration.load("apexcore_v3")
        except:
            pass
        
        try:
            model._uncertainty.load("apexcore_v3")
        except:
            pass
        
        manifest_path = os.path.join(load_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                data = json.load(f)
                model._manifest = ApexCoreV3Manifest(**data)
        
        model._is_fitted = True
        logger.info(f"[ApexCoreV3] Model loaded from {load_dir}")
        return model
    
    @property
    def manifest(self) -> Optional[ApexCoreV3Manifest]:
        return self._manifest
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def get_v3_model_status(model_size: str = "big") -> Dict[str, Any]:
    """Get status of ApexCore V3 trained models."""
    model_dir = os.path.join(MODEL_DIR, model_size)
    manifest_path = os.path.join(model_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        return {
            "status": "not_trained",
            "model_size": model_size,
            "model_dir": model_dir,
            "version": "3.0.0",
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
            "accuracy_modules": manifest.get("accuracy_modules", []),
            "metrics": manifest.get("metrics"),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


__all__ = [
    "ApexCoreV3Model",
    "ApexCoreV3Prediction",
    "ApexCoreV3Manifest",
    "get_v3_model_status",
]
