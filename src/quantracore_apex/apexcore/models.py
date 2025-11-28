"""
ApexCore Model Definitions

Defines ApexCore Full and ApexCore Mini model architectures.
These are lightweight neural assistants trained by ApexLab.

Version: 8.1

Note: This is architecture definition only. Training is handled
by ApexLab and must be run manually (not automatically).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ApexCoreConfig:
    """Base configuration for ApexCore models."""
    model_name: str = "apexcore_base"
    input_dim: int = 30
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    max_iter: int = 200
    random_state: int = 42


@dataclass
class ApexCoreFullConfig(ApexCoreConfig):
    """Configuration for ApexCore Full (desktop) model."""
    model_name: str = "apexcore_full"
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    max_iter: int = 500


@dataclass
class ApexCoreMiniConfig(ApexCoreConfig):
    """Configuration for ApexCore Mini (lightweight) model."""
    model_name: str = "apexcore_mini"
    hidden_layers: List[int] = field(default_factory=lambda: [32, 16])
    max_iter: int = 200


@dataclass
class ApexCoreOutputs:
    """
    Unified outputs from ApexCore models.
    
    Matches the teacher (Apex engine) output structure for validation.
    """
    quantrascore: float = 50.0
    regime_prediction: str = "unknown"
    risk_tier: str = "medium"
    volatility_band: str = "medium"
    score_bucket: str = "neutral"
    continuation_probability: float = 0.5
    confidence: float = 0.0
    is_placeholder: bool = True
    compliance_note: str = "ApexCore output is structural probability, not trading advice"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quantrascore": self.quantrascore,
            "regime_prediction": self.regime_prediction,
            "risk_tier": self.risk_tier,
            "volatility_band": self.volatility_band,
            "score_bucket": self.score_bucket,
            "continuation_probability": self.continuation_probability,
            "confidence": self.confidence,
            "is_placeholder": self.is_placeholder,
            "compliance_note": self.compliance_note,
        }


class ApexCoreBase(ABC):
    """
    Abstract base class for ApexCore models.
    
    All ApexCore models must implement prediction methods
    and maintain deterministic behavior for given weights.
    """
    
    def __init__(self, config: ApexCoreConfig):
        """Initialize with configuration."""
        self.config = config
        self.is_trained = False
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.regime_classifier = None
        self.score_regressor = None
        self.risk_classifier = None
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """
        Make predictions from features.
        
        Args:
            features: Feature array of shape (n_samples, input_dim)
            
        Returns:
            ApexCoreOutputs with predictions
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_score: np.ndarray,
        y_risk: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train the model on labeled data.
        
        Args:
            X: Feature array
            y_regime: Regime labels
            y_score: QuantraScore targets
            y_risk: Risk tier labels
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    def save(self, path: str) -> None:
        """Save model weights to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'scaler': self.scaler,
                'regime_classifier': self.regime_classifier,
                'score_regressor': self.score_regressor,
                'risk_classifier': self.risk_classifier,
                'is_trained': self.is_trained,
            }, f)
    
    def load(self, path: str) -> None:
        """Load model weights from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.scaler = data['scaler']
            self.regime_classifier = data['regime_classifier']
            self.score_regressor = data['score_regressor']
            self.risk_classifier = data['risk_classifier']
            self.is_trained = data['is_trained']


class ApexCoreFull(ApexCoreBase):
    """
    ApexCore Full — Desktop neural assistant.
    
    Larger model for workstation deployment.
    Target specs: 3-20MB, <20ms inference.
    """
    
    def __init__(self, config: Optional[ApexCoreFullConfig] = None):
        """Initialize ApexCore Full model."""
        if config is None:
            config = ApexCoreFullConfig()
        super().__init__(config)
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize sklearn models."""
        if not HAS_SKLEARN:
            return
        
        self.regime_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        
        self.score_regressor = MLPRegressor(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        
        self.risk_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
    
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """Make predictions from features."""
        if not self.is_trained or not HAS_SKLEARN:
            return ApexCoreOutputs(is_placeholder=True)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        X_scaled = self.scaler.transform(features)
        
        regime_pred = self.regime_classifier.predict(X_scaled)[0]
        score_pred = float(np.clip(self.score_regressor.predict(X_scaled)[0], 0, 100))
        risk_pred = self.risk_classifier.predict(X_scaled)[0]
        
        if score_pred >= 70:
            score_bucket = "strong"
        elif score_pred >= 55:
            score_bucket = "moderate"
        elif score_pred >= 40:
            score_bucket = "neutral"
        else:
            score_bucket = "weak"
        
        regime_proba = self.regime_classifier.predict_proba(X_scaled)[0]
        confidence = float(np.max(regime_proba) * 100)
        
        return ApexCoreOutputs(
            quantrascore=round(score_pred, 2),
            regime_prediction=str(regime_pred),
            risk_tier=str(risk_pred),
            score_bucket=score_bucket,
            confidence=round(confidence, 2),
            is_placeholder=False,
        )
    
    def train(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_score: np.ndarray,
        y_risk: np.ndarray,
    ) -> Dict[str, float]:
        """Train the model on labeled data."""
        if not HAS_SKLEARN:
            return {"error": "sklearn not available"}
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.regime_classifier.fit(X_scaled, y_regime)
        self.score_regressor.fit(X_scaled, y_score)
        self.risk_classifier.fit(X_scaled, y_risk)
        
        self.is_trained = True
        
        regime_score = self.regime_classifier.score(X_scaled, y_regime)
        risk_score = self.risk_classifier.score(X_scaled, y_risk)
        score_preds = self.score_regressor.predict(X_scaled)
        score_mae = float(np.mean(np.abs(score_preds - y_score)))
        
        return {
            "regime_accuracy": round(regime_score, 4),
            "risk_accuracy": round(risk_score, 4),
            "score_mae": round(score_mae, 4),
            "samples": len(X),
        }


class ApexCoreMini(ApexCoreBase):
    """
    ApexCore Mini — Lightweight neural assistant.
    
    Smaller model for resource-constrained deployment.
    Target specs: 0.5-3MB, <30ms inference.
    
    Note: Not for Android deployment per project requirements.
    This is a lightweight desktop variant only.
    """
    
    def __init__(self, config: Optional[ApexCoreMiniConfig] = None):
        """Initialize ApexCore Mini model."""
        if config is None:
            config = ApexCoreMiniConfig()
        super().__init__(config)
        self._init_models()
    
    def _init_models(self) -> None:
        """Initialize sklearn models with smaller architecture."""
        if not HAS_SKLEARN:
            return
        
        self.regime_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
        )
        
        self.score_regressor = MLPRegressor(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
        )
        
        self.risk_classifier = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_layers),
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            early_stopping=True,
        )
    
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """Make predictions from features."""
        if not self.is_trained or not HAS_SKLEARN:
            return ApexCoreOutputs(is_placeholder=True)
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        X_scaled = self.scaler.transform(features)
        
        regime_pred = self.regime_classifier.predict(X_scaled)[0]
        score_pred = float(np.clip(self.score_regressor.predict(X_scaled)[0], 0, 100))
        risk_pred = self.risk_classifier.predict(X_scaled)[0]
        
        if score_pred >= 70:
            score_bucket = "strong"
        elif score_pred >= 55:
            score_bucket = "moderate"
        elif score_pred >= 40:
            score_bucket = "neutral"
        else:
            score_bucket = "weak"
        
        return ApexCoreOutputs(
            quantrascore=round(score_pred, 2),
            regime_prediction=str(regime_pred),
            risk_tier=str(risk_pred),
            score_bucket=score_bucket,
            confidence=60.0,
            is_placeholder=False,
        )
    
    def train(
        self,
        X: np.ndarray,
        y_regime: np.ndarray,
        y_score: np.ndarray,
        y_risk: np.ndarray,
    ) -> Dict[str, float]:
        """Train the model on labeled data."""
        if not HAS_SKLEARN:
            return {"error": "sklearn not available"}
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.regime_classifier.fit(X_scaled, y_regime)
        self.score_regressor.fit(X_scaled, y_score)
        self.risk_classifier.fit(X_scaled, y_risk)
        
        self.is_trained = True
        
        regime_score = self.regime_classifier.score(X_scaled, y_regime)
        risk_score = self.risk_classifier.score(X_scaled, y_risk)
        score_preds = self.score_regressor.predict(X_scaled)
        score_mae = float(np.mean(np.abs(score_preds - y_score)))
        
        return {
            "regime_accuracy": round(regime_score, 4),
            "risk_accuracy": round(risk_score, 4),
            "score_mae": round(score_mae, 4),
            "samples": len(X),
            "model_type": "mini",
        }
