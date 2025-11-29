"""
ApexCore Model Interface for QuantraCore Apex.

Defines the standard interface for ApexCore models.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import pickle
import logging

from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.apexlab.features import FeatureExtractor


logger = logging.getLogger(__name__)


@dataclass
class ApexCoreOutputs:
    """Outputs from ApexCore model."""
    regime_prediction: int
    regime_confidence: float
    risk_prediction: int
    risk_confidence: float
    quantrascore_prediction: float
    raw_outputs: Dict[str, Any]


class ApexCoreModel(ABC):
    """
    Abstract base class for ApexCore models.
    """
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """Make predictions from feature vector."""
        pass
    
    @abstractmethod
    def predict_window(self, window: OhlcvWindow) -> ApexCoreOutputs:
        """Make predictions from OHLCV window."""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from disk."""
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class ApexCoreFull(ApexCoreModel):
    """
    ApexCore Full model for desktop (K6).
    
    Size: 3-20MB
    Inference: <20ms
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = None
        self.regime_model = None
        self.risk_model = None
        self.score_model = None
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def load(self, model_path: str) -> None:
        """Load trained model from disk."""
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data["scaler"]
        self.regime_model = model_data["regime_model"]
        self.risk_model = model_data["risk_model"]
        self.score_model = model_data["score_model"]
        self._is_loaded = True
        
        logger.info(f"ApexCore Full loaded from {model_path}")
    
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """Make predictions from feature vector."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        
        regime_pred = self.regime_model.predict(features_scaled)[0]
        regime_proba = self.regime_model.predict_proba(features_scaled)[0]
        regime_conf = float(np.max(regime_proba))
        
        risk_pred = self.risk_model.predict(features_scaled)[0]
        risk_proba = self.risk_model.predict_proba(features_scaled)[0]
        risk_conf = float(np.max(risk_proba))
        
        score_pred = self.score_model.predict(features_scaled)[0]
        score_pred = float(np.clip(score_pred, 0, 100))
        
        return ApexCoreOutputs(
            regime_prediction=int(regime_pred),
            regime_confidence=regime_conf,
            risk_prediction=int(risk_pred),
            risk_confidence=risk_conf,
            quantrascore_prediction=score_pred,
            raw_outputs={
                "regime_proba": regime_proba.tolist(),
                "risk_proba": risk_proba.tolist(),
            }
        )
    
    def predict_window(self, window: OhlcvWindow) -> ApexCoreOutputs:
        """Make predictions from OHLCV window."""
        features = self.feature_extractor.extract(window)
        return self.predict(features)


class ApexCoreMini(ApexCoreModel):
    """
    ApexCore Mini model for mobile (Android/QuantraVision).
    
    Size: 0.5-3MB
    Inference: <30ms
    
    Note: For MVP+, this is a stub that defers to ApexCore Full.
    """
    
    def __init__(self):
        self._full_model = ApexCoreFull()
    
    @property
    def is_loaded(self) -> bool:
        return self._full_model.is_loaded
    
    def load(self, model_path: str) -> None:
        """Load model from disk."""
        self._full_model.load(model_path)
        logger.info("ApexCore Mini loaded (using Full model as backend)")
    
    def predict(self, features: np.ndarray) -> ApexCoreOutputs:
        """Make predictions from feature vector."""
        return self._full_model.predict(features)
    
    def predict_window(self, window: OhlcvWindow) -> ApexCoreOutputs:
        """Make predictions from OHLCV window."""
        return self._full_model.predict_window(window)
