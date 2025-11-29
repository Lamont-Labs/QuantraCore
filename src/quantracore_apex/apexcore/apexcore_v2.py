"""
ApexCore V2 - Multi-Head Neural Assistant Model.

Provides institutional-grade predictive capabilities with:
- QuantraScore regression (mandatory head)
- Runner probability classification
- Quality tier classification
- Avoid-trade probability
- Regime classification
- Ensemble support with uncertainty quantification

Uses scikit-learn for disk efficiency per project requirements.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from enum import Enum
import joblib
import hashlib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin


class ModelVariant(Enum):
    BIG = "big"
    MINI = "mini"


@dataclass
class ApexCoreV2Config:
    """Configuration for ApexCore V2 model."""
    variant: ModelVariant = ModelVariant.BIG
    random_state: int = 42
    n_estimators_big: int = 100
    n_estimators_mini: int = 30
    max_depth_big: int = 6
    max_depth_mini: int = 3
    learning_rate: float = 0.1
    num_quality_tiers: int = 5
    num_regimes: int = 5
    protocol_vector_dim: int = 115


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Feature encoder for ApexCore V2.
    
    Encodes structural features into a fixed-length vector suitable
    for the model ensemble.
    """
    
    def __init__(self, protocol_vector_dim: int = 115):
        self.protocol_vector_dim = protocol_vector_dim
        self.scaler = StandardScaler()
        self._fitted = False
    
    def fit(self, X: np.ndarray, y=None) -> "FeatureEncoder":
        """Fit the encoder on training data."""
        self.scaler.fit(X)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self._fitted:
            return X
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)


class ApexCoreV2Head:
    """Base class for model heads."""
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ApexCoreV2Head":
        """Fit the head on training data."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get probability predictions (for classifiers)."""
        return None


class QuantraScoreHead(ApexCoreV2Head):
    """Regression head for QuantraScore (0-100)."""
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BIG,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        super().__init__("quantra_score", random_state)
        self.variant = variant
        
        if variant == ModelVariant.BIG:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=n_estimators // 3,
                max_depth=max_depth // 2,
                random_state=random_state,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantraScoreHead":
        """Fit the QuantraScore head."""
        y_clipped = np.clip(y, 0, 100)
        self.model.fit(X, y_clipped)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict QuantraScore (0-100)."""
        if not self._fitted:
            return np.full(len(X), 50.0)
        preds = self.model.predict(X)
        return np.clip(preds, 0, 100)


class RunnerProbHead(ApexCoreV2Head):
    """Binary classification head for runner probability."""
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BIG,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        super().__init__("runner_prob", random_state)
        self.variant = variant
        
        if variant == ModelVariant.BIG:
            base_clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            self.model = CalibratedClassifierCV(base_clf, cv=3)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators // 3,
                max_depth=max_depth // 2,
                random_state=random_state,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RunnerProbHead":
        """Fit the runner probability head."""
        y_binary = (y > 0).astype(int)
        self.model.fit(X, y_binary)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary runner class."""
        if not self._fitted:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get runner probability [0, 1]."""
        if not self._fitted:
            return np.full(len(X), 0.1)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class QualityTierHead(ApexCoreV2Head):
    """Multi-class classification for quality tier."""
    
    TIERS = ["A_PLUS", "A", "B", "C", "D"]
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BIG,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        super().__init__("quality_tier", random_state)
        self.variant = variant
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.TIERS)
        
        if variant == ModelVariant.BIG:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators // 3,
                max_depth=max_depth // 2,
                random_state=random_state,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QualityTierHead":
        """Fit the quality tier head."""
        y_encoded = self.label_encoder.transform(y)
        self.model.fit(X, y_encoded)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quality tier labels."""
        if not self._fitted:
            return np.array(["C"] * len(X))
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get quality tier probabilities (logits-style)."""
        if not self._fitted:
            return np.zeros((len(X), len(self.TIERS)))
        return self.model.predict_proba(X)


class AvoidTradeHead(ApexCoreV2Head):
    """Binary classification for avoid-trade probability."""
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BIG,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        super().__init__("avoid_trade", random_state)
        self.variant = variant
        
        if variant == ModelVariant.BIG:
            base_clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            self.model = CalibratedClassifierCV(base_clf, cv=3)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators // 3,
                max_depth=max_depth // 2,
                random_state=random_state,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AvoidTradeHead":
        """Fit the avoid-trade head."""
        self.model.fit(X, y.astype(int))
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary avoid-trade class."""
        if not self._fitted:
            return np.zeros(len(X))
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get avoid-trade probability [0, 1]."""
        if not self._fitted:
            return np.full(len(X), 0.1)
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba


class RegimeHead(ApexCoreV2Head):
    """Multi-class classification for regime."""
    
    REGIMES = ["trend_up", "trend_down", "chop", "squeeze", "crash"]
    
    def __init__(
        self,
        variant: ModelVariant = ModelVariant.BIG,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        super().__init__("regime", random_state)
        self.variant = variant
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.REGIMES)
        
        if variant == ModelVariant.BIG:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=n_estimators // 3,
                max_depth=max_depth // 2,
                random_state=random_state,
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimeHead":
        """Fit the regime head."""
        y_mapped = np.array([r if r in self.REGIMES else "chop" for r in y])
        y_encoded = self.label_encoder.transform(y_mapped)
        self.model.fit(X, y_encoded)
        self._fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels."""
        if not self._fitted:
            return np.array(["chop"] * len(X))
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds.astype(int))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get regime probabilities (logits-style)."""
        if not self._fitted:
            return np.zeros((len(X), len(self.REGIMES)))
        return self.model.predict_proba(X)


class ApexCoreV2Model:
    """
    ApexCore V2 Multi-Head Model.
    
    Combines all prediction heads into a unified model interface
    with deterministic behavior and ensemble support.
    """
    
    def __init__(
        self,
        config: Optional[ApexCoreV2Config] = None,
        variant: ModelVariant = ModelVariant.BIG,
    ):
        self.config = config or ApexCoreV2Config(variant=variant)
        self.variant = self.config.variant
        
        n_est = (
            self.config.n_estimators_big
            if self.variant == ModelVariant.BIG
            else self.config.n_estimators_mini
        )
        max_d = (
            self.config.max_depth_big
            if self.variant == ModelVariant.BIG
            else self.config.max_depth_mini
        )
        
        self.encoder = FeatureEncoder(self.config.protocol_vector_dim)
        
        self.heads = {
            "quantra_score": QuantraScoreHead(self.variant, self.config.random_state, n_est, max_d),
            "runner_prob": RunnerProbHead(self.variant, self.config.random_state, n_est, max_d),
            "quality_tier": QualityTierHead(self.variant, self.config.random_state, n_est, max_d),
            "avoid_trade": AvoidTradeHead(self.variant, self.config.random_state, n_est, max_d),
            "regime": RegimeHead(self.variant, self.config.random_state, n_est, max_d),
        }
        
        self._fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
    ) -> "ApexCoreV2Model":
        """
        Fit all heads on training data.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            targets: Dictionary of target arrays by head name
                - quantra_score: float array
                - hit_runner_threshold: int array
                - future_quality_tier: str array
                - avoid_trade: int array
                - regime_label: str array
        """
        X_encoded = self.encoder.fit_transform(X)
        
        if "quantra_score" in targets:
            self.heads["quantra_score"].fit(X_encoded, targets["quantra_score"])
        
        if "hit_runner_threshold" in targets:
            self.heads["runner_prob"].fit(X_encoded, targets["hit_runner_threshold"])
        
        if "future_quality_tier" in targets:
            self.heads["quality_tier"].fit(X_encoded, targets["future_quality_tier"])
        
        if "avoid_trade" in targets:
            self.heads["avoid_trade"].fit(X_encoded, targets["avoid_trade"])
        
        if "regime_label" in targets:
            self.heads["regime"].fit(X_encoded, targets["regime_label"])
        
        self._fitted = True
        return self
    
    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run forward pass through all heads.
        
        Returns:
            Dictionary with:
            - quantra_score: [batch_size] regression values
            - runner_prob: [batch_size] probabilities
            - quality_logits: [batch_size, 5] class probabilities
            - avoid_trade_prob: [batch_size] probabilities
            - regime_logits: [batch_size, 5] class probabilities
        """
        X_encoded = self.encoder.transform(X) if self._fitted else X
        
        return {
            "quantra_score": self.heads["quantra_score"].predict(X_encoded),
            "runner_prob": self.heads["runner_prob"].predict_proba(X_encoded),
            "quality_logits": self.heads["quality_tier"].predict_proba(X_encoded),
            "avoid_trade_prob": self.heads["avoid_trade"].predict_proba(X_encoded),
            "regime_logits": self.heads["regime"].predict_proba(X_encoded),
        }
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Alias for forward()."""
        return self.forward(X)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "config": self.config,
            "variant": self.variant,
            "encoder": self.encoder,
            "heads": self.heads,
            "fitted": self._fitted,
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "ApexCoreV2Model":
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(config=data["config"], variant=data["variant"])
        model.encoder = data["encoder"]
        model.heads = data["heads"]
        model._fitted = data["fitted"]
        return model
    
    def get_hash(self) -> str:
        """Get deterministic hash of model state."""
        state_str = str(self.config) + str(self._fitted)
        return hashlib.sha256(state_str.encode()).hexdigest()


class ApexCoreV2Big(ApexCoreV2Model):
    """ApexCore V2 Big variant - full precision, more parameters."""
    
    def __init__(self, config: Optional[ApexCoreV2Config] = None):
        if config is None:
            config = ApexCoreV2Config(variant=ModelVariant.BIG)
        super().__init__(config, ModelVariant.BIG)


class ApexCoreV2Mini(ApexCoreV2Model):
    """ApexCore V2 Mini variant - lightweight, mobile-friendly."""
    
    def __init__(self, config: Optional[ApexCoreV2Config] = None):
        if config is None:
            config = ApexCoreV2Config(variant=ModelVariant.MINI)
        super().__init__(config, ModelVariant.MINI)


class ApexCoreV2Ensemble:
    """
    Ensemble of ApexCore V2 models with uncertainty quantification.
    
    Maintains N models and provides:
    - Mean predictions across ensemble
    - Disagreement metrics for fail-closed behavior
    """
    
    def __init__(
        self,
        ensemble_size: int = 3,
        variant: ModelVariant = ModelVariant.BIG,
        base_random_state: int = 42,
    ):
        self.ensemble_size = ensemble_size
        self.variant = variant
        self.base_random_state = base_random_state
        
        self.members: List[ApexCoreV2Model] = []
        for i in range(ensemble_size):
            config = ApexCoreV2Config(
                variant=variant,
                random_state=base_random_state + i,
            )
            if variant == ModelVariant.BIG:
                self.members.append(ApexCoreV2Big(config))
            else:
                self.members.append(ApexCoreV2Mini(config))
        
        self._fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
        bootstrap: bool = True,
    ) -> "ApexCoreV2Ensemble":
        """
        Fit all ensemble members.
        
        Args:
            X: Feature matrix
            targets: Target dictionary
            bootstrap: Whether to use bootstrap sampling for diversity
        """
        n_samples = len(X)
        
        for i, member in enumerate(self.members):
            if bootstrap:
                rng = np.random.RandomState(self.base_random_state + i)
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[indices]
                targets_boot = {k: v[indices] for k, v in targets.items()}
                member.fit(X_boot, targets_boot)
            else:
                member.fit(X, targets)
        
        self._fitted = True
        return self
    
    def forward(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Run forward pass and compute ensemble statistics.
        
        Returns:
            Dictionary with:
            - quantra_score: mean prediction
            - runner_prob: mean probability
            - quality_logits: mean class probabilities
            - avoid_trade_prob: mean probability
            - regime_logits: mean class probabilities
            - disagreement: dictionary of per-head disagreement metrics
        """
        if not self._fitted:
            dummy = self.members[0].forward(X)
            dummy["disagreement"] = {k: np.zeros(len(X)) for k in dummy}
            return dummy
        
        all_outputs = [m.forward(X) for m in self.members]
        
        mean_outputs = {}
        disagreement = {}
        
        for key in all_outputs[0].keys():
            stacked = np.stack([o[key] for o in all_outputs], axis=0)
            mean_outputs[key] = np.mean(stacked, axis=0)
            disagreement[key] = np.std(stacked, axis=0)
            if disagreement[key].ndim > 1:
                disagreement[key] = np.mean(disagreement[key], axis=-1)
        
        mean_outputs["disagreement"] = disagreement
        return mean_outputs
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Alias for forward()."""
        return self.forward(X)
    
    def get_member_hashes(self) -> Dict[str, str]:
        """Get hashes for all ensemble members."""
        return {f"member_{i}": m.get_hash() for i, m in enumerate(self.members)}
    
    def save(self, directory: str) -> None:
        """Save ensemble to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, member in enumerate(self.members):
            member.save(path / f"member_{i}.joblib")
        
        joblib.dump({
            "ensemble_size": self.ensemble_size,
            "variant": self.variant.value,
            "base_random_state": self.base_random_state,
            "fitted": self._fitted,
        }, path / "ensemble_meta.joblib")
    
    @classmethod
    def load(cls, directory: str) -> "ApexCoreV2Ensemble":
        """Load ensemble from directory."""
        path = Path(directory)
        meta = joblib.load(path / "ensemble_meta.joblib")
        
        ensemble = cls(
            ensemble_size=meta["ensemble_size"],
            variant=ModelVariant(meta["variant"]),
            base_random_state=meta["base_random_state"],
        )
        
        ensemble.members = []
        for i in range(meta["ensemble_size"]):
            member = ApexCoreV2Model.load(path / f"member_{i}.joblib")
            ensemble.members.append(member)
        
        ensemble._fitted = meta["fitted"]
        return ensemble


__all__ = [
    "ApexCoreV2Config",
    "ApexCoreV2Model",
    "ApexCoreV2Big",
    "ApexCoreV2Mini",
    "ApexCoreV2Ensemble",
    "ModelVariant",
    "FeatureEncoder",
    "QuantraScoreHead",
    "RunnerProbHead",
    "QualityTierHead",
    "AvoidTradeHead",
    "RegimeHead",
]
