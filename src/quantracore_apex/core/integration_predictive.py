"""
Predictive Advisor - Engine Integration for ApexCore V2.

Provides safe, fail-closed integration of predictive models with
the deterministic Apex engine. The deterministic engine always
has final authority over all predictions.

Key principles:
1. Deterministic-first: Engine overrides model predictions
2. Fail-closed: Invalid models/hashes result in disabled predictions
3. No auto-trade: Predictions are advisory only, research mode
4. Logged evolution: All model usage is tracked
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import numpy as np
import logging

from src.quantracore_apex.apexcore.apexcore_v2 import (
    ApexCoreV2Model,
    ApexCoreV2Ensemble,
    ModelVariant,
)
from src.quantracore_apex.apexcore.manifest import (
    ApexCoreV2Manifest,
    verify_manifest_against_file,
    load_manifest,
    select_best_model,
)

logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """Advisory recommendation from predictive layer."""
    UPRANK = "UPRANK"
    DOWNRANK = "DOWNRANK"
    AVOID = "AVOID"
    NEUTRAL = "NEUTRAL"
    DISABLED = "DISABLED"


@dataclass
class PredictiveAdvisory:
    """
    Complete advisory output from PredictiveAdvisor.
    
    This is purely advisory and does not override
    deterministic engine decisions.
    """
    symbol: str
    base_quantra_score: float
    model_quantra_score: float
    runner_prob: float
    quality_tier: str
    avoid_trade_prob: float
    ensemble_disagreement: float
    recommendation: Recommendation
    confidence: float
    reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "base_quantra_score": self.base_quantra_score,
            "model_quantra_score": self.model_quantra_score,
            "runner_prob": self.runner_prob,
            "quality_tier": self.quality_tier,
            "avoid_trade_prob": self.avoid_trade_prob,
            "ensemble_disagreement": self.ensemble_disagreement,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "reasons": self.reasons,
        }


@dataclass
class PredictiveConfig:
    """Configuration for PredictiveAdvisor."""
    enabled: bool = True
    model_dir: str = "models/apexcore_v2/big"
    manifest_dir: str = "models/apexcore_v2/big/manifests"
    variant: str = "big"
    
    runner_prob_uprank_threshold: float = 0.7
    runner_prob_min_threshold: float = 0.1
    avoid_trade_prob_max: float = 0.3
    max_disagreement_allowed: float = 0.2
    
    strict_hash_verification: bool = False


class PredictiveAdvisor:
    """
    Predictive layer advisor for Apex engine integration.
    
    Loads ApexCore V2 models and provides advisory predictions
    with fail-closed safety behavior.
    
    The deterministic engine ALWAYS has final authority.
    """
    
    def __init__(self, config: Optional[PredictiveConfig] = None):
        self.config = config or PredictiveConfig()
        self.ensemble: Optional[ApexCoreV2Ensemble] = None
        self.model: Optional[ApexCoreV2Model] = None
        self.manifest: Optional[ApexCoreV2Manifest] = None
        self._enabled = False
        self._status = "NOT_LOADED"
        self._load_failure_reason = ""
        
        if self.config.enabled:
            self._try_load_model()
    
    def _try_load_model(self) -> None:
        """Attempt to load model with fail-closed behavior."""
        try:
            manifest_dir = Path(self.config.manifest_dir)
            if not manifest_dir.exists():
                self._status = "MANIFEST_NOT_FOUND"
                self._load_failure_reason = f"Manifest directory not found: {manifest_dir}"
                logger.warning(self._load_failure_reason)
                return
            
            manifest_path, self.manifest = select_best_model(str(manifest_dir))
            
            if not manifest_path:
                self._status = "NO_VALID_MANIFEST"
                self._load_failure_reason = "No valid manifest found"
                logger.warning(self._load_failure_reason)
                return
            
            if self.config.strict_hash_verification:
                is_valid, failures = self.manifest.is_valid_for_promotion()
                if not is_valid:
                    self._status = "METRICS_BELOW_THRESHOLD"
                    self._load_failure_reason = f"Metrics validation failed: {failures}"
                    logger.warning(self._load_failure_reason)
                    return
            
            ensemble_path = Path(self.config.model_dir) / "ensemble"
            if ensemble_path.exists():
                self.ensemble = ApexCoreV2Ensemble.load(str(ensemble_path))
                self._enabled = True
                self._status = "ENSEMBLE_LOADED"
                logger.info(f"Loaded ApexCore V2 ensemble from {ensemble_path}")
            else:
                model_path = Path(self.config.model_dir) / "model.joblib"
                if model_path.exists():
                    self.model = ApexCoreV2Model.load(str(model_path))
                    self._enabled = True
                    self._status = "MODEL_LOADED"
                    logger.info(f"Loaded ApexCore V2 model from {model_path}")
                else:
                    self._status = "MODEL_NOT_FOUND"
                    self._load_failure_reason = f"Model not found at {model_path}"
                    logger.warning(self._load_failure_reason)
                    
        except Exception as e:
            self._status = "LOAD_EXCEPTION"
            self._load_failure_reason = str(e)
            logger.error(f"Failed to load predictive model: {e}")
    
    @property
    def is_enabled(self) -> bool:
        """Check if predictive layer is enabled and loaded."""
        return self._enabled
    
    @property
    def status(self) -> str:
        """Get current status of predictive layer."""
        return self._status
    
    def advise_on_candidate(
        self,
        symbol: str,
        base_quantra_score: float,
        features: np.ndarray,
    ) -> PredictiveAdvisory:
        """
        Generate advisory for a single candidate.
        
        Args:
            symbol: Symbol being analyzed
            base_quantra_score: QuantraScore from deterministic engine
            features: Feature vector for model input
            
        Returns:
            PredictiveAdvisory with recommendation
        """
        if not self._enabled:
            return PredictiveAdvisory(
                symbol=symbol,
                base_quantra_score=base_quantra_score,
                model_quantra_score=base_quantra_score,
                runner_prob=0.0,
                quality_tier="C",
                avoid_trade_prob=0.0,
                ensemble_disagreement=1.0,
                recommendation=Recommendation.DISABLED,
                confidence=0.0,
                reasons=[f"Predictive layer disabled: {self._status}"],
            )
        
        features_2d = features.reshape(1, -1) if features.ndim == 1 else features
        
        if self.ensemble is not None:
            outputs = self.ensemble.forward(features_2d)
            disagreement = float(outputs.get("disagreement", {}).get("runner_prob", np.array([0.0]))[0])
        else:
            outputs = self.model.forward(features_2d)
            disagreement = 0.0
        
        model_score = float(outputs["quantra_score"][0])
        runner_prob = float(outputs["runner_prob"][0])
        avoid_prob = float(outputs["avoid_trade_prob"][0])
        
        quality_logits = outputs["quality_logits"][0]
        quality_idx = np.argmax(quality_logits)
        quality_tiers = ["A_PLUS", "A", "B", "C", "D"]
        quality_tier = quality_tiers[quality_idx] if quality_idx < len(quality_tiers) else "C"
        
        recommendation, reasons, confidence = self._compute_recommendation(
            runner_prob=runner_prob,
            avoid_prob=avoid_prob,
            disagreement=disagreement,
            quality_tier=quality_tier,
        )
        
        return PredictiveAdvisory(
            symbol=symbol,
            base_quantra_score=base_quantra_score,
            model_quantra_score=model_score,
            runner_prob=runner_prob,
            quality_tier=quality_tier,
            avoid_trade_prob=avoid_prob,
            ensemble_disagreement=disagreement,
            recommendation=recommendation,
            confidence=confidence,
            reasons=reasons,
        )
    
    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        features_matrix: np.ndarray,
    ) -> List[Tuple[Dict[str, Any], PredictiveAdvisory]]:
        """
        Rank candidates by predicted runner probability.
        
        Args:
            candidates: List of candidate dictionaries from engine
            features_matrix: Feature matrix [n_candidates, n_features]
            
        Returns:
            List of (candidate, advisory) tuples sorted by runner_prob
        """
        if not self._enabled:
            return [(c, self.advise_on_candidate(
                c.get("symbol", "UNKNOWN"),
                c.get("quantrascore", 50.0),
                features_matrix[i] if i < len(features_matrix) else np.zeros(1),
            )) for i, c in enumerate(candidates)]
        
        advisories = []
        for i, candidate in enumerate(candidates):
            features = features_matrix[i] if i < len(features_matrix) else np.zeros(1)
            advisory = self.advise_on_candidate(
                candidate.get("symbol", "UNKNOWN"),
                candidate.get("quantrascore", 50.0),
                features,
            )
            advisories.append((candidate, advisory))
        
        advisories.sort(key=lambda x: x[1].runner_prob, reverse=True)
        
        return advisories
    
    def _compute_recommendation(
        self,
        runner_prob: float,
        avoid_prob: float,
        disagreement: float,
        quality_tier: str,
    ) -> Tuple[Recommendation, List[str], float]:
        """
        Apply fail-closed rules to compute recommendation.
        
        Returns:
            (recommendation, reasons, confidence)
        """
        reasons = []
        
        if disagreement > self.config.max_disagreement_allowed:
            reasons.append(
                f"High ensemble disagreement ({disagreement:.2f} > "
                f"{self.config.max_disagreement_allowed})"
            )
            return Recommendation.NEUTRAL, reasons, 0.3
        
        if avoid_prob > self.config.avoid_trade_prob_max:
            reasons.append(
                f"High avoid-trade probability ({avoid_prob:.2f} > "
                f"{self.config.avoid_trade_prob_max})"
            )
            return Recommendation.AVOID, reasons, 0.8
        
        if quality_tier == "D":
            reasons.append("Quality tier is D (poor expected outcome)")
            return Recommendation.DOWNRANK, reasons, 0.6
        
        if runner_prob >= self.config.runner_prob_uprank_threshold:
            reasons.append(
                f"High runner probability ({runner_prob:.2f} >= "
                f"{self.config.runner_prob_uprank_threshold})"
            )
            confidence = min(0.9, runner_prob)
            if quality_tier in ("A_PLUS", "A"):
                reasons.append(f"Quality tier is {quality_tier}")
                confidence = min(0.95, confidence + 0.1)
            return Recommendation.UPRANK, reasons, confidence
        
        if runner_prob < self.config.runner_prob_min_threshold:
            reasons.append(
                f"Low runner probability ({runner_prob:.2f} < "
                f"{self.config.runner_prob_min_threshold})"
            )
            return Recommendation.NEUTRAL, reasons, 0.5
        
        reasons.append("No strong signal in either direction")
        return Recommendation.NEUTRAL, reasons, 0.5
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get detailed status report."""
        return {
            "enabled": self._enabled,
            "status": self._status,
            "failure_reason": self._load_failure_reason,
            "config": {
                "model_dir": self.config.model_dir,
                "variant": self.config.variant,
                "strict_hash_verification": self.config.strict_hash_verification,
            },
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "thresholds": {
                "runner_prob_uprank": self.config.runner_prob_uprank_threshold,
                "runner_prob_min": self.config.runner_prob_min_threshold,
                "avoid_trade_max": self.config.avoid_trade_prob_max,
                "max_disagreement": self.config.max_disagreement_allowed,
            },
        }


_default_advisor: Optional[PredictiveAdvisor] = None


def get_predictive_advisor(config: Optional[PredictiveConfig] = None) -> PredictiveAdvisor:
    """Get or create the default predictive advisor."""
    global _default_advisor
    if _default_advisor is None:
        _default_advisor = PredictiveAdvisor(config)
    return _default_advisor


def reset_predictive_advisor() -> None:
    """Reset the default predictive advisor."""
    global _default_advisor
    _default_advisor = None


__all__ = [
    "PredictiveAdvisor",
    "PredictiveAdvisory",
    "PredictiveConfig",
    "Recommendation",
    "get_predictive_advisor",
    "reset_predictive_advisor",
]
