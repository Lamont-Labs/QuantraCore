"""
Redundant Scoring Architecture for v9.0-A
Implements dual-path QuantraScore with shadow scorer cross-check.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ScoreConsistencyStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ScoreConsistencyResult:
    """Result of score consistency check."""
    status: ScoreConsistencyStatus
    primary_score: float
    shadow_score: float
    primary_band: str
    shadow_band: str
    absolute_diff: float
    band_match: bool
    details: Dict[str, Any]


class ShadowScorer:
    """
    Independent shadow scorer for QuantraScore verification.
    Uses slightly different aggregation routes to cross-check the primary scorer.
    """
    
    ALLOWED_ABSOLUTE_DIFF = 5.0
    WARNING_THRESHOLD = 10.0
    
    def __init__(self):
        self.consistency_log = []
    
    def compute_shadow_score(
        self,
        protocol_results: Dict[str, Any],
        regime: str,
        risk_tier: str,
        monster_runner_state: str,
        microtraits: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """
        Compute shadow QuantraScore using alternative aggregation.
        
        Instead of direct weighted sum, groups factors differently:
        1. Structural factors (regime, risk)
        2. Protocol activation factors
        3. MonsterRunner boost/penalty
        
        Returns (shadow_score, shadow_band).
        """
        base_score = 50.0
        
        structural_adjustment = self._compute_structural_adjustment(regime, risk_tier)
        protocol_adjustment = self._compute_protocol_adjustment(protocol_results)
        runner_adjustment = self._compute_runner_adjustment(monster_runner_state)
        
        shadow_score = base_score + structural_adjustment + protocol_adjustment + runner_adjustment
        shadow_score = max(0.0, min(100.0, shadow_score))
        
        shadow_band = self._score_to_band(shadow_score)
        
        return shadow_score, shadow_band
    
    def _compute_structural_adjustment(self, regime: str, risk_tier: str) -> float:
        """Compute adjustment based on regime and risk."""
        adjustment = 0.0
        
        regime_adjustments = {
            "trending_up": 10.0,
            "trending_down": -5.0,
            "range_bound": 0.0,
            "volatile": -10.0,
            "breakout": 15.0,
            "breakdown": -15.0,
        }
        adjustment += regime_adjustments.get(regime, 0.0)
        
        risk_adjustments = {
            "low": 5.0,
            "moderate": 0.0,
            "elevated": -5.0,
            "high": -10.0,
            "extreme": -20.0,
        }
        adjustment += risk_adjustments.get(risk_tier, 0.0)
        
        return adjustment
    
    def _compute_protocol_adjustment(self, protocol_results: Dict[str, Any]) -> float:
        """Compute adjustment based on protocol activations."""
        adjustment = 0.0
        
        fired_count = 0
        high_confidence_count = 0
        
        for protocol_id, result in protocol_results.items():
            if isinstance(result, dict):
                if result.get("fired", False):
                    fired_count += 1
                    confidence = result.get("confidence", 0.0)
                    if confidence >= 0.7:
                        high_confidence_count += 1
                    adjustment += confidence * 2.0
        
        if fired_count > 10:
            adjustment += 5.0
        if high_confidence_count >= 5:
            adjustment += 5.0
            
        return min(adjustment, 20.0)
    
    def _compute_runner_adjustment(self, monster_runner_state: str) -> float:
        """Compute adjustment based on MonsterRunner state."""
        runner_adjustments = {
            "primed": 10.0,
            "active": 15.0,
            "idle": 0.0,
            "cooldown": -5.0,
        }
        return runner_adjustments.get(monster_runner_state, 0.0)
    
    def _score_to_band(self, score: float) -> str:
        """Convert numeric score to band."""
        if score >= 80:
            return "strong"
        elif score >= 60:
            return "moderate"
        elif score >= 40:
            return "weak"
        elif score >= 20:
            return "poor"
        else:
            return "reject"
    
    def check_consistency(
        self,
        primary_score: float,
        primary_band: str,
        shadow_score: float,
        shadow_band: str
    ) -> ScoreConsistencyResult:
        """
        Check consistency between primary and shadow scores.
        """
        absolute_diff = abs(primary_score - shadow_score)
        band_match = primary_band == shadow_band
        
        if absolute_diff <= self.ALLOWED_ABSOLUTE_DIFF and band_match:
            status = ScoreConsistencyStatus.OK
        elif absolute_diff <= self.WARNING_THRESHOLD:
            status = ScoreConsistencyStatus.WARNING
        else:
            status = ScoreConsistencyStatus.FAIL
        
        result = ScoreConsistencyResult(
            status=status,
            primary_score=primary_score,
            shadow_score=shadow_score,
            primary_band=primary_band,
            shadow_band=shadow_band,
            absolute_diff=absolute_diff,
            band_match=band_match,
            details={
                "allowed_diff": self.ALLOWED_ABSOLUTE_DIFF,
                "warning_threshold": self.WARNING_THRESHOLD,
            }
        )
        
        if status != ScoreConsistencyStatus.OK:
            logger.warning(
                f"Score consistency {status.value}: "
                f"primary={primary_score:.1f} ({primary_band}), "
                f"shadow={shadow_score:.1f} ({shadow_band}), "
                f"diff={absolute_diff:.1f}"
            )
        
        self.consistency_log.append(result)
        return result


class RedundantScorer:
    """
    Main redundant scoring interface.
    Wraps primary QuantraScore and shadow scorer for dual-path verification.
    """
    
    def __init__(self):
        self.shadow_scorer = ShadowScorer()
        self.consistency_stats = {
            "total_checks": 0,
            "ok_count": 0,
            "warning_count": 0,
            "fail_count": 0,
        }
    
    def compute_with_verification(
        self,
        primary_score: float,
        primary_band: str,
        protocol_results: Dict[str, Any],
        regime: str,
        risk_tier: str,
        monster_runner_state: str = "idle",
        microtraits: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute verified QuantraScore with shadow check.
        
        Returns dict with primary score, shadow score, and consistency result.
        """
        shadow_score, shadow_band = self.shadow_scorer.compute_shadow_score(
            protocol_results=protocol_results,
            regime=regime,
            risk_tier=risk_tier,
            monster_runner_state=monster_runner_state,
            microtraits=microtraits
        )
        
        consistency = self.shadow_scorer.check_consistency(
            primary_score=primary_score,
            primary_band=primary_band,
            shadow_score=shadow_score,
            shadow_band=shadow_band
        )
        
        self.consistency_stats["total_checks"] += 1
        if consistency.status == ScoreConsistencyStatus.OK:
            self.consistency_stats["ok_count"] += 1
        elif consistency.status == ScoreConsistencyStatus.WARNING:
            self.consistency_stats["warning_count"] += 1
        else:
            self.consistency_stats["fail_count"] += 1
        
        return {
            "primary_score": primary_score,
            "primary_band": primary_band,
            "shadow_score": shadow_score,
            "shadow_band": shadow_band,
            "consistency_status": consistency.status.value,
            "consistency_ok": consistency.status == ScoreConsistencyStatus.OK,
            "absolute_diff": consistency.absolute_diff,
            "band_match": consistency.band_match,
        }
    
    def get_consistency_stats(self) -> Dict[str, Any]:
        """Get summary statistics for consistency checks."""
        total = self.consistency_stats["total_checks"]
        return {
            **self.consistency_stats,
            "ok_rate": self.consistency_stats["ok_count"] / total if total > 0 else 0.0,
            "warning_rate": self.consistency_stats["warning_count"] / total if total > 0 else 0.0,
            "fail_rate": self.consistency_stats["fail_count"] / total if total > 0 else 0.0,
        }
    
    def reset_stats(self):
        """Reset consistency statistics."""
        self.consistency_stats = {
            "total_checks": 0,
            "ok_count": 0,
            "warning_count": 0,
            "fail_count": 0,
        }
