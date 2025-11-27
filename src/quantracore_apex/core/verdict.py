"""
Verdict generation module for QuantraCore Apex.

Generates final verdict based on all analysis components.
All verdicts are framed as structural observations, NOT trading signals.
"""

from typing import List
from .schemas import (
    Verdict, ScoreBucket, RegimeType, RiskTier,
    EntropyState, SuppressionState, DriftState
)


def determine_risk_tier(
    entropy_state: EntropyState,
    drift_state: DriftState,
    suppression_state: SuppressionState,
    quantrascore: float
) -> RiskTier:
    """
    Determine overall risk tier based on analysis.
    """
    risk_points = 0
    
    if entropy_state == EntropyState.CHAOTIC:
        risk_points += 3
    elif entropy_state == EntropyState.ELEVATED:
        risk_points += 1
    
    if drift_state == DriftState.CRITICAL:
        risk_points += 3
    elif drift_state == DriftState.SIGNIFICANT:
        risk_points += 2
    elif drift_state == DriftState.MILD:
        risk_points += 1
    
    if suppression_state == SuppressionState.HEAVY:
        risk_points += 1
    
    if quantrascore < 30 or quantrascore > 85:
        risk_points += 1
    
    if risk_points >= 5:
        return RiskTier.EXTREME
    elif risk_points >= 3:
        return RiskTier.HIGH
    elif risk_points >= 1:
        return RiskTier.MEDIUM
    else:
        return RiskTier.LOW


def identify_risk_factors(
    entropy_state: EntropyState,
    drift_state: DriftState,
    suppression_state: SuppressionState,
    regime: RegimeType,
    quantrascore: float
) -> List[str]:
    """
    Identify specific risk factors in current analysis.
    """
    factors = []
    
    if entropy_state == EntropyState.CHAOTIC:
        factors.append("Chaotic price entropy detected")
    elif entropy_state == EntropyState.ELEVATED:
        factors.append("Elevated price entropy")
    
    if drift_state == DriftState.CRITICAL:
        factors.append("Critical drift from mean")
    elif drift_state == DriftState.SIGNIFICANT:
        factors.append("Significant price drift")
    
    if suppression_state == SuppressionState.HEAVY:
        factors.append("Heavy suppression - potential volatility ahead")
    
    if regime == RegimeType.VOLATILE:
        factors.append("Volatile regime active")
    
    if quantrascore > 85:
        factors.append("Extreme high score - potential mean reversion")
    elif quantrascore < 15:
        factors.append("Extreme low score - structural weakness")
    
    return factors


def determine_action(
    score_bucket: ScoreBucket,
    risk_tier: RiskTier,
    regime: RegimeType
) -> str:
    """
    Determine structural action label.
    These are analytical categories, NOT trading signals.
    """
    if risk_tier == RiskTier.EXTREME:
        return "caution_elevated_risk"
    
    if score_bucket == ScoreBucket.VERY_HIGH:
        if regime in [RegimeType.TRENDING_UP, RegimeType.COMPRESSED]:
            return "structural_probability_elevated"
        else:
            return "elevated_with_conditions"
    elif score_bucket == ScoreBucket.HIGH:
        return "structural_interest"
    elif score_bucket == ScoreBucket.NEUTRAL:
        return "neutral_observation"
    elif score_bucket == ScoreBucket.LOW:
        return "structural_weakness"
    else:
        return "minimal_structural_interest"


def determine_primary_signal(
    regime: RegimeType,
    suppression_state: SuppressionState,
    score_bucket: ScoreBucket
) -> str:
    """
    Determine primary signal description.
    """
    if suppression_state == SuppressionState.HEAVY:
        return "Compression pattern detected"
    
    if regime == RegimeType.TRENDING_UP:
        return "Uptrend structure"
    elif regime == RegimeType.TRENDING_DOWN:
        return "Downtrend structure"
    elif regime == RegimeType.COMPRESSED:
        return "Compressed structure"
    elif regime == RegimeType.VOLATILE:
        return "Volatile structure"
    elif regime == RegimeType.RANGE_BOUND:
        return "Range-bound structure"
    
    return "Mixed structure"


def compute_verdict_confidence(
    quantrascore: float,
    risk_tier: RiskTier,
    entropy_state: EntropyState
) -> float:
    """
    Compute confidence in the verdict.
    """
    base_confidence = 0.7
    
    if entropy_state == EntropyState.STABLE:
        base_confidence += 0.1
    elif entropy_state == EntropyState.CHAOTIC:
        base_confidence -= 0.2
    
    if risk_tier == RiskTier.LOW:
        base_confidence += 0.05
    elif risk_tier == RiskTier.EXTREME:
        base_confidence -= 0.15
    
    score_extremity = abs(quantrascore - 50) / 50
    base_confidence += score_extremity * 0.1
    
    return float(max(0.3, min(0.95, base_confidence)))


def build_verdict(
    quantrascore: float,
    score_bucket: ScoreBucket,
    regime: RegimeType,
    entropy_state: EntropyState,
    suppression_state: SuppressionState,
    drift_state: DriftState,
) -> tuple:
    """
    Build complete verdict from analysis components.
    
    Returns:
        Tuple of (Verdict, RiskTier)
    """
    risk_tier = determine_risk_tier(
        entropy_state, drift_state, suppression_state, quantrascore
    )
    
    risk_factors = identify_risk_factors(
        entropy_state, drift_state, suppression_state, regime, quantrascore
    )
    
    action = determine_action(score_bucket, risk_tier, regime)
    primary_signal = determine_primary_signal(regime, suppression_state, score_bucket)
    confidence = compute_verdict_confidence(quantrascore, risk_tier, entropy_state)
    
    verdict = Verdict(
        action=action,
        confidence=confidence,
        primary_signal=primary_signal,
        risk_factors=risk_factors,
        compliance_note="Structural analysis only - not trading advice"
    )
    
    return verdict, risk_tier
