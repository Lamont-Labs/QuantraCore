"""
QuantraScore computation module for QuantraCore Apex.

QuantraScore is the primary deterministic score (0-100) that
aggregates all structural analysis into a single metric.
"""

import numpy as np
from .schemas import (
    Microtraits, EntropyMetrics, SuppressionMetrics,
    DriftMetrics, ContinuationMetrics, VolumeMetrics,
    RegimeType, ScoreBucket
)


def compute_structure_score(microtraits: Microtraits) -> float:
    """
    Compute structural quality score from microtraits.
    """
    compression_bonus = microtraits.compression_score * 15
    
    noise_penalty = microtraits.noise_score * 10
    
    trend_bonus = abs(microtraits.trend_consistency) * 10
    
    body_quality = microtraits.body_ratio * 10
    
    volume_bonus = min(5, (microtraits.volume_intensity - 1) * 5) if microtraits.volume_intensity > 1 else 0
    
    score = 50 + compression_bonus + trend_bonus + body_quality + volume_bonus - noise_penalty
    
    return float(np.clip(score, 0, 100))


def compute_entropy_adjustment(entropy_metrics: EntropyMetrics) -> float:
    """
    Compute score adjustment based on entropy.
    Stable entropy = positive, chaotic = negative.
    """
    from .schemas import EntropyState
    
    if entropy_metrics.entropy_state == EntropyState.STABLE:
        return 5.0
    elif entropy_metrics.entropy_state == EntropyState.ELEVATED:
        return -5.0
    else:
        return -15.0


def compute_suppression_bonus(suppression_metrics: SuppressionMetrics) -> float:
    """
    Compute bonus for suppression (coiled structures).
    Higher suppression with high coil = higher bonus.
    """
    if suppression_metrics.suppression_level < 0.3:
        return 0.0
    
    base_bonus = suppression_metrics.suppression_level * 10
    coil_multiplier = 1 + (suppression_metrics.coil_factor * 0.3)
    
    return float(min(20, base_bonus * coil_multiplier))


def compute_drift_adjustment(drift_metrics: DriftMetrics) -> float:
    """
    Compute adjustment based on drift.
    High drift with reversion pressure = opportunity.
    """
    from .schemas import DriftState
    
    if drift_metrics.drift_state == DriftState.NONE:
        return 0.0
    
    if drift_metrics.mean_reversion_pressure > 0.6:
        return drift_metrics.drift_magnitude * 10
    elif drift_metrics.drift_state == DriftState.CRITICAL:
        return -10.0
    
    return 0.0


def compute_continuation_adjustment(continuation_metrics: ContinuationMetrics) -> float:
    """
    Compute adjustment based on continuation analysis.
    """
    if continuation_metrics.exhaustion_signal:
        return -10.0
    
    if continuation_metrics.continuation_probability > 0.7:
        return (continuation_metrics.continuation_probability - 0.5) * 20
    
    return 0.0


def compute_volume_adjustment(volume_metrics: VolumeMetrics) -> float:
    """
    Compute adjustment based on volume analysis.
    """
    adjustment = 0.0
    
    if volume_metrics.volume_spike_detected:
        adjustment += min(10, volume_metrics.spike_magnitude * 3)
    
    if volume_metrics.volume_trend == "increasing":
        adjustment += 5
    elif volume_metrics.volume_trend == "decreasing":
        adjustment -= 3
    
    return float(adjustment)


def compute_regime_modifier(regime: RegimeType) -> float:
    """
    Apply regime-based modifier to score.
    """
    modifiers = {
        RegimeType.TRENDING_UP: 1.05,
        RegimeType.TRENDING_DOWN: 1.05,
        RegimeType.RANGE_BOUND: 0.95,
        RegimeType.VOLATILE: 0.90,
        RegimeType.COMPRESSED: 1.10,
        RegimeType.UNKNOWN: 1.00,
    }
    return modifiers.get(regime, 1.0)


def score_to_bucket(score: float) -> ScoreBucket:
    """
    Convert numeric score to bucket category.
    """
    if score <= 20:
        return ScoreBucket.VERY_LOW
    elif score <= 40:
        return ScoreBucket.LOW
    elif score <= 60:
        return ScoreBucket.NEUTRAL
    elif score <= 80:
        return ScoreBucket.HIGH
    else:
        return ScoreBucket.VERY_HIGH


def compute_quantrascore(
    microtraits: Microtraits,
    entropy_metrics: EntropyMetrics,
    suppression_metrics: SuppressionMetrics,
    drift_metrics: DriftMetrics,
    continuation_metrics: ContinuationMetrics,
    volume_metrics: VolumeMetrics,
    regime: RegimeType,
) -> tuple:
    """
    Compute the final QuantraScore (0-100).
    
    Returns:
        Tuple of (score, bucket)
    """
    base_score = compute_structure_score(microtraits)
    
    entropy_adj = compute_entropy_adjustment(entropy_metrics)
    suppression_bonus = compute_suppression_bonus(suppression_metrics)
    drift_adj = compute_drift_adjustment(drift_metrics)
    continuation_adj = compute_continuation_adjustment(continuation_metrics)
    volume_adj = compute_volume_adjustment(volume_metrics)
    
    adjusted_score = base_score + entropy_adj + suppression_bonus + drift_adj + continuation_adj + volume_adj
    
    regime_mod = compute_regime_modifier(regime)
    final_score = adjusted_score * regime_mod
    
    final_score = float(np.clip(final_score, 0, 100))
    bucket = score_to_bucket(final_score)
    
    return final_score, bucket
