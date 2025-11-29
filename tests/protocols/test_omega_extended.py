"""
Tests for Extended Omega Directives (Ω06-Ω20)

Tests the new extended Omega directives for structure and basic behavior.
"""

import pytest
from datetime import datetime

from src.quantracore_apex.core.schemas import (
    ApexResult, Microtraits, EntropyMetrics, SuppressionMetrics,
    DriftMetrics, ContinuationMetrics, VolumeMetrics, Verdict,
    ScoreBucket, RegimeType, RiskTier, EntropyState, SuppressionState, DriftState,
    ProtocolResult,
)
from src.quantracore_apex.protocols.omega.omega import (
    OmegaDirectives, OmegaLevel, OmegaStatus
)


def create_test_apex_result(
    quantrascore: float = 50.0,
    risk_tier: RiskTier = RiskTier.MEDIUM,
    entropy_state: EntropyState = EntropyState.STABLE,
    drift_state: DriftState = DriftState.SIGNIFICANT,
    protocol_count: int = 40,
) -> ApexResult:
    """Create a test ApexResult with correct schema fields."""
    return ApexResult(
        symbol="TEST",
        timestamp=datetime.now(),
        window_hash="test123",
        quantrascore=quantrascore,
        score_bucket=ScoreBucket.NEUTRAL,
        regime=RegimeType.RANGE_BOUND,
        risk_tier=risk_tier,
        entropy_state=entropy_state,
        suppression_state=SuppressionState.NONE,
        drift_state=drift_state,
        microtraits=Microtraits(
            wick_ratio=0.3,
            body_ratio=0.7,
            bullish_pct_last20=0.5,
            compression_score=0.3,
            noise_score=0.3,
            strength_slope=0.0,
            range_density=1.0,
            volume_intensity=1.0,
            trend_consistency=0.0,
            volatility_ratio=1.0,
        ),
        entropy_metrics=EntropyMetrics(
            price_entropy=0.5,
            volume_entropy=0.5,
            combined_entropy=0.5,
            entropy_state=entropy_state,
            entropy_floor=0.1,
        ),
        suppression_metrics=SuppressionMetrics(
            suppression_level=0.3,
            suppression_state=SuppressionState.NONE,
            coil_factor=0.2,
            breakout_probability=0.5,
            is_suppressed=False,
            suppression_score=0.3,
        ),
        drift_metrics=DriftMetrics(
            drift_magnitude=0.1,
            drift_direction=0.0,
            drift_state=drift_state,
            mean_reversion_pressure=0.5,
        ),
        continuation_metrics=ContinuationMetrics(
            continuation_probability=0.5,
            momentum_strength=0.5,
            exhaustion_signal=False,
            reversal_risk=0.3,
        ),
        volume_metrics=VolumeMetrics(
            volume_spike_detected=False,
            spike_magnitude=1.0,
            volume_trend="stable",
            relative_volume=1.0,
        ),
        protocol_results=[
            ProtocolResult(protocol_id=f"T{i:02d}", fired=True, confidence=0.5)
            for i in range(1, protocol_count + 1)
        ],
        verdict=Verdict(
            action="hold",
            confidence=0.5,
            compliance_note="Research only",
        ),
    )


class TestOmega6VolatilityCap:
    """Tests for Ω6 Volatility Cap Override."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_6=False)
        result = create_test_apex_result()
        status = omega.check_omega_6(result)
        
        assert not status.active
        assert status.level == OmegaLevel.INACTIVE
    
    def test_returns_valid_status(self):
        omega = OmegaDirectives()
        result = create_test_apex_result()
        status = omega.check_omega_6(result)
        
        assert isinstance(status, OmegaStatus)
        assert status.level in [OmegaLevel.INACTIVE, OmegaLevel.ADVISORY, OmegaLevel.ENFORCED, OmegaLevel.LOCKED]


class TestOmega7MomentumDivergence:
    """Tests for Ω7 Momentum Divergence Alert."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_7=False)
        result = create_test_apex_result()
        status = omega.check_omega_7(result)
        
        assert not status.active
    
    def test_returns_valid_status(self):
        omega = OmegaDirectives()
        result = create_test_apex_result()
        status = omega.check_omega_7(result)
        
        assert isinstance(status, OmegaStatus)


class TestOmega8BollingerSqueeze:
    """Tests for Ω8 Bollinger Squeeze Warning."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_8=False)
        result = create_test_apex_result()
        status = omega.check_omega_8(result)
        
        assert not status.active


class TestOmega9MACDReversal:
    """Tests for Ω9 MACD Reversal Detection."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_9=False)
        result = create_test_apex_result()
        status = omega.check_omega_9(result)
        
        assert not status.active


class TestOmega10FearSpike:
    """Tests for Ω10 Fear Spike Override."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_10=False)
        result = create_test_apex_result()
        status = omega.check_omega_10(result)
        
        assert not status.active


class TestOmega11RSIExtreme:
    """Tests for Ω11 RSI Extreme Override."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_11=False)
        result = create_test_apex_result()
        status = omega.check_omega_11(result)
        
        assert not status.active


class TestOmega12VolumeSpike:
    """Tests for Ω12 Volume Spike Alert."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_12=False)
        result = create_test_apex_result()
        status = omega.check_omega_12(result)
        
        assert not status.active


class TestOmega13TrendWeakness:
    """Tests for Ω13 Trend Weakness Warning."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_13=False)
        result = create_test_apex_result()
        status = omega.check_omega_13(result)
        
        assert not status.active


class TestOmega14GapRisk:
    """Tests for Ω14 Gap Risk Override."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_14=False)
        result = create_test_apex_result()
        status = omega.check_omega_14(result)
        
        assert not status.active


class TestOmega15TailRisk:
    """Tests for Ω15 Tail Risk Lock."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_15=False)
        result = create_test_apex_result()
        status = omega.check_omega_15(result)
        
        assert not status.active


class TestOmega16OvernightDrift:
    """Tests for Ω16 Overnight Drift Alert."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_16=False)
        result = create_test_apex_result()
        status = omega.check_omega_16(result)
        
        assert not status.active


class TestOmega17FractalChaos:
    """Tests for Ω17 Fractal Chaos Warning."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_17=False)
        result = create_test_apex_result()
        status = omega.check_omega_17(result)
        
        assert not status.active


class TestOmega18LiquidityVoid:
    """Tests for Ω18 Liquidity Void Lock."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_18=False)
        result = create_test_apex_result()
        status = omega.check_omega_18(result)
        
        assert not status.active


class TestOmega19CorrelationBreakdown:
    """Tests for Ω19 Correlation Breakdown Alert."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_19=False)
        result = create_test_apex_result()
        status = omega.check_omega_19(result)
        
        assert not status.active
    
    def test_enforced_on_breakdown(self):
        omega = OmegaDirectives()
        result = create_test_apex_result(
            entropy_state=EntropyState.CHAOTIC,
            drift_state=DriftState.CRITICAL
        )
        status = omega.check_omega_19(result)
        
        assert status.active
        assert status.level == OmegaLevel.ENFORCED


class TestOmega20NuclearKillswitch:
    """Tests for Ω20 Nuclear Killswitch."""
    
    def test_inactive_when_disabled(self):
        omega = OmegaDirectives(enable_omega_20=False)
        result = create_test_apex_result()
        status = omega.check_omega_20(result)
        
        assert not status.active
    
    def test_locked_on_insufficient_protocols(self):
        omega = OmegaDirectives()
        result = create_test_apex_result(protocol_count=5)
        status = omega.check_omega_20(result)
        
        assert status.active
        assert status.level == OmegaLevel.LOCKED
    
    def test_locked_on_anomalous_score(self):
        omega = OmegaDirectives()
        result = create_test_apex_result(quantrascore=0)
        status = omega.check_omega_20(result)
        
        assert status.active
        assert status.level == OmegaLevel.LOCKED


class TestOmegaApplyAll:
    """Tests for apply_all with extended directives."""
    
    def test_applies_all_20_directives(self):
        omega = OmegaDirectives()
        result = create_test_apex_result()
        statuses = omega.apply_all(result)
        
        assert len(statuses) == 20
        assert "omega_1_safety" in statuses
        assert "omega_6_volatility_cap" in statuses
        assert "omega_10_fear_spike" in statuses
        assert "omega_15_tail_risk" in statuses
        assert "omega_20_nuclear_killswitch" in statuses
    
    def test_all_statuses_are_valid(self):
        omega = OmegaDirectives()
        result = create_test_apex_result()
        statuses = omega.apply_all(result)
        
        for key, status in statuses.items():
            assert isinstance(status, OmegaStatus)
            assert status.level in [
                OmegaLevel.INACTIVE, OmegaLevel.ADVISORY, 
                OmegaLevel.ENFORCED, OmegaLevel.LOCKED
            ]
    
    def test_highest_alert_level(self):
        omega = OmegaDirectives()
        result = create_test_apex_result(protocol_count=5)
        statuses = omega.apply_all(result)
        
        highest = omega.get_highest_alert_level(statuses)
        assert highest == OmegaLevel.LOCKED
