"""
Tests for Predictive Advisor engine integration.

Validates:
- Fail-closed behavior
- Recommendation logic
- Engine integration safety
- Deterministic authority preservation
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.quantracore_apex.core.integration_predictive import (
    PredictiveAdvisor,
    PredictiveAdvisory,
    PredictiveConfig,
    Recommendation,
    get_predictive_advisor,
    reset_predictive_advisor,
)


class TestPredictiveAdvisor:
    """Tests for PredictiveAdvisor."""
    
    def test_advisor_creation_disabled(self):
        """Test advisor creation with disabled config."""
        config = PredictiveConfig(enabled=False)
        advisor = PredictiveAdvisor(config)
        
        assert advisor.is_enabled is False
        assert advisor.status == "NOT_LOADED"
    
    def test_advisor_creation_no_model(self):
        """Test advisor creation with missing model."""
        config = PredictiveConfig(
            enabled=True,
            model_dir="/nonexistent/path",
            manifest_dir="/nonexistent/manifests",
        )
        advisor = PredictiveAdvisor(config)
        
        assert advisor.is_enabled is False
        assert "NOT_FOUND" in advisor.status or "NOT_LOADED" in advisor.status
    
    def test_advise_on_candidate_disabled(self):
        """Test advisory when disabled."""
        config = PredictiveConfig(enabled=False)
        advisor = PredictiveAdvisor(config)
        
        advisory = advisor.advise_on_candidate(
            symbol="AAPL",
            base_quantra_score=75.0,
            features=np.random.randn(20),
        )
        
        assert advisory.symbol == "AAPL"
        assert advisory.base_quantra_score == 75.0
        assert advisory.recommendation == Recommendation.DISABLED
        assert len(advisory.reasons) > 0


class TestPredictiveAdvisory:
    """Tests for PredictiveAdvisory dataclass."""
    
    def test_advisory_creation(self):
        """Test advisory creation."""
        advisory = PredictiveAdvisory(
            symbol="MSFT",
            base_quantra_score=80.0,
            model_quantra_score=82.0,
            runner_prob=0.75,
            quality_tier="A",
            avoid_trade_prob=0.05,
            ensemble_disagreement=0.08,
            recommendation=Recommendation.UPRANK,
            confidence=0.85,
            reasons=["High runner probability"],
        )
        
        assert advisory.symbol == "MSFT"
        assert advisory.runner_prob == 0.75
        assert advisory.recommendation == Recommendation.UPRANK
    
    def test_advisory_to_dict(self):
        """Test advisory serialization."""
        advisory = PredictiveAdvisory(
            symbol="GOOGL",
            base_quantra_score=65.0,
            model_quantra_score=68.0,
            runner_prob=0.45,
            quality_tier="B",
            avoid_trade_prob=0.15,
            ensemble_disagreement=0.12,
            recommendation=Recommendation.NEUTRAL,
            confidence=0.5,
            reasons=["No strong signal"],
        )
        
        d = advisory.to_dict()
        
        assert d["symbol"] == "GOOGL"
        assert d["runner_prob"] == 0.45
        assert d["recommendation"] == "NEUTRAL"
        assert isinstance(d["reasons"], list)


class TestRecommendationLogic:
    """Tests for recommendation computation logic."""
    
    def test_config_thresholds(self):
        """Test configuration thresholds."""
        config = PredictiveConfig(
            runner_prob_uprank_threshold=0.8,
            runner_prob_min_threshold=0.15,
            avoid_trade_prob_max=0.25,
            max_disagreement_allowed=0.15,
        )
        
        assert config.runner_prob_uprank_threshold == 0.8
        assert config.runner_prob_min_threshold == 0.15
        assert config.avoid_trade_prob_max == 0.25
        assert config.max_disagreement_allowed == 0.15


class TestFailClosedBehavior:
    """Tests for fail-closed safety behavior."""
    
    def test_disabled_returns_disabled_recommendation(self):
        """Test that disabled advisor returns DISABLED recommendation."""
        config = PredictiveConfig(enabled=False)
        advisor = PredictiveAdvisor(config)
        
        advisory = advisor.advise_on_candidate(
            symbol="TEST",
            base_quantra_score=90.0,
            features=np.zeros(10),
        )
        
        assert advisory.recommendation == Recommendation.DISABLED
        assert advisory.confidence == 0.0
    
    def test_missing_model_fails_closed(self):
        """Test that missing model fails closed."""
        config = PredictiveConfig(
            enabled=True,
            model_dir="/definitely/not/a/real/path",
            manifest_dir="/also/not/real",
        )
        advisor = PredictiveAdvisor(config)
        
        assert advisor.is_enabled is False
        
        advisory = advisor.advise_on_candidate(
            symbol="TEST",
            base_quantra_score=85.0,
            features=np.zeros(10),
        )
        
        assert advisory.recommendation == Recommendation.DISABLED
    
    def test_get_status_report(self):
        """Test status report generation."""
        config = PredictiveConfig(
            enabled=True,
            model_dir="/test/path",
            variant="mini",
        )
        advisor = PredictiveAdvisor(config)
        
        report = advisor.get_status_report()
        
        assert "enabled" in report
        assert "status" in report
        assert "config" in report
        assert "thresholds" in report
        assert report["config"]["variant"] == "mini"


class TestRankCandidates:
    """Tests for candidate ranking functionality."""
    
    def test_rank_candidates_disabled(self):
        """Test ranking when advisor is disabled."""
        config = PredictiveConfig(enabled=False)
        advisor = PredictiveAdvisor(config)
        
        candidates = [
            {"symbol": "AAPL", "quantrascore": 80},
            {"symbol": "MSFT", "quantrascore": 75},
            {"symbol": "GOOGL", "quantrascore": 85},
        ]
        
        features_matrix = np.random.randn(3, 10)
        
        ranked = advisor.rank_candidates(candidates, features_matrix)
        
        assert len(ranked) == 3
        for candidate, advisory in ranked:
            assert advisory.recommendation == Recommendation.DISABLED


class TestGlobalAdvisor:
    """Tests for global advisor singleton."""
    
    def test_get_predictive_advisor(self):
        """Test getting global advisor."""
        reset_predictive_advisor()
        
        config = PredictiveConfig(enabled=False)
        advisor = get_predictive_advisor(config)
        
        assert advisor is not None
    
    def test_reset_predictive_advisor(self):
        """Test resetting global advisor."""
        config = PredictiveConfig(enabled=False)
        advisor1 = get_predictive_advisor(config)
        
        reset_predictive_advisor()
        
        advisor2 = get_predictive_advisor(config)
        
        assert advisor1 is not advisor2


class TestDeterministicAuthority:
    """Tests for deterministic engine authority."""
    
    def test_advisory_never_overrides_engine(self):
        """
        Test that advisory never claims to override engine.
        
        This is a documentation test - the advisory is purely informational
        and the engine always has final authority.
        """
        advisory = PredictiveAdvisory(
            symbol="TEST",
            base_quantra_score=30.0,
            model_quantra_score=85.0,
            runner_prob=0.95,
            quality_tier="A_PLUS",
            avoid_trade_prob=0.01,
            ensemble_disagreement=0.02,
            recommendation=Recommendation.UPRANK,
            confidence=0.95,
            reasons=["Very high runner probability"],
        )
        
        assert advisory.base_quantra_score != advisory.model_quantra_score
        assert "override" not in str(advisory.reasons).lower()
    
    def test_avoid_recommendation_is_advisory_only(self):
        """Test that AVOID is advisory only, not a hard block."""
        advisory = PredictiveAdvisory(
            symbol="RISKY",
            base_quantra_score=70.0,
            model_quantra_score=68.0,
            runner_prob=0.15,
            quality_tier="D",
            avoid_trade_prob=0.85,
            ensemble_disagreement=0.10,
            recommendation=Recommendation.AVOID,
            confidence=0.8,
            reasons=["High avoid-trade probability"],
        )
        
        d = advisory.to_dict()
        assert d["recommendation"] == "AVOID"
        assert "advisory" not in d or d.get("override") is not True


class TestRecommendationEnumValues:
    """Tests for Recommendation enum values."""
    
    def test_all_recommendations_exist(self):
        """Test all expected recommendation values exist."""
        expected = ["UPRANK", "DOWNRANK", "AVOID", "NEUTRAL", "DISABLED"]
        
        for name in expected:
            assert hasattr(Recommendation, name)
    
    def test_recommendation_values(self):
        """Test recommendation enum values."""
        assert Recommendation.UPRANK.value == "UPRANK"
        assert Recommendation.DOWNRANK.value == "DOWNRANK"
        assert Recommendation.AVOID.value == "AVOID"
        assert Recommendation.NEUTRAL.value == "NEUTRAL"
        assert Recommendation.DISABLED.value == "DISABLED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
