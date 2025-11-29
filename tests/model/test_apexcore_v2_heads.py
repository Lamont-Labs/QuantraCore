"""
Tests for ApexCore V2 model heads and architectures.

Validates:
- Model instantiation (Big and Mini variants)
- Forward pass output shapes and types
- Head-specific predictions
- Ensemble behavior
"""

import pytest
import numpy as np

from src.quantracore_apex.apexcore.apexcore_v2 import (
    ApexCoreV2Model,
    ApexCoreV2Big,
    ApexCoreV2Mini,
    ApexCoreV2Ensemble,
    ApexCoreV2Config,
    ModelVariant,
    QuantraScoreHead,
    RunnerProbHead,
    QualityTierHead,
    AvoidTradeHead,
    RegimeHead,
    FeatureEncoder,
)


class TestApexCoreV2Big:
    """Tests for ApexCoreV2Big model."""
    
    def test_instantiation(self):
        """Test model instantiation."""
        model = ApexCoreV2Big()
        
        assert model is not None
        assert model.variant == ModelVariant.BIG
        assert len(model.heads) == 5
    
    def test_forward_unfitted(self):
        """Test forward pass on unfitted model returns defaults."""
        model = ApexCoreV2Big()
        X = np.random.randn(10, 20)
        
        outputs = model.forward(X)
        
        assert "quantra_score" in outputs
        assert "runner_prob" in outputs
        assert "quality_logits" in outputs
        assert "avoid_trade_prob" in outputs
        assert "regime_logits" in outputs
        
        assert outputs["quantra_score"].shape == (10,)
        assert np.all(outputs["quantra_score"] == 50.0)
    
    def test_forward_fitted(self):
        """Test forward pass on fitted model."""
        model = ApexCoreV2Big()
        
        n_samples = 100
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.randint(0, 2, n_samples),
            "future_quality_tier": np.random.choice(
                ["A_PLUS", "A", "B", "C", "D"], n_samples
            ),
            "avoid_trade": np.random.randint(0, 2, n_samples),
            "regime_label": np.random.choice(
                ["trend_up", "trend_down", "chop", "squeeze", "crash"], n_samples
            ),
        }
        
        model.fit(X, targets)
        
        X_test = np.random.randn(5, n_features)
        outputs = model.forward(X_test)
        
        assert outputs["quantra_score"].shape == (5,)
        assert np.all((outputs["quantra_score"] >= 0) & (outputs["quantra_score"] <= 100))
        
        assert outputs["runner_prob"].shape == (5,)
        assert np.all((outputs["runner_prob"] >= 0) & (outputs["runner_prob"] <= 1))
        
        assert outputs["quality_logits"].shape == (5, 5)
        
        assert outputs["avoid_trade_prob"].shape == (5,)
        
        assert outputs["regime_logits"].shape == (5, 5)
    
    def test_all_heads_present(self):
        """Verify all required heads exist."""
        model = ApexCoreV2Big()
        
        required_heads = [
            "quantra_score",
            "runner_prob",
            "quality_tier",
            "avoid_trade",
            "regime",
        ]
        
        for head_name in required_heads:
            assert head_name in model.heads, f"Missing head: {head_name}"


class TestApexCoreV2Mini:
    """Tests for ApexCoreV2Mini model."""
    
    def test_instantiation(self):
        """Test mini model instantiation."""
        model = ApexCoreV2Mini()
        
        assert model is not None
        assert model.variant == ModelVariant.MINI
    
    def test_forward_shapes(self):
        """Test forward pass output shapes."""
        model = ApexCoreV2Mini()
        X = np.random.randn(8, 15)
        
        outputs = model.forward(X)
        
        assert outputs["quantra_score"].shape == (8,)
        assert outputs["runner_prob"].shape == (8,)
    
    def test_mini_vs_big_same_interface(self):
        """Verify Mini and Big have same interface."""
        big = ApexCoreV2Big()
        mini = ApexCoreV2Mini()
        
        X = np.random.randn(5, 10)
        
        big_outputs = big.forward(X)
        mini_outputs = mini.forward(X)
        
        assert set(big_outputs.keys()) == set(mini_outputs.keys())


class TestApexCoreV2Ensemble:
    """Tests for ApexCoreV2Ensemble."""
    
    def test_ensemble_creation(self):
        """Test ensemble instantiation."""
        ensemble = ApexCoreV2Ensemble(ensemble_size=3)
        
        assert len(ensemble.members) == 3
        assert ensemble.ensemble_size == 3
    
    def test_ensemble_forward_unfitted(self):
        """Test ensemble forward before fitting."""
        ensemble = ApexCoreV2Ensemble(ensemble_size=3)
        X = np.random.randn(10, 20)
        
        outputs = ensemble.forward(X)
        
        assert "quantra_score" in outputs
        assert "disagreement" in outputs
        assert outputs["quantra_score"].shape == (10,)
    
    def test_ensemble_forward_fitted(self):
        """Test ensemble forward after fitting."""
        ensemble = ApexCoreV2Ensemble(ensemble_size=3, variant=ModelVariant.MINI)
        
        n_samples = 50
        n_features = 15
        X = np.random.randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.randint(0, 2, n_samples),
            "future_quality_tier": np.random.choice(
                ["A_PLUS", "A", "B", "C", "D"], n_samples
            ),
            "avoid_trade": np.random.randint(0, 2, n_samples),
            "regime_label": np.random.choice(
                ["trend_up", "trend_down", "chop", "squeeze", "crash"], n_samples
            ),
        }
        
        ensemble.fit(X, targets, bootstrap=True)
        
        X_test = np.random.randn(5, n_features)
        outputs = ensemble.forward(X_test)
        
        assert "disagreement" in outputs
        assert "runner_prob" in outputs["disagreement"]
        
        assert outputs["quantra_score"].shape == (5,)
        assert outputs["runner_prob"].shape == (5,)
    
    def test_ensemble_disagreement_calculation(self):
        """Test that disagreement is computed correctly."""
        ensemble = ApexCoreV2Ensemble(ensemble_size=5, variant=ModelVariant.MINI)
        
        n_samples = 100
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.randint(0, 2, n_samples),
            "future_quality_tier": np.random.choice(["A", "B", "C"], n_samples),
            "avoid_trade": np.random.randint(0, 2, n_samples),
            "regime_label": np.random.choice(["chop", "trend_up"], n_samples),
        }
        
        ensemble.fit(X, targets, bootstrap=True)
        
        X_test = np.random.randn(10, n_features)
        outputs = ensemble.forward(X_test)
        
        assert "disagreement" in outputs
        for key in ["quantra_score", "runner_prob"]:
            assert key in outputs["disagreement"]
            assert outputs["disagreement"][key].shape == (10,)
    
    def test_ensemble_member_hashes(self):
        """Test ensemble member hash generation."""
        ensemble = ApexCoreV2Ensemble(ensemble_size=3)
        
        hashes = ensemble.get_member_hashes()
        
        assert len(hashes) == 3
        assert "member_0" in hashes
        assert "member_1" in hashes
        assert "member_2" in hashes


class TestIndividualHeads:
    """Tests for individual model heads."""
    
    def test_quantra_score_head(self):
        """Test QuantraScore regression head."""
        head = QuantraScoreHead(variant=ModelVariant.MINI, n_estimators=10, max_depth=3)
        
        X = np.random.randn(50, 10)
        y = np.random.uniform(0, 100, 50)
        
        head.fit(X, y)
        preds = head.predict(X)
        
        assert preds.shape == (50,)
        assert np.all((preds >= 0) & (preds <= 100))
    
    def test_runner_prob_head(self):
        """Test runner probability head."""
        head = RunnerProbHead(variant=ModelVariant.MINI, n_estimators=10, max_depth=3)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        head.fit(X, y)
        proba = head.predict_proba(X)
        
        assert proba.shape == (50,)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_quality_tier_head(self):
        """Test quality tier classification head."""
        head = QualityTierHead(variant=ModelVariant.MINI, n_estimators=10, max_depth=3)
        
        X = np.random.randn(50, 10)
        y = np.random.choice(["A_PLUS", "A", "B", "C", "D"], 50)
        
        head.fit(X, y)
        preds = head.predict(X)
        proba = head.predict_proba(X)
        
        assert len(preds) == 50
        assert proba.shape == (50, 5)
    
    def test_avoid_trade_head(self):
        """Test avoid trade classification head."""
        head = AvoidTradeHead(variant=ModelVariant.MINI, n_estimators=10, max_depth=3)
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        head.fit(X, y)
        proba = head.predict_proba(X)
        
        assert proba.shape == (50,)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_regime_head(self):
        """Test regime classification head."""
        head = RegimeHead(variant=ModelVariant.MINI, n_estimators=10, max_depth=3)
        
        X = np.random.randn(50, 10)
        y = np.random.choice(["trend_up", "trend_down", "chop", "squeeze", "crash"], 50)
        
        head.fit(X, y)
        preds = head.predict(X)
        proba = head.predict_proba(X)
        
        assert len(preds) == 50
        assert proba.shape == (50, 5)


class TestFeatureEncoder:
    """Tests for feature encoder."""
    
    def test_encoder_fit_transform(self):
        """Test encoder fit and transform."""
        encoder = FeatureEncoder()
        
        X = np.random.randn(100, 20)
        X_transformed = encoder.fit_transform(X)
        
        assert X_transformed.shape == X.shape
    
    def test_encoder_transform_after_fit(self):
        """Test encoder transform on new data."""
        encoder = FeatureEncoder()
        
        X_train = np.random.randn(100, 20)
        encoder.fit(X_train)
        
        X_test = np.random.randn(10, 20)
        X_transformed = encoder.transform(X_test)
        
        assert X_transformed.shape == X_test.shape


class TestModelConfig:
    """Tests for model configuration."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ApexCoreV2Config()
        
        assert config.variant == ModelVariant.BIG
        assert config.random_state == 42
        assert config.n_estimators_big == 100
        assert config.n_estimators_mini == 30
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = ApexCoreV2Config(
            variant=ModelVariant.MINI,
            random_state=123,
            n_estimators_mini=50,
        )
        
        assert config.variant == ModelVariant.MINI
        assert config.random_state == 123
        assert config.n_estimators_mini == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
