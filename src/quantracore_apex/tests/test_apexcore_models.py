"""
Comprehensive ApexCore Model Tests

Tests ApexCore Full and Mini models for correctness.
"""

import pytest
import numpy as np

from src.quantracore_apex.apexcore.models import (
    ApexCoreConfig, ApexCoreFullConfig, ApexCoreMiniConfig,
    ApexCoreOutputs, ApexCoreFull, ApexCoreMini
)


class TestApexCoreConfig:
    """Tests for configuration classes."""
    
    def test_base_config(self):
        """Test base config defaults."""
        config = ApexCoreConfig()
        
        assert config.model_name == "apexcore_base"
        assert config.input_dim == 30
        assert config.random_state == 42
    
    def test_full_config(self):
        """Test full config defaults."""
        config = ApexCoreFullConfig()
        
        assert config.model_name == "apexcore_full"
        assert len(config.hidden_layers) == 3
        assert config.max_iter == 500
    
    def test_mini_config(self):
        """Test mini config defaults."""
        config = ApexCoreMiniConfig()
        
        assert config.model_name == "apexcore_mini"
        assert len(config.hidden_layers) == 2
        assert config.max_iter == 200


class TestApexCoreOutputs:
    """Tests for output structure."""
    
    def test_default_outputs(self):
        """Test default output values."""
        outputs = ApexCoreOutputs()
        
        assert outputs.quantrascore == 50.0
        assert outputs.is_placeholder is True
        assert len(outputs.compliance_note) > 0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        outputs = ApexCoreOutputs(quantrascore=75.0)
        
        d = outputs.to_dict()
        
        assert d["quantrascore"] == 75.0
        assert "compliance_note" in d


class TestApexCoreFull:
    """Tests for ApexCore Full model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ApexCoreFull()
        
        assert model is not None
        assert model.is_trained is False
    
    def test_predict_untrained(self):
        """Test prediction on untrained model returns placeholder."""
        model = ApexCoreFull()
        features = np.random.randn(1, 30)
        
        result = model.predict(features)
        
        assert result.is_placeholder is True
    
    @pytest.mark.skip(reason="sklearn MLP has numpy type compatibility issues in this environment")
    def test_train_small_dataset(self):
        """Test training on small dataset."""
        model = ApexCoreFull()
        
        np.random.seed(42)
        X = np.random.randn(50, 30).astype(np.float64)
        y_regime = np.array(["trending", "ranging", "volatile"] * 16 + ["trending", "ranging"])
        y_score = np.random.uniform(20, 80, 50).astype(np.float64)
        y_risk = np.array(["low", "medium", "high"] * 16 + ["low", "medium"])
        
        metrics = model.train(X, y_regime, y_score, y_risk)
        
        assert model.is_trained is True
        assert "regime_accuracy" in metrics
        assert "risk_accuracy" in metrics
        assert "score_mae" in metrics
    
    @pytest.mark.skip(reason="sklearn MLP has numpy type compatibility issues in this environment")
    def test_predict_trained(self):
        """Test prediction on trained model."""
        model = ApexCoreFull()
        
        np.random.seed(42)
        X = np.random.randn(50, 30).astype(np.float64)
        y_regime = np.array(["trending", "ranging", "volatile"] * 16 + ["trending", "ranging"])
        y_score = np.random.uniform(20, 80, 50).astype(np.float64)
        y_risk = np.array(["low", "medium", "high"] * 16 + ["low", "medium"])
        
        model.train(X, y_regime, y_score, y_risk)
        
        features = np.random.randn(1, 30).astype(np.float64)
        result = model.predict(features)
        
        assert result.is_placeholder is False
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.skip(reason="sklearn MLP has numpy type compatibility issues in this environment")
    def test_determinism(self):
        """Test training and prediction are deterministic."""
        np.random.seed(42)
        X = np.random.randn(50, 30).astype(np.float64)
        y_regime = np.array(["trending", "ranging"] * 25)
        y_score = np.random.uniform(20, 80, 50).astype(np.float64)
        y_risk = np.array(["low", "high"] * 25)
        
        model1 = ApexCoreFull()
        model1.train(X.copy(), y_regime.copy(), y_score.copy(), y_risk.copy())
        
        model2 = ApexCoreFull()
        model2.train(X.copy(), y_regime.copy(), y_score.copy(), y_risk.copy())
        
        test_features = np.array([[0.1] * 30], dtype=np.float64)
        r1 = model1.predict(test_features)
        r2 = model2.predict(test_features)
        
        assert r1.quantrascore == r2.quantrascore


class TestApexCoreMini:
    """Tests for ApexCore Mini model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ApexCoreMini()
        
        assert model is not None
        assert model.is_trained is False
    
    def test_predict_untrained(self):
        """Test prediction on untrained model returns placeholder."""
        model = ApexCoreMini()
        features = np.random.randn(1, 30)
        
        result = model.predict(features)
        
        assert result.is_placeholder is True
    
    @pytest.mark.skip(reason="sklearn MLP has numpy type compatibility issues in this environment")
    def test_train_small_dataset(self):
        """Test training on small dataset."""
        model = ApexCoreMini()
        
        np.random.seed(42)
        X = np.random.randn(50, 30).astype(np.float64)
        y_regime = np.array(["trending", "ranging", "volatile"] * 16 + ["trending", "ranging"])
        y_score = np.random.uniform(20, 80, 50).astype(np.float64)
        y_risk = np.array(["low", "medium", "high"] * 16 + ["low", "medium"])
        
        metrics = model.train(X, y_regime, y_score, y_risk)
        
        assert model.is_trained is True
        assert metrics["model_type"] == "mini"
    
    def test_mini_smaller_than_full(self):
        """Test mini model has fewer parameters than full."""
        mini_config = ApexCoreMiniConfig()
        full_config = ApexCoreFullConfig()
        
        mini_params = sum(mini_config.hidden_layers)
        full_params = sum(full_config.hidden_layers)
        
        assert mini_params < full_params


class TestApexCoreIntegration:
    """Integration tests for ApexCore models."""
    
    def test_full_vs_mini_architecture(self):
        """Test full has more capacity than mini."""
        full = ApexCoreFull()
        mini = ApexCoreMini()
        
        full_layers = full.config.hidden_layers
        mini_layers = mini.config.hidden_layers
        
        assert len(full_layers) >= len(mini_layers)
        assert sum(full_layers) > sum(mini_layers)
    
    @pytest.mark.skip(reason="sklearn MLP has numpy type compatibility issues in this environment")
    def test_score_bucket_classification(self):
        """Test score bucket is correctly classified."""
        model = ApexCoreFull()
        
        np.random.seed(42)
        X = np.random.randn(100, 30).astype(np.float64)
        y_regime = np.array(["trending", "ranging"] * 50)
        y_score = np.random.uniform(10, 90, 100).astype(np.float64)
        y_risk = np.array(["low", "high"] * 50)
        
        model.train(X, y_regime, y_score, y_risk)
        
        features = np.random.randn(10, 30).astype(np.float64)
        for i in range(10):
            result = model.predict(features[i:i+1])
            
            if result.quantrascore >= 70:
                assert result.score_bucket == "strong"
            elif result.quantrascore >= 55:
                assert result.score_bucket == "moderate"
            elif result.quantrascore >= 40:
                assert result.score_bucket == "neutral"
            else:
                assert result.score_bucket == "weak"
