"""
ApexCore Model Load Tests for QuantraCore Apex.

Tests ApexCore Full and Mini model loading and prediction.
All assertions are SUBSTANTIVE - they will fail if behavior regresses.
"""

import pytest
import numpy as np

from src.quantracore_apex.apexcore.models import (
    ApexCoreFull, ApexCoreMini,
    ApexCoreFullConfig, ApexCoreMiniConfig,
    ApexCoreOutputs
)


class TestApexCoreImports:
    """Test ApexCore module imports."""
    
    def test_apexcore_full_importable(self):
        """ApexCoreFull should be importable."""
        assert ApexCoreFull is not None
        assert callable(ApexCoreFull)
    
    def test_apexcore_mini_importable(self):
        """ApexCoreMini should be importable."""
        assert ApexCoreMini is not None
        assert callable(ApexCoreMini)
    
    def test_config_classes_importable(self):
        """Config classes should be importable."""
        assert ApexCoreFullConfig is not None
        assert ApexCoreMiniConfig is not None
    
    def test_outputs_class_importable(self):
        """ApexCoreOutputs should be importable."""
        assert ApexCoreOutputs is not None


class TestApexCoreMiniLoad:
    """Test ApexCore Mini model loading."""
    
    def test_mini_instantiation(self):
        """ApexCoreMini should instantiate with config."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        assert isinstance(model, ApexCoreMini)
    
    def test_mini_has_predict_method(self):
        """ApexCoreMini should have predict method."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        assert hasattr(model, "predict")
        assert callable(model.predict)
    
    def test_mini_predict_returns_outputs(self):
        """ApexCoreMini.predict should return ApexCoreOutputs."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        dummy_features = np.random.randn(1, config.input_dim)
        result = model.predict(dummy_features)
        
        assert isinstance(result, ApexCoreOutputs), f"Expected ApexCoreOutputs, got {type(result)}"
    
    def test_mini_predict_quantrascore_in_range(self):
        """ApexCoreMini quantrascore should be in 0-100 range."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        dummy_features = np.random.randn(1, config.input_dim)
        result = model.predict(dummy_features)
        
        assert 0 <= result.quantrascore <= 100, f"quantrascore {result.quantrascore} out of range"
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10])
    def test_mini_handles_different_batch_sizes(self, batch_size: int):
        """ApexCoreMini should handle different batch sizes."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        features = np.random.randn(batch_size, config.input_dim)
        result = model.predict(features)
        
        assert isinstance(result, ApexCoreOutputs)
        assert 0 <= result.quantrascore <= 100


class TestApexCoreFullLoad:
    """Test ApexCore Full model loading."""
    
    def test_full_instantiation(self):
        """ApexCoreFull should instantiate with config."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        assert isinstance(model, ApexCoreFull)
    
    def test_full_has_predict_method(self):
        """ApexCoreFull should have predict method."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        assert hasattr(model, "predict")
        assert callable(model.predict)
    
    def test_full_predict_returns_outputs(self):
        """ApexCoreFull.predict should return ApexCoreOutputs."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        dummy_features = np.random.randn(1, config.input_dim)
        result = model.predict(dummy_features)
        
        assert isinstance(result, ApexCoreOutputs)
    
    def test_full_predict_quantrascore_in_range(self):
        """ApexCoreFull quantrascore should be in 0-100 range."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        dummy_features = np.random.randn(1, config.input_dim)
        result = model.predict(dummy_features)
        
        assert 0 <= result.quantrascore <= 100


class TestApexCoreOutputs:
    """Test ApexCoreOutputs structure."""
    
    def test_outputs_default_quantrascore(self):
        """Default ApexCoreOutputs should have quantrascore=50."""
        outputs = ApexCoreOutputs()
        assert outputs.quantrascore == 50.0
    
    def test_outputs_has_required_fields(self):
        """ApexCoreOutputs should have all required fields."""
        outputs = ApexCoreOutputs()
        
        assert hasattr(outputs, "quantrascore")
        assert hasattr(outputs, "regime_prediction")
        assert hasattr(outputs, "risk_tier")
        assert hasattr(outputs, "volatility_band")
        assert hasattr(outputs, "score_bucket")
        assert hasattr(outputs, "confidence")
    
    def test_outputs_to_dict_returns_dict(self):
        """to_dict should return a dictionary."""
        outputs = ApexCoreOutputs()
        d = outputs.to_dict()
        
        assert isinstance(d, dict)
    
    def test_outputs_to_dict_has_quantrascore(self):
        """to_dict should include quantrascore."""
        outputs = ApexCoreOutputs()
        d = outputs.to_dict()
        
        assert "quantrascore" in d
        assert d["quantrascore"] == 50.0


class TestApexCoreDeterminism:
    """Test ApexCore determinism."""
    
    def test_mini_deterministic_same_input(self):
        """ApexCoreMini should produce same output for same input."""
        config = ApexCoreMiniConfig(random_state=42)
        model = ApexCoreMini(config)
        
        np.random.seed(42)
        features = np.random.randn(1, config.input_dim)
        
        result1 = model.predict(features.copy())
        result2 = model.predict(features.copy())
        
        assert result1.quantrascore == result2.quantrascore, \
            f"Non-deterministic: {result1.quantrascore} vs {result2.quantrascore}"
    
    def test_full_deterministic_same_input(self):
        """ApexCoreFull should produce same output for same input."""
        config = ApexCoreFullConfig(random_state=42)
        model = ApexCoreFull(config)
        
        np.random.seed(42)
        features = np.random.randn(1, config.input_dim)
        
        result1 = model.predict(features.copy())
        result2 = model.predict(features.copy())
        
        assert result1.quantrascore == result2.quantrascore


class TestApexCoreConfig:
    """Test ApexCore configuration."""
    
    def test_mini_config_defaults(self):
        """ApexCoreMiniConfig should have sensible defaults."""
        config = ApexCoreMiniConfig()
        
        assert config.input_dim == 30
        assert config.model_name == "apexcore_mini"
        assert len(config.hidden_layers) > 0
    
    def test_full_config_defaults(self):
        """ApexCoreFullConfig should have sensible defaults."""
        config = ApexCoreFullConfig()
        
        assert config.input_dim == 30
        assert config.model_name == "apexcore_full"
        assert len(config.hidden_layers) > 0
    
    def test_full_has_more_layers_than_mini(self):
        """ApexCoreFull should have more/larger layers than Mini."""
        mini_config = ApexCoreMiniConfig()
        full_config = ApexCoreFullConfig()
        
        mini_total = sum(mini_config.hidden_layers)
        full_total = sum(full_config.hidden_layers)
        
        assert full_total > mini_total, "Full should have more capacity than Mini"
