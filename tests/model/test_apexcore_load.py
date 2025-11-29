"""
ApexCore Model Load Tests for QuantraCore Apex.

Tests ApexCore Full and Mini model loading and prediction.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.quantracore_apex.apexcore.models import (
    ApexCoreFull, ApexCoreMini,
    ApexCoreFullConfig, ApexCoreMiniConfig,
    ApexCoreOutputs
)


class TestApexCoreImport:
    """Test ApexCore module imports."""
    
    def test_apexcore_full_import(self):
        """ApexCoreFull should be importable."""
        assert ApexCoreFull is not None
    
    def test_apexcore_mini_import(self):
        """ApexCoreMini should be importable."""
        assert ApexCoreMini is not None
    
    def test_config_import(self):
        """Config classes should be importable."""
        assert ApexCoreFullConfig is not None
        assert ApexCoreMiniConfig is not None
    
    def test_outputs_import(self):
        """ApexCoreOutputs should be importable."""
        assert ApexCoreOutputs is not None


class TestApexCoreMiniLoad:
    """Test ApexCore Mini model loading."""
    
    def test_mini_instantiation(self):
        """ApexCoreMini should instantiate."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        assert model is not None
    
    def test_mini_has_predict(self):
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
        
        assert result is not None
        assert isinstance(result, ApexCoreOutputs)
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_mini_predict_valid_score(self, batch_size: int):
        """ApexCoreMini should return valid QuantraScore."""
        config = ApexCoreMiniConfig()
        model = ApexCoreMini(config)
        
        dummy_features = np.random.randn(batch_size, config.input_dim)
        result = model.predict(dummy_features)
        
        assert 0 <= result.quantrascore <= 100


class TestApexCoreFullLoad:
    """Test ApexCore Full model loading."""
    
    def test_full_instantiation(self):
        """ApexCoreFull should instantiate."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        assert model is not None
    
    def test_full_has_predict(self):
        """ApexCoreFull should have predict method."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        assert hasattr(model, "predict")
    
    def test_full_predict_returns_outputs(self):
        """ApexCoreFull.predict should return ApexCoreOutputs."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        dummy_features = np.random.randn(1, config.input_dim)
        result = model.predict(dummy_features)
        
        assert isinstance(result, ApexCoreOutputs)
    
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_full_predict_valid_score(self, batch_size: int):
        """ApexCoreFull should return valid QuantraScore."""
        config = ApexCoreFullConfig()
        model = ApexCoreFull(config)
        
        dummy_features = np.random.randn(batch_size, config.input_dim)
        result = model.predict(dummy_features)
        
        assert 0 <= result.quantrascore <= 100


class TestApexCoreOutputs:
    """Test ApexCoreOutputs structure."""
    
    def test_outputs_has_quantrascore(self):
        """ApexCoreOutputs should have quantrascore."""
        outputs = ApexCoreOutputs()
        assert hasattr(outputs, "quantrascore")
    
    def test_outputs_has_regime(self):
        """ApexCoreOutputs should have regime_prediction."""
        outputs = ApexCoreOutputs()
        assert hasattr(outputs, "regime_prediction")
    
    def test_outputs_has_risk_tier(self):
        """ApexCoreOutputs should have risk_tier."""
        outputs = ApexCoreOutputs()
        assert hasattr(outputs, "risk_tier")
    
    def test_outputs_to_dict(self):
        """ApexCoreOutputs should convert to dict."""
        outputs = ApexCoreOutputs()
        d = outputs.to_dict()
        
        assert isinstance(d, dict)
        assert "quantrascore" in d
        assert "regime_prediction" in d
        assert "risk_tier" in d


class TestApexCoreDeterminism:
    """Test ApexCore determinism."""
    
    def test_mini_deterministic(self):
        """ApexCoreMini should be deterministic."""
        config = ApexCoreMiniConfig(random_state=42)
        model = ApexCoreMini(config)
        
        np.random.seed(42)
        features = np.random.randn(1, config.input_dim)
        
        result1 = model.predict(features)
        result2 = model.predict(features)
        
        assert result1.quantrascore == result2.quantrascore
    
    def test_full_deterministic(self):
        """ApexCoreFull should be deterministic."""
        config = ApexCoreFullConfig(random_state=42)
        model = ApexCoreFull(config)
        
        np.random.seed(42)
        features = np.random.randn(1, config.input_dim)
        
        result1 = model.predict(features)
        result2 = model.predict(features)
        
        assert result1.quantrascore == result2.quantrascore
