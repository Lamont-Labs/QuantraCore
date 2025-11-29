"""
ApexLab Label Generation Tests for QuantraCore Apex.

Tests ApexLab label generation pipeline.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from src.quantracore_apex.apexlab.labels import LabelGenerator, generate_labels
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.apexlab.dataset_builder import DatasetBuilder
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
ALL_SYMBOLS = LIQUID_SYMBOLS + ["META", "NVDA", "AMD", "NFLX", "INTC"]


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestLabelGeneratorImport:
    """Test LabelGenerator imports."""
    
    def test_label_generator_import(self):
        """LabelGenerator should be importable."""
        assert LabelGenerator is not None
    
    def test_feature_extractor_import(self):
        """FeatureExtractor should be importable."""
        assert FeatureExtractor is not None
    
    def test_window_builder_import(self):
        """WindowBuilder should be importable."""
        assert WindowBuilder is not None
    
    def test_dataset_builder_import(self):
        """DatasetBuilder should be importable."""
        assert DatasetBuilder is not None


class TestLabelGeneratorInstantiation:
    """Test LabelGenerator instantiation."""
    
    def test_label_generator_instantiation(self):
        """LabelGenerator should instantiate."""
        generator = LabelGenerator(enable_logging=False)
        assert generator is not None
    
    def test_label_generator_has_generate(self):
        """LabelGenerator should have generate method."""
        generator = LabelGenerator(enable_logging=False)
        assert hasattr(generator, "generate")
        assert callable(generator.generate)


class TestLabelGeneration:
    """Test label generation."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_returns_dict(self, symbol: str):
        """generate should return a dictionary."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert labels is not None
        assert isinstance(labels, dict)
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_has_quantrascore(self, symbol: str):
        """Labels should include quantrascore_numeric."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "quantrascore_numeric" in labels
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_quantrascore_valid(self, symbol: str):
        """QuantraScore should be in 0-100 range."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        score = labels["quantrascore_numeric"]
        assert 0 <= score <= 100
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_generate_has_regime(self, symbol: str):
        """Labels should include regime classification."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "regime_class" in labels
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_generate_has_risk_tier(self, symbol: str):
        """Labels should include risk tier."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "risk_tier" in labels


class TestBatchLabelGeneration:
    """Test batch label generation."""
    
    def test_generate_batch(self):
        """generate_batch should work for multiple windows."""
        generator = LabelGenerator(enable_logging=False)
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        batch_labels = generator.generate_batch(windows)
        
        assert batch_labels is not None
        assert isinstance(batch_labels, dict)
        assert "quantrascore_numeric" in batch_labels
        assert len(batch_labels["quantrascore_numeric"]) == 3


class TestModuleLevelGenerate:
    """Test module-level generate_labels function."""
    
    def test_module_generate_labels(self):
        """Module-level generate_labels should work."""
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        labels = generate_labels(windows)
        
        assert labels is not None
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 3


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_feature_extractor_instantiation(self):
        """FeatureExtractor should instantiate."""
        extractor = FeatureExtractor()
        assert extractor is not None
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_feature_extraction(self, symbol: str):
        """FeatureExtractor should extract features."""
        extractor = FeatureExtractor()
        window = _get_test_window(symbol)
        
        features = extractor.extract(window)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestLabelGeneratorDeterminism:
    """Test label generator determinism."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_labels_deterministic(self, symbol: str):
        """Labels should be deterministic for same input."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels1 = generator.generate(window)
        labels2 = generator.generate(window)
        
        assert labels1["quantrascore_numeric"] == labels2["quantrascore_numeric"]
        assert labels1["regime_class"] == labels2["regime_class"]
