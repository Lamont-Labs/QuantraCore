"""
ApexLab Label Generation Tests for QuantraCore Apex.

Tests ApexLab label generation pipeline with SUBSTANTIVE assertions.
"""

import pytest
import numpy as np

from src.quantracore_apex.apexlab.labels import LabelGenerator, generate_labels
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestLabelGeneratorImport:
    """Test LabelGenerator imports."""
    
    def test_label_generator_importable(self):
        """LabelGenerator should be importable."""
        assert LabelGenerator is not None
        assert callable(LabelGenerator)
    
    def test_feature_extractor_importable(self):
        """FeatureExtractor should be importable."""
        assert FeatureExtractor is not None
    
    def test_generate_labels_function_importable(self):
        """generate_labels function should be importable."""
        assert generate_labels is not None
        assert callable(generate_labels)


class TestLabelGeneratorInstantiation:
    """Test LabelGenerator instantiation."""
    
    def test_instantiation(self):
        """LabelGenerator should instantiate."""
        generator = LabelGenerator(enable_logging=False)
        assert isinstance(generator, LabelGenerator)
    
    def test_has_generate_method(self):
        """LabelGenerator should have generate method."""
        generator = LabelGenerator(enable_logging=False)
        assert hasattr(generator, "generate")
        assert callable(generator.generate)
    
    def test_has_generate_batch_method(self):
        """LabelGenerator should have generate_batch method."""
        generator = LabelGenerator(enable_logging=False)
        assert hasattr(generator, "generate_batch")
        assert callable(generator.generate_batch)


class TestLabelGeneration:
    """Test label generation."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_returns_dict(self, symbol: str):
        """generate should return a dictionary."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert isinstance(labels, dict), f"Expected dict, got {type(labels)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_has_quantrascore_numeric(self, symbol: str):
        """Labels should include quantrascore_numeric key."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "quantrascore_numeric" in labels, "Missing quantrascore_numeric"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_quantrascore_in_valid_range(self, symbol: str):
        """quantrascore_numeric should be in 0-100 range."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        score = labels["quantrascore_numeric"]
        
        assert 0 <= score <= 100, f"quantrascore {score} out of range"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_has_regime_class(self, symbol: str):
        """Labels should include regime_class key."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "regime_class" in labels
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_generate_has_risk_tier(self, symbol: str):
        """Labels should include risk_tier key."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels = generator.generate(window)
        
        assert "risk_tier" in labels


class TestBatchLabelGeneration:
    """Test batch label generation."""
    
    def test_generate_batch_returns_dict(self):
        """generate_batch should return a dictionary."""
        generator = LabelGenerator(enable_logging=False)
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        batch_labels = generator.generate_batch(windows)
        
        assert isinstance(batch_labels, dict)
    
    def test_generate_batch_has_quantrascore(self):
        """Batch labels should include quantrascore_numeric."""
        generator = LabelGenerator(enable_logging=False)
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        batch_labels = generator.generate_batch(windows)
        
        assert "quantrascore_numeric" in batch_labels
    
    def test_generate_batch_correct_length(self):
        """Batch labels should have correct array lengths."""
        generator = LabelGenerator(enable_logging=False)
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        batch_labels = generator.generate_batch(windows)
        
        scores = batch_labels["quantrascore_numeric"]
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"


class TestModuleLevelGenerate:
    """Test module-level generate_labels function."""
    
    def test_returns_numpy_array(self):
        """generate_labels should return numpy array."""
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        labels = generate_labels(windows)
        
        assert isinstance(labels, np.ndarray), f"Expected ndarray, got {type(labels)}"
    
    def test_correct_length(self):
        """generate_labels should return correct number of labels."""
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        labels = generate_labels(windows)
        
        assert len(labels) == 3
    
    def test_values_in_range(self):
        """All label values should be in 0-100 range."""
        windows = [_get_test_window(s) for s in LIQUID_SYMBOLS[:3]]
        
        labels = generate_labels(windows)
        
        for score in labels:
            assert 0 <= score <= 100, f"Score {score} out of range"


class TestFeatureExtraction:
    """Test feature extraction."""
    
    def test_feature_extractor_instantiation(self):
        """FeatureExtractor should instantiate."""
        extractor = FeatureExtractor()
        assert isinstance(extractor, FeatureExtractor)
    
    def test_feature_extractor_has_extract(self):
        """FeatureExtractor should have extract method."""
        extractor = FeatureExtractor()
        assert hasattr(extractor, "extract")
        assert callable(extractor.extract)
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_extract_returns_array(self, symbol: str):
        """extract should return numpy array."""
        extractor = FeatureExtractor()
        window = _get_test_window(symbol)
        
        features = extractor.extract(window)
        
        assert isinstance(features, np.ndarray), f"Expected ndarray, got {type(features)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_extract_non_empty(self, symbol: str):
        """Extracted features should be non-empty."""
        extractor = FeatureExtractor()
        window = _get_test_window(symbol)
        
        features = extractor.extract(window)
        
        assert len(features) > 0, "Empty feature array"


class TestLabelGeneratorDeterminism:
    """Test label generator determinism."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_labels_deterministic(self, symbol: str):
        """Labels should be deterministic for same input."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels1 = generator.generate(window)
        labels2 = generator.generate(window)
        
        assert labels1["quantrascore_numeric"] == labels2["quantrascore_numeric"], \
            f"Non-deterministic: {labels1['quantrascore_numeric']} vs {labels2['quantrascore_numeric']}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_regime_deterministic(self, symbol: str):
        """Regime class should be deterministic."""
        generator = LabelGenerator(enable_logging=False)
        window = _get_test_window(symbol)
        
        labels1 = generator.generate(window)
        labels2 = generator.generate(window)
        
        assert labels1["regime_class"] == labels2["regime_class"]
