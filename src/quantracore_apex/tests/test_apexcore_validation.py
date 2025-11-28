"""
ApexLab + ApexCore Validation Tests for QuantraCore Apex v9.0-A.

Validates the offline training pipeline and neural inference loop:
- ApexLab feature extraction
- ApexCore model training/inference
- Deterministic outputs
- QuantraScore head validation
"""

import pytest
import os
from datetime import datetime, timedelta
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.apexlab.labels import LabelGenerator
from src.quantracore_apex.apexcore.interface import ApexCoreOutputs


class TestApexLabFeatureExtraction:
    """Test ApexLab feature extraction pipeline."""
    
    @pytest.fixture
    def sample_windows(self) -> List:
        """Generate sample windows for testing."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        windows = []
        for i in range(10):
            symbol = f"FEAT_{i}"
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            windows.append(window)
        
        return windows
    
    def test_feature_extractor_initialization(self):
        """Verify feature extractor initializes correctly."""
        extractor = FeatureExtractor()
        assert extractor is not None
    
    def test_feature_extraction_produces_output(self, sample_windows):
        """Verify feature extraction produces valid output."""
        extractor = FeatureExtractor()
        
        for window in sample_windows:
            features = extractor.extract(window)
            assert features is not None
            assert len(features) > 0
    
    def test_feature_extraction_deterministic(self, sample_windows):
        """Verify feature extraction is deterministic."""
        extractor = FeatureExtractor()
        
        for window in sample_windows:
            features1 = extractor.extract(window)
            features2 = extractor.extract(window)
            
            for f1, f2 in zip(features1, features2):
                assert f1 == f2, f"Feature mismatch for {window.symbol}"


class TestApexLabLabelGeneration:
    """Test ApexLab label generation pipeline."""
    
    @pytest.fixture
    def sample_windows(self) -> List:
        """Generate sample windows for testing."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        windows = []
        for i in range(10):
            symbol = f"LABEL_{i}"
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            windows.append(window)
        
        return windows
    
    def test_label_generator_initialization(self):
        """Verify label generator initializes correctly."""
        generator = LabelGenerator(enable_logging=False)
        assert generator is not None
    
    def test_label_generation_produces_output(self, sample_windows):
        """Verify label generation produces valid output."""
        generator = LabelGenerator(enable_logging=False)
        
        for window in sample_windows:
            label = generator.generate(window)
            assert label is not None
    
    def test_label_generation_deterministic(self, sample_windows):
        """Verify label generation is deterministic."""
        generator = LabelGenerator(enable_logging=False)
        
        for window in sample_windows:
            label1 = generator.generate(window)
            label2 = generator.generate(window)
            
            assert label1["quantrascore_numeric"] == label2["quantrascore_numeric"]
            assert label1["regime_class"] == label2["regime_class"]


class TestApexCoreInterface:
    """Test ApexCore model interface."""
    
    def test_outputs_structure(self):
        """Verify ApexCoreOutputs has correct structure."""
        outputs = ApexCoreOutputs(
            regime_prediction=0,
            regime_confidence=0.85,
            risk_prediction=1,
            risk_confidence=0.75,
            quantrascore_prediction=65.0,
            raw_outputs={"test": True}
        )
        
        assert outputs.quantrascore_prediction == 65.0
        assert 0 <= outputs.regime_confidence <= 1
        assert 0 <= outputs.risk_confidence <= 1
    
    def test_outputs_validation(self):
        """Verify outputs are validated correctly."""
        outputs = ApexCoreOutputs(
            regime_prediction=0,
            regime_confidence=0.9,
            risk_prediction=0,
            risk_confidence=0.5,
            quantrascore_prediction=50.0,
            raw_outputs={}
        )
        
        assert 0 <= outputs.quantrascore_prediction <= 100


class TestApexLabPipeline:
    """End-to-end ApexLab pipeline tests."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    def test_full_pipeline_synthetic_data(self, engine):
        """Test full ApexLab pipeline with synthetic data."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        feature_extractor = FeatureExtractor()
        label_generator = LabelGenerator(enable_logging=False)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        dataset = []
        
        for i in range(20):
            symbol = f"PIPE_{i}"
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            
            features = feature_extractor.extract(window)
            label = label_generator.generate(window)
            
            dataset.append({
                "symbol": symbol,
                "features": features,
                "quantrascore": label["quantrascore_numeric"],
                "regime_class": label["regime_class"]
            })
        
        assert len(dataset) == 20
        
        for item in dataset:
            assert 0 <= item["quantrascore"] <= 100
            assert item["regime_class"] is not None
    
    def test_pipeline_stability(self, engine):
        """Verify pipeline produces stable outputs."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        feature_extractor = FeatureExtractor()
        label_generator = LabelGenerator(enable_logging=False)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        symbol = "STABLE_TEST"
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, symbol)
        
        run1_features = feature_extractor.extract(window)
        run1_label = label_generator.generate(window)
        
        run2_features = feature_extractor.extract(window)
        run2_label = label_generator.generate(window)
        
        import numpy as np
        assert np.allclose(run1_features, run2_features)
        assert run1_label["quantrascore_numeric"] == run2_label["quantrascore_numeric"]


class TestSmallCapFeatures:
    """Test small-cap specific feature extraction."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    def test_volatility_features_computed(self, engine):
        """Verify volatility features are computed for all windows."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        feature_extractor = FeatureExtractor()
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("SMALL_CAP", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, "SMALL_CAP")
        
        features = feature_extractor.extract(window)
        
        assert len(features) > 10


class TestQuantraScoreValidation:
    """Validate QuantraScore computations."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    def test_quantrascore_range(self, engine):
        """Verify QuantraScore is always in 0-100 range."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        context = ApexContext(seed=42, compliance_mode=True)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        for i in range(50):
            bars = adapter.fetch_ohlcv(f"RANGE_{i}", start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, f"RANGE_{i}")
            
            result = engine.run(window, context)
            
            assert 0 <= result.quantrascore <= 100, \
                f"QuantraScore {result.quantrascore} out of range for RANGE_{i}"
    
    def test_quantrascore_deterministic(self, engine):
        """Verify QuantraScore is deterministic."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        context = ApexContext(seed=42, compliance_mode=True)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("DETERM", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, "DETERM")
        
        scores = []
        for _ in range(10):
            result = engine.run(window, context)
            scores.append(result.quantrascore)
        
        assert all(s == scores[0] for s in scores), "QuantraScore not deterministic"
