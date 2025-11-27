"""
ApexLab Pipeline Tests for QuantraCore Apex.

Smoke tests for the training pipeline.
"""

import pytest
from datetime import datetime, timedelta

from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.apexlab.labels import LabelGenerator
from src.quantracore_apex.apexlab.dataset_builder import DatasetBuilder
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv


class TestApexLabPipeline:
    """Test ApexLab training pipeline."""
    
    @pytest.fixture
    def sample_windows(self):
        """Create sample windows for testing."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100, step=20)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=200)
        
        bars = adapter.fetch_ohlcv("PIPELINE_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        return window_builder.build(normalized_bars, "PIPELINE_TEST")
    
    def test_feature_extraction(self, sample_windows):
        """Test feature extraction produces correct shape."""
        extractor = FeatureExtractor()
        
        if len(sample_windows) > 0:
            features = extractor.extract(sample_windows[0])
            
            assert features.shape[0] == extractor.feature_dim
            assert features.shape[0] == len(extractor.FEATURE_NAMES)
    
    def test_label_generation(self, sample_windows):
        """Test label generation produces required labels."""
        generator = LabelGenerator(enable_logging=False)
        
        if len(sample_windows) > 0:
            labels = generator.generate(sample_windows[0])
            
            required_labels = ["regime_class", "risk_tier", "quantrascore_numeric"]
            for label in required_labels:
                assert label in labels, f"Missing required label: {label}"
    
    def test_dataset_builder(self, sample_windows):
        """Test dataset builder creates valid dataset."""
        builder = DatasetBuilder(enable_logging=False)
        
        if len(sample_windows) >= 3:
            test_windows = sample_windows[:3]
            dataset = builder.build(test_windows, "test_dataset")
            
            assert "features" in dataset
            assert "labels" in dataset
            assert "metadata" in dataset
            assert dataset["features"].shape[0] == len(test_windows)
    
    def test_dataset_split(self, sample_windows):
        """Test dataset splitting."""
        builder = DatasetBuilder(enable_logging=False)
        
        if len(sample_windows) >= 5:
            test_windows = sample_windows[:5]
            dataset = builder.build(test_windows, "split_test")
            
            train_ds, val_ds = builder.split(dataset, train_ratio=0.8)
            
            total = train_ds["features"].shape[0] + val_ds["features"].shape[0]
            assert total == dataset["features"].shape[0]


class TestFeatureExtractor:
    """Test feature extractor in detail."""
    
    def test_feature_names_match_dimension(self):
        """Test feature names match feature dimension."""
        extractor = FeatureExtractor()
        
        assert len(extractor.FEATURE_NAMES) == extractor.feature_dim
    
    def test_features_no_nan(self):
        """Test features contain no NaN values."""
        import numpy as np
        
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        extractor = FeatureExtractor()
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("NAN_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, "NAN_TEST")
        
        if window:
            features = extractor.extract(window)
            assert not np.any(np.isnan(features)), "Features contain NaN"
