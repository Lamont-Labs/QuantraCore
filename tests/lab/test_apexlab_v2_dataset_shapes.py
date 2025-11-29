"""
Tests for ApexLab V2 dataset shapes and feature extraction.

Validates:
- Feature matrix dimensions
- Target array shapes
- Protocol vector encoding
- Dataset builder output consistency
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.quantracore_apex.apexlab.apexlab_v2 import (
    ApexLabV2Row,
    ApexLabV2Builder,
    ApexLabV2DatasetBuilder,
    encode_protocol_vector,
)
from src.quantracore_apex.apexlab.training import (
    extract_features,
    extract_targets,
    create_walk_forward_splits,
)
from src.quantracore_apex.core.schemas import OhlcvWindow


class TestProtocolVectorShapes:
    """Tests for protocol vector encoding shapes."""
    
    def test_vector_length(self):
        """Test protocol vector has correct length."""
        vector = encode_protocol_vector([])
        
        assert len(vector) == 115
    
    def test_vector_with_protocols(self):
        """Test vector shape with active protocols."""
        vector = encode_protocol_vector(["T01", "T50", "LP10", "MR03"])
        
        assert len(vector) == 115
        assert sum(vector) == 4
    
    def test_vector_all_zeros_empty_input(self):
        """Test all zeros for empty input."""
        vector = encode_protocol_vector([])
        
        assert all(v == 0.0 for v in vector)
    
    def test_vector_dtype(self):
        """Test vector values are floats."""
        vector = encode_protocol_vector(["T05"])
        
        assert all(isinstance(v, float) for v in vector)


class TestDatasetBuilderShapes:
    """Tests for dataset builder output shapes."""
    
    def test_builder_instantiation(self):
        """Test builder can be instantiated."""
        builder = ApexLabV2Builder(enable_logging=False)
        assert builder is not None


class TestFeatureExtraction:
    """Tests for training feature extraction."""
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        df = pd.DataFrame({
            "quantra_score": [50.0, 60.0, 70.0],
            "risk_tier": ["low", "medium", "high"],
            "volatility_band": ["low", "mid", "high"],
        })
        
        features = extract_features(df)
        
        assert features.shape[0] == 3
        assert features.shape[1] >= 1
    
    def test_extract_features_with_protocol_vector(self):
        """Test feature extraction with protocol vectors."""
        protocol_vectors = [
            [1.0] + [0.0] * 114,
            [0.0, 1.0] + [0.0] * 113,
            [0.0, 0.0, 1.0] + [0.0] * 112,
        ]
        
        df = pd.DataFrame({
            "quantra_score": [50.0, 60.0, 70.0],
            "protocol_vector": protocol_vectors,
        })
        
        features = extract_features(df)
        
        assert features.shape[0] == 3
    
    def test_extract_features_empty_df(self):
        """Test feature extraction with minimal data."""
        df = pd.DataFrame({
            "quantra_score": [50.0],
        })
        
        features = extract_features(df)
        
        assert features.shape[0] == 1


class TestTargetExtraction:
    """Tests for training target extraction."""
    
    def test_extract_targets_all_present(self):
        """Test target extraction with all targets."""
        df = pd.DataFrame({
            "quantra_score": [50.0, 60.0, 70.0],
            "hit_runner_threshold": [0, 1, 1],
            "future_quality_tier": ["C", "A", "B"],
            "avoid_trade": [0, 0, 1],
            "regime_label": ["chop", "trend_up", "trend_down"],
        })
        
        targets = extract_targets(df)
        
        assert "quantra_score" in targets
        assert "hit_runner_threshold" in targets
        assert "future_quality_tier" in targets
        assert "avoid_trade" in targets
        assert "regime_label" in targets
        
        assert targets["quantra_score"].shape == (3,)
        assert targets["hit_runner_threshold"].shape == (3,)
    
    def test_extract_targets_partial(self):
        """Test target extraction with partial targets."""
        df = pd.DataFrame({
            "quantra_score": [50.0, 60.0],
            "hit_runner_threshold": [0, 1],
        })
        
        targets = extract_targets(df)
        
        assert "quantra_score" in targets
        assert "hit_runner_threshold" in targets
        assert "future_quality_tier" not in targets


class TestWalkForwardSplits:
    """Tests for walk-forward split creation."""
    
    def test_basic_split(self):
        """Test basic walk-forward split."""
        df = pd.DataFrame({
            "event_time": pd.date_range("2024-01-01", periods=100, freq="D"),
            "value": range(100),
        })
        
        splits = create_walk_forward_splits(df, n_folds=5, val_ratio=0.2)
        
        assert len(splits) >= 1
        
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            assert len(set(train_idx) & set(val_idx)) == 0
    
    def test_split_no_time_column(self):
        """Test split without time column."""
        df = pd.DataFrame({
            "value": range(50),
        })
        
        splits = create_walk_forward_splits(df, n_folds=3, val_ratio=0.2)
        
        assert len(splits) >= 1
    
    def test_split_proportions(self):
        """Test split maintains approximate proportions."""
        df = pd.DataFrame({
            "event_time": pd.date_range("2024-01-01", periods=100, freq="D"),
            "value": range(100),
        })
        
        splits = create_walk_forward_splits(df, n_folds=1, val_ratio=0.2)
        
        train_idx, val_idx = splits[0]
        
        total = len(train_idx) + len(val_idx)
        val_ratio_actual = len(val_idx) / total
        
        assert 0.1 <= val_ratio_actual <= 0.4


class TestApexLabV2DatasetBuilder:
    """Tests for high-level dataset builder."""
    
    def test_builder_creation(self):
        """Test builder instantiation."""
        builder = ApexLabV2DatasetBuilder()
        
        assert builder is not None
        assert builder.rows == []
    
    def test_add_rows(self):
        """Test adding rows to dataset."""
        builder = ApexLabV2DatasetBuilder()
        
        row1 = ApexLabV2Row(
            symbol="AAPL",
            event_time=datetime.utcnow(),
            timeframe="1d",
            engine_snapshot_id="e1",
            scanner_snapshot_id="s1",
        )
        
        row2 = ApexLabV2Row(
            symbol="MSFT",
            event_time=datetime.utcnow(),
            timeframe="1d",
            engine_snapshot_id="e2",
            scanner_snapshot_id="s2",
        )
        
        builder.add_rows([row1, row2])
        
        assert len(builder.rows) == 2
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        builder = ApexLabV2DatasetBuilder()
        
        for i in range(3):
            row = ApexLabV2Row(
                symbol=f"SYM{i}",
                event_time=datetime.utcnow(),
                timeframe="1d",
                engine_snapshot_id=f"e{i}",
                scanner_snapshot_id=f"s{i}",
                quantra_score=50.0 + i * 10,
            )
            builder.add_row(row)
        
        df = builder.to_dataframe()
        
        assert len(df) == 3
        assert "symbol" in df.columns
        assert "quantra_score" in df.columns
    
    def test_feature_target_split(self):
        """Test feature/target split."""
        builder = ApexLabV2DatasetBuilder()
        
        for i in range(5):
            row = ApexLabV2Row(
                symbol="TEST",
                event_time=datetime.utcnow(),
                timeframe="1d",
                engine_snapshot_id="e",
                scanner_snapshot_id="s",
                quantra_score=50.0 + i,
                hit_runner_threshold=i % 2,
                future_quality_tier=["A", "B", "C"][i % 3],
            )
            builder.add_row(row)
        
        features, targets = builder.get_feature_target_split()
        
        assert features.shape[0] == 5
        assert "quantra_score" in targets
        assert "hit_runner_threshold" in targets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
