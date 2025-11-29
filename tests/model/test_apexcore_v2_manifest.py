"""
Tests for ApexCore V2 manifest system.

Validates:
- Manifest creation and serialization
- Hash verification
- Threshold validation for promotion
- Best model selection
"""

import pytest
import tempfile
import os
import json
from pathlib import Path

from src.quantracore_apex.apexcore.manifest import (
    ApexCoreV2Manifest,
    ManifestMetrics,
    ManifestThresholds,
    load_manifest,
    select_best_model,
    verify_manifest_against_file,
    compute_file_hash,
    create_manifest_for_model,
)


class TestManifestMetrics:
    """Tests for ManifestMetrics dataclass."""
    
    def test_default_values(self):
        """Test default metric values."""
        metrics = ManifestMetrics()
        
        assert metrics.val_brier_runner == 1.0
        assert metrics.val_auc_runner == 0.5
        assert metrics.val_calibration_error_runner == 1.0
        assert metrics.val_accuracy_quality_tier == 0.2
        assert metrics.val_accuracy_regime == 0.2
        assert metrics.val_mse_quantra_score == 100.0
    
    def test_custom_values(self):
        """Test custom metric values."""
        metrics = ManifestMetrics(
            val_auc_runner=0.85,
            val_calibration_error_runner=0.05,
        )
        
        assert metrics.val_auc_runner == 0.85
        assert metrics.val_calibration_error_runner == 0.05
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = ManifestMetrics(val_auc_runner=0.75)
        d = metrics.to_dict()
        
        assert d["val_auc_runner"] == 0.75
        assert "val_brier_runner" in d
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"val_auc_runner": 0.80, "val_mse_quantra_score": 25.0}
        metrics = ManifestMetrics.from_dict(d)
        
        assert metrics.val_auc_runner == 0.80
        assert metrics.val_mse_quantra_score == 25.0


class TestManifestThresholds:
    """Tests for ManifestThresholds dataclass."""
    
    def test_default_values(self):
        """Test default threshold values."""
        thresholds = ManifestThresholds()
        
        assert thresholds.runner_prob_min_for_a_plus_flag == 0.7
        assert thresholds.avoid_trade_prob_max_to_allow == 0.3
        assert thresholds.max_disagreement_allowed == 0.2
        assert thresholds.min_auc_runner_to_promote == 0.6
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        thresholds = ManifestThresholds()
        d = thresholds.to_dict()
        
        assert "runner_prob_min_for_a_plus_flag" in d
        assert "max_disagreement_allowed" in d


class TestApexCoreV2Manifest:
    """Tests for ApexCoreV2Manifest."""
    
    def test_default_creation(self):
        """Test manifest creation with defaults."""
        manifest = ApexCoreV2Manifest()
        
        assert manifest.model_family == "apexcore_v2"
        assert manifest.variant == "big"
        assert manifest.ensemble_size == 1
        assert manifest.created_utc != ""
    
    def test_custom_creation(self):
        """Test manifest creation with custom values."""
        manifest = ApexCoreV2Manifest(
            variant="mini",
            ensemble_size=5,
            engine_snapshot_id="snap123",
            lab_dataset_id="data456",
        )
        
        assert manifest.variant == "mini"
        assert manifest.ensemble_size == 5
        assert manifest.engine_snapshot_id == "snap123"
        assert manifest.lab_dataset_id == "data456"
    
    def test_to_dict(self):
        """Test manifest serialization."""
        manifest = ApexCoreV2Manifest(
            variant="big",
            hashes={"member_0": "sha256:abc123"},
        )
        
        d = manifest.to_dict()
        
        assert d["variant"] == "big"
        assert d["hashes"]["member_0"] == "sha256:abc123"
        assert "metrics" in d
        assert "thresholds" in d
    
    def test_from_dict(self):
        """Test manifest creation from dictionary."""
        d = {
            "variant": "mini",
            "ensemble_size": 3,
            "hashes": {"member_0": "sha256:xyz"},
            "metrics": {"val_auc_runner": 0.9},
            "thresholds": {"max_disagreement_allowed": 0.15},
        }
        
        manifest = ApexCoreV2Manifest.from_dict(d)
        
        assert manifest.variant == "mini"
        assert manifest.ensemble_size == 3
        assert manifest.metrics.val_auc_runner == 0.9
        assert manifest.thresholds.max_disagreement_allowed == 0.15
    
    def test_save_and_load(self):
        """Test manifest save and load."""
        manifest = ApexCoreV2Manifest(
            variant="big",
            ensemble_size=3,
            hashes={"member_0": "sha256:test"},
        )
        manifest.metrics = ManifestMetrics(val_auc_runner=0.88)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "manifest.json")
            manifest.save(path)
            
            loaded = ApexCoreV2Manifest.load(path)
        
        assert loaded.variant == "big"
        assert loaded.ensemble_size == 3
        assert loaded.metrics.val_auc_runner == 0.88
    
    def test_is_valid_for_promotion_valid(self):
        """Test promotion validation with valid metrics."""
        manifest = ApexCoreV2Manifest()
        manifest.metrics = ManifestMetrics(
            val_auc_runner=0.75,
            val_calibration_error_runner=0.08,
        )
        manifest.thresholds = ManifestThresholds(
            min_auc_runner_to_promote=0.6,
            max_calibration_error_to_promote=0.15,
        )
        
        is_valid, failures = manifest.is_valid_for_promotion()
        
        assert is_valid is True
        assert len(failures) == 0
    
    def test_is_valid_for_promotion_invalid(self):
        """Test promotion validation with invalid metrics."""
        manifest = ApexCoreV2Manifest()
        manifest.metrics = ManifestMetrics(
            val_auc_runner=0.45,
            val_calibration_error_runner=0.25,
        )
        manifest.thresholds = ManifestThresholds(
            min_auc_runner_to_promote=0.6,
            max_calibration_error_to_promote=0.15,
        )
        
        is_valid, failures = manifest.is_valid_for_promotion()
        
        assert is_valid is False
        assert len(failures) == 2


class TestManifestFunctions:
    """Tests for manifest utility functions."""
    
    def test_load_manifest(self):
        """Test load_manifest function."""
        manifest = ApexCoreV2Manifest(variant="mini")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            manifest.save(path)
            
            loaded = load_manifest(path)
        
        assert loaded.variant == "mini"
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content for hashing")
            temp_path = f.name
        
        try:
            hash_result = compute_file_hash(temp_path)
            
            assert hash_result.startswith("sha256:")
            assert len(hash_result) > 10
            
            hash_result2 = compute_file_hash(temp_path)
            assert hash_result == hash_result2
        finally:
            os.unlink(temp_path)
    
    def test_select_best_model(self):
        """Test best model selection from manifests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest1 = ApexCoreV2Manifest()
            manifest1.metrics = ManifestMetrics(val_auc_runner=0.70)
            manifest1.thresholds = ManifestThresholds(min_auc_runner_to_promote=0.5)
            manifest1.save(os.path.join(tmpdir, "model1.json"))
            
            manifest2 = ApexCoreV2Manifest()
            manifest2.metrics = ManifestMetrics(val_auc_runner=0.85)
            manifest2.thresholds = ManifestThresholds(min_auc_runner_to_promote=0.5)
            manifest2.save(os.path.join(tmpdir, "model2.json"))
            
            manifest3 = ApexCoreV2Manifest()
            manifest3.metrics = ManifestMetrics(val_auc_runner=0.60)
            manifest3.thresholds = ManifestThresholds(min_auc_runner_to_promote=0.5)
            manifest3.save(os.path.join(tmpdir, "model3.json"))
            
            best_path, best_manifest = select_best_model(tmpdir)
        
        assert best_manifest.metrics.val_auc_runner >= 0.60
    
    def test_select_best_model_empty_dir(self):
        """Test best model selection from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            best_path, best_manifest = select_best_model(tmpdir)
        
        assert best_path == ""
        assert best_manifest.model_family == "apexcore_v2"
    
    def test_verify_manifest_against_file_valid(self):
        """Test hash verification with valid file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
            f.write("model data")
            temp_path = f.name
        
        try:
            file_hash = compute_file_hash(temp_path)
            manifest = ApexCoreV2Manifest(hashes={"model": file_hash})
            
            is_valid = verify_manifest_against_file(temp_path, manifest)
            
            assert is_valid is True
        finally:
            os.unlink(temp_path)
    
    def test_verify_manifest_against_file_invalid(self):
        """Test hash verification with invalid hash."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
            f.write("model data")
            temp_path = f.name
        
        try:
            manifest = ApexCoreV2Manifest(hashes={"model": "sha256:wronghash"})
            
            is_valid = verify_manifest_against_file(temp_path, manifest)
            
            assert is_valid is False
        finally:
            os.unlink(temp_path)
    
    def test_verify_manifest_missing_file(self):
        """Test hash verification with missing file."""
        manifest = ApexCoreV2Manifest(hashes={"model": "sha256:test"})
        
        is_valid = verify_manifest_against_file("/nonexistent/path.joblib", manifest)
        
        assert is_valid is False
    
    def test_create_manifest_for_model(self):
        """Test manifest creation for a model file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
            f.write("model data for manifest test")
            temp_path = f.name
        
        try:
            metrics = ManifestMetrics(val_auc_runner=0.82)
            
            manifest = create_manifest_for_model(
                model_path=temp_path,
                variant="mini",
                ensemble_size=3,
                metrics=metrics,
                engine_snapshot_id="engine123",
                lab_dataset_id="data456",
            )
            
            assert manifest.variant == "mini"
            assert manifest.ensemble_size == 3
            assert manifest.engine_snapshot_id == "engine123"
            assert manifest.metrics.val_auc_runner == 0.82
            assert "model" in manifest.hashes
            assert manifest.hashes["model"].startswith("sha256:")
        finally:
            os.unlink(temp_path)


class TestManifestEdgeCases:
    """Tests for manifest edge cases."""
    
    def test_manifest_with_empty_hashes(self):
        """Test manifest with empty hashes dictionary."""
        manifest = ApexCoreV2Manifest(hashes={})
        
        d = manifest.to_dict()
        loaded = ApexCoreV2Manifest.from_dict(d)
        
        assert loaded.hashes == {}
    
    def test_manifest_round_trip(self):
        """Test complete manifest round trip."""
        original = ApexCoreV2Manifest(
            variant="big",
            ensemble_size=5,
            engine_snapshot_id="engine",
            lab_dataset_id="dataset",
            hashes={
                "member_0": "sha256:abc",
                "member_1": "sha256:def",
            },
            version="2.0.0",
            symbols_trained=100,
            samples_trained=50000,
        )
        original.metrics = ManifestMetrics(
            val_auc_runner=0.92,
            val_calibration_error_runner=0.03,
        )
        original.thresholds = ManifestThresholds(
            max_disagreement_allowed=0.1,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "full_manifest.json")
            original.save(path)
            loaded = ApexCoreV2Manifest.load(path)
        
        assert loaded.variant == original.variant
        assert loaded.ensemble_size == original.ensemble_size
        assert loaded.metrics.val_auc_runner == original.metrics.val_auc_runner
        assert loaded.thresholds.max_disagreement_allowed == original.thresholds.max_disagreement_allowed
        assert loaded.symbols_trained == original.symbols_trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
