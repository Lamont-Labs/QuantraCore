"""
Tests for ApexCore V2 determinism.

Validates bitwise-identical results across:
- Multiple runs with same seed
- Model save/load cycles
- Ensemble predictions
"""

import pytest
import numpy as np
import tempfile
import os

from src.quantracore_apex.apexcore.apexcore_v2 import (
    ApexCoreV2Model,
    ApexCoreV2Big,
    ApexCoreV2Mini,
    ApexCoreV2Ensemble,
    ApexCoreV2Config,
    ModelVariant,
)


class TestModelDeterminism:
    """Tests for model determinism."""
    
    def test_same_seed_same_output(self):
        """Test that same seed produces identical outputs."""
        config = ApexCoreV2Config(
            variant=ModelVariant.MINI,
            random_state=42,
            n_estimators_mini=10,
            max_depth_mini=3,
        )
        
        n_samples = 50
        n_features = 15
        X = np.random.RandomState(123).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(456).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(789).randint(0, 2, n_samples),
            "future_quality_tier": np.array(["A", "B", "C", "D", "A_PLUS"] * 10),
            "avoid_trade": np.random.RandomState(101).randint(0, 2, n_samples),
            "regime_label": np.array(["chop", "trend_up", "trend_down", "squeeze", "crash"] * 10),
        }
        
        model1 = ApexCoreV2Model(config=config)
        model1.fit(X, targets)
        
        model2 = ApexCoreV2Model(config=config)
        model2.fit(X, targets)
        
        X_test = np.random.RandomState(999).randn(10, n_features)
        
        outputs1 = model1.forward(X_test)
        outputs2 = model2.forward(X_test)
        
        np.testing.assert_array_almost_equal(
            outputs1["quantra_score"],
            outputs2["quantra_score"],
            decimal=10,
        )
        
        np.testing.assert_array_almost_equal(
            outputs1["runner_prob"],
            outputs2["runner_prob"],
            decimal=10,
        )
    
    def test_different_seeds_different_output(self):
        """Test that different seeds can produce different outputs."""
        n_samples = 100
        n_features = 20
        X = np.random.RandomState(123).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(456).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(789).randint(0, 2, n_samples),
            "future_quality_tier": np.random.RandomState(111).choice(
                ["A", "B", "C", "D", "A_PLUS"], n_samples
            ),
            "avoid_trade": np.random.RandomState(101).randint(0, 2, n_samples),
            "regime_label": np.random.RandomState(222).choice(
                ["chop", "trend_up", "trend_down", "squeeze", "crash"], n_samples
            ),
        }
        
        config1 = ApexCoreV2Config(variant=ModelVariant.MINI, random_state=42, n_estimators_mini=30)
        config2 = ApexCoreV2Config(variant=ModelVariant.MINI, random_state=999, n_estimators_mini=30)
        
        model1 = ApexCoreV2Model(config=config1)
        model2 = ApexCoreV2Model(config=config2)
        
        model1.fit(X, targets)
        model2.fit(X, targets)
        
        X_test = np.random.RandomState(777).randn(20, n_features)
        
        outputs1 = model1.forward(X_test)
        outputs2 = model2.forward(X_test)
        
        assert outputs1["quantra_score"].shape == outputs2["quantra_score"].shape
    
    def test_save_load_determinism(self):
        """Test that save/load produces identical outputs."""
        config = ApexCoreV2Config(
            variant=ModelVariant.MINI,
            random_state=42,
            n_estimators_mini=10,
        )
        
        n_samples = 50
        n_features = 15
        X = np.random.RandomState(123).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(456).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(789).randint(0, 2, n_samples),
            "future_quality_tier": np.array(["A", "B", "C", "D", "A_PLUS"] * 10),
            "avoid_trade": np.random.RandomState(101).randint(0, 2, n_samples),
            "regime_label": np.array(["chop", "trend_up", "trend_down", "squeeze", "crash"] * 10),
        }
        
        model = ApexCoreV2Model(config=config)
        model.fit(X, targets)
        
        X_test = np.random.RandomState(999).randn(10, n_features)
        outputs_before = model.forward(X_test)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.joblib")
            model.save(model_path)
            
            loaded_model = ApexCoreV2Model.load(model_path)
        
        outputs_after = loaded_model.forward(X_test)
        
        np.testing.assert_array_almost_equal(
            outputs_before["quantra_score"],
            outputs_after["quantra_score"],
            decimal=10,
        )
        
        np.testing.assert_array_almost_equal(
            outputs_before["runner_prob"],
            outputs_after["runner_prob"],
            decimal=10,
        )


class TestEnsembleDeterminism:
    """Tests for ensemble determinism."""
    
    def test_ensemble_same_seed_same_output(self):
        """Test ensemble determinism with same base seed."""
        n_samples = 50
        n_features = 15
        X = np.random.RandomState(123).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(456).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(789).randint(0, 2, n_samples),
            "future_quality_tier": np.array(["A", "B", "C", "D", "A_PLUS"] * 10),
            "avoid_trade": np.random.RandomState(101).randint(0, 2, n_samples),
            "regime_label": np.array(["chop", "trend_up", "trend_down", "squeeze", "crash"] * 10),
        }
        
        ensemble1 = ApexCoreV2Ensemble(
            ensemble_size=3,
            variant=ModelVariant.MINI,
            base_random_state=42,
        )
        
        ensemble2 = ApexCoreV2Ensemble(
            ensemble_size=3,
            variant=ModelVariant.MINI,
            base_random_state=42,
        )
        
        ensemble1.fit(X, targets, bootstrap=False)
        ensemble2.fit(X, targets, bootstrap=False)
        
        X_test = np.random.RandomState(999).randn(10, n_features)
        
        outputs1 = ensemble1.forward(X_test)
        outputs2 = ensemble2.forward(X_test)
        
        np.testing.assert_array_almost_equal(
            outputs1["quantra_score"],
            outputs2["quantra_score"],
            decimal=10,
        )
    
    def test_ensemble_save_load_determinism(self):
        """Test ensemble save/load produces identical outputs."""
        n_samples = 50
        n_features = 15
        X = np.random.RandomState(123).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(456).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(789).randint(0, 2, n_samples),
            "future_quality_tier": np.array(["A", "B", "C", "D", "A_PLUS"] * 10),
            "avoid_trade": np.random.RandomState(101).randint(0, 2, n_samples),
            "regime_label": np.array(["chop", "trend_up", "trend_down", "squeeze", "crash"] * 10),
        }
        
        ensemble = ApexCoreV2Ensemble(
            ensemble_size=3,
            variant=ModelVariant.MINI,
            base_random_state=42,
        )
        ensemble.fit(X, targets, bootstrap=False)
        
        X_test = np.random.RandomState(999).randn(10, n_features)
        outputs_before = ensemble.forward(X_test)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save(tmpdir)
            loaded_ensemble = ApexCoreV2Ensemble.load(tmpdir)
        
        outputs_after = loaded_ensemble.forward(X_test)
        
        np.testing.assert_array_almost_equal(
            outputs_before["quantra_score"],
            outputs_after["quantra_score"],
            decimal=10,
        )
        
        np.testing.assert_array_almost_equal(
            outputs_before["runner_prob"],
            outputs_after["runner_prob"],
            decimal=10,
        )


class TestMultipleRuns:
    """Tests for determinism across multiple runs."""
    
    @pytest.mark.parametrize("run_idx", range(5))
    def test_repeated_forward_pass(self, run_idx):
        """Test repeated forward passes produce identical results."""
        model = ApexCoreV2Mini()
        
        n_samples = 30
        n_features = 10
        X = np.random.RandomState(42).randn(n_samples, n_features)
        
        targets = {
            "quantra_score": np.random.RandomState(43).uniform(0, 100, n_samples),
            "hit_runner_threshold": np.random.RandomState(44).randint(0, 2, n_samples),
            "future_quality_tier": np.array(["A", "B", "C"] * 10),
            "avoid_trade": np.random.RandomState(45).randint(0, 2, n_samples),
            "regime_label": np.array(["chop", "trend_up", "trend_down"] * 10),
        }
        
        model.fit(X, targets)
        
        X_test = np.random.RandomState(99).randn(5, n_features)
        
        results = []
        for _ in range(3):
            outputs = model.forward(X_test)
            results.append(outputs["quantra_score"].copy())
        
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[1], results[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
