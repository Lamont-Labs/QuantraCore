"""
Tests for ApexLab V2 evaluation metrics.

Validates:
- Calibration curve computation
- Ranking quality analysis
- Regime-segmented performance
- Report generation
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.quantracore_apex.apexlab.evaluation import (
    CalibrationResult,
    RankingResult,
    RegimeResult,
    EvaluationReport,
    evaluate_runner_calibration,
    evaluate_ranking_quality,
    evaluate_regime_performance,
    ApexCoreV2Evaluator,
)


class TestCalibrationMetrics:
    """Tests for calibration curve computation."""
    
    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9])
        
        result = evaluate_runner_calibration(y_true, y_prob, n_bins=5)
        
        assert isinstance(result, CalibrationResult)
        assert result.expected_calibration_error < 0.5
        assert len(result.bin_edges) == 6
    
    def test_uncalibrated_predictions(self):
        """Test calibration with uncalibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        result = evaluate_runner_calibration(y_true, y_prob, n_bins=5)
        
        assert result.expected_calibration_error > 0.3
    
    def test_calibration_bin_counts(self):
        """Test that bin counts sum to total samples."""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)
        
        result = evaluate_runner_calibration(y_true, y_prob, n_bins=10)
        
        assert sum(result.bin_counts) == n_samples
    
    def test_calibration_result_to_dict(self):
        """Test calibration result serialization."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        
        result = evaluate_runner_calibration(y_true, y_prob, n_bins=5)
        d = result.to_dict()
        
        assert "bin_edges" in d
        assert "expected_calibration_error" in d
        assert "max_calibration_error" in d


class TestRankingMetrics:
    """Tests for ranking quality analysis."""
    
    def test_perfect_ranking(self):
        """Test ranking with perfect predictions."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        returns = np.array([0.1, 0.08, 0.05, -0.02, -0.03, -0.05])
        
        result = evaluate_ranking_quality(y_true, y_prob, returns, n_deciles=3)
        
        assert isinstance(result, RankingResult)
        assert result.top_decile_lift >= 1.0
    
    def test_ranking_decile_counts(self):
        """Test that deciles are properly computed."""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)
        returns = np.random.uniform(-0.1, 0.1, n_samples)
        
        result = evaluate_ranking_quality(y_true, y_prob, returns, n_deciles=10)
        
        assert len(result.decile_returns) == 10
        assert len(result.decile_runner_rates) == 10
    
    def test_ranking_auc(self):
        """Test AUC computation in ranking."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.6, 0.5])
        returns = np.zeros(8)
        
        result = evaluate_ranking_quality(y_true, y_prob, returns)
        
        assert 0.0 <= result.auc_runner <= 1.0
    
    def test_ranking_result_to_dict(self):
        """Test ranking result serialization."""
        y_true = np.random.randint(0, 2, 50)
        y_prob = np.random.uniform(0, 1, 50)
        returns = np.random.uniform(-0.1, 0.1, 50)
        
        result = evaluate_ranking_quality(y_true, y_prob, returns)
        d = result.to_dict()
        
        assert "decile_returns" in d
        assert "auc_runner" in d
        assert "top_decile_lift" in d


class TestRegimePerformance:
    """Tests for regime-segmented performance."""
    
    def test_regime_performance_basic(self):
        """Test basic regime performance computation."""
        df = pd.DataFrame({
            "regime_label": ["chop"] * 30 + ["trend_up"] * 30,
            "hit_runner_threshold": [0, 1] * 30,
            "future_quality_tier": ["A", "B", "C"] * 20,
            "quantra_score": np.random.uniform(40, 80, 60),
        })
        
        outputs = {
            "runner_prob": np.random.uniform(0, 1, 60),
            "quality_logits": np.random.uniform(0, 1, (60, 5)),
            "quantra_score": np.random.uniform(40, 80, 60),
        }
        
        results = evaluate_regime_performance(df, outputs)
        
        assert len(results) >= 1
        for result in results:
            assert isinstance(result, RegimeResult)
    
    def test_regime_performance_no_regime_column(self):
        """Test regime performance without regime column."""
        df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
        })
        
        outputs = {}
        
        results = evaluate_regime_performance(df, outputs)
        
        assert len(results) == 0
    
    def test_regime_result_to_dict(self):
        """Test regime result serialization."""
        result = RegimeResult(
            regime="trend_up",
            n_samples=100,
            runner_auc=0.75,
            runner_precision=0.65,
            quality_accuracy=0.45,
            avg_quantra_score_error=5.2,
        )
        
        d = result.to_dict()
        
        assert d["regime"] == "trend_up"
        assert d["n_samples"] == 100
        assert d["runner_auc"] == 0.75


class TestEvaluationReport:
    """Tests for evaluation report generation."""
    
    def test_report_creation(self):
        """Test evaluation report creation."""
        calibration = CalibrationResult(
            bin_edges=[0.0, 0.5, 1.0],
            bin_accuracies=[0.3, 0.7],
            bin_confidences=[0.25, 0.75],
            bin_counts=[50, 50],
            expected_calibration_error=0.05,
            max_calibration_error=0.08,
        )
        
        ranking = RankingResult(
            decile_returns=[0.1, 0.08, 0.05, 0.02, 0.0, -0.01, -0.03, -0.05, -0.07, -0.10],
            decile_runner_rates=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05],
            top_decile_lift=1.6,
            auc_runner=0.78,
            precision_at_k={10: 0.9, 50: 0.75},
        )
        
        report = EvaluationReport(
            timestamp="2024-01-15T12:00:00",
            model_variant="big",
            n_samples=1000,
            calibration=calibration,
            ranking=ranking,
            regime_results=[],
            overall_metrics={"runner_auc": 0.78},
        )
        
        assert report.model_variant == "big"
        assert report.n_samples == 1000
    
    def test_report_to_dict(self):
        """Test report serialization."""
        calibration = CalibrationResult(
            bin_edges=[0.0, 1.0],
            bin_accuracies=[0.5],
            bin_confidences=[0.5],
            bin_counts=[100],
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
        )
        
        ranking = RankingResult(
            decile_returns=[0.0] * 10,
            decile_runner_rates=[0.5] * 10,
            top_decile_lift=1.0,
            auc_runner=0.5,
            precision_at_k={},
        )
        
        report = EvaluationReport(
            timestamp="2024-01-15T12:00:00",
            model_variant="mini",
            n_samples=500,
            calibration=calibration,
            ranking=ranking,
            regime_results=[],
            overall_metrics={},
        )
        
        d = report.to_dict()
        
        assert d["model_variant"] == "mini"
        assert "calibration" in d
        assert "ranking" in d
    
    def test_report_save(self):
        """Test report save to file."""
        calibration = CalibrationResult(
            bin_edges=[0.0, 1.0],
            bin_accuracies=[0.5],
            bin_confidences=[0.5],
            bin_counts=[100],
            expected_calibration_error=0.0,
            max_calibration_error=0.0,
        )
        
        ranking = RankingResult(
            decile_returns=[0.0] * 10,
            decile_runner_rates=[0.5] * 10,
            top_decile_lift=1.0,
            auc_runner=0.5,
            precision_at_k={},
        )
        
        report = EvaluationReport(
            timestamp="2024-01-15T12:00:00",
            model_variant="big",
            n_samples=200,
            calibration=calibration,
            ranking=ranking,
            regime_results=[],
            overall_metrics={"test": 1.0},
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            report.save(path)
            
            assert os.path.exists(path)


class TestApexCoreV2Evaluator:
    """Tests for the evaluator class."""
    
    def test_evaluator_creation(self):
        """Test evaluator instantiation."""
        evaluator = ApexCoreV2Evaluator()
        
        assert evaluator is not None
        assert evaluator.model is None
        assert evaluator.ensemble is None
    
    def test_evaluator_no_model_raises(self):
        """Test that evaluation without model raises error."""
        evaluator = ApexCoreV2Evaluator()
        
        df = pd.DataFrame({"test": [1, 2, 3]})
        features = np.random.randn(3, 10)
        
        with pytest.raises(ValueError):
            evaluator.evaluate(df, features)
    
    def test_text_summary_generation(self):
        """Test text summary generation."""
        calibration = CalibrationResult(
            bin_edges=[0.0, 1.0],
            bin_accuracies=[0.5],
            bin_confidences=[0.5],
            bin_counts=[100],
            expected_calibration_error=0.08,
            max_calibration_error=0.12,
        )
        
        ranking = RankingResult(
            decile_returns=[0.05, 0.03, 0.01, 0.0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
            decile_runner_rates=[0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
            top_decile_lift=1.4,
            auc_runner=0.72,
            precision_at_k={10: 0.8},
        )
        
        report = EvaluationReport(
            timestamp="2024-01-15T12:00:00",
            model_variant="big",
            n_samples=1000,
            calibration=calibration,
            ranking=ranking,
            regime_results=[],
            overall_metrics={"runner_auc": 0.72, "calibration_ece": 0.08},
        )
        
        evaluator = ApexCoreV2Evaluator()
        summary = evaluator.generate_text_summary(report)
        
        assert "ApexCore V2 Evaluation Report" in summary
        assert "Runner AUC" in summary
        assert "Calibration ECE" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
