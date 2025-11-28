"""
Tests for v9.0-A Engine Hardening modules.
"""

import pytest
from typing import Dict, Any, List


class TestRedundantScorer:
    """Tests for redundant scoring architecture."""
    
    def test_shadow_scorer_computes_score(self):
        from src.quantracore_apex.core.redundant_scorer import ShadowScorer
        
        scorer = ShadowScorer()
        protocol_results = {
            "T01": {"fired": True, "confidence": 0.8},
            "T02": {"fired": True, "confidence": 0.6},
            "T03": {"fired": False, "confidence": 0.2},
        }
        
        score, band = scorer.compute_shadow_score(
            protocol_results=protocol_results,
            regime="trending_up",
            risk_tier="moderate",
            monster_runner_state="idle"
        )
        
        assert 0 <= score <= 100
        assert band in ["strong", "moderate", "weak", "poor", "reject"]
    
    def test_consistency_check_within_threshold(self):
        from src.quantracore_apex.core.redundant_scorer import ShadowScorer, ScoreConsistencyStatus
        
        scorer = ShadowScorer()
        
        result = scorer.check_consistency(
            primary_score=65.0,
            primary_band="moderate",
            shadow_score=63.0,
            shadow_band="moderate"
        )
        
        assert result.status == ScoreConsistencyStatus.OK
        assert result.band_match is True
        assert result.absolute_diff == 2.0
    
    def test_consistency_check_warning_on_deviation(self):
        from src.quantracore_apex.core.redundant_scorer import ShadowScorer, ScoreConsistencyStatus
        
        scorer = ShadowScorer()
        
        result = scorer.check_consistency(
            primary_score=65.0,
            primary_band="moderate",
            shadow_score=58.0,
            shadow_band="weak"
        )
        
        assert result.status == ScoreConsistencyStatus.WARNING
        assert result.band_match is False
    
    def test_consistency_check_fail_on_major_deviation(self):
        from src.quantracore_apex.core.redundant_scorer import ShadowScorer, ScoreConsistencyStatus
        
        scorer = ShadowScorer()
        
        result = scorer.check_consistency(
            primary_score=70.0,
            primary_band="moderate",
            shadow_score=55.0,
            shadow_band="weak"
        )
        
        assert result.status == ScoreConsistencyStatus.FAIL
        assert result.absolute_diff == 15.0
    
    def test_redundant_scorer_with_verification(self):
        from src.quantracore_apex.core.redundant_scorer import RedundantScorer
        
        scorer = RedundantScorer()
        protocol_results = {
            "T01": {"fired": True, "confidence": 0.7},
        }
        
        result = scorer.compute_with_verification(
            primary_score=60.0,
            primary_band="moderate",
            protocol_results=protocol_results,
            regime="range_bound",
            risk_tier="low"
        )
        
        assert "primary_score" in result
        assert "shadow_score" in result
        assert "consistency_status" in result
        assert "consistency_ok" in result
    
    def test_consistency_stats_tracking(self):
        from src.quantracore_apex.core.redundant_scorer import RedundantScorer
        
        scorer = RedundantScorer()
        
        for i in range(5):
            scorer.compute_with_verification(
                primary_score=60.0 + i,
                primary_band="moderate",
                protocol_results={},
                regime="range_bound",
                risk_tier="low"
            )
        
        stats = scorer.get_consistency_stats()
        assert stats["total_checks"] == 5
        assert "ok_rate" in stats


class TestDriftDetector:
    """Tests for drift detection framework."""
    
    def test_drift_detector_init(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector, DriftMode
        
        detector = DriftDetector()
        
        assert detector.current_mode == DriftMode.NORMAL
        assert len(detector.baselines) == 0
    
    def test_update_rolling_values(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        for i in range(10):
            detector.update_rolling("test_metric", float(i))
        
        assert "test_metric" in detector.rolling_values
        assert len(detector.rolling_values["test_metric"]) == 10
    
    def test_compute_baseline_from_rolling(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector
        
        detector = DriftDetector()
        
        for i in range(20):
            detector.update_rolling("test_metric", 50.0 + i * 0.5)
        
        baseline = detector.compute_baseline_from_rolling("test_metric")
        
        assert baseline is not None
        assert baseline.metric_name == "test_metric"
        assert baseline.sample_count == 20
        assert baseline.mean > 0
        assert baseline.std_dev >= 0
    
    def test_check_drift_no_baseline(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector, DriftSeverity
        
        detector = DriftDetector()
        
        event = detector.check_drift("unknown_metric", 50.0)
        
        assert event.severity == DriftSeverity.NONE
        assert event.details.get("reason") == "no_baseline"
    
    def test_check_drift_with_baseline(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector, DriftBaseline, DriftSeverity
        
        detector = DriftDetector()
        
        detector.baselines["test_metric"] = DriftBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=5.0,
            sample_count=100
        )
        
        event = detector.check_drift("test_metric", 51.0)
        assert event.severity == DriftSeverity.NONE
        
        event = detector.check_drift("test_metric", 65.0)
        assert event.severity == DriftSeverity.SEVERE
    
    def test_evaluate_mode_transitions(self):
        from src.quantracore_apex.core.drift_detector import (
            DriftDetector, DriftBaseline, DriftMode
        )
        
        detector = DriftDetector()
        
        detector.baselines["test_metric"] = DriftBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=5.0,
            sample_count=100
        )
        
        for _ in range(5):
            detector.check_drift("test_metric", 80.0)
        
        mode = detector.evaluate_mode()
        assert mode == DriftMode.DRIFT_GUARDED
    
    def test_get_status(self):
        from src.quantracore_apex.core.drift_detector import DriftDetector
        
        detector = DriftDetector()
        status = detector.get_status()
        
        assert "mode" in status
        assert "baselines_loaded" in status
        assert "total_drift_events" in status


class TestDecisionGates:
    """Tests for fail-closed decision gates."""
    
    def test_data_integrity_gate_passes_valid_data(self):
        from src.quantracore_apex.core.decision_gates import DataIntegrityGate, GateStatus
        
        gate = DataIntegrityGate()
        
        ohlcv_data = [
            {"date": "2024-01-01", "open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000},
            {"date": "2024-01-02", "open": 103, "high": 108, "low": 101, "close": 106, "volume": 1200000},
        ]
        
        result = gate.check(ohlcv_data)
        
        assert result.status == GateStatus.PASSED
        assert result.action_taken == "proceed"
        assert "no_negative_values" in result.checks_passed
    
    def test_data_integrity_gate_fails_empty_data(self):
        from src.quantracore_apex.core.decision_gates import DataIntegrityGate, GateStatus
        
        gate = DataIntegrityGate()
        
        result = gate.check([])
        
        assert result.status == GateStatus.FAILED
        assert result.action_taken == "reject_signal"
        assert "no_data" in result.checks_failed
    
    def test_data_integrity_gate_fails_negative_prices(self):
        from src.quantracore_apex.core.decision_gates import DataIntegrityGate, GateStatus
        
        gate = DataIntegrityGate()
        
        ohlcv_data = [
            {"date": "2024-01-01", "open": -100, "high": 105, "low": 98, "close": 103, "volume": 1000000},
        ]
        
        result = gate.check(ohlcv_data)
        
        assert result.status == GateStatus.FAILED
        assert "negative_values" in result.checks_failed
    
    def test_model_integrity_gate_passes(self):
        from src.quantracore_apex.core.decision_gates import ModelIntegrityGate, GateStatus
        
        gate = ModelIntegrityGate(engine_version="9.0-A")
        
        result = gate.check(
            model_id="apexcore_full_v1",
            model_hash="abc123def456",
            model_version="9.0-A",
            manifest_hash="abc123def456"
        )
        
        assert result.status == GateStatus.PASSED
        assert "hash_matches_manifest" in result.checks_passed
        assert "version_compatible" in result.checks_passed
    
    def test_model_integrity_gate_fails_hash_mismatch(self):
        from src.quantracore_apex.core.decision_gates import ModelIntegrityGate, GateStatus
        
        gate = ModelIntegrityGate()
        
        result = gate.check(
            model_id="apexcore_full_v1",
            model_hash="abc123",
            model_version="9.0-A",
            manifest_hash="different_hash"
        )
        
        assert result.status == GateStatus.FAILED
        assert "hash_mismatch" in result.checks_failed
        assert result.action_taken == "fallback_deterministic"
    
    def test_risk_guard_gate_passes(self):
        from src.quantracore_apex.core.decision_gates import RiskGuardGate, GateStatus
        
        gate = RiskGuardGate()
        
        result = gate.check({
            "position_risk": 0.05,
            "portfolio_risk": 0.15,
            "concentration": 0.20,
            "liquidity_score": 0.50,
        })
        
        assert result.status == GateStatus.PASSED
        assert result.action_taken == "proceed"
    
    def test_risk_guard_gate_fails_high_risk(self):
        from src.quantracore_apex.core.decision_gates import RiskGuardGate, GateStatus
        
        gate = RiskGuardGate()
        
        result = gate.check({
            "position_risk": 0.25,
            "portfolio_risk": 0.40,
        })
        
        assert result.status == GateStatus.FAILED
        assert result.action_taken == "blocked_by_risk"
    
    def test_risk_guard_gate_fails_no_metrics(self):
        from src.quantracore_apex.core.decision_gates import RiskGuardGate, GateStatus
        
        gate = RiskGuardGate()
        
        result = gate.check({})
        
        assert result.status == GateStatus.FAILED
        assert "no_risk_metrics" in result.checks_failed
    
    def test_decision_gate_runner_all_pass(self):
        from src.quantracore_apex.core.decision_gates import DecisionGateRunner
        
        runner = DecisionGateRunner()
        
        ohlcv_data = [
            {"date": "2024-01-01", "open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000000},
        ]
        risk_metrics = {"position_risk": 0.05}
        
        all_passed, results = runner.run_all_gates(
            ohlcv_data=ohlcv_data,
            risk_metrics=risk_metrics
        )
        
        assert all_passed is True
        assert "data_integrity" in results
        assert "risk_guard" in results


class TestReplayEngine:
    """Tests for sandbox replay engine."""
    
    def test_replay_engine_init(self):
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        
        engine = ReplayEngine()
        
        assert engine.replay_history == []
    
    def test_replay_config_defaults(self):
        from src.quantracore_apex.replay.replay_engine import ReplayConfig
        
        config = ReplayConfig()
        
        assert config.universe == "demo"
        assert config.timeframe == "1d"
        assert config.initial_capital == 100000.0
    
    def test_run_demo_replay(self):
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        
        engine = ReplayEngine()
        result = engine.run_demo_replay()
        
        assert result.symbols_processed > 0
        assert len(result.equity_curve) > 0
        assert result.duration_seconds >= 0
    
    def test_replay_result_to_dict(self):
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        
        engine = ReplayEngine()
        result = engine.run_demo_replay()
        
        result_dict = result.to_dict()
        
        assert "config" in result_dict
        assert "equity_curve" in result_dict
        assert "signals_generated" in result_dict
    
    def test_replay_summary(self):
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        
        engine = ReplayEngine()
        engine.run_demo_replay()
        engine.run_demo_replay()
        
        summary = engine.get_replay_summary()
        
        assert summary["total_replays"] == 2
        assert summary["total_symbols_processed"] > 0


class TestConfigFiles:
    """Tests for v9.0-A configuration files."""
    
    def test_mode_yaml_exists_and_valid(self):
        from pathlib import Path
        import yaml
        
        mode_file = Path("config/mode.yaml")
        assert mode_file.exists(), "config/mode.yaml must exist"
        
        with open(mode_file) as f:
            config = yaml.safe_load(f)
        
        assert config["default_mode"] == "research"
        assert "modes" in config
        assert "research" in config["modes"]
    
    def test_symbol_universe_yaml_exists_and_valid(self):
        from pathlib import Path
        import yaml
        
        universe_file = Path("config/symbol_universe.yaml")
        assert universe_file.exists(), "config/symbol_universe.yaml must exist"
        
        with open(universe_file) as f:
            config = yaml.safe_load(f)
        
        assert "universes" in config
        assert "demo" in config["universes"]
        assert len(config["universes"]["demo"]["symbols"]) > 0
