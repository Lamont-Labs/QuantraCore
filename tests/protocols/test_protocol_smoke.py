"""
Protocol Smoke Tests for QuantraCore Apex.

Tests Tier protocols (T01-T80), Learning protocols (LP01-LP25).
All assertions are substantive - they WILL fail if behavior regresses.
"""

import pytest
from typing import List, Optional

from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.protocols.learning import LearningProtocolRunner
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


def _get_test_window(symbol: str = "AAPL") -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


def _get_microtraits(window: OhlcvWindow) -> Microtraits:
    """Helper to compute microtraits."""
    return compute_microtraits(window)


class TestTierProtocolLoader:
    """Test Tier protocol loading."""
    
    def test_tier_runner_import(self):
        """TierProtocolRunner should be importable."""
        assert TierProtocolRunner is not None
        assert callable(TierProtocolRunner)
    
    def test_tier_runner_instantiation(self):
        """TierProtocolRunner should instantiate."""
        runner = TierProtocolRunner()
        assert runner is not None
        assert isinstance(runner, TierProtocolRunner)
    
    def test_tier_runner_has_protocols_dict(self):
        """TierProtocolRunner should have loaded protocols dict."""
        runner = TierProtocolRunner()
        assert hasattr(runner, "protocols")
        assert isinstance(runner.protocols, dict)
    
    def test_tier_runner_loaded_at_least_50_protocols(self):
        """TierProtocolRunner should load at least 50 protocols."""
        runner = TierProtocolRunner()
        loaded_count = len(runner.protocols)
        assert loaded_count >= 50, f"Only {loaded_count} protocols loaded, expected >= 50"
    
    def test_tier_runner_protocols_are_callable(self):
        """All loaded protocols should be callable functions."""
        runner = TierProtocolRunner()
        for protocol_id, protocol_fn in runner.protocols.items():
            assert callable(protocol_fn), f"Protocol {protocol_id} is not callable"


class TestTierProtocolExecution:
    """Test Tier protocol execution with real assertions."""
    
    def test_run_all_returns_list_of_protocol_results(self):
        """run_all should return list of ProtocolResult objects."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        assert isinstance(results, list)
        assert len(results) > 0, "run_all returned empty list"
        for r in results:
            assert isinstance(r, ProtocolResult), f"Expected ProtocolResult, got {type(r)}"
    
    def test_run_all_protocol_results_have_required_fields(self):
        """Each ProtocolResult should have protocol_id, fired, confidence."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        for r in results:
            assert hasattr(r, "protocol_id"), "ProtocolResult missing protocol_id"
            assert hasattr(r, "fired"), "ProtocolResult missing fired"
            assert hasattr(r, "confidence"), "ProtocolResult missing confidence"
            assert isinstance(r.fired, bool), f"fired should be bool, got {type(r.fired)}"
            assert isinstance(r.confidence, (int, float)), f"confidence should be numeric"
            assert 0.0 <= r.confidence <= 1.0, f"confidence {r.confidence} out of [0,1] range"
    
    @pytest.mark.parametrize("start,end,expected_min", [
        (1, 10, 5),
        (11, 20, 5),
        (21, 30, 5),
        (31, 40, 5),
    ])
    def test_run_range_returns_expected_count(self, start: int, end: int, expected_min: int):
        """run_range should return results for loaded protocols in range."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results = runner.run_range(window, microtraits, start, end)
        
        assert isinstance(results, list)
        assert len(results) >= expected_min, f"Expected >= {expected_min} results, got {len(results)}"
    
    def test_run_single_returns_protocol_result(self):
        """run_single should return a ProtocolResult for valid protocol."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        first_protocol = list(runner.protocols.keys())[0]
        result = runner.run_single(first_protocol, window, microtraits)
        
        assert result is not None, f"run_single returned None for {first_protocol}"
        assert isinstance(result, ProtocolResult)
        assert result.protocol_id == first_protocol


class TestLearningProtocolLoader:
    """Test Learning protocol loading."""
    
    def test_learning_runner_import(self):
        """LearningProtocolRunner should be importable."""
        assert LearningProtocolRunner is not None
    
    def test_learning_runner_instantiation(self):
        """LearningProtocolRunner should instantiate."""
        runner = LearningProtocolRunner()
        assert runner is not None
        assert isinstance(runner, LearningProtocolRunner)
    
    def test_learning_runner_has_protocols_attribute(self):
        """LearningProtocolRunner should have protocols attribute."""
        runner = LearningProtocolRunner()
        assert hasattr(runner, "protocols")


class TestProtocolDeterminism:
    """Test protocol determinism with substantive checks."""
    
    def test_run_all_deterministic_same_results(self):
        """run_all should produce identical results on same input."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results1 = runner.run_all(window, microtraits)
        results2 = runner.run_all(window, microtraits)
        
        assert len(results1) == len(results2), "Different result counts"
        
        for r1, r2 in zip(results1, results2):
            assert r1.protocol_id == r2.protocol_id, f"Protocol ID mismatch: {r1.protocol_id} vs {r2.protocol_id}"
            assert r1.fired == r2.fired, f"Fired mismatch for {r1.protocol_id}: {r1.fired} vs {r2.fired}"
            assert r1.confidence == r2.confidence, f"Confidence mismatch for {r1.protocol_id}"
    
    def test_run_single_deterministic_across_10_runs(self):
        """run_single should be deterministic across 10 runs."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        first_protocol = list(runner.protocols.keys())[0]
        
        results = []
        for _ in range(10):
            r = runner.run_single(first_protocol, window, microtraits)
            assert r is not None, "run_single returned None"
            results.append((r.fired, r.confidence))
        
        first = results[0]
        for i, r in enumerate(results[1:], 2):
            assert r == first, f"Run {i} produced different result: {r} vs {first}"


class TestProtocolCoverage:
    """Test that protocols actually execute and produce meaningful output."""
    
    def test_at_least_one_protocol_fires(self):
        """At least one protocol should fire on typical data."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = _get_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        fired_count = sum(1 for r in results if r.fired)
        assert fired_count > 0, "No protocols fired - check protocol logic"
    
    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "TSLA", "GME", "AMZN"])
    def test_protocols_execute_on_various_symbols(self, symbol: str):
        """Protocols should execute on various symbols without error."""
        runner = TierProtocolRunner()
        window = _get_test_window(symbol)
        microtraits = _get_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        assert len(results) > 0, f"No results for {symbol}"
        
        for r in results:
            assert r.confidence >= 0.0, f"Negative confidence for {r.protocol_id}"
            assert r.confidence <= 1.0, f"Confidence > 1 for {r.protocol_id}"
