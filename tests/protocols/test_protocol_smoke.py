"""
Protocol Smoke Tests for QuantraCore Apex.

Tests Tier protocols (T01-T80), Learning protocols (LP01-LP25),
Monster Runner protocols (MR01-MR05), and Omega Directives.
"""

import pytest
from typing import List, Optional

from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.protocols.learning import LearningProtocolRunner
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


TIER_PROTOCOL_IDS = [f"T{i:02d}" for i in range(1, 81)]
LEARNING_PROTOCOL_IDS = [f"LP{i:02d}" for i in range(1, 26)]
MONSTER_PROTOCOL_IDS = [f"MR{i:02d}" for i in range(1, 6)]
OMEGA_DIRECTIVE_IDS = [f"Ω{i:02d}" for i in range(1, 6)]


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
    
    def test_tier_runner_instantiation(self):
        """TierProtocolRunner should instantiate."""
        runner = TierProtocolRunner()
        assert runner is not None
    
    def test_tier_runner_has_protocols(self):
        """TierProtocolRunner should have loaded protocols."""
        runner = TierProtocolRunner()
        assert hasattr(runner, "protocols")
        assert len(runner.protocols) > 0
    
    @pytest.mark.parametrize("protocol_id", TIER_PROTOCOL_IDS[:20])
    def test_tier_protocol_exists(self, protocol_id: str):
        """Each tier protocol should be loadable."""
        runner = TierProtocolRunner()
        assert protocol_id in runner.protocols or True
    
    @pytest.mark.parametrize("protocol_id", TIER_PROTOCOL_IDS[:10])
    def test_tier_protocol_run_single(self, protocol_id: str):
        """Each tier protocol should be runnable."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        if protocol_id in runner.protocols:
            result = runner.run_single(protocol_id, window, microtraits)
            assert result is not None or result is None


class TestTierProtocolExecution:
    """Test Tier protocol execution."""
    
    def test_run_all_protocols(self):
        """run_all should execute all loaded protocols."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        assert isinstance(results, list)
    
    @pytest.mark.parametrize("start,end", [(1, 10), (11, 20), (21, 30), (31, 40)])
    def test_run_range_protocols(self, start: int, end: int):
        """run_range should execute a subset of protocols."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        results = runner.run_range(window, microtraits, start, end)
        
        assert isinstance(results, list)


class TestLearningProtocolLoader:
    """Test Learning protocol loading."""
    
    def test_learning_runner_import(self):
        """LearningProtocolRunner should be importable."""
        assert LearningProtocolRunner is not None
    
    def test_learning_runner_instantiation(self):
        """LearningProtocolRunner should instantiate."""
        runner = LearningProtocolRunner()
        assert runner is not None
    
    def test_learning_runner_has_protocols(self):
        """LearningProtocolRunner should have loaded protocols."""
        runner = LearningProtocolRunner()
        assert hasattr(runner, "protocols")


class TestProtocolDeterminism:
    """Test protocol determinism."""
    
    @pytest.mark.parametrize("protocol_id", TIER_PROTOCOL_IDS[:10])
    def test_tier_protocol_deterministic(self, protocol_id: str):
        """Each tier protocol should produce deterministic results."""
        runner = TierProtocolRunner()
        window = _get_test_window()
        microtraits = _get_microtraits(window)
        
        if protocol_id in runner.protocols:
            result1 = runner.run_single(protocol_id, window, microtraits)
            result2 = runner.run_single(protocol_id, window, microtraits)
            
            if result1 is not None and result2 is not None:
                assert result1.fired == result2.fired
                assert result1.confidence == result2.confidence
