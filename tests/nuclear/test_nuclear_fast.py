"""
Nuclear Fast Determinism Tests for QuantraCore Apex.

Fast determinism validation (2-3 runs per test).
Heavy nuclear testing is handled by separate scripts.
"""

import pytest
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexContext
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
VOLATILE_SYMBOLS = ["GME", "AMC", "RIVN", "LCID", "SPCE"]
ALL_NUCLEAR_SYMBOLS = LIQUID_SYMBOLS + VOLATILE_SYMBOLS


def _get_test_window(symbol: str, seed: int = 42) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=seed)
    bars = adapter.fetch(symbol, num_bars=100, seed=seed)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestNuclearScoreDeterminism:
    """Test QuantraScore determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_score_identical_2_runs(self, symbol: str):
        """QuantraScore should be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_score_identical_3_runs(self, symbol: str):
        """QuantraScore should be identical across 3 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        scores = []
        for _ in range(3):
            result = engine.run(window)
            scores.append(result.quantrascore)
        
        assert len(set(scores)) == 1


class TestNuclearRegimeDeterminism:
    """Test regime determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_regime_identical_2_runs(self, symbol: str):
        """Regime should be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.regime == result2.regime
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_risk_tier_identical_2_runs(self, symbol: str):
        """Risk tier should be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.risk_tier == result2.risk_tier


class TestNuclearStateDeterminism:
    """Test state determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_entropy_state_identical(self, symbol: str):
        """Entropy state should be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.entropy_state == result2.entropy_state
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_suppression_state_identical(self, symbol: str):
        """Suppression state should be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.suppression_state == result2.suppression_state
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_drift_state_identical(self, symbol: str):
        """Drift state should be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.drift_state == result2.drift_state


class TestNuclearProtocolDeterminism:
    """Test protocol determinism."""
    
    @pytest.mark.parametrize("protocol_id", [f"T{i:02d}" for i in range(1, 11)])
    def test_protocol_deterministic(self, protocol_id: str):
        """Each protocol should be deterministic."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        if protocol_id in runner.protocols:
            result1 = runner.run_single(protocol_id, window, microtraits)
            result2 = runner.run_single(protocol_id, window, microtraits)
            
            if result1 is not None and result2 is not None:
                assert result1.fired == result2.fired
                assert result1.confidence == result2.confidence


class TestNuclearCrossEngine:
    """Test determinism across engine instances."""
    
    @pytest.mark.parametrize("symbol", ALL_NUCLEAR_SYMBOLS)
    def test_cross_engine_deterministic(self, symbol: str):
        """Different engine instances should produce same results."""
        engine1 = ApexEngine(enable_logging=False)
        engine2 = ApexEngine(enable_logging=False)
        
        window = _get_test_window(symbol)
        
        result1 = engine1.run(window)
        result2 = engine2.run(window)
        
        assert result1.quantrascore == result2.quantrascore
        assert result1.regime == result2.regime
        assert result1.risk_tier == result2.risk_tier


class TestNuclearWithContext:
    """Test determinism with context."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_context_deterministic(self, symbol: str):
        """Same context should produce same results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(seed=42, compliance_mode=True)
        
        result1 = engine.run(window, context)
        result2 = engine.run(window, context)
        
        assert result1.quantrascore == result2.quantrascore
