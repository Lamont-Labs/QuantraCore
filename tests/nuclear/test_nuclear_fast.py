"""
Nuclear Fast Determinism Tests for QuantraCore Apex.

Fast determinism validation with SUBSTANTIVE assertions.
All tests verify actual behavioral guarantees.
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
ALL_SYMBOLS = LIQUID_SYMBOLS + VOLATILE_SYMBOLS


def _get_test_window(symbol: str, seed: int = 42) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=seed)
    bars = adapter.fetch(symbol, num_bars=100, seed=seed)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestNuclearScoreDeterminism:
    """Test QuantraScore determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_score_identical_2_runs(self, symbol: str):
        """QuantraScore MUST be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore, \
            f"DETERMINISM VIOLATION for {symbol}: {result1.quantrascore} vs {result2.quantrascore}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_score_identical_5_runs(self, symbol: str):
        """QuantraScore MUST be identical across 5 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        scores = []
        for i in range(5):
            result = engine.run(window)
            scores.append(result.quantrascore)
        
        first = scores[0]
        for i, score in enumerate(scores[1:], 2):
            assert score == first, \
                f"DETERMINISM VIOLATION: Run {i} = {score}, expected {first}"


class TestNuclearRegimeDeterminism:
    """Test regime determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_regime_identical_2_runs(self, symbol: str):
        """Regime MUST be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.regime == result2.regime, \
            f"REGIME VIOLATION for {symbol}: {result1.regime} vs {result2.regime}"
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_risk_tier_identical_2_runs(self, symbol: str):
        """Risk tier MUST be identical across 2 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.risk_tier == result2.risk_tier, \
            f"RISK TIER VIOLATION for {symbol}: {result1.risk_tier} vs {result2.risk_tier}"


class TestNuclearStateDeterminism:
    """Test state determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_entropy_state_identical(self, symbol: str):
        """Entropy state MUST be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.entropy_state == result2.entropy_state
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_suppression_state_identical(self, symbol: str):
        """Suppression state MUST be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.suppression_state == result2.suppression_state
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_drift_state_identical(self, symbol: str):
        """Drift state MUST be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.drift_state == result2.drift_state


class TestNuclearProtocolDeterminism:
    """Test protocol determinism."""
    
    def test_protocol_results_identical(self):
        """Protocol results MUST be identical across runs."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        results1 = runner.run_all(window, microtraits)
        results2 = runner.run_all(window, microtraits)
        
        assert len(results1) == len(results2), "Different result counts"
        
        for r1, r2 in zip(results1, results2):
            assert r1.protocol_id == r2.protocol_id
            assert r1.fired == r2.fired, \
                f"Protocol {r1.protocol_id} fired mismatch: {r1.fired} vs {r2.fired}"
            assert r1.confidence == r2.confidence, \
                f"Protocol {r1.protocol_id} confidence mismatch"


class TestNuclearCrossEngine:
    """Test determinism across engine instances."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_cross_engine_identical_score(self, symbol: str):
        """Different engine instances MUST produce same score."""
        engine1 = ApexEngine(enable_logging=False)
        engine2 = ApexEngine(enable_logging=False)
        
        window = _get_test_window(symbol)
        
        result1 = engine1.run(window)
        result2 = engine2.run(window)
        
        assert result1.quantrascore == result2.quantrascore, \
            f"CROSS-ENGINE VIOLATION: {result1.quantrascore} vs {result2.quantrascore}"
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_cross_engine_identical_regime(self, symbol: str):
        """Different engine instances MUST produce same regime."""
        engine1 = ApexEngine(enable_logging=False)
        engine2 = ApexEngine(enable_logging=False)
        
        window = _get_test_window(symbol)
        
        result1 = engine1.run(window)
        result2 = engine2.run(window)
        
        assert result1.regime == result2.regime


class TestNuclearMicrotraitsDeterminism:
    """Test microtraits determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_identical(self, symbol: str):
        """Microtraits MUST be identical across runs."""
        window = _get_test_window(symbol)
        
        mt1 = compute_microtraits(window)
        mt2 = compute_microtraits(window)
        
        assert mt1.volatility_ratio == mt2.volatility_ratio, \
            f"volatility_ratio mismatch: {mt1.volatility_ratio} vs {mt2.volatility_ratio}"
        assert mt1.trend_consistency == mt2.trend_consistency


class TestNuclearWindowHash:
    """Test window hash determinism."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_window_hash_identical(self, symbol: str):
        """Window hash MUST be identical across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.window_hash == result2.window_hash, \
            f"HASH VIOLATION: {result1.window_hash} vs {result2.window_hash}"
