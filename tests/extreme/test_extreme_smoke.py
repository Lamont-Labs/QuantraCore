"""
Extreme Condition Smoke Tests for QuantraCore Apex.

Fast tests simulating extreme market conditions.
All tests complete in under 2 seconds.
"""

import pytest
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexContext, RiskTier
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


VOLATILE_SYMBOLS = ["TSLA", "GME", "AMC", "RIVN", "LCID", "BBBY", "SPCE", "PLTR"]
CRYPTO_ADJACENT = ["RIOT", "MARA", "COIN", "HOOD", "SQ", "PYPL"]
MEME_STOCKS = ["GME", "AMC", "BBBY", "BB", "NOK", "EXPR"]

ALL_EXTREME_SYMBOLS = list(set(VOLATILE_SYMBOLS + CRYPTO_ADJACENT + MEME_STOCKS))


def _get_test_window(symbol: str, seed: int = 42) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=seed)
    bars = adapter.fetch(symbol, num_bars=100, seed=seed)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestExtremeVolatileSymbols:
    """Test engine on volatile symbols."""
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_valid_score(self, symbol: str):
        """Volatile symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_has_risk_tier(self, symbol: str):
        """Volatile symbols should have risk tier assigned."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.risk_tier is not None
        assert isinstance(result.risk_tier, RiskTier)


class TestExtremeCryptoAdjacent:
    """Test engine on crypto-adjacent symbols."""
    
    @pytest.mark.parametrize("symbol", CRYPTO_ADJACENT)
    def test_crypto_adjacent_valid(self, symbol: str):
        """Crypto-adjacent symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100


class TestExtremeMemeStocks:
    """Test engine on meme stocks."""
    
    @pytest.mark.parametrize("symbol", MEME_STOCKS)
    def test_meme_valid_score(self, symbol: str):
        """Meme stocks should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", MEME_STOCKS)
    def test_meme_deterministic(self, symbol: str):
        """Meme stocks should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore


class TestExtremeAllSymbols:
    """Test all extreme symbols."""
    
    @pytest.mark.parametrize("symbol", ALL_EXTREME_SYMBOLS)
    def test_all_extreme_valid(self, symbol: str):
        """All extreme symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
        assert result.regime is not None
        assert result.risk_tier is not None


class TestExtremeDifferentSeeds:
    """Test extreme conditions with different seeds."""
    
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    def test_different_seeds_valid(self, seed: int):
        """Different seeds should all produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("TSLA", seed=seed)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    def test_seed_deterministic(self, seed: int):
        """Same seed should produce same result."""
        engine = ApexEngine(enable_logging=False)
        
        window1 = _get_test_window("TSLA", seed=seed)
        window2 = _get_test_window("TSLA", seed=seed)
        
        result1 = engine.run(window1)
        result2 = engine.run(window2)
        
        assert result1.quantrascore == result2.quantrascore


class TestExtremeComplianceMode:
    """Test extreme conditions with compliance mode."""
    
    @pytest.mark.parametrize("symbol", ALL_EXTREME_SYMBOLS[:5])
    def test_compliance_mode_extreme(self, symbol: str):
        """Compliance mode should work on extreme symbols."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(compliance_mode=True)
        
        result = engine.run(window, context)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
