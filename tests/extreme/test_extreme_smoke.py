"""
Extreme Condition Smoke Tests for QuantraCore Apex.

Fast tests on volatile/extreme symbols with SUBSTANTIVE assertions.
"""

import pytest

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexContext, RiskTier, ApexResult
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


VOLATILE_SYMBOLS = ["TSLA", "GME", "AMC", "RIVN", "LCID"]
MEME_STOCKS = ["GME", "AMC", "BB", "NOK", "EXPR"]
CRYPTO_ADJACENT = ["RIOT", "MARA", "COIN", "HOOD", "SQ"]

ALL_EXTREME = list(set(VOLATILE_SYMBOLS + MEME_STOCKS + CRYPTO_ADJACENT))


def _get_test_window(symbol: str, seed: int = 42) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=seed)
    bars = adapter.fetch(symbol, num_bars=100, seed=seed)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestExtremeVolatileSymbols:
    """Test engine on volatile symbols."""
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_returns_apex_result(self, symbol: str):
        """Volatile symbols should return ApexResult."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult), f"Expected ApexResult, got {type(result)}"
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_valid_score(self, symbol: str):
        """Volatile symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100, f"Score {result.quantrascore} out of range"
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_has_risk_tier(self, symbol: str):
        """Volatile symbols should have RiskTier assigned."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.risk_tier, RiskTier), f"Expected RiskTier, got {type(result.risk_tier)}"


class TestExtremeMemeStocks:
    """Test engine on meme stocks."""
    
    @pytest.mark.parametrize("symbol", MEME_STOCKS)
    def test_meme_valid_result(self, symbol: str):
        """Meme stocks should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", MEME_STOCKS)
    def test_meme_deterministic(self, symbol: str):
        """Meme stocks should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore, \
            f"Non-deterministic for {symbol}: {result1.quantrascore} vs {result2.quantrascore}"


class TestExtremeCryptoAdjacent:
    """Test engine on crypto-adjacent symbols."""
    
    @pytest.mark.parametrize("symbol", CRYPTO_ADJACENT)
    def test_crypto_adjacent_valid(self, symbol: str):
        """Crypto-adjacent symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100


class TestExtremeAllSymbols:
    """Test all extreme symbols."""
    
    @pytest.mark.parametrize("symbol", ALL_EXTREME)
    def test_extreme_produces_result(self, symbol: str):
        """All extreme symbols should produce results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert isinstance(result, ApexResult)
    
    @pytest.mark.parametrize("symbol", ALL_EXTREME)
    def test_extreme_has_required_fields(self, symbol: str):
        """All extreme symbols should have required fields."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.regime is not None
        assert result.risk_tier is not None
        assert result.entropy_state is not None


class TestExtremeDifferentSeeds:
    """Test extreme conditions with different seeds."""
    
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    def test_different_seeds_valid_results(self, seed: int):
        """Different seeds should all produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("TSLA", seed=seed)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("seed", [1, 42, 100, 999, 12345])
    def test_same_seed_deterministic(self, seed: int):
        """Same seed should produce deterministic results."""
        engine = ApexEngine(enable_logging=False)
        
        window1 = _get_test_window("TSLA", seed=seed)
        window2 = _get_test_window("TSLA", seed=seed)
        
        result1 = engine.run(window1)
        result2 = engine.run(window2)
        
        assert result1.quantrascore == result2.quantrascore


class TestExtremeComplianceMode:
    """Test extreme conditions with compliance mode."""
    
    @pytest.mark.parametrize("symbol", ALL_EXTREME[:5])
    def test_compliance_mode_works(self, symbol: str):
        """Compliance mode should work on extreme symbols."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(compliance_mode=True)
        
        result = engine.run(window, context)
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
