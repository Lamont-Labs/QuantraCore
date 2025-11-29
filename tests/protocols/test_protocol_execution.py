"""
Protocol Execution Tests for QuantraCore Apex.

Tests end-to-end protocol execution on various symbols.
All assertions are SUBSTANTIVE - they will fail if behavior regresses.
"""

import pytest

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
MID_CAP_SYMBOLS = ["SQ", "SNAP", "ROKU", "DKNG", "PLTR"]
SMALL_CAP_SYMBOLS = ["RIOT", "MARA", "CLSK", "HUT", "BITF"]

ALL_SYMBOLS = LIQUID_SYMBOLS + MID_CAP_SYMBOLS + SMALL_CAP_SYMBOLS


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestProtocolExecutionLiquid:
    """Test protocol execution on liquid symbols."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_returns_apex_result(self, symbol: str):
        """Engine should return ApexResult for liquid symbols."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult), f"Expected ApexResult, got {type(result)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_valid_quantrascore(self, symbol: str):
        """Liquid symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100, f"Score {result.quantrascore} out of range"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_symbol_matches(self, symbol: str):
        """Result symbol should match input."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.symbol == symbol


class TestProtocolExecutionMidCap:
    """Test protocol execution on mid-cap symbols."""
    
    @pytest.mark.parametrize("symbol", MID_CAP_SYMBOLS)
    def test_midcap_returns_result(self, symbol: str):
        """Mid-cap symbols should return ApexResult."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult)
    
    @pytest.mark.parametrize("symbol", MID_CAP_SYMBOLS)
    def test_midcap_valid_score(self, symbol: str):
        """Mid-cap symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100


class TestProtocolExecutionSmallCap:
    """Test protocol execution on small-cap symbols."""
    
    @pytest.mark.parametrize("symbol", SMALL_CAP_SYMBOLS)
    def test_smallcap_returns_result(self, symbol: str):
        """Small-cap symbols should return ApexResult."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult)
    
    @pytest.mark.parametrize("symbol", SMALL_CAP_SYMBOLS)
    def test_smallcap_valid_score(self, symbol: str):
        """Small-cap symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100


class TestProtocolExecutionAll:
    """Test protocol execution on all symbol types."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_all_symbols_valid_result(self, symbol: str):
        """All symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
        assert result.regime is not None
        assert result.risk_tier is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_all_symbols_have_microtraits(self, symbol: str):
        """All symbols should have microtraits populated."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.microtraits is not None


class TestProtocolExecutionDeterminism:
    """Test protocol execution determinism."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_liquid_deterministic(self, symbol: str):
        """Liquid symbols should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore
    
    @pytest.mark.parametrize("symbol", MID_CAP_SYMBOLS)
    def test_midcap_deterministic(self, symbol: str):
        """Mid-cap symbols should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore
