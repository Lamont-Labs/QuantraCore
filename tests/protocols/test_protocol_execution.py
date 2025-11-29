"""
Protocol Execution Tests for QuantraCore Apex.

Tests end-to-end protocol execution on various symbols.
"""

import pytest
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD"]
MID_CAP_SYMBOLS = ["SQ", "SNAP", "ROKU", "DKNG", "PLTR", "HOOD", "SOFI", "COIN"]
SMALL_CAP_SYMBOLS = ["RIOT", "MARA", "CLSK", "HUT", "BITF", "CIFR", "BTBT", "EBON"]

ALL_SYMBOLS = LIQUID_SYMBOLS + MID_CAP_SYMBOLS + SMALL_CAP_SYMBOLS


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestProtocolExecutionLiquid:
    """Test protocol execution on liquid symbols."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_executes_protocols(self, symbol: str):
        """Engine should execute protocols and produce valid result."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_protocol_trace_non_empty(self, symbol: str):
        """Protocol trace should be populated."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert hasattr(result, "quantrascore")


class TestProtocolExecutionMidCap:
    """Test protocol execution on mid-cap symbols."""
    
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
    def test_smallcap_valid_score(self, symbol: str):
        """Small-cap symbols should produce valid QuantraScore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100


class TestProtocolExecutionAll:
    """Test protocol execution on all symbol types."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_all_symbols_valid(self, symbol: str):
        """All symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
        assert result.regime is not None
        assert result.risk_tier is not None
