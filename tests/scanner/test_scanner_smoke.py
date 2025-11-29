"""
Scanner Smoke Tests for QuantraCore Apex.

Tests the data layer and scanner functionality.
"""

import pytest
from typing import List
from datetime import datetime, timedelta

from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
ALL_SYMBOLS = LIQUID_SYMBOLS + ["META", "NVDA", "AMD", "NFLX", "INTC"]


class TestSyntheticAdapterImport:
    """Test synthetic adapter imports."""
    
    def test_adapter_import(self):
        """SyntheticAdapter should be importable."""
        assert SyntheticAdapter is not None
    
    def test_adapter_instantiation(self):
        """SyntheticAdapter should instantiate."""
        adapter = SyntheticAdapter(seed=42)
        assert adapter is not None


class TestSyntheticAdapterFetch:
    """Test synthetic adapter data fetching."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_returns_bars(self, symbol: str):
        """Adapter should return OHLCV bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert bars is not None
        assert isinstance(bars, list)
        assert len(bars) > 0
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_sufficient_bars(self, symbol: str):
        """Adapter should return at least 50 bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert len(bars) >= 50
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_bars_have_ohlcv_fields(self, symbol: str):
        """Each bar should have OHLCV fields."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert len(bars) > 0
        bar = bars[0]
        
        assert hasattr(bar, "open")
        assert hasattr(bar, "high")
        assert hasattr(bar, "low")
        assert hasattr(bar, "close")
        assert hasattr(bar, "volume")
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_bars_have_valid_prices(self, symbol: str):
        """Bar prices should be positive."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        for bar in bars:
            assert bar.open > 0
            assert bar.high > 0
            assert bar.low > 0
            assert bar.close > 0
            assert bar.high >= bar.low
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close


class TestScannerDeterminism:
    """Test scanner determinism."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_deterministic(self, symbol: str):
        """Same seed should produce identical bars."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=42)
        
        bars1 = adapter1.fetch(symbol, num_bars=100, seed=42)
        bars2 = adapter2.fetch(symbol, num_bars=100, seed=42)
        
        assert len(bars1) == len(bars2)
        for b1, b2 in zip(bars1, bars2):
            assert b1.close == b2.close
            assert b1.volume == b2.volume
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_different_seeds_different_data(self, symbol: str):
        """Different seeds should produce different bars."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=999)
        
        bars1 = adapter1.fetch(symbol, num_bars=100, seed=42)
        bars2 = adapter2.fetch(symbol, num_bars=100, seed=999)
        
        closes1 = [b.close for b in bars1]
        closes2 = [b.close for b in bars2]
        
        assert closes1 != closes2


class TestOhlcvWindowCreation:
    """Test OhlcvWindow creation from bars."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_window_creation(self, symbol: str):
        """Should create valid OhlcvWindow from bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert window is not None
        assert window.symbol == symbol
        assert window.timeframe == "1d"
        assert len(window.bars) == len(bars)
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_window_has_bars(self, symbol: str):
        """Window should have accessible bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert hasattr(window, "bars")
        assert len(window.bars) > 0
