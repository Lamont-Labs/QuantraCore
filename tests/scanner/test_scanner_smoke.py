"""
Scanner Smoke Tests for QuantraCore Apex.

Tests the data layer and scanner functionality with SUBSTANTIVE assertions.
"""

import pytest

from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


class TestSyntheticAdapterBasics:
    """Test synthetic adapter basic functionality."""
    
    def test_adapter_instantiation(self):
        """SyntheticAdapter should instantiate."""
        adapter = SyntheticAdapter(seed=42)
        assert isinstance(adapter, SyntheticAdapter)
    
    def test_adapter_has_fetch_method(self):
        """SyntheticAdapter should have fetch method."""
        adapter = SyntheticAdapter(seed=42)
        assert hasattr(adapter, "fetch")
        assert callable(adapter.fetch)
    
    def test_adapter_is_available(self):
        """SyntheticAdapter should report as available."""
        adapter = SyntheticAdapter(seed=42)
        assert adapter.is_available()
    
    def test_adapter_name(self):
        """SyntheticAdapter should have name 'synthetic'."""
        adapter = SyntheticAdapter(seed=42)
        assert adapter.name == "synthetic"


class TestSyntheticAdapterFetch:
    """Test synthetic adapter data fetching."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_returns_list(self, symbol: str):
        """fetch should return a list."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert isinstance(bars, list)
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_returns_ohlcv_bars(self, symbol: str):
        """fetch should return OhlcvBar objects."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert len(bars) > 0
        for bar in bars:
            assert isinstance(bar, OhlcvBar), f"Expected OhlcvBar, got {type(bar)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_fetch_returns_sufficient_bars(self, symbol: str):
        """fetch should return at least 50 bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        assert len(bars) >= 50, f"Only {len(bars)} bars returned"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_bars_have_valid_ohlcv(self, symbol: str):
        """Each bar should have valid OHLCV values."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        for i, bar in enumerate(bars):
            assert bar.open > 0, f"Bar {i}: open <= 0"
            assert bar.high > 0, f"Bar {i}: high <= 0"
            assert bar.low > 0, f"Bar {i}: low <= 0"
            assert bar.close > 0, f"Bar {i}: close <= 0"
            assert bar.volume >= 0, f"Bar {i}: negative volume"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_bars_ohlc_relationship_valid(self, symbol: str):
        """High >= max(open, close) and Low <= min(open, close)."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        for i, bar in enumerate(bars):
            assert bar.high >= bar.open, f"Bar {i}: high < open"
            assert bar.high >= bar.close, f"Bar {i}: high < close"
            assert bar.low <= bar.open, f"Bar {i}: low > open"
            assert bar.low <= bar.close, f"Bar {i}: low > close"
            assert bar.high >= bar.low, f"Bar {i}: high < low"


class TestSyntheticAdapterDeterminism:
    """Test synthetic adapter determinism."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_same_seed_same_data(self, symbol: str):
        """Same seed should produce identical bars."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=42)
        
        bars1 = adapter1.fetch(symbol, num_bars=100, seed=42)
        bars2 = adapter2.fetch(symbol, num_bars=100, seed=42)
        
        assert len(bars1) == len(bars2), "Different bar counts"
        
        for i, (b1, b2) in enumerate(zip(bars1, bars2)):
            assert b1.open == b2.open, f"Bar {i}: open mismatch"
            assert b1.high == b2.high, f"Bar {i}: high mismatch"
            assert b1.low == b2.low, f"Bar {i}: low mismatch"
            assert b1.close == b2.close, f"Bar {i}: close mismatch"
            assert b1.volume == b2.volume, f"Bar {i}: volume mismatch"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_different_seeds_different_data(self, symbol: str):
        """Different seeds should produce different bars."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=999)
        
        bars1 = adapter1.fetch(symbol, num_bars=100, seed=42)
        bars2 = adapter2.fetch(symbol, num_bars=100, seed=999)
        
        closes1 = [b.close for b in bars1]
        closes2 = [b.close for b in bars2]
        
        assert closes1 != closes2, "Different seeds produced identical data"


class TestOhlcvWindowCreation:
    """Test OhlcvWindow creation from bars."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_window_creation(self, symbol: str):
        """OhlcvWindow should be creatable from bars."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert isinstance(window, OhlcvWindow)
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_window_symbol_stored(self, symbol: str):
        """Window should store symbol correctly."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert window.symbol == symbol
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_window_timeframe_stored(self, symbol: str):
        """Window should store timeframe correctly."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert window.timeframe == "1d"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_window_bars_accessible(self, symbol: str):
        """Window bars should be accessible."""
        adapter = SyntheticAdapter(seed=42)
        bars = adapter.fetch(symbol, num_bars=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)
        
        assert len(window.bars) == len(bars)
        assert window.bars[0].close == bars[0].close


class TestDifferentSymbolsProduceDifferentData:
    """Test that different symbols produce different synthetic data."""
    
    def test_symbols_have_different_prices(self):
        """Different symbols should have different price ranges."""
        adapter = SyntheticAdapter(seed=42)
        
        aapl_bars = adapter.fetch("AAPL", num_bars=100)
        msft_bars = adapter.fetch("MSFT", num_bars=100)
        tsla_bars = adapter.fetch("TSLA", num_bars=100)
        
        aapl_avg = sum(b.close for b in aapl_bars) / len(aapl_bars)
        msft_avg = sum(b.close for b in msft_bars) / len(msft_bars)
        tsla_avg = sum(b.close for b in tsla_bars) / len(tsla_bars)
        
        averages = [aapl_avg, msft_avg, tsla_avg]
        assert len(set(averages)) == 3, "Symbols have identical price averages"
