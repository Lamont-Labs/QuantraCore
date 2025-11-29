"""
Protocol Latency Tests for QuantraCore Apex.

Performance tests with SUBSTANTIVE timing assertions.
"""

import pytest
import time

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
MAX_ENGINE_TIME_SECONDS = 2.0
MAX_BATCH_TIME_SECONDS = 10.0


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestEngineLatency:
    """Test engine execution latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_single_symbol_under_2_seconds(self, symbol: str):
        """Engine should analyze single symbol in under 2 seconds."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        start = time.time()
        result = engine.run(window)
        elapsed = time.time() - start
        
        assert result is not None, "Engine returned None"
        assert elapsed < MAX_ENGINE_TIME_SECONDS, \
            f"Engine took {elapsed:.2f}s, max is {MAX_ENGINE_TIME_SECONDS}s"
    
    def test_batch_5_symbols_under_10_seconds(self):
        """Engine should handle 5 symbols in under 10 seconds."""
        engine = ApexEngine(enable_logging=False)
        
        start = time.time()
        for symbol in LIQUID_SYMBOLS:
            window = _get_test_window(symbol)
            result = engine.run(window)
            assert result is not None
        elapsed = time.time() - start
        
        assert elapsed < MAX_BATCH_TIME_SECONDS, \
            f"Batch took {elapsed:.2f}s, max is {MAX_BATCH_TIME_SECONDS}s"
    
    def test_10_runs_same_symbol(self):
        """10 runs on same symbol should average under 1 second each."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("AAPL")
        
        start = time.time()
        for _ in range(10):
            result = engine.run(window)
            assert result is not None
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        assert avg_time < 1.0, f"Average time {avg_time:.2f}s, expected < 1.0s"


class TestProtocolLatency:
    """Test protocol execution latency."""
    
    def test_all_protocols_under_5_seconds(self):
        """All protocols should execute in under 5 seconds."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        start = time.time()
        results = runner.run_all(window, microtraits)
        elapsed = time.time() - start
        
        assert len(results) > 0, "No protocol results"
        assert elapsed < 5.0, f"Protocols took {elapsed:.2f}s, max is 5.0s"


class TestMicrotraitsLatency:
    """Test microtraits computation latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_microtraits_under_half_second(self, symbol: str):
        """Microtraits should compute in under 0.5 seconds."""
        window = _get_test_window(symbol)
        
        start = time.time()
        microtraits = compute_microtraits(window)
        elapsed = time.time() - start
        
        assert microtraits is not None
        assert elapsed < 0.5, f"Microtraits took {elapsed:.2f}s"


class TestDataLayerLatency:
    """Test data layer latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_synthetic_fetch_under_100ms(self, symbol: str):
        """Synthetic data fetch should be under 100ms."""
        adapter = SyntheticAdapter(seed=42)
        
        start = time.time()
        bars = adapter.fetch(symbol, num_bars=100)
        elapsed = time.time() - start
        
        assert len(bars) > 0
        assert elapsed < 0.1, f"Fetch took {elapsed:.3f}s, expected < 0.1s"
    
    def test_batch_fetch_all_symbols(self):
        """Batch fetch should be fast."""
        adapter = SyntheticAdapter(seed=42)
        
        start = time.time()
        for symbol in LIQUID_SYMBOLS:
            bars = adapter.fetch(symbol, num_bars=100)
            assert len(bars) > 0
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Batch fetch took {elapsed:.2f}s"
