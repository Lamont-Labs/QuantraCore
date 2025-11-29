"""
Protocol Latency Tests for QuantraCore Apex.

Fast performance tests to ensure engine runs within acceptable time limits.
"""

import pytest
import time
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
MAX_ENGINE_TIME_SECONDS = 2.0
MAX_PROTOCOL_TIME_SECONDS = 0.5
MAX_BATCH_TIME_SECONDS = 5.0


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestEngineLatency:
    """Test engine execution latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_single_symbol_fast(self, symbol: str):
        """Engine should complete single symbol analysis quickly."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        start = time.time()
        result = engine.run(window)
        elapsed = time.time() - start
        
        assert result is not None
        assert elapsed < MAX_ENGINE_TIME_SECONDS, f"Engine took {elapsed:.2f}s, max is {MAX_ENGINE_TIME_SECONDS}s"
    
    def test_engine_batch_5_symbols_fast(self):
        """Engine should handle 5 symbols quickly."""
        engine = ApexEngine(enable_logging=False)
        
        start = time.time()
        for symbol in LIQUID_SYMBOLS:
            window = _get_test_window(symbol)
            result = engine.run(window)
            assert result is not None
        elapsed = time.time() - start
        
        assert elapsed < MAX_BATCH_TIME_SECONDS, f"Batch took {elapsed:.2f}s, max is {MAX_BATCH_TIME_SECONDS}s"
    
    def test_engine_10_runs_same_symbol_fast(self):
        """Engine should handle 10 runs on same symbol quickly."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("AAPL")
        
        start = time.time()
        for _ in range(10):
            result = engine.run(window)
            assert result is not None
        elapsed = time.time() - start
        
        per_run = elapsed / 10
        assert per_run < MAX_ENGINE_TIME_SECONDS, f"Per-run: {per_run:.2f}s"


class TestProtocolLatency:
    """Test individual protocol latency."""
    
    @pytest.mark.parametrize("protocol_id", [f"T{i:02d}" for i in range(1, 11)])
    def test_single_protocol_fast(self, protocol_id: str):
        """Single protocol should execute quickly."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        if protocol_id in runner.protocols:
            start = time.time()
            result = runner.run_single(protocol_id, window, microtraits)
            elapsed = time.time() - start
            
            assert elapsed < MAX_PROTOCOL_TIME_SECONDS
    
    def test_all_protocols_batch_fast(self):
        """All protocols should execute quickly."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        start = time.time()
        results = runner.run_all(window, microtraits)
        elapsed = time.time() - start
        
        assert results is not None
        assert elapsed < MAX_BATCH_TIME_SECONDS


class TestMicrotraitsLatency:
    """Test microtraits computation latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_microtraits_fast(self, symbol: str):
        """Microtraits should compute quickly."""
        window = _get_test_window(symbol)
        
        start = time.time()
        microtraits = compute_microtraits(window)
        elapsed = time.time() - start
        
        assert microtraits is not None
        assert elapsed < 0.5, f"Microtraits took {elapsed:.2f}s"


class TestDataLayerLatency:
    """Test data layer latency."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_synthetic_fetch_fast(self, symbol: str):
        """Synthetic data fetch should be fast."""
        adapter = SyntheticAdapter(seed=42)
        
        start = time.time()
        bars = adapter.fetch(symbol, num_bars=100)
        elapsed = time.time() - start
        
        assert len(bars) > 0
        assert elapsed < 0.5, f"Fetch took {elapsed:.2f}s"
    
    def test_synthetic_fetch_batch_fast(self):
        """Batch fetch should be fast."""
        adapter = SyntheticAdapter(seed=42)
        
        start = time.time()
        for symbol in LIQUID_SYMBOLS:
            bars = adapter.fetch(symbol, num_bars=100)
            assert len(bars) > 0
        elapsed = time.time() - start
        
        assert elapsed < 2.0, f"Batch fetch took {elapsed:.2f}s"
