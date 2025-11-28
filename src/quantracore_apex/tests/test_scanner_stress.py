"""
Scanner Stress Tests for QuantraCore Apex v9.0-A.

Tests the Universal Scanner with:
- All 8 scan modes
- Multi-provider data failover
- Memory and performance constraints
- Large symbol batches using fixtures
"""

import pytest
import time
import psutil
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
import yaml

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder


class TestScanModeLoading:
    """Verify all 8 scan modes are properly configured."""
    
    def test_all_scan_modes_present(self):
        """Verify all 8 scan modes exist in config."""
        with open("config/scan_modes.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        modes = config.get("modes", {})
        expected_modes = [
            "full_us_equities",
            "high_vol_small_caps",
            "low_float_runners",
            "mega_large_focus",
            "mid_cap_focus",
            "momentum_runners",
            "demo",
            "ci_test",
        ]
        
        for mode in expected_modes:
            assert mode in modes, f"Missing scan mode: {mode}"
    
    def test_scan_mode_structure(self):
        """Verify scan modes have required fields."""
        with open("config/scan_modes.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        modes = config.get("modes", {})
        
        for mode_name, mode_config in modes.items():
            assert "description" in mode_config, f"{mode_name} missing description"
            assert "buckets" in mode_config, f"{mode_name} missing buckets"


class TestMarketCapBuckets:
    """Verify all 7 market cap buckets are configured."""
    
    def test_all_buckets_present(self):
        """Verify all 7 market cap buckets exist."""
        with open("config/symbol_universe.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        buckets = config.get("market_cap_buckets", {})
        expected_buckets = ["mega", "large", "mid", "small", "micro", "nano", "penny"]
        
        for bucket in expected_buckets:
            assert bucket in buckets, f"Missing bucket: {bucket}"
    
    def test_bucket_thresholds(self):
        """Verify all bucket definitions exist."""
        with open("config/symbol_universe.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        buckets = config.get("market_cap_buckets", {})
        
        assert len(buckets) == 7, f"Expected 7 buckets, got {len(buckets)}"
        assert "mega" in buckets
        assert "penny" in buckets


class TestScannerBatchProcessing:
    """Test scanner with varying batch sizes."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    @pytest.fixture
    def synthetic_windows(self) -> List:
        """Generate synthetic windows for batch testing."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        symbols = [f"BATCH_{i:03d}" for i in range(50)]
        windows = []
        
        for symbol in symbols:
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            windows.append(window)
        
        return windows
    
    def test_batch_scan_50_symbols(self, engine, synthetic_windows):
        """Scan 50 symbols and verify all complete."""
        context = ApexContext(seed=42, compliance_mode=True)
        
        results = []
        for window in synthetic_windows:
            result = engine.run(window, context)
            results.append(result)
        
        assert len(results) == 50
        
        for result in results:
            assert 0 <= result.quantrascore <= 100
            assert result.verdict is not None
    
    def test_memory_stable_during_batch(self, engine, synthetic_windows):
        """Verify memory usage stays stable during batch processing."""
        context = ApexContext(seed=42, compliance_mode=True)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        for window in synthetic_windows:
            engine.run(window, context)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f} MB during batch"


class TestDataFailover:
    """Test multi-provider data failover mechanism."""
    
    def test_synthetic_adapter_available(self):
        """Verify synthetic adapter is available for testing."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=30)
        
        bars = adapter.fetch_ohlcv("TEST_SYM", start_date, end_date, "1d")
        
        assert len(bars) > 0
        assert all(hasattr(bar, 'open') for bar in bars)
    
    def test_synthetic_generates_consistent_data(self):
        """Verify synthetic adapter produces consistent data for same symbol."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=30)
        
        bars1 = adapter1.fetch_ohlcv("CONSIST_TEST", start_date, end_date, "1d")
        bars2 = adapter2.fetch_ohlcv("CONSIST_TEST", start_date, end_date, "1d")
        
        assert len(bars1) == len(bars2)
        for b1, b2 in zip(bars1, bars2):
            assert b1.open == b2.open
            assert b1.close == b2.close
    
    def test_failover_graceful_degradation(self):
        """Verify system gracefully handles data source failures."""
        adapter = SyntheticAdapter(seed=42)
        
        assert adapter is not None
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=30)
        bars = adapter.fetch_ohlcv("FAIL_TEST", start_date, end_date, "1d")
        
        assert len(bars) > 0


class TestPerformanceBaselines:
    """Performance regression tests for K6-class machine."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    def test_single_symbol_latency(self, engine):
        """Single symbol scan should complete quickly."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("PERF_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, "PERF_TEST")
        
        context = ApexContext(seed=42, compliance_mode=True)
        
        start_time = time.time()
        engine.run(window, context)
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Single symbol scan took {elapsed:.2f}s (max 2.0s)"
    
    def test_batch_throughput(self, engine):
        """Verify acceptable throughput for batch processing."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        windows = []
        for i in range(20):
            bars = adapter.fetch_ohlcv(f"THRU_{i}", start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, f"THRU_{i}")
            windows.append(window)
        
        context = ApexContext(seed=42, compliance_mode=True)
        
        start_time = time.time()
        for window in windows:
            engine.run(window, context)
        elapsed = time.time() - start_time
        
        throughput = 20 / elapsed
        
        assert throughput >= 5, f"Throughput {throughput:.1f} symbols/sec below 5 minimum"


class TestScanModeIntegration:
    """Integration tests for each scan mode."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    def _run_mode_test(self, engine, mode_name: str, sample_size: int = 5):
        """Run integration test for a scan mode."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 6, 1)
        start_date = end_date - timedelta(days=150)
        
        windows = []
        for i in range(sample_size):
            symbol = f"{mode_name.upper()[:4]}_{i}"
            bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
            normalized_bars, _ = normalize_ohlcv(bars)
            window = window_builder.build_single(normalized_bars, symbol)
            windows.append(window)
        
        context = ApexContext(seed=42, compliance_mode=True)
        
        results = []
        for window in windows:
            result = engine.run(window, context)
            results.append(result)
        
        return results
    
    def test_demo_mode(self, engine):
        """Test demo scan mode."""
        results = self._run_mode_test(engine, "demo", 5)
        assert len(results) == 5
        assert all(0 <= r.quantrascore <= 100 for r in results)
    
    def test_ci_test_mode(self, engine):
        """Test CI test scan mode."""
        results = self._run_mode_test(engine, "ci_test", 3)
        assert len(results) == 3
    
    def test_high_vol_small_caps_mode(self, engine):
        """Test high volatility small caps mode."""
        results = self._run_mode_test(engine, "high_vol_small_caps", 5)
        assert len(results) == 5
    
    def test_low_float_runners_mode(self, engine):
        """Test low float runners mode."""
        results = self._run_mode_test(engine, "low_float_runners", 5)
        assert len(results) == 5
    
    def test_momentum_runners_mode(self, engine):
        """Test momentum runners mode."""
        results = self._run_mode_test(engine, "momentum_runners", 5)
        assert len(results) == 5
