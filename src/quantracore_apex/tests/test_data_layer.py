"""
Data Layer Tests for QuantraCore Apex.

Tests caching, hashing, and normalization.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta

from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv, build_windows
from src.quantracore_apex.data_layer.caching import OhlcvCache
from src.quantracore_apex.data_layer.hashing import compute_data_hash, verify_hash


class TestSyntheticAdapter:
    """Test synthetic data adapter."""
    
    def test_fetch_returns_bars(self):
        """Test adapter returns OHLCV bars."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=100)
        
        bars = adapter.fetch_ohlcv("TEST", start_date, end_date, "1d")
        
        assert len(bars) > 0
    
    def test_bars_have_valid_structure(self):
        """Test bars have valid OHLC relationships."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=50)
        
        bars = adapter.fetch_ohlcv("VALID", start_date, end_date, "1d")
        
        for bar in bars:
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.volume > 0
    
    def test_deterministic_output(self):
        """Test adapter produces deterministic output."""
        adapter1 = SyntheticAdapter(seed=42)
        adapter2 = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=50)
        
        bars1 = adapter1.fetch_ohlcv("DET", start_date, end_date, "1d")
        bars2 = adapter2.fetch_ohlcv("DET", start_date, end_date, "1d")
        
        assert len(bars1) == len(bars2)
        for b1, b2 in zip(bars1, bars2):
            assert b1.close == b2.close


class TestNormalization:
    """Test data normalization."""
    
    def test_normalize_removes_zero_volume(self):
        """Test normalization removes zero volume bars."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=50)
        
        bars = adapter.fetch_ohlcv("NORM", start_date, end_date, "1d")
        normalized, _ = normalize_ohlcv(bars, remove_invalid=True)
        
        for bar in normalized:
            assert bar.volume > 0
    
    def test_build_windows_correct_size(self):
        """Test window builder creates correct size windows."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("WIN", start_date, end_date, "1d")
        normalized, _ = normalize_ohlcv(bars)
        
        windows = build_windows(normalized, "WIN", "1d", window_size=100, step=10)
        
        for window in windows:
            assert len(window.bars) == 100


class TestCaching:
    """Test data caching."""
    
    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        cache = OhlcvCache(
            cache_dir=f"{temp_dir}/cache",
            historical_dir=f"{temp_dir}/historical"
        )
        yield cache
        shutil.rmtree(temp_dir)
    
    def test_cache_put_and_get(self, temp_cache):
        """Test cache put and get operations."""
        adapter = SyntheticAdapter(seed=42)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=50)
        
        bars = adapter.fetch_ohlcv("CACHE", start_date, end_date, "1d")
        
        temp_cache.put("CACHE", start_date, end_date, "1d", bars)
        
        cached_bars = temp_cache.get("CACHE", start_date, end_date, "1d")
        
        assert cached_bars is not None
        assert len(cached_bars) == len(bars)
    
    def test_cache_miss_returns_none(self, temp_cache):
        """Test cache miss returns None."""
        result = temp_cache.get(
            "NONEXISTENT",
            datetime(2024, 1, 1),
            datetime(2024, 2, 1),
            "1d"
        )
        assert result is None


class TestHashing:
    """Test data hashing."""
    
    def test_hash_deterministic(self):
        """Test hash is deterministic."""
        data = "test data for hashing"
        
        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)
        
        assert hash1 == hash2
    
    def test_verify_hash_correct(self):
        """Test hash verification works."""
        data = "verification test"
        data_hash = compute_data_hash(data)
        
        assert verify_hash(data, data_hash)
    
    def test_verify_hash_wrong(self):
        """Test hash verification fails for wrong hash."""
        data = "original data"
        
        assert not verify_hash(data, "wrong_hash")
