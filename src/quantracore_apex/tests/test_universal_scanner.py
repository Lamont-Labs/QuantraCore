"""
Tests for Universal Scanner Upgrade - v9.0-A

Tests symbol universe loader, scan modes, data client failover,
universe scanner, and MonsterRunner fuse score.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


class TestSymbolUniverseLoader:
    """Tests for config/symbol_universe.yaml loader."""
    
    def test_load_symbol_universe(self):
        """Test that symbol universe config loads correctly."""
        from src.quantracore_apex.config.symbol_universe import load_symbol_universe
        
        config = load_symbol_universe()
        
        assert config.version == "9.0-A"
        assert len(config.symbols) > 0
        assert len(config.universes) > 0
    
    def test_get_all_symbols(self):
        """Test getting all symbols."""
        from src.quantracore_apex.config.symbol_universe import get_all_symbols
        
        symbols = get_all_symbols()
        
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
    
    def test_get_symbols_by_bucket(self):
        """Test filtering by market cap bucket."""
        from src.quantracore_apex.config.symbol_universe import get_symbols_by_bucket
        
        mega_symbols = get_symbols_by_bucket(["mega"])
        small_symbols = get_symbols_by_bucket(["small", "micro", "nano"])
        
        assert isinstance(mega_symbols, list)
        assert isinstance(small_symbols, list)
    
    def test_get_symbols_for_mode(self):
        """Test getting symbols for scan modes."""
        from src.quantracore_apex.config.symbol_universe import get_symbols_for_mode
        
        demo_symbols = get_symbols_for_mode("demo")
        full_symbols = get_symbols_for_mode("full_us_equities")
        
        assert len(demo_symbols) > 0
        assert len(demo_symbols) <= 20
    
    def test_get_symbol_info(self):
        """Test getting detailed symbol info."""
        from src.quantracore_apex.config.symbol_universe import get_symbol_info
        
        info = get_symbol_info("AAPL")
        
        if info:
            assert info.symbol == "AAPL"
            assert info.market_cap_bucket in ["mega", "large", "mid", "small", "micro", "nano", "penny", "unknown"]
            assert isinstance(info.is_smallcap, bool)
            assert info.risk_category in ["low", "medium", "high", "extreme"]
    
    def test_is_smallcap(self):
        """Test smallcap detection."""
        from src.quantracore_apex.config.symbol_universe import is_smallcap, get_symbol_info
        
        result = is_smallcap("AAPL")
        assert isinstance(result, bool)
    
    def test_all_buckets_recognized(self):
        """Test that all market cap buckets are recognized."""
        from src.quantracore_apex.config.symbol_universe import ALL_BUCKETS, SMALLCAP_BUCKETS
        
        assert "mega" in ALL_BUCKETS
        assert "large" in ALL_BUCKETS
        assert "mid" in ALL_BUCKETS
        assert "small" in ALL_BUCKETS
        assert "micro" in ALL_BUCKETS
        assert "nano" in ALL_BUCKETS
        assert "penny" in ALL_BUCKETS
        
        assert "small" in SMALLCAP_BUCKETS
        assert "micro" in SMALLCAP_BUCKETS
        assert "nano" in SMALLCAP_BUCKETS
        assert "penny" in SMALLCAP_BUCKETS


class TestScanModesConfig:
    """Tests for config/scan_modes.yaml loader."""
    
    def test_load_scan_mode(self):
        """Test loading a scan mode."""
        from src.quantracore_apex.config.scan_modes import load_scan_mode
        
        config = load_scan_mode("demo")
        
        assert config.name == "demo"
        assert len(config.buckets) > 0
        assert config.max_symbols > 0
        assert config.chunk_size > 0
    
    def test_list_scan_modes(self):
        """Test listing all scan modes."""
        from src.quantracore_apex.config.scan_modes import list_scan_modes
        
        modes = list_scan_modes()
        
        assert len(modes) > 0
        assert "demo" in modes
        assert "mega_large_focus" in modes
        assert "high_vol_small_caps" in modes
        assert "low_float_runners" in modes
    
    def test_mode_configs_valid(self):
        """Test that all mode configs are valid."""
        from src.quantracore_apex.config.scan_modes import list_scan_modes, load_scan_mode
        
        for mode_name in list_scan_modes():
            config = load_scan_mode(mode_name)
            
            assert config.name == mode_name
            assert isinstance(config.buckets, list)
            assert config.max_symbols > 0
            assert config.chunk_size > 0
            assert config.batch_delay_seconds >= 0
    
    def test_smallcap_mode_detection(self):
        """Test detection of smallcap-focused modes."""
        from src.quantracore_apex.config.scan_modes import load_scan_mode
        
        smallcap_mode = load_scan_mode("high_vol_small_caps")
        mega_mode = load_scan_mode("mega_large_focus")
        
        assert smallcap_mode.is_smallcap_focused == True
        assert mega_mode.is_smallcap_focused == False
    
    def test_extreme_risk_detection(self):
        """Test detection of extreme risk modes."""
        from src.quantracore_apex.config.scan_modes import load_scan_mode
        
        runner_mode = load_scan_mode("low_float_runners")
        
        assert runner_mode.is_extreme_risk == True
    
    def test_performance_config(self):
        """Test K6 performance configuration."""
        from src.quantracore_apex.config.scan_modes import get_performance_config
        
        config = get_performance_config()
        
        assert config.max_concurrent_requests > 0
        assert config.memory_limit_mb > 0
        assert config.cpu_throttle_percent > 0
        assert config.disk_cache_max_gb > 0
    
    def test_filter_config(self):
        """Test filter configuration parsing."""
        from src.quantracore_apex.config.scan_modes import load_scan_mode
        
        config = load_scan_mode("high_vol_small_caps")
        
        assert config.filters is not None
        assert config.filters.min_price is not None
        assert config.filters.max_price is not None


class TestMonsterRunnerFuseScore:
    """Tests for MonsterRunner fuse score calculation."""
    
    @pytest.fixture
    def sample_bars(self):
        """Generate sample OHLCV bars."""
        from src.quantracore_apex.core.schemas import OhlcvBar
        
        bars = []
        base_price = 100.0
        np.random.seed(42)
        
        for i in range(100):
            noise = np.random.randn() * 2
            close = base_price + noise + (i * 0.1)
            bars.append(OhlcvBar(
                timestamp=datetime.now() - timedelta(days=100-i),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000000 + np.random.randint(-100000, 100000)
            ))
        
        return bars
    
    def test_calculate_mr_fuse_score(self, sample_bars):
        """Test fuse score calculation."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import calculate_mr_fuse_score
        
        result = calculate_mr_fuse_score(sample_bars)
        
        assert 0 <= result.fuse_score <= 100
        assert result.runner_tier in ["none", "watching", "warming", "primed"]
        assert isinstance(result.runner_candidate, bool)
        assert isinstance(result.risk_flags, list)
    
    def test_volatility_expansion(self, sample_bars):
        """Test volatility expansion calculation."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import calculate_volatility_expansion
        
        vol_exp = calculate_volatility_expansion(sample_bars)
        
        assert vol_exp > 0
        assert isinstance(vol_exp, float)
    
    def test_liquidity_score(self, sample_bars):
        """Test liquidity score calculation."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import calculate_liquidity_score
        
        liq_score = calculate_liquidity_score(sample_bars)
        
        assert 0 <= liq_score <= 100
    
    def test_float_pressure(self):
        """Test float pressure calculation."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import calculate_float_pressure
        
        low_float = calculate_float_pressure(5.0, 2000000, 10.0)
        high_float = calculate_float_pressure(500.0, 2000000, 10.0)
        
        assert low_float > high_float
        assert 0 <= low_float <= 100
        assert 0 <= high_float <= 100
    
    def test_trend_strength(self, sample_bars):
        """Test trend strength calculation."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import calculate_trend_strength
        
        strength = calculate_trend_strength(sample_bars)
        
        assert -100 <= strength <= 100
    
    def test_quick_fuse_check(self, sample_bars):
        """Test quick fuse check function."""
        from src.quantracore_apex.protocols.monster_runner.fuse_score import quick_fuse_check
        
        score = quick_fuse_check(sample_bars)
        
        assert 0 <= score <= 100


class TestDataClient:
    """Tests for multi-provider data client."""
    
    def test_create_data_client(self):
        """Test data client creation."""
        from src.quantracore_apex.data_layer.client import create_data_client
        
        client = create_data_client(use_cache=True)
        
        assert client is not None
        assert client.use_cache == True
    
    def test_get_available_providers(self):
        """Test listing available providers."""
        from src.quantracore_apex.data_layer.client import create_data_client
        
        client = create_data_client()
        providers = client.get_available_providers()
        
        assert isinstance(providers, list)
    
    def test_health_check(self):
        """Test client health check."""
        from src.quantracore_apex.data_layer.client import create_data_client
        
        client = create_data_client()
        health = client.health_check()
        
        assert "cache_enabled" in health
        assert "cache_dir" in health


class TestUniverseScannerSmallcapsDemo:
    """Tests for universe scanner with small-cap demo."""
    
    def test_universe_scanner_creation(self):
        """Test scanner creation."""
        from src.quantracore_apex.core.universe_scan import create_universe_scanner
        
        scanner = create_universe_scanner()
        
        assert scanner is not None
    
    def test_scan_result_dataclass(self):
        """Test scan result structure."""
        from src.quantracore_apex.core.universe_scan import ScanResult
        
        result = ScanResult(
            symbol="TEST",
            quantrascore=50.0,
            market_cap_bucket="small",
            smallcap_flag=True,
            mr_fuse_score=45.0,
            speculative_flag=True,
        )
        
        assert result.symbol == "TEST"
        assert result.smallcap_flag == True
        assert result.speculative_flag == True
        assert result.mr_fuse_score == 45.0
    
    def test_universe_scan_result_dataclass(self):
        """Test universe scan result structure."""
        from src.quantracore_apex.core.universe_scan import UniverseScanResult
        
        result = UniverseScanResult(
            mode="demo",
            scan_count=10,
            success_count=8,
            error_count=2,
            smallcap_count=3,
            extreme_risk_count=1,
            runner_candidate_count=2,
        )
        
        assert result.mode == "demo"
        assert result.scan_count == 10
        assert result.smallcap_count == 3


class TestCSVBundleAdapter:
    """Tests for CSV bundle adapter."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        try:
            from src.quantracore_apex.data_layer.adapters.csv_bundle_adapter import CSVBundleAdapter
            
            adapter = CSVBundleAdapter()
            
            assert adapter.name == "csv_bundle"
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_is_available(self):
        """Test availability check."""
        try:
            from src.quantracore_apex.data_layer.adapters.csv_bundle_adapter import CSVBundleAdapter
            
            adapter = CSVBundleAdapter()
            
            assert isinstance(adapter.is_available(), bool)
        except ImportError:
            pytest.skip("pandas not available")


class TestIntegration:
    """Integration tests for universal scanner."""
    
    def test_config_to_scanner_flow(self):
        """Test flow from config to scanner."""
        from src.quantracore_apex.config.scan_modes import load_scan_mode
        from src.quantracore_apex.config.symbol_universe import get_symbols_for_mode
        
        mode_config = load_scan_mode("demo")
        symbols = get_symbols_for_mode("demo")
        
        assert mode_config.name == "demo"
        assert len(symbols) > 0
        assert len(symbols) <= mode_config.max_symbols
    
    def test_smallcap_identification_consistency(self):
        """Test that smallcap flags are consistent."""
        from src.quantracore_apex.config.symbol_universe import (
            get_symbol_info,
            is_smallcap,
            SMALLCAP_BUCKETS,
        )
        
        info = get_symbol_info("AAPL")
        if info:
            is_small_from_func = is_smallcap("AAPL")
            is_small_from_bucket = info.market_cap_bucket in SMALLCAP_BUCKETS
            
            assert is_small_from_func == is_small_from_bucket
