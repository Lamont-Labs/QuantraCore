"""
Protocol Signature Tests for QuantraCore Apex.

Tests protocol firing patterns for known inputs.
"""

import pytest
from datetime import datetime, timedelta

from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.protocols.tier import TierProtocolRunner
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder


class TestProtocolSignatures:
    """Test protocol firing signatures."""
    
    @pytest.fixture
    def protocol_runner(self):
        return TierProtocolRunner()
    
    @pytest.fixture
    def sample_window(self):
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("PROTO_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        return window_builder.build_single(normalized_bars, "PROTO_TEST")
    
    def test_t01_returns_result(self, protocol_runner, sample_window):
        """Test T01 protocol returns valid result."""
        microtraits = compute_microtraits(sample_window)
        result = protocol_runner.run_single("T01", sample_window, microtraits)
        
        assert result is not None
        assert result.protocol_id == "T01"
        assert 0 <= result.confidence <= 1
    
    def test_t01_to_t20_all_return_results(self, protocol_runner, sample_window):
        """Test T01-T20 all return valid results."""
        microtraits = compute_microtraits(sample_window)
        results = protocol_runner.run_range(sample_window, microtraits, start=1, end=20)
        
        assert len(results) == 20
        
        for result in results:
            assert result.protocol_id.startswith("T")
            assert 0 <= result.confidence <= 1
    
    def test_protocol_result_has_details(self, protocol_runner, sample_window):
        """Test protocol results include details."""
        microtraits = compute_microtraits(sample_window)
        result = protocol_runner.run_single("T01", sample_window, microtraits)
        
        assert result.details is not None
        assert isinstance(result.details, dict)
    
    def test_stub_protocols_return_stub_status(self, protocol_runner, sample_window):
        """Test stub protocols (T21+) return stub status."""
        microtraits = compute_microtraits(sample_window)
        result = protocol_runner.run_single("T30", sample_window, microtraits)
        
        if result:
            assert result.fired == False
            assert "stub" in str(result.details.get("status", "")).lower() or \
                   "stub" in str(result.details.get("message", "")).lower()


class TestProtocolDeterminism:
    """Test protocol determinism."""
    
    def test_same_input_same_output(self):
        """Verify protocols are deterministic."""
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        runner = TierProtocolRunner()
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("DET_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, "DET_TEST")
        
        microtraits = compute_microtraits(window)
        
        result1 = runner.run_single("T01", window, microtraits)
        result2 = runner.run_single("T01", window, microtraits)
        
        assert result1.fired == result2.fired
        assert result1.confidence == result2.confidence
        assert result1.signal_type == result2.signal_type
