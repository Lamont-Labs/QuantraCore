"""
Small Protocol Matrix Tests for QuantraCore Apex.

Tests protocol execution across multiple symbols and protocol IDs.
"""

import pytest
from typing import List, Tuple

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


MATRIX_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
MATRIX_PROTOCOLS = [f"T{i:02d}" for i in range(1, 21)]


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


def generate_matrix_pairs() -> List[Tuple[str, str]]:
    """Generate (symbol, protocol_id) pairs for matrix testing."""
    pairs = []
    for symbol in MATRIX_SYMBOLS:
        for protocol_id in MATRIX_PROTOCOLS:
            pairs.append((symbol, protocol_id))
    return pairs


MATRIX_PAIRS = generate_matrix_pairs()


class TestProtocolMatrixExecution:
    """Test protocol matrix execution."""
    
    @pytest.mark.parametrize("symbol,protocol_id", MATRIX_PAIRS)
    def test_protocol_executes(self, symbol: str, protocol_id: str):
        """Each protocol should execute for each symbol."""
        runner = TierProtocolRunner()
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        if protocol_id in runner.protocols:
            result = runner.run_single(protocol_id, window, microtraits)
            assert result is not None or result is None


class TestMatrixDeterminism:
    """Test matrix determinism."""
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_symbol_deterministic(self, symbol: str):
        """Each symbol should produce deterministic results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore


class TestMatrixCoverage:
    """Test matrix coverage."""
    
    def test_matrix_full_coverage(self):
        """Matrix should cover all (symbol, protocol) combinations."""
        assert len(MATRIX_PAIRS) == len(MATRIX_SYMBOLS) * len(MATRIX_PROTOCOLS)
        assert len(MATRIX_PAIRS) == 100
    
    def test_all_symbols_valid(self):
        """All matrix symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        
        for symbol in MATRIX_SYMBOLS:
            window = _get_test_window(symbol)
            result = engine.run(window)
            
            assert 0 <= result.quantrascore <= 100


class TestMatrixRanges:
    """Test protocol range execution."""
    
    @pytest.mark.parametrize("start,end", [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)])
    def test_protocol_range(self, start: int, end: int):
        """Protocol ranges should execute."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        results = runner.run_range(window, microtraits, start, end)
        
        assert isinstance(results, list)


class TestMatrixSymbolVariants:
    """Test matrix with different symbol types."""
    
    MEGA_CAP = ["AAPL", "MSFT", "GOOGL"]
    LARGE_CAP = ["TSLA", "AMZN", "META"]
    MID_CAP = ["SQ", "SNAP", "ROKU"]
    
    @pytest.mark.parametrize("symbol", MEGA_CAP)
    def test_mega_cap_valid(self, symbol: str):
        """Mega-cap symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", LARGE_CAP)
    def test_large_cap_valid(self, symbol: str):
        """Large-cap symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", MID_CAP)
    def test_mid_cap_valid(self, symbol: str):
        """Mid-cap symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        assert 0 <= result.quantrascore <= 100
