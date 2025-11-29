"""
Small Protocol Matrix Tests for QuantraCore Apex.

Tests protocol execution across multiple symbols and protocol ranges.
All assertions are substantive - they WILL fail if behavior regresses.
"""

import pytest

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, ApexResult, ProtocolResult
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.protocols.tier.tier_loader import TierProtocolRunner
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


MATRIX_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestMatrixEngineExecution:
    """Test engine execution across matrix of symbols."""
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_engine_produces_valid_result(self, symbol: str):
        """Engine should produce ApexResult with valid quantrascore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult), f"Expected ApexResult, got {type(result)}"
        assert isinstance(result.quantrascore, (int, float)), "quantrascore should be numeric"
        assert 0 <= result.quantrascore <= 100, f"quantrascore {result.quantrascore} out of range"
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_engine_result_has_all_required_fields(self, symbol: str):
        """ApexResult should have all required fields populated."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.symbol == symbol, "Symbol mismatch in result"
        assert result.regime is not None, "regime is None"
        assert result.risk_tier is not None, "risk_tier is None"
        assert result.entropy_state is not None, "entropy_state is None"
        assert result.suppression_state is not None, "suppression_state is None"
        assert result.drift_state is not None, "drift_state is None"
        assert result.verdict is not None, "verdict is None"
        assert result.microtraits is not None, "microtraits is None"


class TestMatrixProtocolExecution:
    """Test protocol execution across matrix."""
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_protocols_execute_for_symbol(self, symbol: str):
        """Protocols should execute and return results for each symbol."""
        runner = TierProtocolRunner()
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        results = runner.run_all(window, microtraits)
        
        assert len(results) >= 50, f"Only {len(results)} protocols ran for {symbol}"
        
        for r in results:
            assert isinstance(r, ProtocolResult)
            assert r.protocol_id.startswith("T"), f"Invalid protocol ID: {r.protocol_id}"
    
    @pytest.mark.parametrize("start,end", [(1, 20), (21, 40), (41, 60), (61, 80)])
    def test_protocol_range_execution(self, start: int, end: int):
        """Each protocol range should execute successfully."""
        runner = TierProtocolRunner()
        window = _get_test_window("AAPL")
        microtraits = compute_microtraits(window)
        
        results = runner.run_range(window, microtraits, start, end)
        
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r.confidence, (int, float))
            assert 0 <= r.confidence <= 1.0


class TestMatrixDeterminism:
    """Test determinism across matrix."""
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_symbol_deterministic_score(self, symbol: str):
        """Each symbol should produce identical score across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore, \
            f"Non-deterministic score for {symbol}: {result1.quantrascore} vs {result2.quantrascore}"
    
    @pytest.mark.parametrize("symbol", MATRIX_SYMBOLS)
    def test_symbol_deterministic_regime(self, symbol: str):
        """Each symbol should produce identical regime across runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.regime == result2.regime, \
            f"Non-deterministic regime for {symbol}: {result1.regime} vs {result2.regime}"


class TestMatrixCoverage:
    """Test coverage across matrix."""
    
    def test_all_symbols_produce_different_scores(self):
        """Different symbols should generally produce different scores."""
        engine = ApexEngine(enable_logging=False)
        
        scores = {}
        for symbol in MATRIX_SYMBOLS:
            window = _get_test_window(symbol)
            result = engine.run(window)
            scores[symbol] = result.quantrascore
        
        unique_scores = set(scores.values())
        assert len(unique_scores) >= 3, f"Too few unique scores: {scores}"
    
    def test_matrix_total_protocol_executions(self):
        """Matrix should execute significant number of protocols."""
        runner = TierProtocolRunner()
        total_executions = 0
        
        for symbol in MATRIX_SYMBOLS:
            window = _get_test_window(symbol)
            microtraits = compute_microtraits(window)
            results = runner.run_all(window, microtraits)
            total_executions += len(results)
        
        expected = len(MATRIX_SYMBOLS) * 50
        assert total_executions >= expected, \
            f"Only {total_executions} total executions, expected >= {expected}"


class TestMatrixSymbolCategories:
    """Test matrix with different symbol categories."""
    
    MEGA_CAP = ["AAPL", "MSFT", "GOOGL"]
    VOLATILE = ["TSLA", "GME", "AMC"]
    
    @pytest.mark.parametrize("symbol", MEGA_CAP)
    def test_mega_cap_valid_result(self, symbol: str):
        """Mega-cap symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100
        assert result.regime is not None
        assert result.microtraits is not None
    
    @pytest.mark.parametrize("symbol", VOLATILE)
    def test_volatile_valid_result(self, symbol: str):
        """Volatile symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert 0 <= result.quantrascore <= 100
        assert result.risk_tier is not None
