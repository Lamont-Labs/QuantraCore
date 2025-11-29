"""
Core Engine Smoke Tests for QuantraCore Apex.

Tests the main ApexEngine functionality with SUBSTANTIVE assertions.
Every test here will FAIL if behavior regresses.
"""

import pytest

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import (
    OhlcvWindow, ApexContext, ApexResult,
    RegimeType, RiskTier, EntropyState
)
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
VOLATILE_SYMBOLS = ["GME", "AMC", "RIVN", "LCID", "SPCE"]


def _get_test_window(symbol: str, num_bars: int = 100, seed: int = 42) -> OhlcvWindow:
    """Helper to create test window using synthetic adapter."""
    adapter = SyntheticAdapter(seed=seed)
    bars = adapter.fetch(symbol, num_bars=num_bars, seed=seed)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestEngineImport:
    """Test engine module imports correctly."""
    
    def test_engine_class_exists(self):
        """ApexEngine class should exist and be callable."""
        assert ApexEngine is not None
        assert callable(ApexEngine)
    
    def test_engine_instantiation_creates_object(self):
        """ApexEngine() should return an ApexEngine instance."""
        engine = ApexEngine(enable_logging=False)
        assert isinstance(engine, ApexEngine)
    
    def test_engine_has_run_method(self):
        """ApexEngine should have callable run method."""
        engine = ApexEngine(enable_logging=False)
        assert hasattr(engine, "run")
        assert callable(engine.run)
    
    def test_engine_has_run_scan_method(self):
        """ApexEngine should have callable run_scan method."""
        engine = ApexEngine(enable_logging=False)
        assert hasattr(engine, "run_scan")
        assert callable(engine.run_scan)


class TestEngineSingleSymbol:
    """Test engine on single symbols with real assertions."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_returns_apex_result(self, symbol: str):
        """Engine should return ApexResult instance."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult), f"Expected ApexResult, got {type(result)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_quantrascore_in_valid_range(self, symbol: str):
        """QuantraScore must be in 0-100 range."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.quantrascore, (int, float)), "quantrascore should be numeric"
        assert 0 <= result.quantrascore <= 100, f"quantrascore {result.quantrascore} out of [0,100]"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_symbol_matches_input(self, symbol: str):
        """Result symbol should match input symbol."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.symbol == symbol, f"Expected {symbol}, got {result.symbol}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_has_regime_enum(self, symbol: str):
        """Result regime should be a RegimeType enum."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.regime, RegimeType), f"Expected RegimeType, got {type(result.regime)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_has_risk_tier_enum(self, symbol: str):
        """Result risk_tier should be a RiskTier enum."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.risk_tier, RiskTier), f"Expected RiskTier, got {type(result.risk_tier)}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_has_entropy_state_enum(self, symbol: str):
        """Result entropy_state should be an EntropyState enum."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.entropy_state, EntropyState)
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_result_has_microtraits(self, symbol: str):
        """Result should have non-null microtraits."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.microtraits is not None, "microtraits is None"


class TestEngineDeterminism:
    """Test engine determinism with substantive checks."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_identical_scores_across_runs(self, symbol: str):
        """Same input should produce identical quantrascore."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore, \
            f"Non-deterministic: {result1.quantrascore} vs {result2.quantrascore}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_identical_regime_across_runs(self, symbol: str):
        """Same input should produce identical regime."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.regime == result2.regime, \
            f"Non-deterministic regime: {result1.regime} vs {result2.regime}"
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_identical_risk_tier_across_runs(self, symbol: str):
        """Same input should produce identical risk_tier."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.risk_tier == result2.risk_tier
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS + VOLATILE_SYMBOLS)
    def test_determinism_across_10_runs(self, symbol: str):
        """Engine should produce identical results across 10 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        first_result = engine.run(window)
        
        for i in range(9):
            result = engine.run(window)
            assert result.quantrascore == first_result.quantrascore, \
                f"Run {i+2} differs: {result.quantrascore} vs {first_result.quantrascore}"
    
    def test_different_engines_same_result(self):
        """Different engine instances should produce same result."""
        engine1 = ApexEngine(enable_logging=False)
        engine2 = ApexEngine(enable_logging=False)
        
        window = _get_test_window("AAPL")
        
        result1 = engine1.run(window)
        result2 = engine2.run(window)
        
        assert result1.quantrascore == result2.quantrascore
        assert result1.regime == result2.regime


class TestEngineWithContext:
    """Test engine with ApexContext."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS[:3])
    def test_engine_accepts_context(self, symbol: str):
        """Engine should accept ApexContext without error."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(seed=42, compliance_mode=True)
        
        result = engine.run(window, context)
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS[:3])
    def test_context_determinism(self, symbol: str):
        """Same context should produce same results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context1 = ApexContext(seed=42, compliance_mode=True)
        context2 = ApexContext(seed=42, compliance_mode=True)
        
        result1 = engine.run(window, context1)
        result2 = engine.run(window, context2)
        
        assert result1.quantrascore == result2.quantrascore


class TestEngineVolatileSymbols:
    """Test engine on volatile symbols."""
    
    @pytest.mark.parametrize("symbol", VOLATILE_SYMBOLS)
    def test_volatile_produces_valid_result(self, symbol: str):
        """Volatile symbols should produce valid results."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
        assert result.regime is not None
        assert result.risk_tier is not None


class TestEngineWindowHash:
    """Test window hashing functionality."""
    
    def test_result_has_window_hash(self):
        """Result should have non-empty window_hash."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("AAPL")
        
        result = engine.run(window)
        
        assert result.window_hash is not None
        assert len(result.window_hash) > 0
    
    def test_same_input_same_hash(self):
        """Same input should produce same window_hash."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window("AAPL")
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.window_hash == result2.window_hash
