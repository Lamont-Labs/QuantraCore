"""
Core Engine Smoke Tests for QuantraCore Apex.

Tests the main ApexEngine functionality including:
- Import verification
- Single symbol analysis
- Determinism validation
"""

import pytest
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar, ApexContext, ApexResult
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


LIQUID_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
VOLATILE_SYMBOLS = ["GME", "AMC", "BBBY", "RIVN", "LCID"]
ALL_TEST_SYMBOLS = LIQUID_SYMBOLS + VOLATILE_SYMBOLS


def _get_test_window(symbol: str, num_bars: int = 100) -> OhlcvWindow:
    """Helper to create test window using synthetic adapter."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=num_bars, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestEngineImport:
    """Test engine module imports correctly."""
    
    def test_engine_import(self):
        """ApexEngine should be importable from core.engine."""
        assert ApexEngine is not None
        assert callable(ApexEngine)
    
    def test_engine_instantiation(self):
        """ApexEngine should instantiate without errors."""
        engine = ApexEngine(enable_logging=False)
        assert engine is not None
        assert isinstance(engine, ApexEngine)
    
    def test_engine_has_run_method(self):
        """ApexEngine should have a run method."""
        engine = ApexEngine(enable_logging=False)
        assert hasattr(engine, "run")
        assert callable(engine.run)
    
    def test_engine_has_run_scan_method(self):
        """ApexEngine should have a run_scan method."""
        engine = ApexEngine(enable_logging=False)
        assert hasattr(engine, "run_scan")
        assert callable(engine.run_scan)


class TestEngineSingleSymbol:
    """Test engine on single symbols."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_single_symbol_smoke(self, symbol: str):
        """Engine should successfully analyze a single symbol."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result is not None
        assert isinstance(result, ApexResult)
        assert hasattr(result, "quantrascore")
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_result_has_required_fields(self, symbol: str):
        """Engine result should contain all required fields."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert hasattr(result, "quantrascore")
        assert hasattr(result, "regime")
        assert hasattr(result, "risk_tier")
        assert hasattr(result, "entropy_state")
        assert hasattr(result, "suppression_state")
        assert hasattr(result, "drift_state")
        assert hasattr(result, "verdict")
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_engine_score_in_valid_range(self, symbol: str):
        """QuantraScore should always be in 0-100 range."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.quantrascore, (int, float))
        assert 0 <= result.quantrascore <= 100


class TestEngineDeterminism:
    """Test engine determinism across multiple runs."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS[:3])
    def test_protocol_order_deterministic(self, symbol: str):
        """Protocol execution should be deterministic across runs."""
        engine1 = ApexEngine(enable_logging=False)
        engine2 = ApexEngine(enable_logging=False)
        
        window = _get_test_window(symbol)
        
        result1 = engine1.run(window)
        result2 = engine2.run(window)
        
        assert result1.quantrascore == result2.quantrascore
        assert result1.regime == result2.regime
        assert result1.risk_tier == result2.risk_tier
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS)
    def test_score_deterministic_same_engine(self, symbol: str):
        """Same engine should produce identical results for same input."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.quantrascore == result2.quantrascore
    
    @pytest.mark.parametrize("symbol", ALL_TEST_SYMBOLS)
    def test_determinism_across_10_runs(self, symbol: str):
        """Engine should produce identical results across 10 runs."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        scores = []
        for _ in range(10):
            result = engine.run(window)
            scores.append(result.quantrascore)
        
        assert len(set(scores)) == 1, f"Non-deterministic scores for {symbol}: {scores}"


class TestEngineWithContext:
    """Test engine with different contexts."""
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS[:3])
    def test_engine_with_context(self, symbol: str):
        """Engine should accept ApexContext parameter."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(seed=42, compliance_mode=True)
        
        result = engine.run(window, context)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
    
    @pytest.mark.parametrize("symbol", LIQUID_SYMBOLS[:3])
    def test_engine_compliance_mode(self, symbol: str):
        """Engine should work in compliance mode."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        context = ApexContext(compliance_mode=True)
        
        result = engine.run(window, context)
        
        assert result is not None
        assert 0 <= result.quantrascore <= 100
        assert result.verdict is not None
