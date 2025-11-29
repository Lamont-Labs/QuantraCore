"""
Volatility Tag Tests for QuantraCore Apex.

Tests volatility classification and metadata analysis.
"""

import pytest
import numpy as np
from typing import List

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, RegimeType
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


STABLE_SYMBOLS = ["JNJ", "PG", "KO", "PEP", "WMT"]
VOLATILE_SYMBOLS = ["TSLA", "GME", "AMC", "RIVN", "LCID"]
MIXED_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

ALL_SYMBOLS = STABLE_SYMBOLS + VOLATILE_SYMBOLS + MIXED_SYMBOLS


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestMicrotraitsComputation:
    """Test microtraits computation."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_computes(self, symbol: str):
        """Microtraits should compute without error."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        assert microtraits is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_has_volatility(self, symbol: str):
        """Microtraits should include volatility metrics."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        assert hasattr(microtraits, "volatility_ratio") or hasattr(microtraits, "atr_pct")
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_volatility_positive(self, symbol: str):
        """Volatility metrics should be non-negative."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        if hasattr(microtraits, "volatility_ratio"):
            assert microtraits.volatility_ratio >= 0
        if hasattr(microtraits, "atr_pct"):
            assert microtraits.atr_pct >= 0


class TestRegimeClassification:
    """Test regime classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_regime_valid_type(self, symbol: str):
        """Engine should classify regime with valid type."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.regime is not None
        assert isinstance(result.regime, RegimeType)
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_regime_in_valid_set(self, symbol: str):
        """Regime should be one of valid enum values."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        valid_regimes = [r for r in RegimeType]
        assert result.regime in valid_regimes


class TestVolatilityBands:
    """Test volatility band classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_entropy_state_valid(self, symbol: str):
        """Entropy state should be valid."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.entropy_state is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_suppression_state_valid(self, symbol: str):
        """Suppression state should be valid."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.suppression_state is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_drift_state_valid(self, symbol: str):
        """Drift state should be valid."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.drift_state is not None
