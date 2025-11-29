"""
Volatility Tag Tests for QuantraCore Apex.

Tests volatility classification and metadata analysis with SUBSTANTIVE assertions.
"""

import pytest

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, RegimeType, Microtraits
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter


ALL_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "GME", "AMZN", "META"]


def _get_test_window(symbol: str) -> OhlcvWindow:
    """Helper to create test window."""
    adapter = SyntheticAdapter(seed=42)
    bars = adapter.fetch(symbol, num_bars=100, seed=42)
    return OhlcvWindow(symbol=symbol, timeframe="1d", bars=bars)


class TestMicrotraitsComputation:
    """Test microtraits computation."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_returns_microtraits_object(self, symbol: str):
        """compute_microtraits should return Microtraits instance."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        assert isinstance(microtraits, Microtraits), f"Expected Microtraits, got {type(microtraits)}"
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_has_volatility_ratio(self, symbol: str):
        """Microtraits should have volatility_ratio attribute."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        assert hasattr(microtraits, "volatility_ratio"), "Missing volatility_ratio"
        assert isinstance(microtraits.volatility_ratio, (int, float))
        assert microtraits.volatility_ratio >= 0, "volatility_ratio should be non-negative"
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_microtraits_has_trend_consistency(self, symbol: str):
        """Microtraits should have trend_consistency attribute."""
        window = _get_test_window(symbol)
        microtraits = compute_microtraits(window)
        
        assert hasattr(microtraits, "trend_consistency")
        assert isinstance(microtraits.trend_consistency, (int, float))
        assert -1 <= microtraits.trend_consistency <= 1


class TestRegimeClassification:
    """Test regime classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_regime_is_valid_enum(self, symbol: str):
        """Engine should classify regime as valid RegimeType."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert isinstance(result.regime, RegimeType), f"Expected RegimeType, got {type(result.regime)}"
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_regime_deterministic(self, symbol: str):
        """Regime should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.regime == result2.regime


class TestEntropyState:
    """Test entropy state classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_entropy_state_exists(self, symbol: str):
        """Result should have entropy_state."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.entropy_state is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_entropy_state_deterministic(self, symbol: str):
        """Entropy state should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.entropy_state == result2.entropy_state


class TestSuppressionState:
    """Test suppression state classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_suppression_state_exists(self, symbol: str):
        """Result should have suppression_state."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.suppression_state is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_suppression_state_deterministic(self, symbol: str):
        """Suppression state should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.suppression_state == result2.suppression_state


class TestDriftState:
    """Test drift state classification."""
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_drift_state_exists(self, symbol: str):
        """Result should have drift_state."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result = engine.run(window)
        
        assert result.drift_state is not None
    
    @pytest.mark.parametrize("symbol", ALL_SYMBOLS)
    def test_drift_state_deterministic(self, symbol: str):
        """Drift state should be deterministic."""
        engine = ApexEngine(enable_logging=False)
        window = _get_test_window(symbol)
        
        result1 = engine.run(window)
        result2 = engine.run(window)
        
        assert result1.drift_state == result2.drift_state
