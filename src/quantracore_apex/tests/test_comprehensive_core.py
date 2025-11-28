"""
Comprehensive Core Engine Tests

Tests all core engine modules for correctness and determinism.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List

from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow, ApexResult
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.entropy import compute_entropy
from src.quantracore_apex.core.suppression import compute_suppression
from src.quantracore_apex.core.drift import compute_drift
from src.quantracore_apex.core.continuation import compute_continuation
from src.quantracore_apex.core.volume_spike import compute_volume_spike
from src.quantracore_apex.core.regime import classify_regime
from src.quantracore_apex.core.quantrascore import compute_quantrascore
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.core.sector_context import apply_sector_context


def generate_bars(n: int = 100, seed: int = 42) -> List[OhlcvBar]:
    """Generate deterministic test bars."""
    np.random.seed(seed)
    bars = []
    price = 100.0
    
    for i in range(n):
        change = np.random.randn() * 0.02
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.005))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.005))
        volume = 1000000 + np.random.randint(-200000, 200000)
        
        bars.append(OhlcvBar(
            timestamp=datetime(2024, 1, 1),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=float(volume),
        ))
        price = close_price
    
    return bars


class TestApexEngine:
    """Tests for ApexEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = ApexEngine()
        assert engine is not None
    
    def test_engine_scan(self):
        """Test full engine scan."""
        engine = ApexEngine()
        bars = generate_bars(100)
        
        result = engine.scan(bars, symbol="TEST")
        
        assert isinstance(result, ApexResult)
        assert 0 <= result.quantrascore <= 100
        assert result.symbol == "TEST"
    
    def test_engine_determinism(self):
        """Test engine produces identical results for same input."""
        engine = ApexEngine()
        bars = generate_bars(100, seed=42)
        
        result1 = engine.scan(bars, symbol="TEST")
        result2 = engine.scan(bars, symbol="TEST")
        
        assert result1.quantrascore == result2.quantrascore
        assert result1.regime == result2.regime
        assert result1.risk_tier == result2.risk_tier
    
    def test_engine_with_different_data(self):
        """Test engine with different data produces different results."""
        engine = ApexEngine()
        
        bars1 = generate_bars(100, seed=42)
        bars2 = generate_bars(100, seed=99)
        
        result1 = engine.scan(bars1, symbol="TEST1")
        result2 = engine.scan(bars2, symbol="TEST2")
        
        assert result1.quantrascore != result2.quantrascore or result1.regime != result2.regime


class TestEntropyModule:
    """Tests for entropy computation."""
    
    def test_compute_entropy(self):
        """Test entropy computation."""
        bars = generate_bars(100)
        window = OhlcvWindow(bars=bars)
        metrics = compute_entropy(window)
        
        assert metrics is not None
        assert 0 <= metrics.combined_entropy <= 1
    
    def test_entropy_determinism(self):
        """Test entropy is deterministic."""
        bars = generate_bars(100, seed=42)
        window = OhlcvWindow(bars=bars)
        
        m1 = compute_entropy(window)
        m2 = compute_entropy(window)
        
        assert m1.combined_entropy == m2.combined_entropy


class TestSuppressionModule:
    """Tests for suppression detection."""
    
    def test_compute_suppression(self):
        """Test suppression computation."""
        bars = generate_bars(100)
        window = OhlcvWindow(bars=bars)
        metrics = compute_suppression(window)
        
        assert metrics is not None
        assert 0 <= metrics.suppression_score <= 1
    
    def test_suppression_determinism(self):
        """Test suppression is deterministic."""
        bars = generate_bars(100, seed=42)
        window = OhlcvWindow(bars=bars)
        
        m1 = compute_suppression(window)
        m2 = compute_suppression(window)
        
        assert m1.suppression_score == m2.suppression_score


class TestDriftModule:
    """Tests for drift analysis."""
    
    def test_compute_drift(self):
        """Test drift computation."""
        bars = generate_bars(100)
        window = OhlcvWindow(bars=bars)
        metrics = compute_drift(window)
        
        assert metrics is not None
        assert metrics.drift_magnitude >= 0


class TestContinuationModule:
    """Tests for continuation analysis."""
    
    def test_compute_continuation(self):
        """Test continuation computation."""
        bars = generate_bars(100)
        window = OhlcvWindow(bars=bars)
        metrics = compute_continuation(window)
        
        assert metrics is not None
        assert 0 <= metrics.continuation_probability <= 1


class TestVolumeModule:
    """Tests for volume analysis."""
    
    def test_compute_volume(self):
        """Test volume computation."""
        bars = generate_bars(100)
        window = OhlcvWindow(bars=bars)
        metrics = compute_volume_spike(window)
        
        assert metrics is not None


class TestRegimeModule:
    """Tests for regime classification."""
    
    def test_classify_regime(self):
        """Test regime classification."""
        bars = generate_bars(100)
        regime = classify_regime(bars)
        
        assert regime is not None


class TestQuantraScoreModule:
    """Tests for QuantraScore computation."""
    
    def test_compute_quantrascore(self):
        """Test QuantraScore computation."""
        engine = ApexEngine()
        bars = generate_bars(100)
        result = engine.scan(bars, symbol="TEST")
        
        assert 0 <= result.quantrascore <= 100
    
    def test_quantrascore_range(self):
        """Test QuantraScore stays in 0-100 range."""
        engine = ApexEngine()
        
        for seed in range(10):
            bars = generate_bars(100, seed=seed)
            result = engine.scan(bars, symbol=f"TEST{seed}")
            assert 0 <= result.quantrascore <= 100


class TestMicrotraitsModule:
    """Tests for microtrait computation."""
    
    def test_compute_microtraits(self):
        """Test microtrait computation."""
        bars = generate_bars(100)
        traits = compute_microtraits(bars)
        
        assert traits is not None


class TestSectorContext:
    """Tests for sector context adjustment."""
    
    def test_apply_sector_context(self):
        """Test sector context application."""
        engine = ApexEngine()
        bars = generate_bars(100)
        result = engine.scan(bars, symbol="AAPL")
        
        assert result is not None
