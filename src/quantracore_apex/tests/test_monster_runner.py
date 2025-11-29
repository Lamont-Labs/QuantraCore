"""
Test MonsterRunner Protocols

Verifies MR01-MR05 protocols execute correctly and
produce deterministic results.
"""

import numpy as np
from datetime import datetime

from src.quantracore_apex.core.schemas import OhlcvBar
from src.quantracore_apex.protocols.monster_runner import (
    run_MR01,
    run_MR02,
    run_MR03,
    run_MR04,
    run_MR05,
    MonsterRunnerLoader,
)


def generate_test_bars(n: int = 100, seed: int = 42) -> list:
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


class TestMR01CompressionDetector:
    """Tests for MR01 Compression Explosion Detector."""
    
    def test_basic_execution(self):
        """Test MR01 executes without error."""
        bars = generate_test_bars(100)
        result = run_MR01(bars)
        
        assert result.protocol_id == "MR01"
        assert 0.0 <= result.compression_score <= 1.0
        assert result.direction_bias in ["bullish", "bearish", "neutral"]
    
    def test_insufficient_data(self):
        """Test MR01 handles insufficient data."""
        bars = generate_test_bars(20)
        result = run_MR01(bars)
        
        assert "Insufficient" in result.notes
    
    def test_determinism(self):
        """Test MR01 produces consistent results."""
        bars = generate_test_bars(100, seed=42)
        
        result1 = run_MR01(bars)
        result2 = run_MR01(bars)
        
        assert result1.compression_score == result2.compression_score
        assert result1.fired == result2.fired


class TestMR02VolumeAnomaly:
    """Tests for MR02 Volume Anomaly Detector."""
    
    def test_basic_execution(self):
        """Test MR02 executes without error."""
        bars = generate_test_bars(100)
        result = run_MR02(bars)
        
        assert result.protocol_id == "MR02"
        assert 0.0 <= result.volume_anomaly_score <= 1.0
    
    def test_determinism(self):
        """Test MR02 produces consistent results."""
        bars = generate_test_bars(100, seed=42)
        
        result1 = run_MR02(bars)
        result2 = run_MR02(bars)
        
        assert result1.volume_anomaly_score == result2.volume_anomaly_score


class TestMR03RegimeShift:
    """Tests for MR03 Volatility Regime Shift Detector."""
    
    def test_basic_execution(self):
        """Test MR03 executes without error."""
        bars = generate_test_bars(100)
        result = run_MR03(bars)
        
        assert result.protocol_id == "MR03"
        assert result.current_regime in ["low", "normal", "high"]
    
    def test_determinism(self):
        """Test MR03 produces consistent results."""
        bars = generate_test_bars(100, seed=42)
        
        result1 = run_MR03(bars)
        result2 = run_MR03(bars)
        
        assert result1.regime_shift_score == result2.regime_shift_score


class TestMR04InstitutionalFootprint:
    """Tests for MR04 Institutional Footprint Detector."""
    
    def test_basic_execution(self):
        """Test MR04 executes without error."""
        bars = generate_test_bars(100)
        result = run_MR04(bars)
        
        assert result.protocol_id == "MR04"
        assert 0.0 <= result.institutional_score <= 1.0


class TestMR05MultiTimeframe:
    """Tests for MR05 Multi-Timeframe Alignment Detector."""
    
    def test_basic_execution(self):
        """Test MR05 executes without error."""
        bars = generate_test_bars(100)
        result = run_MR05(bars)
        
        assert result.protocol_id == "MR05"
        assert result.dominant_direction in ["bullish", "bearish", "neutral"]


class TestMonsterRunnerLoader:
    """Tests for MonsterRunner protocol loader."""
    
    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = MonsterRunnerLoader()
        protocols = loader.get_loaded_protocols()
        
        assert len(protocols) >= 1
    
    def test_run_all(self):
        """Test running all protocols."""
        loader = MonsterRunnerLoader()
        bars = generate_test_bars(100)
        
        result = loader.run_all(bars)
        
        assert isinstance(result.monster_score, float)
        assert 0.0 <= result.monster_score <= 1.0
        assert "compliance_note" in result.to_dict()
    
    def test_determinism(self):
        """Test loader produces consistent results."""
        loader = MonsterRunnerLoader()
        bars = generate_test_bars(100, seed=42)
        
        result1 = loader.run_all(bars)
        result2 = loader.run_all(bars)
        
        assert result1.monster_score == result2.monster_score
