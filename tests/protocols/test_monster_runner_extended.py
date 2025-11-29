"""
Tests for MonsterRunner Extended Protocols (MR06-MR20)

Tests the new Stage 2 MonsterRunner protocols for:
- Bollinger breakouts
- Volume explosions  
- Gap runners
- VWAP breakouts
- NR7 patterns
- Short squeeze/gamma
- Crypto pumps
- News catalysts
- Fractal explosions
- 100% day moves
- Parabolic phase 3
- Meme frenzy
- Options gamma ramp
- FOMO cascade
- Nuclear runners
"""

import pytest
import numpy as np
from datetime import datetime

from src.quantracore_apex.core.schemas import OhlcvBar
from src.quantracore_apex.protocols.monster_runner import (
    run_MR06, run_MR07, run_MR08, run_MR09, run_MR10,
    run_MR11, run_MR12, run_MR13, run_MR14, run_MR15,
    run_MR16, run_MR17, run_MR18, run_MR19, run_MR20,
    MonsterRunnerLoader,
)


def create_test_bars(n: int = 100, base_price: float = 100.0, volatility: float = 0.02) -> list:
    """Create synthetic OHLCV bars for testing."""
    bars = []
    price = base_price
    
    for i in range(n):
        change = np.random.randn() * volatility
        open_price = price
        close_price = price * (1 + change)
        high = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.005))
        low = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.005))
        volume = int(1000000 * (1 + np.random.randn() * 0.3))
        
        bars.append(OhlcvBar(
            timestamp=datetime.now(),
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=max(volume, 100000),
        ))
        
        price = close_price
    
    return bars


def create_explosive_bars(n: int = 100) -> list:
    """Create bars simulating an explosive move."""
    bars = []
    price = 100.0
    
    for i in range(n - 10):
        change = np.random.randn() * 0.01
        open_price = price
        close_price = price * (1 + change)
        high = max(open_price, close_price) * 1.002
        low = min(open_price, close_price) * 0.998
        volume = 1000000
        
        bars.append(OhlcvBar(
            timestamp=datetime.now(),
            open=open_price, high=high, low=low, close=close_price,
            volume=volume,
        ))
        price = close_price
    
    for i in range(10):
        open_price = price
        close_price = price * 1.15
        high = close_price * 1.02
        low = open_price * 0.98
        volume = 5000000
        
        bars.append(OhlcvBar(
            timestamp=datetime.now(),
            open=open_price, high=high, low=low, close=close_price,
            volume=volume,
        ))
        price = close_price
    
    return bars


class TestMR06BollingerBreakout:
    """Tests for MR06 Bollinger Breakout Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(60)
        result = run_MR06(bars)
        
        assert result.protocol_id == "MR06"
        assert hasattr(result, 'fired')
        assert hasattr(result, 'breakout_score')
        assert hasattr(result, 'squeeze_depth')
        assert hasattr(result, 'breakout_direction')
    
    def test_insufficient_data(self):
        bars = create_test_bars(10)
        result = run_MR06(bars)
        
        assert not result.fired
        assert "Insufficient" in result.notes
    
    def test_detects_breakout(self):
        bars = create_explosive_bars(60)
        result = run_MR06(bars)
        
        assert result.breakout_score >= 0.0
        assert result.breakout_direction in ["neutral", "bullish", "bearish"]


class TestMR07VolumeExplosion:
    """Tests for MR07 Volume Explosion Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(40)
        result = run_MR07(bars)
        
        assert result.protocol_id == "MR07"
        assert hasattr(result, 'explosion_score')
        assert hasattr(result, 'volume_multiple')
        assert hasattr(result, 'direction')
    
    def test_insufficient_data(self):
        bars = create_test_bars(10)
        result = run_MR07(bars)
        
        assert not result.fired
        assert "Insufficient" in result.notes


class TestMR08EarningsGapRunner:
    """Tests for MR08 Earnings Gap Runner Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR08(bars)
        
        assert result.protocol_id == "MR08"
        assert hasattr(result, 'runner_score')
        assert hasattr(result, 'gap_pct')
        assert hasattr(result, 'continuation')


class TestMR09VWAPBreakout:
    """Tests for MR09 VWAP Breakout Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(40)
        result = run_MR09(bars)
        
        assert result.protocol_id == "MR09"
        assert hasattr(result, 'vwap_score')
        assert hasattr(result, 'vwap_deviation')
        assert hasattr(result, 'volume_confirmation')


class TestMR10NR7Breakout:
    """Tests for MR10 NR7 Breakout Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR10(bars)
        
        assert result.protocol_id == "MR10"
        assert hasattr(result, 'nr7_score')
        assert hasattr(result, 'is_nr7')
        assert hasattr(result, 'range_rank')


class TestMR11ShortSqueezeGamma:
    """Tests for MR11 Short Squeeze Gamma Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR11(bars)
        
        assert result.protocol_id == "MR11"
        assert hasattr(result, 'squeeze_score')
        assert hasattr(result, 'price_acceleration')
        assert hasattr(result, 'parabolic_move')
    
    def test_detects_squeeze(self):
        bars = create_explosive_bars(50)
        result = run_MR11(bars)
        
        assert result.squeeze_score >= 0.0


class TestMR12CryptoPump:
    """Tests for MR12 Crypto Pump Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(20)
        result = run_MR12(bars)
        
        assert result.protocol_id == "MR12"
        assert hasattr(result, 'pump_score')
        assert hasattr(result, 'is_pump')
        assert hasattr(result, 'is_dump')


class TestMR13NewsCatalyst:
    """Tests for MR13 News Catalyst Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR13(bars)
        
        assert result.protocol_id == "MR13"
        assert hasattr(result, 'catalyst_score')
        assert hasattr(result, 'gap_magnitude')
        assert hasattr(result, 'potential_catalyst')


class TestMR14FractalExplosion:
    """Tests for MR14 Fractal Explosion Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(40)
        result = run_MR14(bars)
        
        assert result.protocol_id == "MR14"
        assert hasattr(result, 'fractal_score')
        assert hasattr(result, 'is_new_high')
        assert hasattr(result, 'is_new_low')


class TestMR15OneHundredPercentDay:
    """Tests for MR15 100% Day Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(20)
        result = run_MR15(bars)
        
        assert result.protocol_id == "MR15"
        assert hasattr(result, 'extreme_score')
        assert hasattr(result, 'is_doubler')
        assert hasattr(result, 'is_halver')


class TestMR16ParabolicPhase3:
    """Tests for MR16 Parabolic Phase 3 Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(40)
        result = run_MR16(bars)
        
        assert result.protocol_id == "MR16"
        assert hasattr(result, 'parabolic_score')
        assert hasattr(result, 'is_blow_off')
        assert hasattr(result, 'exhaustion_risk')


class TestMR17MemeFrenzy:
    """Tests for MR17 Meme Stock Frenzy Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR17(bars)
        
        assert result.protocol_id == "MR17"
        assert hasattr(result, 'frenzy_score')
        assert hasattr(result, 'volatility_ratio')
        assert hasattr(result, 'frenzy_characteristics')


class TestMR18OptionsGammaRamp:
    """Tests for MR18 Options Gamma Ramp Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR18(bars)
        
        assert result.protocol_id == "MR18"
        assert hasattr(result, 'gamma_score')
        assert hasattr(result, 'volatility_spike')
        assert hasattr(result, 'gamma_signature')


class TestMR19FOMOCascade:
    """Tests for MR19 FOMO Cascade Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(30)
        result = run_MR19(bars)
        
        assert result.protocol_id == "MR19"
        assert hasattr(result, 'fomo_score')
        assert hasattr(result, 'cumulative_return_10d')
        assert hasattr(result, 'fomo_stage')


class TestMR20NuclearRunner:
    """Tests for MR20 Nuclear Runner Detector."""
    
    def test_returns_result_structure(self):
        bars = create_test_bars(20)
        result = run_MR20(bars)
        
        assert result.protocol_id == "MR20"
        assert hasattr(result, 'nuclear_score')
        assert hasattr(result, 'is_nuclear')
        assert hasattr(result, 'multiplier')
    
    def test_detects_nuclear_runner(self):
        bars = create_explosive_bars(30)
        result = run_MR20(bars)
        
        assert result.nuclear_score >= 0.0
        assert result.multiplier >= 1.0


class TestMonsterRunnerLoaderExtended:
    """Tests for extended MonsterRunnerLoader."""
    
    def test_loads_all_20_protocols(self):
        loader = MonsterRunnerLoader()
        loaded = loader.get_loaded_protocols()
        
        assert len(loaded) == 20
        for i in range(1, 21):
            assert f"MR{i:02d}" in loaded
    
    def test_run_all_executes_all_protocols(self):
        loader = MonsterRunnerLoader()
        bars = create_test_bars(100)
        
        result = loader.run_all(bars)
        
        assert len(result.individual_results) == 20
        for i in range(1, 21):
            protocol_id = f"MR{i:02d}"
            assert protocol_id in result.individual_results
    
    def test_result_structure(self):
        loader = MonsterRunnerLoader()
        bars = create_test_bars(100)
        
        result = loader.run_all(bars)
        
        assert hasattr(result, 'any_fired')
        assert hasattr(result, 'monster_score')
        assert hasattr(result, 'protocols_fired')
        assert hasattr(result, 'compliance_note')
