"""
Tests for ApexLab V2 label schema and helper functions.

Validates:
- ApexLabV2Row structure and serialization
- Quality tier assignment logic
- Runner flag computation
- Regime label mapping
- Future returns calculation
"""

import pytest
import numpy as np
from datetime import datetime

from src.quantracore_apex.apexlab.apexlab_v2 import (
    ApexLabV2Row,
    ApexLabV2Builder,
    QualityTier,
    MarketCapBand,
    VolatilityBand,
    RegimeType,
    assign_quality_tier,
    compute_runner_flags,
    compute_regime_label,
    compute_future_returns,
    compute_volatility_band,
    compute_liquidity_band,
    compute_entropy_band,
    compute_avoid_trade,
    encode_protocol_vector,
)
from src.quantracore_apex.core.schemas import OhlcvWindow


class TestApexLabV2Row:
    """Tests for ApexLabV2Row dataclass."""
    
    def test_row_creation_defaults(self):
        """Test row creation with default values."""
        row = ApexLabV2Row(
            symbol="AAPL",
            event_time=datetime.utcnow(),
            timeframe="1d",
            engine_snapshot_id="abc123",
            scanner_snapshot_id="def456",
        )
        
        assert row.symbol == "AAPL"
        assert row.quantra_score == 50.0
        assert row.risk_tier == "medium"
        assert row.future_quality_tier == QualityTier.C.value
        assert row.hit_runner_threshold == 0
        assert row.avoid_trade == 0
    
    def test_row_to_dict(self):
        """Test row serialization to dictionary."""
        row = ApexLabV2Row(
            symbol="MSFT",
            event_time=datetime(2024, 1, 15, 12, 0, 0),
            timeframe="1h",
            engine_snapshot_id="snap1",
            scanner_snapshot_id="snap2",
            quantra_score=75.0,
            risk_tier="low",
        )
        
        d = row.to_dict()
        
        assert d["symbol"] == "MSFT"
        assert d["quantra_score"] == 75.0
        assert d["risk_tier"] == "low"
        assert "protocol_ids" in d
        assert "protocol_vector" in d
    
    def test_row_from_dict(self):
        """Test row creation from dictionary."""
        data = {
            "symbol": "GOOGL",
            "event_time": datetime(2024, 2, 1),
            "timeframe": "5m",
            "engine_snapshot_id": "e1",
            "scanner_snapshot_id": "s1",
            "quantra_score": 85.0,
            "hit_runner_threshold": 1,
        }
        
        row = ApexLabV2Row.from_dict(data)
        
        assert row.symbol == "GOOGL"
        assert row.quantra_score == 85.0
        assert row.hit_runner_threshold == 1
    
    def test_row_all_fields_present(self):
        """Verify all required fields exist in row."""
        row = ApexLabV2Row(
            symbol="TEST",
            event_time=datetime.utcnow(),
            timeframe="1d",
            engine_snapshot_id="e",
            scanner_snapshot_id="s",
        )
        
        d = row.to_dict()
        
        required_fields = [
            "symbol", "event_time", "timeframe", "engine_snapshot_id",
            "scanner_snapshot_id", "sector", "market_cap_band",
            "quantra_score", "risk_tier", "entropy_band", "suppression_state",
            "regime_type", "volatility_band", "liquidity_band",
            "protocol_ids", "protocol_vector",
            "ret_1d", "ret_3d", "ret_5d", "ret_10d",
            "max_runup_5d", "max_drawdown_5d", "time_to_peak_5d",
            "future_quality_tier", "hit_runner_threshold",
            "hit_monster_runner_threshold", "avoid_trade",
            "regime_label", "sector_regime",
        ]
        
        for field in required_fields:
            assert field in d, f"Missing field: {field}"


class TestQualityTierAssignment:
    """Tests for quality tier assignment logic."""
    
    def test_a_plus_tier(self):
        """Test A_PLUS tier assignment."""
        tier = assign_quality_tier(
            max_runup_5d=0.25,
            max_drawdown_5d=-0.03,
            ret_5d=0.20,
        )
        assert tier == QualityTier.A_PLUS.value
    
    def test_a_tier(self):
        """Test A tier assignment."""
        tier = assign_quality_tier(
            max_runup_5d=0.12,
            max_drawdown_5d=-0.05,
            ret_5d=0.08,
        )
        assert tier == QualityTier.A.value
    
    def test_b_tier(self):
        """Test B tier assignment."""
        tier = assign_quality_tier(
            max_runup_5d=0.07,
            max_drawdown_5d=-0.08,
            ret_5d=0.03,
        )
        assert tier == QualityTier.B.value
    
    def test_c_tier(self):
        """Test C tier assignment (default)."""
        tier = assign_quality_tier(
            max_runup_5d=0.03,
            max_drawdown_5d=-0.03,
            ret_5d=0.01,
        )
        assert tier == QualityTier.C.value
    
    def test_d_tier(self):
        """Test D tier assignment (loss)."""
        tier = assign_quality_tier(
            max_runup_5d=0.02,
            max_drawdown_5d=-0.10,
            ret_5d=-0.08,
        )
        assert tier == QualityTier.D.value
    
    def test_edge_case_exactly_threshold(self):
        """Test edge case at exact threshold."""
        tier = assign_quality_tier(
            max_runup_5d=0.20,
            max_drawdown_5d=-0.05,
            ret_5d=0.15,
        )
        assert tier == QualityTier.A_PLUS.value


class TestRunnerFlags:
    """Tests for runner flag computation."""
    
    def test_no_runner(self):
        """Test no runner detected."""
        hit_runner, hit_monster = compute_runner_flags(0.05)
        assert hit_runner == 0
        assert hit_monster == 0
    
    def test_runner_detected(self):
        """Test runner detected but not monster."""
        hit_runner, hit_monster = compute_runner_flags(0.18)
        assert hit_runner == 1
        assert hit_monster == 0
    
    def test_monster_runner_detected(self):
        """Test monster runner detected."""
        hit_runner, hit_monster = compute_runner_flags(0.30)
        assert hit_runner == 1
        assert hit_monster == 1
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        hit_runner, hit_monster = compute_runner_flags(
            0.12,
            runner_threshold=0.10,
            monster_threshold=0.20,
        )
        assert hit_runner == 1
        assert hit_monster == 0


class TestRegimeLabels:
    """Tests for regime label computation."""
    
    def test_trend_up(self):
        """Test trend up mapping."""
        label = compute_regime_label("trending_up")
        assert label == RegimeType.TREND_UP.value
    
    def test_trend_down(self):
        """Test trend down mapping."""
        label = compute_regime_label("trending_down")
        assert label == RegimeType.TREND_DOWN.value
    
    def test_range_bound(self):
        """Test range bound -> chop mapping."""
        label = compute_regime_label("range_bound")
        assert label == RegimeType.CHOP.value
    
    def test_unknown_regime(self):
        """Test unknown regime defaults to chop."""
        label = compute_regime_label("unknown_regime")
        assert label == RegimeType.CHOP.value


class TestFutureReturns:
    """Tests for future returns calculation."""
    
    def test_basic_returns(self):
        """Test basic return calculation."""
        entry_price = 100.0
        future_prices = np.array([105, 110, 108, 112, 115])
        
        returns = compute_future_returns(entry_price, future_prices, bars_per_day=1)
        
        assert returns["ret_1d"] == pytest.approx(0.05, abs=0.01)
        assert returns["ret_3d"] == pytest.approx(0.08, abs=0.01)
        assert returns["ret_5d"] == pytest.approx(0.15, abs=0.01)
        assert returns["max_runup_5d"] == pytest.approx(0.15, abs=0.01)
        assert returns["max_drawdown_5d"] == pytest.approx(0.05, abs=0.01)
    
    def test_empty_prices(self):
        """Test with empty price array."""
        returns = compute_future_returns(100.0, np.array([]))
        
        assert returns["ret_1d"] == 0.0
        assert returns["ret_5d"] == 0.0
    
    def test_zero_entry_price(self):
        """Test with zero entry price."""
        returns = compute_future_returns(0.0, np.array([100, 105]))
        
        assert returns["ret_1d"] == 0.0
    
    def test_time_to_peak(self):
        """Test time to peak calculation."""
        entry_price = 100.0
        future_prices = np.array([102, 105, 108, 106, 104])
        
        returns = compute_future_returns(entry_price, future_prices, bars_per_day=1)
        
        assert returns["time_to_peak_5d"] == 2


class TestVolatilityBands:
    """Tests for volatility band classification."""
    
    def test_low_volatility(self):
        """Test low volatility classification."""
        band = compute_volatility_band(0.10)
        assert band == VolatilityBand.LOW.value
    
    def test_mid_volatility(self):
        """Test mid volatility classification."""
        band = compute_volatility_band(0.20)
        assert band == VolatilityBand.MID.value
    
    def test_high_volatility(self):
        """Test high volatility classification."""
        band = compute_volatility_band(0.40)
        assert band == VolatilityBand.HIGH.value


class TestLiquidityBands:
    """Tests for liquidity band classification."""
    
    def test_low_liquidity(self):
        """Test low liquidity classification."""
        band = compute_liquidity_band(1000, 50000)
        assert band == "low"
    
    def test_mid_liquidity(self):
        """Test mid liquidity classification."""
        band = compute_liquidity_band(50000, 5000000)
        assert band == "mid"
    
    def test_high_liquidity(self):
        """Test high liquidity classification."""
        band = compute_liquidity_band(1000000, 50000000)
        assert band == "high"


class TestAvoidTrade:
    """Tests for avoid trade logic."""
    
    def test_avoid_high_volatility(self):
        """Test avoid trade on high volatility."""
        avoid = compute_avoid_trade("high", "mid", "none", "medium")
        assert avoid == 1
    
    def test_avoid_low_liquidity(self):
        """Test avoid trade on low liquidity."""
        avoid = compute_avoid_trade("mid", "low", "none", "medium")
        assert avoid == 1
    
    def test_avoid_blocked_suppression(self):
        """Test avoid trade on blocked suppression."""
        avoid = compute_avoid_trade("mid", "mid", "blocked", "medium")
        assert avoid == 1
    
    def test_avoid_extreme_risk(self):
        """Test avoid trade on extreme risk."""
        avoid = compute_avoid_trade("mid", "mid", "none", "extreme")
        assert avoid == 1
    
    def test_no_avoid(self):
        """Test no avoid trade conditions."""
        avoid = compute_avoid_trade("mid", "mid", "none", "medium")
        assert avoid == 0


class TestProtocolVector:
    """Tests for protocol vector encoding."""
    
    def test_tier_protocol_encoding(self):
        """Test Tier protocol encoding."""
        vector = encode_protocol_vector(["T03", "T17"])
        
        assert vector[2] == 1.0
        assert vector[16] == 1.0
        assert sum(vector) == 2.0
    
    def test_learning_protocol_encoding(self):
        """Test Learning protocol encoding."""
        vector = encode_protocol_vector(["LP05", "LP10"])
        
        assert vector[84] == 1.0
        assert vector[89] == 1.0
    
    def test_monster_runner_encoding(self):
        """Test Monster Runner protocol encoding."""
        vector = encode_protocol_vector(["MR01", "MR03"])
        
        assert vector[105] == 1.0
        assert vector[107] == 1.0
    
    def test_empty_protocols(self):
        """Test empty protocol list."""
        vector = encode_protocol_vector([])
        
        assert sum(vector) == 0.0
        assert len(vector) == 115


class TestApexLabV2Builder:
    """Tests for ApexLabV2Builder."""
    
    def test_builder_creation(self):
        """Test builder instantiation."""
        builder = ApexLabV2Builder(enable_logging=False)
        
        assert builder is not None
        assert builder.runner_threshold == 0.15
        assert builder.monster_threshold == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
