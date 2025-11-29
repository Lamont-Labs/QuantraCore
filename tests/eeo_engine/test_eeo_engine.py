"""
Comprehensive tests for the Entry/Exit Optimization Engine.

Tests cover:
- Data models and structures
- Entry optimization strategies
- Exit optimization (stops, targets, trailing)
- Profile system
- Full plan building
- Integration with execution engine
"""

import pytest
from datetime import datetime

from src.quantracore_apex.eeo_engine import (
    EntryExitOptimizer,
    EntryOptimizer,
    ExitOptimizer,
    SignalContext,
    PredictiveContext,
    MarketMicrostructure,
    RiskContext,
    EEOContext,
    SignalDirection,
    QualityTier,
    ProfileType,
    VolatilityBand,
    LiquidityBand,
    RegimeType,
    EntryMode,
    ExitMode,
    OrderTypeEEO,
    TrailingStopMode,
    CONSERVATIVE_PROFILE,
    BALANCED_PROFILE,
    AGGRESSIVE_RESEARCH_PROFILE,
    get_profile,
    EntryExitPlan,
    BaseEntry,
    EntryZone,
    ProtectiveStop,
    ProfitTarget,
)


class TestEntryZone:
    """Tests for EntryZone model."""
    
    def test_entry_zone_creation(self):
        """Test entry zone creation."""
        zone = EntryZone(lower=95.0, upper=105.0)
        assert zone.lower == 95.0
        assert zone.upper == 105.0
    
    def test_entry_zone_contains(self):
        """Test price containment check."""
        zone = EntryZone(lower=95.0, upper=105.0)
        assert zone.contains(100.0)
        assert zone.contains(95.0)
        assert zone.contains(105.0)
        assert not zone.contains(94.9)
        assert not zone.contains(105.1)
    
    def test_entry_zone_width(self):
        """Test zone width calculation."""
        zone = EntryZone(lower=95.0, upper=105.0)
        assert zone.width() == 10.0
    
    def test_entry_zone_midpoint(self):
        """Test zone midpoint calculation."""
        zone = EntryZone(lower=95.0, upper=105.0)
        assert zone.midpoint() == 100.0


class TestSignalContext:
    """Tests for SignalContext model."""
    
    def test_signal_context_creation(self):
        """Test signal context creation with defaults."""
        ctx = SignalContext(symbol="AAPL")
        assert ctx.symbol == "AAPL"
        assert ctx.direction == SignalDirection.LONG
        assert ctx.quantra_score == 50.0
    
    def test_signal_context_favorable_regime(self):
        """Test favorable regime detection."""
        ctx = SignalContext(symbol="AAPL", regime_type=RegimeType.TREND_UP)
        assert ctx.is_favorable_regime()
        
        ctx = SignalContext(symbol="AAPL", regime_type=RegimeType.CHOP)
        assert not ctx.is_favorable_regime()
    
    def test_signal_context_high_conviction(self):
        """Test high conviction detection."""
        ctx = SignalContext(symbol="AAPL", quantra_score=75.0)
        assert ctx.is_high_conviction()
        
        ctx = SignalContext(symbol="AAPL", quantra_score=65.0)
        assert not ctx.is_high_conviction()


class TestPredictiveContext:
    """Tests for PredictiveContext model."""
    
    def test_predictive_context_runner_candidate(self):
        """Test runner candidate detection."""
        ctx = PredictiveContext(runner_prob=0.65)
        assert ctx.is_runner_candidate()
        
        ctx = PredictiveContext(runner_prob=0.55)
        assert not ctx.is_runner_candidate()
    
    def test_predictive_context_high_quality(self):
        """Test high quality detection."""
        ctx = PredictiveContext(future_quality_tier=QualityTier.A_PLUS)
        assert ctx.is_high_quality()
        
        ctx = PredictiveContext(future_quality_tier=QualityTier.A)
        assert ctx.is_high_quality()
        
        ctx = PredictiveContext(future_quality_tier=QualityTier.B)
        assert not ctx.is_high_quality()
    
    def test_predictive_context_runner_bucket(self):
        """Test runner probability bucketing."""
        assert PredictiveContext(runner_prob=0.85).get_runner_bucket() == "very_high"
        assert PredictiveContext(runner_prob=0.65).get_runner_bucket() == "high"
        assert PredictiveContext(runner_prob=0.45).get_runner_bucket() == "medium"
        assert PredictiveContext(runner_prob=0.25).get_runner_bucket() == "low"
        assert PredictiveContext(runner_prob=0.15).get_runner_bucket() == "very_low"


class TestMarketMicrostructure:
    """Tests for MarketMicrostructure model."""
    
    def test_from_price(self):
        """Test creation from simple price."""
        micro = MarketMicrostructure.from_price(100.0, atr=2.0)
        assert micro.mid_price == 100.0
        assert micro.atr_14 == 2.0
        assert micro.spread > 0
    
    def test_wide_spread_detection(self):
        """Test wide spread detection."""
        micro = MarketMicrostructure(
            mid_price=100.0,
            spread=1.0,
            spread_pct=1.0,
        )
        assert micro.is_wide_spread(0.5)
        assert not micro.is_wide_spread(1.5)


class TestRiskContext:
    """Tests for RiskContext model."""
    
    def test_max_risk_dollars(self):
        """Test max risk calculation."""
        ctx = RiskContext(
            account_equity=100000.0,
            per_trade_risk_fraction=0.01,
        )
        assert ctx.max_risk_dollars() == 1000.0
    
    def test_can_open_position(self):
        """Test position opening check."""
        ctx = RiskContext(
            open_position_count=5,
            max_positions=10,
        )
        assert ctx.can_open_position()
        
        ctx = RiskContext(
            open_position_count=10,
            max_positions=10,
        )
        assert not ctx.can_open_position()


class TestEEOContext:
    """Tests for EEOContext model."""
    
    @pytest.fixture
    def base_context(self):
        """Create base context for tests."""
        return EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(),
            microstructure=MarketMicrostructure.from_price(100.0),
            risk=RiskContext(),
        )
    
    def test_should_proceed_normal(self, base_context):
        """Test proceed check in normal conditions."""
        assert base_context.should_proceed()
    
    def test_should_not_proceed_avoid_trade(self):
        """Test proceed check with high avoid_trade_prob."""
        ctx = EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(avoid_trade_prob=0.7),
            microstructure=MarketMicrostructure.from_price(100.0),
            risk=RiskContext(),
        )
        assert not ctx.should_proceed()
    
    def test_should_not_proceed_max_positions(self):
        """Test proceed check when max positions reached."""
        ctx = EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(),
            microstructure=MarketMicrostructure.from_price(100.0),
            risk=RiskContext(open_position_count=10, max_positions=10),
        )
        assert not ctx.should_proceed()


class TestProfiles:
    """Tests for EEO profile system."""
    
    def test_conservative_profile(self):
        """Test conservative profile values."""
        assert CONSERVATIVE_PROFILE.per_trade_risk_fraction == 0.005
        assert CONSERVATIVE_PROFILE.entry_aggressiveness == "LOW"
        assert CONSERVATIVE_PROFILE.use_trailing_stop == False
        assert CONSERVATIVE_PROFILE.is_conservative()
    
    def test_balanced_profile(self):
        """Test balanced profile values."""
        assert BALANCED_PROFILE.per_trade_risk_fraction == 0.01
        assert BALANCED_PROFILE.entry_aggressiveness == "MEDIUM"
        assert BALANCED_PROFILE.use_trailing_stop == True
    
    def test_aggressive_research_profile(self):
        """Test aggressive research profile values."""
        assert AGGRESSIVE_RESEARCH_PROFILE.per_trade_risk_fraction == 0.02
        assert AGGRESSIVE_RESEARCH_PROFILE.entry_aggressiveness == "HIGH"
        assert AGGRESSIVE_RESEARCH_PROFILE.is_aggressive()
    
    def test_get_profile(self):
        """Test profile retrieval."""
        assert get_profile(ProfileType.CONSERVATIVE) == CONSERVATIVE_PROFILE
        assert get_profile(ProfileType.BALANCED) == BALANCED_PROFILE
        assert get_profile(ProfileType.AGGRESSIVE_RESEARCH) == AGGRESSIVE_RESEARCH_PROFILE


class TestEntryOptimizer:
    """Tests for EntryOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer with balanced profile."""
        return EntryOptimizer(BALANCED_PROFILE)
    
    @pytest.fixture
    def base_context(self):
        """Create base context for tests."""
        return EEOContext(
            signal=SignalContext(symbol="AAPL", direction=SignalDirection.LONG),
            predictive=PredictiveContext(),
            microstructure=MarketMicrostructure.from_price(100.0, atr=2.0),
            risk=RiskContext(),
        )
    
    def test_optimize_baseline_long(self, optimizer, base_context):
        """Test baseline long optimization."""
        result = optimizer.optimize(base_context)
        
        assert result.base_entry is not None
        assert result.base_entry.entry_price > 0
        assert result.base_entry.entry_zone.lower < result.base_entry.entry_zone.upper
    
    def test_optimize_high_volatility(self, optimizer):
        """Test high volatility optimization."""
        context = EEOContext(
            signal=SignalContext(
                symbol="AAPL",
                volatility_band=VolatilityBand.HIGH,
            ),
            predictive=PredictiveContext(),
            microstructure=MarketMicrostructure.from_price(100.0, atr=5.0),
            risk=RiskContext(),
        )
        
        result = optimizer.optimize(context)
        
        assert result.strategy_used == "high_volatility_mode"
        assert result.base_entry.order_type == OrderTypeEEO.LIMIT
    
    def test_optimize_runner_candidate(self, optimizer):
        """Test runner anticipation strategy."""
        context = EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(
                runner_prob=0.75,
                future_quality_tier=QualityTier.A,
            ),
            microstructure=MarketMicrostructure.from_price(100.0, atr=2.0),
            risk=RiskContext(),
        )
        
        result = optimizer.optimize(context)
        
        assert result.strategy_used == "runner_anticipation_adjust"


class TestExitOptimizer:
    """Tests for ExitOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer with balanced profile."""
        return ExitOptimizer(BALANCED_PROFILE)
    
    @pytest.fixture
    def base_context(self):
        """Create base context for tests."""
        return EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(
                estimated_move_median=5.0,
                estimated_move_max=8.0,
            ),
            microstructure=MarketMicrostructure.from_price(100.0, atr=2.0),
            risk=RiskContext(),
        )
    
    def test_protective_stop_calculation(self, optimizer, base_context):
        """Test protective stop is calculated."""
        result = optimizer.optimize(base_context, entry_price=100.0)
        
        assert result.protective_stop is not None
        assert result.protective_stop.enabled
        assert result.protective_stop.stop_price < 100.0
    
    def test_profit_targets_calculated(self, optimizer, base_context):
        """Test profit targets are calculated."""
        result = optimizer.optimize(base_context, entry_price=100.0)
        
        assert len(result.profit_targets) == 2
        assert result.profit_targets[0].target_price > 100.0
        assert result.profit_targets[1].target_price > result.profit_targets[0].target_price
    
    def test_trailing_stop_enabled(self, optimizer, base_context):
        """Test trailing stop is enabled with balanced profile."""
        result = optimizer.optimize(base_context, entry_price=100.0)
        
        assert result.trailing_stop is not None
        assert result.trailing_stop.enabled
    
    def test_time_based_exit(self, optimizer, base_context):
        """Test time-based exit is configured."""
        result = optimizer.optimize(base_context, entry_price=100.0)
        
        assert result.time_based_exit is not None
        assert result.time_based_exit.enabled
        assert result.time_based_exit.max_bars_in_trade == BALANCED_PROFILE.time_based_exit_bars
    
    def test_abort_conditions(self, optimizer, base_context):
        """Test abort conditions are generated."""
        result = optimizer.optimize(base_context, entry_price=100.0)
        
        assert len(result.abort_conditions) >= 2


class TestEntryExitOptimizer:
    """Tests for main EntryExitOptimizer coordinator."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer with balanced profile."""
        return EntryExitOptimizer(profile_type=ProfileType.BALANCED)
    
    @pytest.fixture
    def full_context(self):
        """Create complete context for tests."""
        return EEOContext(
            signal=SignalContext(
                symbol="NVDA",
                direction=SignalDirection.LONG,
                quantra_score=75.0,
                signal_id="test_signal_001",
            ),
            predictive=PredictiveContext(
                runner_prob=0.65,
                future_quality_tier=QualityTier.A,
                estimated_move_median=10.0,
                estimated_move_max=15.0,
            ),
            microstructure=MarketMicrostructure.from_price(500.0, atr=10.0),
            risk=RiskContext(
                account_equity=100000.0,
                per_trade_risk_fraction=0.01,
            ),
        )
    
    def test_build_plan_creates_valid_plan(self, optimizer, full_context):
        """Test that build_plan creates a valid plan."""
        plan = optimizer.build_plan(full_context)
        
        assert plan.is_valid()
        assert plan.symbol == "NVDA"
        assert plan.direction == SignalDirection.LONG
        assert plan.source_signal_id == "test_signal_001"
    
    def test_plan_has_entry(self, optimizer, full_context):
        """Test that plan has entry configuration."""
        plan = optimizer.build_plan(full_context)
        
        assert plan.base_entry is not None
        assert plan.base_entry.entry_price > 0
        assert plan.base_entry.entry_zone is not None
    
    def test_plan_has_protective_stop(self, optimizer, full_context):
        """Test that plan has protective stop."""
        plan = optimizer.build_plan(full_context)
        
        assert plan.has_protective_stop()
        assert plan.protective_stop.stop_price < plan.base_entry.entry_price
    
    def test_plan_has_targets(self, optimizer, full_context):
        """Test that plan has profit targets."""
        plan = optimizer.build_plan(full_context)
        
        assert len(plan.profit_targets) >= 2
        assert plan.total_target_fraction() == 1.0
    
    def test_plan_risk_reward_ratio(self, optimizer, full_context):
        """Test that plan calculates R:R ratio."""
        plan = optimizer.build_plan(full_context)
        
        rr = plan.risk_reward_ratio()
        assert rr is not None
        assert rr > 0
    
    def test_plan_metadata(self, optimizer, full_context):
        """Test that plan has correct metadata."""
        plan = optimizer.build_plan(full_context)
        
        assert plan.metadata.profile_used == "Balanced"
        assert plan.metadata.quality_label == QualityTier.A
        assert plan.metadata.runner_bucket == "high"
    
    def test_plan_aborts_when_avoid_trade_high(self, optimizer):
        """Test that plan aborts when avoid_trade_prob is high."""
        context = EEOContext(
            signal=SignalContext(symbol="AAPL"),
            predictive=PredictiveContext(avoid_trade_prob=0.8),
            microstructure=MarketMicrostructure.from_price(100.0),
            risk=RiskContext(),
        )
        
        plan = optimizer.build_plan(context)
        
        assert not plan.is_valid()
        assert len(plan.abort_conditions) > 0
    
    def test_build_plan_from_signal(self, optimizer):
        """Test convenience method for building plan from signal."""
        signal = SignalContext(
            symbol="TSLA",
            direction=SignalDirection.LONG,
            quantra_score=80.0,
        )
        
        plan = optimizer.build_plan_from_signal(
            signal_context=signal,
            current_price=250.0,
            account_equity=50000.0,
            atr=8.0,
        )
        
        assert plan.is_valid()
        assert plan.symbol == "TSLA"
    
    def test_profile_affects_risk_fraction(self):
        """Test that different profiles use different risk fractions."""
        conservative = EntryExitOptimizer(profile_type=ProfileType.CONSERVATIVE)
        aggressive = EntryExitOptimizer(profile_type=ProfileType.AGGRESSIVE_RESEARCH)
        
        assert conservative.profile.per_trade_risk_fraction == 0.005
        assert aggressive.profile.per_trade_risk_fraction == 0.02
        assert aggressive.profile.per_trade_risk_fraction > conservative.profile.per_trade_risk_fraction


class TestEntryExitPlan:
    """Tests for EntryExitPlan model."""
    
    def test_plan_to_dict(self):
        """Test plan serialization."""
        plan = EntryExitPlan(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            size_notional=5000.0,
            base_entry=BaseEntry(
                order_type=OrderTypeEEO.LIMIT,
                entry_price=100.0,
                entry_zone=EntryZone(lower=98.0, upper=102.0),
            ),
            protective_stop=ProtectiveStop(
                enabled=True,
                stop_price=96.0,
            ),
            profit_targets=[
                ProfitTarget(target_price=105.0, fraction_to_exit=0.5),
                ProfitTarget(target_price=110.0, fraction_to_exit=0.5),
            ],
        )
        
        d = plan.to_dict()
        
        assert d["symbol"] == "AAPL"
        assert d["direction"] == "LONG"
        assert d["is_valid"] == True
        assert "base_entry" in d
        assert "protective_stop" in d
        assert len(d["profit_targets"]) == 2
    
    def test_plan_is_valid(self):
        """Test plan validation."""
        valid_plan = EntryExitPlan(
            symbol="AAPL",
            size_notional=1000.0,
            base_entry=BaseEntry(
                order_type=OrderTypeEEO.MARKET,
                entry_price=100.0,
                entry_zone=EntryZone(lower=99.0, upper=101.0),
            ),
        )
        assert valid_plan.is_valid()
        
        invalid_plan = EntryExitPlan(symbol="")
        assert not invalid_plan.is_valid()
        
        no_size_plan = EntryExitPlan(symbol="AAPL", size_notional=0)
        assert not no_size_plan.is_valid()


class TestIntegrationWithExecutionEngine:
    """Tests for EEO integration with execution engine."""
    
    def test_plan_execution_imports(self):
        """Test that execution engine can import EEO components."""
        from src.quantracore_apex.broker.execution_engine import (
            ExecutionEngine,
            EEO_AVAILABLE,
        )
        
        assert EEO_AVAILABLE == True
    
    def test_execute_plan_method_exists(self):
        """Test that ExecutionEngine has execute_plan method."""
        from src.quantracore_apex.broker.execution_engine import ExecutionEngine
        
        assert hasattr(ExecutionEngine, 'execute_plan')
