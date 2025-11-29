"""
Tests for Risk Engine, OMS, Portfolio, and Signal Builder modules.
"""


from src.quantracore_apex.risk.engine import RiskEngine, RiskPermission
from src.quantracore_apex.broker.oms import OrderManagementSystem, OrderSide, OrderType, OrderStatus
from src.quantracore_apex.portfolio.portfolio import Portfolio
from src.quantracore_apex.signal.signal_builder import SignalBuilder, SignalDirection, SignalStrength


class TestRiskEngine:
    """Tests for RiskEngine class."""
    
    def test_engine_initialization(self):
        engine = RiskEngine()
        assert engine.denial_threshold == 0.8
        assert engine.restriction_threshold == 0.6
    
    def test_assess_low_risk(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="AAPL",
            quantra_score=75,
            regime="trending_up",
            entropy_state="stable",
            volatility_ratio=0.8,
            spread_pct=0.001,
        )
        
        assert assessment.symbol == "AAPL"
        assert assessment.risk_tier in ["low", "medium"]
        assert assessment.permission == RiskPermission.ALLOW
        assert assessment.compliance_note != ""
    
    def test_assess_high_risk_volatile(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="TSLA",
            quantra_score=35,
            regime="volatile",
            entropy_state="chaotic",
            volatility_ratio=3.5,
            spread_pct=0.02,
        )
        
        assert assessment.risk_tier in ["high", "extreme"]
        assert assessment.permission == RiskPermission.DENY
        assert len(assessment.denial_reasons) > 0
    
    def test_assess_earnings_proximity(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="AAPL",
            quantra_score=60,
            regime="range_bound",
            entropy_state="stable",
            earnings_days_away=1,
        )
        
        assert assessment.fundamental_risk == 1.0
    
    def test_omega_override_entropy(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="XYZ",
            quantra_score=70,
            regime="trending_up",
            entropy_state="EntropyState.CHAOTIC",
        )
        
        assert assessment.permission == RiskPermission.DENY
        assert assessment.override_code == "OMEGA_2_ENTROPY_OVERRIDE"
    
    def test_omega_override_drift(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="ABC",
            quantra_score=70,
            regime="trending_up",
            entropy_state="stable",
            drift_state="DriftState.CRITICAL",
        )
        
        assert assessment.permission == RiskPermission.DENY
        assert assessment.override_code == "OMEGA_3_DRIFT_OVERRIDE"
    
    def test_composite_risk_calculation(self):
        engine = RiskEngine()
        assessment = engine.assess(
            symbol="TEST",
            quantra_score=50,
            regime="range_bound",
            entropy_state="elevated",
            drift_state="moderate",
            suppression_state="mild",
            volatility_ratio=1.5,
            spread_pct=0.005,
        )
        
        assert 0.0 <= assessment.composite_risk <= 1.0
        assert 0.0 <= assessment.volatility_risk <= 1.0
        assert 0.0 <= assessment.entropy_risk <= 1.0


class TestOrderManagementSystem:
    """Tests for OrderManagementSystem class."""
    
    def test_oms_initialization(self):
        oms = OrderManagementSystem(initial_cash=50000.0)
        assert oms.cash == 50000.0
        assert oms.simulation_mode
        assert len(oms.orders) == 0
    
    def test_place_market_order(self):
        oms = OrderManagementSystem()
        order = oms.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.simulation_mode
    
    def test_place_limit_order(self):
        oms = OrderManagementSystem()
        order = oms.place_order(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=250.00,
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 250.00
    
    def test_submit_order(self):
        oms = OrderManagementSystem()
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        submitted = oms.submit_order(order.order_id)
        
        assert submitted.status == OrderStatus.SUBMITTED
    
    def test_simulate_fill(self):
        oms = OrderManagementSystem(initial_cash=100000.0)
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        oms.submit_order(order.order_id)
        
        fill = oms.simulate_fill(order.order_id, fill_price=150.0)
        
        assert fill.quantity == 100
        assert fill.price == 150.0
        assert order.status == OrderStatus.FILLED
        assert oms.get_position("AAPL") == 100
        assert oms.cash == 100000.0 - (100 * 150.0)
    
    def test_partial_fill(self):
        oms = OrderManagementSystem()
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        oms.submit_order(order.order_id)
        
        oms.simulate_fill(order.order_id, fill_price=150.0, fill_quantity=50)
        
        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 50
    
    def test_cancel_order(self):
        oms = OrderManagementSystem()
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        cancelled = oms.cancel_order(order.order_id)
        
        assert cancelled.status == OrderStatus.CANCELLED
    
    def test_reject_order(self):
        oms = OrderManagementSystem()
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        rejected = oms.reject_order(order.order_id, "Risk limit exceeded")
        
        assert rejected.status == OrderStatus.REJECTED
        assert "Risk limit" in rejected.notes
    
    def test_order_hash(self):
        oms = OrderManagementSystem()
        order = oms.place_order("AAPL", OrderSide.BUY, 100)
        
        hash1 = order.hash()
        hash2 = order.hash()
        
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_get_open_orders(self):
        oms = OrderManagementSystem()
        oms.place_order("AAPL", OrderSide.BUY, 100)
        oms.place_order("TSLA", OrderSide.SELL, 50)
        
        open_orders = oms.get_open_orders()
        assert len(open_orders) == 2
        
        aapl_orders = oms.get_open_orders(symbol="AAPL")
        assert len(aapl_orders) == 1
    
    def test_reset(self):
        oms = OrderManagementSystem(initial_cash=50000.0)
        oms.place_order("AAPL", OrderSide.BUY, 100)
        oms.reset()
        
        assert len(oms.orders) == 0
        assert oms.cash == 50000.0


class TestPortfolio:
    """Tests for Portfolio class."""
    
    def test_portfolio_initialization(self):
        portfolio = Portfolio(initial_cash=200000.0)
        assert portfolio.cash == 200000.0
        assert portfolio.initial_cash == 200000.0
        assert len(portfolio.positions) == 0
    
    def test_update_position_buy(self):
        portfolio = Portfolio(initial_cash=100000.0)
        pos = portfolio.update_position("AAPL", quantity_change=100, price=150.0)
        
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        assert portfolio.cash == 100000.0 - (100 * 150.0)
    
    def test_update_position_sell(self):
        portfolio = Portfolio(initial_cash=100000.0)
        portfolio.update_position("AAPL", quantity_change=100, price=150.0)
        pos = portfolio.update_position("AAPL", quantity_change=-50, price=160.0)
        
        assert pos.quantity == 50
        assert pos.realized_pnl == 50 * (160.0 - 150.0)
    
    def test_update_prices(self):
        portfolio = Portfolio()
        portfolio.update_position("AAPL", quantity_change=100, price=150.0)
        portfolio.update_prices({"AAPL": 160.0})
        
        pos = portfolio.get_position("AAPL")
        assert pos.market_value == 100 * 160.0
        assert pos.unrealized_pnl == 100 * (160.0 - 150.0)
    
    def test_sector_exposure(self):
        portfolio = Portfolio()
        portfolio.set_sector_map({"AAPL": "Technology", "XOM": "Energy"})
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.update_position("XOM", 50, 80.0)
        portfolio.update_prices({"AAPL": 150.0, "XOM": 80.0})
        
        exposure = portfolio.get_sector_exposure()
        assert "Technology" in exposure
        assert "Energy" in exposure
    
    def test_take_snapshot(self):
        portfolio = Portfolio(initial_cash=100000.0)
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.update_prices({"AAPL": 155.0})
        
        snapshot = portfolio.take_snapshot()
        
        assert snapshot.cash == 100000.0 - (100 * 150.0)
        assert snapshot.positions_value == 100 * 155.0
        assert snapshot.num_positions == 1
        assert snapshot.compliance_note != ""
    
    def test_get_heat_map(self):
        portfolio = Portfolio()
        portfolio.set_sector_map({"AAPL": "Technology"})
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.update_prices({"AAPL": 160.0})
        
        heat_map = portfolio.get_heat_map()
        assert "Technology" in heat_map
        assert "AAPL" in heat_map["Technology"]
    
    def test_long_short_exposure(self):
        portfolio = Portfolio()
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.update_position("TSLA", -50, 200.0)
        portfolio.update_prices({"AAPL": 150.0, "TSLA": 200.0})
        
        assert portfolio.get_long_exposure() == 15000.0
        assert portfolio.get_short_exposure() == 10000.0
        assert portfolio.get_net_exposure() == 5000.0
    
    def test_reset(self):
        portfolio = Portfolio(initial_cash=50000.0)
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.reset()
        
        assert len(portfolio.positions) == 0
        assert portfolio.cash == 50000.0


class TestSignalBuilder:
    """Tests for SignalBuilder class."""
    
    def test_builder_initialization(self):
        builder = SignalBuilder()
        assert len(builder.signals) == 0
    
    def test_build_long_signal(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="AAPL",
            quantra_score=75,
            regime="trending_up",
            risk_tier="low",
            entropy_state="stable",
            current_price=150.0,
        )
        
        assert signal.symbol == "AAPL"
        assert signal.direction == SignalDirection.LONG
        assert signal.quantra_score == 75
        assert signal.compliance_note != ""
    
    def test_build_short_signal(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="TSLA",
            quantra_score=25,
            regime="trending_down",
            risk_tier="medium",
            entropy_state="elevated",
            current_price=200.0,
        )
        
        assert signal.direction == SignalDirection.SHORT
    
    def test_build_neutral_signal(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="IBM",
            quantra_score=50,
            regime="range_bound",
            risk_tier="low",
            entropy_state="stable",
        )
        
        assert signal.direction == SignalDirection.NEUTRAL
    
    def test_signal_strength_strong(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="AAPL",
            quantra_score=80,
            regime="trending_up",
            risk_tier="low",
            entropy_state="stable",
        )
        
        assert signal.strength == SignalStrength.STRONG
    
    def test_signal_strength_weak_extreme_risk(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="XYZ",
            quantra_score=80,
            regime="trending_up",
            risk_tier="extreme",
            entropy_state="chaotic",
        )
        
        assert signal.strength == SignalStrength.WEAK
    
    def test_signal_levels_calculated(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="AAPL",
            quantra_score=75,
            regime="trending_up",
            risk_tier="low",
            entropy_state="stable",
            current_price=150.0,
            volatility_pct=2.0,
        )
        
        assert signal.entry_zone_low is not None
        assert signal.entry_zone_high is not None
        assert signal.stop_loss is not None
        assert signal.target_1 is not None
    
    def test_signal_confidence(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="AAPL",
            quantra_score=85,
            regime="trending_up",
            risk_tier="low",
            entropy_state="stable",
        )
        
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.confidence > 0.5
    
    def test_supporting_factors(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="AAPL",
            quantra_score=75,
            regime="trending_up",
            risk_tier="low",
            entropy_state="stable",
            fired_protocols=["T01", "T02", "T03", "T04", "T05"],
        )
        
        assert len(signal.supporting_factors) > 0
    
    def test_warning_factors_volatile(self):
        builder = SignalBuilder()
        signal = builder.build_signal(
            symbol="XYZ",
            quantra_score=50,
            regime="volatile",
            risk_tier="high",
            entropy_state="chaotic",
        )
        
        assert len(signal.warning_factors) > 0
    
    def test_get_filtered_signals(self):
        builder = SignalBuilder()
        builder.build_signal("AAPL", 75, "trending_up", "low", "stable")
        builder.build_signal("TSLA", 25, "volatile", "high", "chaotic")
        builder.build_signal("IBM", 50, "range_bound", "medium", "elevated")
        
        long_signals = builder.get_signals(direction=SignalDirection.LONG)
        assert len(long_signals) == 1
        
        high_conf = builder.get_signals(min_confidence=0.5)
        assert len(high_conf) <= 3
    
    def test_clear_signals(self):
        builder = SignalBuilder()
        builder.build_signal("AAPL", 75, "trending_up", "low", "stable")
        builder.clear()
        
        assert len(builder.signals) == 0
    
    def test_signal_to_dict(self):
        builder = SignalBuilder()
        signal = builder.build_signal("AAPL", 75, "trending_up", "low", "stable")
        
        d = signal.to_dict()
        assert isinstance(d, dict)
        assert "symbol" in d
        assert "direction" in d
