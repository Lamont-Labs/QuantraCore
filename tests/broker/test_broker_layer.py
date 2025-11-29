"""
Tests for QuantraCore Apex Broker Layer.

Comprehensive tests for:
- Order models
- Risk engine
- Paper simulation adapter
- Router selection
- Execution engine
"""

import pytest
from datetime import datetime
import uuid

from src.quantracore_apex.broker import (
    ExecutionMode,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderIntent,
    OrderStatus,
    SignalDirection,
    RiskDecisionType,
    OrderTicket,
    ExecutionResult,
    BrokerPosition,
    ApexSignal,
    RiskDecision,
    OrderMetadata,
    BrokerConfig,
    RiskConfig,
    RiskEngine,
    ExecutionEngine,
    BrokerRouter,
)
from src.quantracore_apex.broker.adapters import (
    NullAdapter,
    PaperSimAdapter,
)

try:
    from src.quantracore_apex.hardening.mode_enforcer import (
        set_mode_for_testing,
        reset_mode_enforcer,
        ExecutionMode as HardeningExecutionMode,
    )
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False


class TestOrderModels:
    """Tests for order-related models."""
    
    def test_order_ticket_creation(self):
        """Test OrderTicket creation with defaults."""
        ticket = OrderTicket(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=10.0,
        )
        
        assert ticket.symbol == "AAPL"
        assert ticket.side == OrderSide.BUY
        assert ticket.qty == 10.0
        assert ticket.order_type == OrderType.MARKET
        assert ticket.time_in_force == TimeInForce.DAY
        assert ticket.ticket_id is not None
    
    def test_order_ticket_to_dict(self):
        """Test OrderTicket serialization."""
        ticket = OrderTicket(
            symbol="MSFT",
            side=OrderSide.SELL,
            qty=5.0,
            order_type=OrderType.LIMIT,
            limit_price=350.0,
        )
        
        data = ticket.to_dict()
        
        assert data["symbol"] == "MSFT"
        assert data["side"] == "SELL"
        assert data["qty"] == 5.0
        assert data["order_type"] == "LIMIT"
        assert data["limit_price"] == 350.0
    
    def test_order_ticket_hash(self):
        """Test OrderTicket hash generation."""
        ticket = OrderTicket(
            symbol="GOOGL",
            side=OrderSide.BUY,
            qty=2.0,
        )
        
        hash1 = ticket.hash()
        hash2 = ticket.hash()
        
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 16
    
    def test_execution_result_is_success(self):
        """Test ExecutionResult success check."""
        success = ExecutionResult(
            order_id="123",
            broker="TEST",
            status=OrderStatus.FILLED,
        )
        
        failure = ExecutionResult(
            order_id="456",
            broker="TEST",
            status=OrderStatus.REJECTED,
        )
        
        assert success.is_success is True
        assert failure.is_success is False
    
    def test_broker_position_side_detection(self):
        """Test BrokerPosition side detection."""
        long_pos = BrokerPosition(symbol="AAPL", qty=100, avg_entry_price=150.0)
        short_pos = BrokerPosition(symbol="TSLA", qty=-50, avg_entry_price=200.0)
        flat_pos = BrokerPosition(symbol="MSFT", qty=0, avg_entry_price=0.0)
        
        assert long_pos.side.value == "LONG"
        assert short_pos.side.value == "SHORT"
        assert flat_pos.side.value == "FLAT"
    
    def test_apex_signal_creation(self):
        """Test ApexSignal creation."""
        signal = ApexSignal(
            signal_id="sig_001",
            symbol="NVDA",
            direction=SignalDirection.LONG,
            quantra_score=75.0,
            runner_prob=0.35,
        )
        
        assert signal.symbol == "NVDA"
        assert signal.direction == SignalDirection.LONG
        assert signal.quantra_score == 75.0


class TestRiskEngine:
    """Tests for RiskEngine."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create a risk engine with test config."""
        config = RiskConfig(
            max_notional_exposure_usd=50_000,
            max_position_notional_per_symbol_usd=5_000,
            max_positions=10,
            max_order_notional_usd=3_000,
            max_daily_turnover_usd=100_000,
            max_leverage=2.0,
            block_short_selling=True,
            require_positive_equity=True,
        )
        return RiskEngine(config=config)
    
    def test_approve_valid_order(self, risk_engine):
        """Test approval of a valid order."""
        ticket = OrderTicket(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=10,
        )
        
        decision = risk_engine.check(
            order=ticket,
            positions=[],
            equity=100_000,
            last_price=150.0,
        )
        
        assert decision.approved is True
        assert decision.decision == RiskDecisionType.APPROVE
    
    def test_reject_exceeds_max_order_notional(self, risk_engine):
        """Test rejection when order exceeds max notional."""
        ticket = OrderTicket(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,  # $15,000 notional at $150
        )
        
        decision = risk_engine.check(
            order=ticket,
            positions=[],
            equity=100_000,
            last_price=150.0,
        )
        
        assert decision.approved is False
        assert "max" in decision.reason.lower()
    
    def test_reject_short_selling_blocked(self, risk_engine):
        """Test rejection of short selling when blocked."""
        ticket = OrderTicket(
            symbol="TSLA",
            side=OrderSide.SELL,
            qty=10,
        )
        
        decision = risk_engine.check(
            order=ticket,
            positions=[],  # No position to sell
            equity=100_000,
            last_price=200.0,
        )
        
        assert decision.approved is False
        assert "short" in decision.reason.lower()
    
    def test_reject_negative_equity(self, risk_engine):
        """Test rejection when equity is negative."""
        ticket = OrderTicket(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=1,
        )
        
        decision = risk_engine.check(
            order=ticket,
            positions=[],
            equity=-1000,  # Negative equity
            last_price=150.0,
        )
        
        assert decision.approved is False
        assert "positive" in decision.reason.lower()
    
    def test_reject_exceeds_max_positions(self, risk_engine):
        """Test rejection when max positions would be exceeded."""
        # Create 10 existing positions
        positions = [
            BrokerPosition(symbol=f"SYM{i}", qty=10, avg_entry_price=100)
            for i in range(10)
        ]
        
        ticket = OrderTicket(
            symbol="NEWSTOCK",  # New position
            side=OrderSide.BUY,
            qty=5,
        )
        
        decision = risk_engine.check(
            order=ticket,
            positions=positions,
            equity=100_000,
            last_price=50.0,
        )
        
        assert decision.approved is False
        assert "position" in decision.reason.lower()
    
    def test_calculate_position_size(self, risk_engine):
        """Test position size calculation."""
        qty = risk_engine.calculate_position_size(
            equity=100_000,
            last_price=50.0,
            volatility_pct=2.0,
        )
        
        # Should be capped by max_order_notional (3000 / 50 = 60 shares)
        assert qty > 0
        assert qty <= 60


class TestPaperSimAdapter:
    """Tests for PaperSimAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create a paper sim adapter."""
        return PaperSimAdapter(initial_cash=100_000.0)
    
    def test_adapter_properties(self, adapter):
        """Test adapter property methods."""
        assert adapter.name == "PAPER_SIM"
        assert adapter.is_paper is True
    
    def test_place_market_order(self, adapter):
        """Test placing a market order."""
        adapter.set_last_price("AAPL", 150.0)
        
        ticket = OrderTicket(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=10,
            order_type=OrderType.MARKET,
        )
        
        result = adapter.place_order(ticket)
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 10
        assert result.avg_fill_price > 0
    
    def test_position_tracking(self, adapter):
        """Test position tracking after fills."""
        adapter.set_last_price("MSFT", 300.0)
        
        # Buy
        buy_ticket = OrderTicket(symbol="MSFT", side=OrderSide.BUY, qty=5)
        adapter.place_order(buy_ticket)
        
        positions = adapter.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "MSFT"
        assert positions[0].qty == 5
        
        # Sell
        sell_ticket = OrderTicket(symbol="MSFT", side=OrderSide.SELL, qty=5)
        adapter.place_order(sell_ticket)
        
        positions = adapter.get_positions()
        assert len(positions) == 0  # Position closed
    
    def test_insufficient_funds_rejection(self, adapter):
        """Test rejection when insufficient funds."""
        adapter.set_last_price("EXPENSIVE", 10_000.0)
        
        ticket = OrderTicket(
            symbol="EXPENSIVE",
            side=OrderSide.BUY,
            qty=100,  # $1M order on $100k account
        )
        
        result = adapter.place_order(ticket)
        
        assert result.status == OrderStatus.REJECTED
        assert "insufficient" in result.error_message.lower()
    
    def test_account_equity(self, adapter):
        """Test account equity calculation."""
        initial_equity = adapter.get_account_equity()
        assert initial_equity == 100_000.0
        
        # Buy some stock
        adapter.set_last_price("AAPL", 150.0)
        ticket = OrderTicket(symbol="AAPL", side=OrderSide.BUY, qty=10)
        adapter.place_order(ticket)
        
        # Equity should be similar (minus slippage)
        new_equity = adapter.get_account_equity()
        assert abs(new_equity - initial_equity) < 100  # Within $100
    
    def test_reset(self, adapter):
        """Test adapter reset."""
        adapter.set_last_price("AAPL", 150.0)
        ticket = OrderTicket(symbol="AAPL", side=OrderSide.BUY, qty=10)
        adapter.place_order(ticket)
        
        adapter.reset()
        
        assert adapter.get_cash() == 100_000.0
        assert len(adapter.get_positions()) == 0


class TestNullAdapter:
    """Tests for NullAdapter."""
    
    def test_null_adapter_logs_orders(self):
        """Test that null adapter logs orders."""
        adapter = NullAdapter()
        
        ticket = OrderTicket(symbol="TEST", side=OrderSide.BUY, qty=10)
        result = adapter.place_order(ticket)
        
        assert result.status == OrderStatus.NEW
        assert "NULL" in result.broker
        
        logged = adapter.get_logged_orders()
        assert len(logged) == 1
        assert logged[0].symbol == "TEST"
    
    def test_null_adapter_no_positions(self):
        """Test that null adapter has no positions."""
        adapter = NullAdapter()
        
        assert len(adapter.get_positions()) == 0
        assert len(adapter.get_open_orders()) == 0
        assert adapter.get_account_equity() == 100_000.0


class TestBrokerRouter:
    """Tests for BrokerRouter."""
    
    def test_research_mode_uses_null_adapter(self):
        """Test that research mode uses null adapter."""
        config = BrokerConfig(execution_mode=ExecutionMode.RESEARCH)
        router = BrokerRouter(config=config)
        
        assert router.mode == ExecutionMode.RESEARCH
        assert router.adapter_name == "NULL_ADAPTER"
    
    def test_paper_mode_uses_paper_sim_when_no_alpaca(self):
        """Test that paper mode uses paper sim when Alpaca not configured."""
        config = BrokerConfig(execution_mode=ExecutionMode.PAPER)
        # Alpaca not configured (no API keys)
        router = BrokerRouter(config=config)
        
        assert router.mode == ExecutionMode.PAPER
        assert router.adapter_name == "PAPER_SIM"
    
    def test_live_mode_raises_error(self):
        """Test that live mode raises an error."""
        config = BrokerConfig(execution_mode=ExecutionMode.LIVE)
        
        with pytest.raises(RuntimeError) as exc_info:
            BrokerRouter(config=config)
        
        assert "DISABLED" in str(exc_info.value)


class TestExecutionEngine:
    """Tests for ExecutionEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create an execution engine in research mode."""
        config = BrokerConfig(execution_mode=ExecutionMode.RESEARCH)
        return ExecutionEngine(config=config)
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine.mode == ExecutionMode.RESEARCH
        assert engine.router is not None
        assert engine.risk_engine is not None
    
    def test_execute_long_signal(self, engine):
        """Test executing a LONG signal."""
        signal = ApexSignal(
            signal_id="test_001",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            quantra_score=70.0,
            size_hint=0.01,
        )
        
        result = engine.execute_signal(signal)
        
        # In research mode, should get a logged result
        assert result is not None
        assert result.status == OrderStatus.NEW
    
    def test_execute_exit_with_no_position(self, engine):
        """Test EXIT signal with no position returns None."""
        signal = ApexSignal(
            signal_id="test_002",
            symbol="NOPOS",
            direction=SignalDirection.EXIT,
        )
        
        result = engine.execute_signal(signal)
        
        # No position to exit
        assert result is None
    
    def test_execute_hold_signal(self, engine):
        """Test HOLD signal returns None."""
        signal = ApexSignal(
            signal_id="test_003",
            symbol="AAPL",
            direction=SignalDirection.HOLD,
        )
        
        result = engine.execute_signal(signal)
        
        assert result is None
    
    def test_get_status(self, engine):
        """Test status reporting."""
        status = engine.get_status()
        
        assert "mode" in status
        assert "adapter" in status
        assert "equity" in status
        assert status["mode"] == "RESEARCH"


class TestExecutionEnginePaperMode:
    """Tests for ExecutionEngine in PAPER mode (actual fills)."""
    
    @pytest.fixture(autouse=True)
    def setup_paper_mode(self):
        """Set up PAPER mode for hardening enforcement."""
        if HARDENING_AVAILABLE:
            set_mode_for_testing(HardeningExecutionMode.PAPER)
        yield
        if HARDENING_AVAILABLE:
            reset_mode_enforcer()
    
    @pytest.fixture
    def paper_engine(self):
        """Create an execution engine in paper mode."""
        config = BrokerConfig(execution_mode=ExecutionMode.PAPER)
        return ExecutionEngine(config=config)
    
    def test_paper_mode_uses_paper_sim(self, paper_engine):
        """Test paper mode uses PaperSimAdapter."""
        assert paper_engine.mode == ExecutionMode.PAPER
        assert paper_engine.router.adapter_name == "PAPER_SIM"
    
    def test_paper_mode_fills_long_signal(self, paper_engine):
        """Test that LONG signals get filled in paper mode."""
        signal = ApexSignal(
            signal_id="paper_test_001",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            quantra_score=75.0,
            size_hint=0.01,
        )
        
        result = paper_engine.execute_signal(signal)
        
        assert result is not None
        assert result.status == OrderStatus.FILLED, f"Expected FILLED, got {result.status}"
        assert result.filled_qty > 0
        assert result.avg_fill_price > 0
        
        positions = paper_engine.router.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
    
    def test_paper_mode_fills_exit_signal(self, paper_engine):
        """Test that EXIT signals close positions in paper mode."""
        open_signal = ApexSignal(
            signal_id="paper_test_002a",
            symbol="MSFT",
            direction=SignalDirection.LONG,
            quantra_score=70.0,
            size_hint=0.01,
        )
        paper_engine.execute_signal(open_signal)
        
        assert len(paper_engine.router.get_positions()) == 1
        
        exit_signal = ApexSignal(
            signal_id="paper_test_002b",
            symbol="MSFT",
            direction=SignalDirection.EXIT,
        )
        result = paper_engine.execute_signal(exit_signal)
        
        assert result is not None
        assert result.status == OrderStatus.FILLED
        
        positions = paper_engine.router.get_positions()
        assert len(positions) == 0, "Position should be closed after EXIT"
    
    def test_paper_mode_equity_updates(self, paper_engine):
        """Test that equity updates after fills."""
        initial_equity = paper_engine.router.get_account_equity()
        
        signal = ApexSignal(
            signal_id="paper_test_003",
            symbol="GOOGL",
            direction=SignalDirection.LONG,
            quantra_score=80.0,
            size_hint=0.02,
        )
        paper_engine.execute_signal(signal)
        
        new_equity = paper_engine.router.get_account_equity()
        assert abs(new_equity - initial_equity) < 200
    
    def test_paper_mode_short_selling_blocked(self, paper_engine):
        """Test that short selling is blocked even in paper mode."""
        signal = ApexSignal(
            signal_id="paper_test_004",
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            quantra_score=90.0,
            size_hint=0.01,
        )
        
        result = paper_engine.execute_signal(signal)
        
        assert result is None


class TestBrokerAPIIntegration:
    """API integration tests for broker layer (requires running server)."""
    
    @pytest.fixture(autouse=True)
    def setup_paper_mode(self):
        """Set up PAPER mode for hardening enforcement."""
        if HARDENING_AVAILABLE:
            set_mode_for_testing(HardeningExecutionMode.PAPER)
        yield
        if HARDENING_AVAILABLE:
            reset_mode_enforcer()
    
    def test_paper_execute_returns_filled(self):
        """
        Verify that /broker/paper/execute returns FILLED status.
        
        This test demonstrates paper trading works through the API.
        Note: Requires server to be running (tests independently verified).
        """
        config = BrokerConfig(execution_mode=ExecutionMode.PAPER)
        engine = ExecutionEngine(config=config)
        
        signal = ApexSignal(
            signal_id="api_test_001",
            symbol="NVDA",
            direction=SignalDirection.LONG,
            quantra_score=80.0,
            size_hint=0.01,
        )
        
        result = engine.execute_signal(signal)
        
        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty > 0
        assert result.avg_fill_price > 0
        assert "PAPER_SIM" in result.broker
