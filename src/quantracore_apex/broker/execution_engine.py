"""
Execution Engine for QuantraCore Apex Broker Layer.

Maps Apex signals and EEO plans to orders, applies risk checks, and routes to broker.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from .router import BrokerRouter
from .risk_engine import RiskEngine
from .config import BrokerConfig, load_broker_config
from .models import (
    OrderTicket,
    ExecutionResult,
    ApexSignal,
    RiskDecision,
    OrderMetadata,
)
from .enums import (
    ExecutionMode,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderIntent,
    OrderStatus,
    SignalDirection,
)
from .execution_logger import ExecutionLogger

try:
    from ..eeo_engine import (
        EntryExitPlan,
        EntryExitOptimizer,
        SignalContext,
        PredictiveContext,
        MarketMicrostructure,
        RiskContext,
        EEOContext,
        SignalDirection as EEOSignalDirection,
        OrderTypeEEO,
        ProfileType,
    )
    EEO_AVAILABLE = True
except ImportError:
    EEO_AVAILABLE = False

try:
    from ..hardening.mode_enforcer import get_mode_enforcer, ModeViolationError
    from ..hardening.kill_switch import get_kill_switch_manager
    from ..hardening.incident_logger import get_incident_logger, IncidentClass, IncidentSeverity
    HARDENING_AVAILABLE = True
except ImportError:
    HARDENING_AVAILABLE = False


logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Execution Engine for QuantraCore Apex.
    
    Pipeline:
    1. Receive ApexSignal
    2. Build OrderTicket from signal
    3. Run RiskEngine.check(order)
    4. If REJECT → log and return None
    5. If APPROVE → BrokerRouter.place_order(ticket)
    6. Log ExecutionResult
    """
    
    def __init__(
        self,
        config: Optional[BrokerConfig] = None,
        config_path: str = "config/broker.yaml",
    ):
        self._config = config or load_broker_config(config_path)
        self._router = BrokerRouter(config=self._config)
        self._risk_engine = RiskEngine(config=self._config.risk)
        self._logger = ExecutionLogger(config=self._config.logging)
        
        logger.info(
            f"[ExecutionEngine] Initialized in {self._config.execution_mode.value} mode "
            f"with {self._router.adapter_name} adapter"
        )
    
    @property
    def mode(self) -> ExecutionMode:
        """Get current execution mode."""
        return self._config.execution_mode
    
    @property
    def router(self) -> BrokerRouter:
        """Get the broker router."""
        return self._router
    
    @property
    def risk_engine(self) -> RiskEngine:
        """Get the risk engine."""
        return self._risk_engine
    
    def execute_signal(self, signal: ApexSignal) -> Optional[ExecutionResult]:
        """
        Execute a single Apex signal.
        
        Args:
            signal: ApexSignal from the engine
            
        Returns:
            ExecutionResult if order was placed, None if rejected/skipped
        """
        logger.info(f"[ExecutionEngine] Processing signal: {signal.signal_id} - {signal.symbol}")
        
        if HARDENING_AVAILABLE:
            kill_switch = get_kill_switch_manager()
            allowed, reason = kill_switch.check_order_allowed()
            if not allowed:
                logger.warning(f"[ExecutionEngine] Signal blocked by kill switch: {reason}")
                return ExecutionResult(
                    order_id="",
                    broker=self._router.adapter_name,
                    status=OrderStatus.REJECTED,
                    error_message=f"Kill switch engaged: {reason}",
                    ticket_id=signal.signal_id,
                )
            
            mode_enforcer = get_mode_enforcer()
            is_paper = self._config.execution_mode == ExecutionMode.PAPER
            is_live = self._config.execution_mode == ExecutionMode.LIVE
            
            if is_paper and not mode_enforcer.check_permission("place_paper_order"):
                return ExecutionResult(
                    order_id="",
                    broker=self._router.adapter_name,
                    status=OrderStatus.REJECTED,
                    error_message="Paper orders not permitted in current mode",
                    ticket_id=signal.signal_id,
                )
            
            if is_live and not mode_enforcer.check_permission("place_live_order"):
                return ExecutionResult(
                    order_id="",
                    broker=self._router.adapter_name,
                    status=OrderStatus.REJECTED,
                    error_message="Live orders not permitted",
                    ticket_id=signal.signal_id,
                )
        
        # Get current state
        positions = self._router.get_positions()
        equity = self._router.get_account_equity()
        last_price = self._router.get_last_price(signal.symbol)
        
        # Build order ticket from signal
        ticket = self._build_order_ticket(signal, positions, equity, last_price)
        
        if ticket is None:
            logger.info(f"[ExecutionEngine] Signal {signal.signal_id} produced no order (e.g., EXIT with no position)")
            return None
        
        # Run risk checks
        risk_decision = self._risk_engine.check(
            order=ticket,
            positions=positions,
            equity=equity,
            last_price=last_price,
        )
        
        # Log the risk decision
        self._logger.log_risk_decision(ticket, risk_decision, equity)
        
        if not risk_decision.approved:
            logger.warning(
                f"[ExecutionEngine] Order REJECTED by risk engine: {risk_decision.reason}"
            )
            return ExecutionResult(
                order_id="",
                broker=self._router.adapter_name,
                status=OrderStatus.REJECTED,
                error_message=f"Risk rejected: {risk_decision.reason}",
                ticket_id=ticket.ticket_id,
            )
        
        # Place the order
        result = self._router.place_order(ticket)
        
        # Log the execution
        self._logger.log_execution(ticket, result, risk_decision)
        
        return result
    
    def execute_signals_batch(self, signals: List[ApexSignal]) -> List[ExecutionResult]:
        """
        Execute multiple signals in batch.
        
        Args:
            signals: List of ApexSignals to process
            
        Returns:
            List of ExecutionResults (one per signal that produced an order)
        """
        results = []
        
        for signal in signals:
            result = self.execute_signal(signal)
            if result is not None:
                results.append(result)
        
        return results
    
    def execute_plan(self, plan: "EntryExitPlan") -> Dict[str, Any]:
        """
        Execute an Entry/Exit Plan from the EEO Engine.
        
        Converts the plan into multiple OrderTickets:
        - Entry orders (single or scaled)
        - Stop loss order (if enabled)
        - Profit target orders (if any)
        
        Args:
            plan: EntryExitPlan from EEO Engine
            
        Returns:
            Dictionary with execution results for each order type
        """
        if not EEO_AVAILABLE:
            return {
                "success": False,
                "error": "EEO Engine not available",
                "entry_result": None,
                "stop_result": None,
                "target_results": [],
            }
        
        if HARDENING_AVAILABLE:
            kill_switch = get_kill_switch_manager()
            allowed, reason = kill_switch.check_order_allowed()
            if not allowed:
                logger.warning(f"[ExecutionEngine] Order blocked by kill switch: {reason}")
                return {
                    "success": False,
                    "error": f"Kill switch engaged: {reason}",
                    "entry_result": None,
                    "stop_result": None,
                    "target_results": [],
                }
            
            mode_enforcer = get_mode_enforcer()
            is_paper = self._config.execution_mode == ExecutionMode.PAPER
            is_live = self._config.execution_mode == ExecutionMode.LIVE
            
            if is_paper and not mode_enforcer.check_permission("place_paper_order"):
                logger.warning("[ExecutionEngine] Paper orders not permitted in current mode")
                return {
                    "success": False,
                    "error": "Paper orders not permitted in current mode",
                    "entry_result": None,
                    "stop_result": None,
                    "target_results": [],
                }
            
            if is_live and not mode_enforcer.check_permission("place_live_order"):
                logger.warning("[ExecutionEngine] Live orders not permitted in current mode")
                return {
                    "success": False,
                    "error": "Live orders not permitted - institutional sign-off required",
                    "entry_result": None,
                    "stop_result": None,
                    "target_results": [],
                }
        
        if not plan.is_valid():
            return {
                "success": False,
                "error": "Invalid plan - missing required components",
                "entry_result": None,
                "stop_result": None,
                "target_results": [],
            }
        
        logger.info(f"[ExecutionEngine] Executing plan {plan.plan_id} for {plan.symbol}")
        
        results = {
            "success": True,
            "plan_id": plan.plan_id,
            "symbol": plan.symbol,
            "entry_result": None,
            "stop_result": None,
            "target_results": [],
            "orders_placed": 0,
        }
        
        entry_ticket = self._build_entry_ticket_from_plan(plan)
        if entry_ticket:
            entry_result = self._execute_and_log_ticket(entry_ticket)
            results["entry_result"] = entry_result.to_dict() if entry_result else None
            if entry_result and entry_result.status == OrderStatus.FILLED:
                results["orders_placed"] += 1
        
        return results
    
    def _build_entry_ticket_from_plan(self, plan: "EntryExitPlan") -> Optional[OrderTicket]:
        """Build entry OrderTicket from EntryExitPlan."""
        if not plan.base_entry:
            return None
        
        base = plan.base_entry
        
        order_type_map = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP": OrderType.STOP,
            "STOP_LIMIT": OrderType.STOP_LIMIT,
        }
        order_type = order_type_map.get(base.order_type.value, OrderType.MARKET)
        
        qty = plan.size_notional / base.entry_price if base.entry_price > 0 else 0
        qty = round(qty, 2)
        
        if qty <= 0:
            return None
        
        if plan.direction.value == "LONG":
            side = OrderSide.BUY
            intent = OrderIntent.OPEN_LONG
        else:
            side = OrderSide.SELL
            intent = OrderIntent.OPEN_SHORT
        
        return OrderTicket(
            symbol=plan.symbol,
            side=side,
            qty=qty,
            order_type=order_type,
            limit_price=base.entry_price if order_type == OrderType.LIMIT else None,
            time_in_force=TimeInForce.DAY,
            intent=intent,
            source_signal_id=plan.source_signal_id,
            strategy_id=f"eeo_{plan.metadata.profile_used}",
            metadata=OrderMetadata(
                quantra_score=0.0,
                runner_prob=0.0,
                regime=plan.metadata.runner_bucket,
            ),
        )
    
    def _execute_and_log_ticket(self, ticket: OrderTicket) -> Optional[ExecutionResult]:
        """Execute a single ticket with risk checks and logging."""
        positions = self._router.get_positions()
        equity = self._router.get_account_equity()
        last_price = self._router.get_last_price(ticket.symbol)
        
        risk_decision = self._risk_engine.check(
            order=ticket,
            positions=positions,
            equity=equity,
            last_price=last_price,
        )
        
        self._logger.log_risk_decision(ticket, risk_decision, equity)
        
        if not risk_decision.approved:
            logger.warning(f"[ExecutionEngine] Order REJECTED: {risk_decision.reason}")
            return ExecutionResult(
                order_id="",
                broker=self._router.adapter_name,
                status=OrderStatus.REJECTED,
                error_message=f"Risk rejected: {risk_decision.reason}",
                ticket_id=ticket.ticket_id,
            )
        
        result = self._router.place_order(ticket)
        self._logger.log_execution(ticket, result, risk_decision)
        
        return result
    
    def _build_order_ticket(
        self,
        signal: ApexSignal,
        positions: List,
        equity: float,
        last_price: float,
    ) -> Optional[OrderTicket]:
        """
        Build an OrderTicket from an ApexSignal.
        
        Args:
            signal: The signal to convert
            positions: Current positions
            equity: Current equity
            last_price: Last known price
            
        Returns:
            OrderTicket or None if no order should be placed
        """
        symbol = signal.symbol.upper()
        
        # Use default price if not available (for research mode)
        effective_price = last_price if last_price > 0 else 100.0
        
        # Find existing position
        current_position = None
        for pos in positions:
            if pos.symbol.upper() == symbol:
                current_position = pos
                break
        
        current_qty = current_position.qty if current_position else 0
        
        # Determine order parameters based on signal direction
        if signal.direction == SignalDirection.LONG:
            if current_qty > 0:
                # Already long, skip
                logger.debug(f"Already long {symbol}, skipping LONG signal")
                return None
            
            # Calculate size
            if signal.size_hint:
                qty = self._calculate_qty_from_hint(signal.size_hint, equity, effective_price)
            else:
                qty = self._risk_engine.calculate_position_size(equity, effective_price)
            
            if qty <= 0:
                return None
            
            return OrderTicket(
                symbol=symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                intent=OrderIntent.OPEN_LONG,
                source_signal_id=signal.signal_id,
                strategy_id=signal.metadata.get("strategy_id", "default"),
                metadata=OrderMetadata(
                    quantra_score=signal.quantra_score,
                    runner_prob=signal.runner_prob,
                    estimated_move=signal.estimated_move,
                    regime=signal.regime,
                    volatility_band=signal.volatility_band,
                ),
            )
        
        elif signal.direction == SignalDirection.EXIT:
            if current_qty <= 0:
                # No position to exit
                logger.debug(f"No position in {symbol} to exit")
                return None
            
            return OrderTicket(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=abs(current_qty),
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY,
                intent=OrderIntent.CLOSE_LONG,
                source_signal_id=signal.signal_id,
                strategy_id=signal.metadata.get("strategy_id", "default"),
                metadata=OrderMetadata(
                    quantra_score=signal.quantra_score,
                    runner_prob=signal.runner_prob,
                    regime=signal.regime,
                ),
            )
        
        elif signal.direction == SignalDirection.SHORT:
            # Short selling blocked by default
            if self._config.risk.block_short_selling:
                logger.debug("Short selling is blocked")
                return None
            
            # Would implement short logic here
            return None
        
        elif signal.direction == SignalDirection.HOLD:
            # No action for HOLD
            return None
        
        return None
    
    def _calculate_qty_from_hint(
        self,
        size_hint: float,
        equity: float,
        last_price: float,
    ) -> float:
        """Calculate quantity from a size hint (fraction of capital)."""
        if last_price <= 0:
            return 0
        
        notional = equity * size_hint
        notional = min(notional, self._config.risk.max_order_notional_usd)
        
        return max(0, round(notional / last_price, 2))
    
    def get_status(self) -> dict:
        """Get execution engine status."""
        return {
            "mode": self._config.execution_mode.value,
            "adapter": self._router.adapter_name,
            "is_paper": self._router.is_paper,
            "equity": self._router.get_account_equity(),
            "position_count": len(self._router.get_positions()),
            "open_order_count": len(self._router.get_open_orders()),
            "daily_turnover": self._risk_engine.get_daily_turnover(),
            "config": self._config.to_dict(),
        }
