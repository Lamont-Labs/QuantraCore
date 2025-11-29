"""
Position Monitor for Active Trade Management.

Institutional-grade position tracking and management:
- Real-time position state tracking
- Stop-loss monitoring
- Profit target monitoring
- Trailing stop adjustments
- Time-based exit enforcement
- Broker position synchronization
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from copy import deepcopy

from .models import (
    PositionState,
    PositionStatus,
    ExitReason,
    TradeOutcome,
    OrchestratorConfig,
)

logger = logging.getLogger(__name__)


class PositionMonitor:
    """
    Monitors and manages active trading positions.
    
    Responsibilities:
    1. Track all open positions with real-time state
    2. Monitor for stop-loss hits
    3. Monitor for profit target hits
    4. Adjust trailing stops based on price movement
    5. Enforce time-based exits
    6. Sync with broker positions
    7. Generate exit signals when conditions are met
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        on_exit_triggered: Optional[Callable[[PositionState, ExitReason], None]] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.on_exit_triggered = on_exit_triggered
        
        self._positions: Dict[str, PositionState] = {}
        self._closed_positions: List[PositionState] = []
        
        self._trailing_stop_atr_multiple: float = 1.5
        self._trailing_stop_activation_pct: float = 0.01
        
        self._metrics = {
            "positions_opened": 0,
            "positions_closed": 0,
            "stops_hit": 0,
            "targets_hit": 0,
            "trailing_stops_hit": 0,
            "time_exits": 0,
            "omega_exits": 0,
        }
    
    @property
    def open_positions(self) -> Dict[str, PositionState]:
        """Get all open positions."""
        return {
            k: v for k, v in self._positions.items() 
            if v.status == PositionStatus.OPEN
        }
    
    @property
    def open_symbols(self) -> Set[str]:
        """Get symbols with open positions."""
        return {p.symbol for p in self._positions.values() if p.status == PositionStatus.OPEN}
    
    @property
    def position_count(self) -> int:
        """Get count of open positions."""
        return len(self.open_positions)
    
    @property
    def total_exposure(self) -> float:
        """Get total notional exposure of open positions."""
        return sum(p.notional_value for p in self.open_positions.values())
    
    def add_position(self, position: PositionState) -> None:
        """
        Add a new position to monitor.
        
        Called when an order is filled.
        """
        position.status = PositionStatus.OPEN
        self._positions[position.position_id] = position
        self._metrics["positions_opened"] += 1
        
        logger.info(
            f"[PositionMonitor] Added position: {position.symbol} | "
            f"Dir={position.direction} | Entry={position.entry_price:.2f} | "
            f"Stop={position.protective_stop:.2f} | T1={position.profit_target_1:.2f}"
        )
    
    def create_position_from_fill(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        protective_stop: float,
        profit_target_1: float,
        profit_target_2: float = 0.0,
        trailing_stop: float = 0.0,
        max_bars: int = 50,
        source_signal_id: str = "",
        broker_order_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PositionState:
        """Create and add a position from execution fill data."""
        position = PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.utcnow(),
            quantity=quantity,
            notional_value=entry_price * quantity,
            current_price=entry_price,
            protective_stop=protective_stop,
            trailing_stop=trailing_stop,
            profit_target_1=profit_target_1,
            profit_target_2=profit_target_2,
            max_bars_allowed=max_bars,
            source_signal_id=source_signal_id,
            broker_order_id=broker_order_id,
            metadata=metadata or {},
        )
        
        self.add_position(position)
        return position
    
    def update_price(self, symbol: str, price: float) -> List[tuple]:
        """
        Update price for all positions of a symbol.
        
        Returns list of (position, exit_reason) for positions that need to exit.
        """
        exits_triggered: List[tuple] = []
        
        for position in self._positions.values():
            if position.symbol != symbol or position.status != PositionStatus.OPEN:
                continue
            
            position.update_price(price)
            
            exit_reason = self._check_exit_conditions(position)
            if exit_reason:
                exits_triggered.append((position, exit_reason))
        
        return exits_triggered
    
    def update_all_prices(self, prices: Dict[str, float]) -> List[tuple]:
        """
        Update prices for all symbols.
        
        Returns list of (position, exit_reason) for positions that need to exit.
        """
        all_exits: List[tuple] = []
        
        for symbol, price in prices.items():
            exits = self.update_price(symbol, price)
            all_exits.extend(exits)
        
        return all_exits
    
    def increment_bars(self) -> List[tuple]:
        """
        Increment bar count for all positions.
        
        Called at each new bar. Returns positions that hit time exit.
        """
        exits_triggered: List[tuple] = []
        
        for position in self.open_positions.values():
            position.bars_in_trade += 1
            
            if position.check_time_exit():
                exits_triggered.append((position, ExitReason.TIME_EXIT))
        
        return exits_triggered
    
    def _check_exit_conditions(self, position: PositionState) -> Optional[ExitReason]:
        """
        Check all exit conditions for a position.
        
        Returns the exit reason if triggered, None otherwise.
        Priority order: Stop > Trailing > Target1 > Target2 > Time
        """
        if position.check_stop_hit():
            self._metrics["stops_hit"] += 1
            return ExitReason.STOP_LOSS
        
        self._update_trailing_stop(position)
        
        if position.check_trailing_stop_hit():
            self._metrics["trailing_stops_hit"] += 1
            return ExitReason.TRAILING_STOP
        
        if position.check_target_hit(1):
            self._metrics["targets_hit"] += 1
            return ExitReason.PROFIT_TARGET_1
        
        if position.check_target_hit(2):
            return ExitReason.PROFIT_TARGET_2
        
        if position.check_time_exit():
            self._metrics["time_exits"] += 1
            return ExitReason.TIME_EXIT
        
        return None
    
    def _update_trailing_stop(self, position: PositionState) -> None:
        """
        Update trailing stop based on price movement.
        
        Only activates after position is in profit by activation threshold.
        """
        if not self.config.enable_trailing_stops:
            return
        
        if position.current_pnl_pct < self._trailing_stop_activation_pct:
            return
        
        if position.direction == "long":
            high_since_entry = position.entry_price + position.max_favorable_excursion
            new_trailing = high_since_entry * (1 - self._trailing_stop_atr_multiple * 0.01)
            
            if new_trailing > position.trailing_stop:
                position.trailing_stop = new_trailing
                logger.debug(
                    f"[PositionMonitor] Trailing stop raised: {position.symbol} -> {new_trailing:.2f}"
                )
        else:
            low_since_entry = position.entry_price - position.max_favorable_excursion
            new_trailing = low_since_entry * (1 + self._trailing_stop_atr_multiple * 0.01)
            
            if position.trailing_stop == 0 or new_trailing < position.trailing_stop:
                position.trailing_stop = new_trailing
                logger.debug(
                    f"[PositionMonitor] Trailing stop lowered: {position.symbol} -> {new_trailing:.2f}"
                )
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: ExitReason,
    ) -> Optional[TradeOutcome]:
        """
        Close a position and generate trade outcome.
        
        Returns TradeOutcome for feedback loop.
        """
        if position_id not in self._positions:
            logger.warning(f"[PositionMonitor] Position not found: {position_id}")
            return None
        
        position = self._positions[position_id]
        position.status = PositionStatus.CLOSED
        position.current_price = exit_price
        position.update_price(exit_price)
        
        if position.direction == "long":
            realized_pnl = (exit_price - position.entry_price) * position.quantity
            realized_pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            realized_pnl = (position.entry_price - exit_price) * position.quantity
            realized_pnl_pct = (position.entry_price - exit_price) / position.entry_price
        
        outcome = TradeOutcome(
            trade_id=position.position_id,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            entry_time=position.entry_time,
            exit_price=exit_price,
            exit_time=datetime.utcnow(),
            quantity=position.quantity,
            notional_value=position.notional_value,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            bars_held=position.bars_in_trade,
            exit_reason=exit_reason,
            original_quantrascore=position.metadata.get("quantrascore", 0.0),
            original_quality_tier=position.metadata.get("quality_tier", ""),
            original_risk_tier=position.metadata.get("risk_tier", ""),
            original_runner_prob=position.metadata.get("runner_prob", 0.0),
            source_signal_id=position.source_signal_id,
            position_id=position.position_id,
            hit_first_target=exit_reason in [ExitReason.PROFIT_TARGET_1, ExitReason.PROFIT_TARGET_2],
            metadata=position.metadata,
        )
        
        self._closed_positions.append(deepcopy(position))
        self._metrics["positions_closed"] += 1
        
        logger.info(
            f"[PositionMonitor] Closed position: {position.symbol} | "
            f"Reason={exit_reason.value} | PnL=${realized_pnl:.2f} ({realized_pnl_pct*100:.2f}%)"
        )
        
        if self.on_exit_triggered:
            self.on_exit_triggered(position, exit_reason)
        
        return outcome
    
    def force_close_all(self, reason: ExitReason = ExitReason.OMEGA_OVERRIDE) -> List[TradeOutcome]:
        """
        Force close all open positions.
        
        Used for emergency shutdown or Omega directive enforcement.
        """
        outcomes: List[TradeOutcome] = []
        
        for position in list(self.open_positions.values()):
            outcome = self.close_position(
                position.position_id,
                position.current_price,
                reason,
            )
            if outcome:
                outcomes.append(outcome)
                self._metrics["omega_exits"] += 1
        
        logger.warning(
            f"[PositionMonitor] Force closed {len(outcomes)} positions: {reason.value}"
        )
        
        return outcomes
    
    def sync_with_broker(self, broker_positions: Dict[str, Dict[str, Any]]) -> None:
        """
        Synchronize position state with broker.
        
        Handles discrepancies between local state and broker state.
        """
        local_symbols = self.open_symbols
        broker_symbols = set(broker_positions.keys())
        
        missing_locally = broker_symbols - local_symbols
        if missing_locally:
            logger.warning(
                f"[PositionMonitor] Broker has positions not tracked locally: {missing_locally}"
            )
        
        missing_at_broker = local_symbols - broker_symbols
        if missing_at_broker:
            logger.warning(
                f"[PositionMonitor] Local positions not found at broker: {missing_at_broker}"
            )
            for symbol in missing_at_broker:
                for position in self.open_positions.values():
                    if position.symbol == symbol:
                        logger.info(f"[PositionMonitor] Marking {symbol} as closed (broker sync)")
                        position.status = PositionStatus.CLOSED
        
        for symbol, broker_pos in broker_positions.items():
            for position in self.open_positions.values():
                if position.symbol == symbol:
                    broker_qty = broker_pos.get("quantity", 0)
                    if abs(broker_qty - position.quantity) > 0.01:
                        logger.warning(
                            f"[PositionMonitor] Quantity mismatch for {symbol}: "
                            f"local={position.quantity}, broker={broker_qty}"
                        )
                        position.quantity = broker_qty
    
    def get_position(self, position_id: str) -> Optional[PositionState]:
        """Get position by ID."""
        return self._positions.get(position_id)
    
    def get_position_for_symbol(self, symbol: str) -> Optional[PositionState]:
        """Get open position for a symbol."""
        for position in self.open_positions.values():
            if position.symbol == symbol:
                return position
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get position monitor metrics."""
        return {
            **self._metrics,
            "current_open_positions": self.position_count,
            "total_exposure": self.total_exposure,
            "open_symbols": list(self.open_symbols),
        }
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all positions as dictionaries."""
        return [p.to_dict() for p in self._positions.values()]
    
    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """Get closed positions as dictionaries."""
        return [p.to_dict() for p in self._closed_positions]
