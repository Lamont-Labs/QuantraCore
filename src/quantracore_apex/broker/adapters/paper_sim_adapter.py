"""
Paper Simulation Adapter for Offline Testing.

A simple, deterministic fill simulation that doesn't require external APIs.
Used for CI tests, unit tests, and offline development.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, OrderSide, OrderType, PositionSide


logger = logging.getLogger(__name__)


class PaperSimAdapter(BrokerAdapter):
    """
    Internal paper trading simulator.
    
    Runs a simple, deterministic fill simulation:
    - MARKET orders fill immediately at last known price
    - LIMIT orders fill if price crosses limit
    - Tracks positions and P&L in memory
    - NO external HTTP calls
    """
    
    def __init__(self, initial_cash: float = 100_000.0):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: Dict[str, Dict] = {}  # symbol -> {qty, avg_price, cost_basis}
        self._orders: Dict[str, Dict] = {}  # order_id -> order data
        self._last_prices: Dict[str, float] = {}  # symbol -> last price
        self._order_counter = 0
        self._fill_counter = 0
    
    @property
    def name(self) -> str:
        return "PAPER_SIM"
    
    @property
    def is_paper(self) -> bool:
        return True
    
    def set_last_price(self, symbol: str, price: float):
        """Set the last known price for a symbol (for testing)."""
        self._last_prices[symbol.upper()] = price
    
    def get_last_price(self, symbol: str) -> float:
        """Get last known price for a symbol."""
        return self._last_prices.get(symbol.upper(), 100.0)  # Default to $100
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """
        Place and immediately fill a simulated order.
        
        For MARKET orders: fills immediately at last known price.
        For LIMIT orders: fills if current price is favorable.
        """
        self._order_counter += 1
        order_id = f"SIM_{self._order_counter:08d}"
        symbol = order.symbol.upper()
        
        # Get current price
        current_price = self.get_last_price(symbol)
        
        # Determine fill price based on order type
        fill_price = self._determine_fill_price(order, current_price)
        
        if fill_price is None:
            # Order can't be filled (e.g., limit not met)
            self._orders[order_id] = {
                "ticket": order,
                "status": OrderStatus.NEW,
                "filled_qty": 0,
                "avg_fill_price": 0,
                "created_at": datetime.utcnow(),
            }
            
            logger.info(f"[PAPER_SIM] Order pending: {order_id} - {order.side.value} {order.qty} {symbol}")
            
            return ExecutionResult(
                order_id=order_id,
                broker=self.name,
                status=OrderStatus.NEW,
                filled_qty=0.0,
                avg_fill_price=0.0,
                timestamp_utc=datetime.utcnow().isoformat(),
                ticket_id=order.ticket_id,
            )
        
        # Execute the fill
        notional = order.qty * fill_price
        
        if order.side == OrderSide.BUY:
            if notional > self._cash:
                # Insufficient funds
                logger.warning(f"[PAPER_SIM] Insufficient funds: need ${notional:.2f}, have ${self._cash:.2f}")
                return ExecutionResult(
                    order_id=order_id,
                    broker=self.name,
                    status=OrderStatus.REJECTED,
                    error_message="Insufficient funds",
                    ticket_id=order.ticket_id,
                )
            
            self._cash -= notional
            self._update_position(symbol, order.qty, fill_price)
        
        else:  # SELL
            current_qty = self._positions.get(symbol, {}).get("qty", 0)
            if order.qty > current_qty:
                logger.warning(f"[PAPER_SIM] Insufficient shares: need {order.qty}, have {current_qty}")
                return ExecutionResult(
                    order_id=order_id,
                    broker=self.name,
                    status=OrderStatus.REJECTED,
                    error_message="Insufficient shares",
                    ticket_id=order.ticket_id,
                )
            
            self._cash += notional
            self._update_position(symbol, -order.qty, fill_price)
        
        # Record the order
        self._orders[order_id] = {
            "ticket": order,
            "status": OrderStatus.FILLED,
            "filled_qty": order.qty,
            "avg_fill_price": fill_price,
            "created_at": datetime.utcnow(),
            "filled_at": datetime.utcnow(),
        }
        
        logger.info(
            f"[PAPER_SIM] Order filled: {order_id} - {order.side.value} {order.qty} {symbol} @ ${fill_price:.2f}"
        )
        
        return ExecutionResult(
            order_id=order_id,
            broker=self.name,
            status=OrderStatus.FILLED,
            filled_qty=order.qty,
            avg_fill_price=fill_price,
            timestamp_utc=datetime.utcnow().isoformat(),
            ticket_id=order.ticket_id,
            raw_broker_payload={
                "simulation": True,
                "cash_after": self._cash,
            },
        )
    
    def _determine_fill_price(self, order: OrderTicket, current_price: float) -> Optional[float]:
        """Determine the fill price based on order type."""
        if order.order_type == OrderType.MARKET:
            # Market orders fill at current price with small slippage
            slippage = 0.0001 if order.side == OrderSide.BUY else -0.0001
            return current_price * (1 + slippage)
        
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            
            # Buy limit: fill if current price <= limit
            if order.side == OrderSide.BUY and current_price <= order.limit_price:
                return order.limit_price
            
            # Sell limit: fill if current price >= limit
            if order.side == OrderSide.SELL and current_price >= order.limit_price:
                return order.limit_price
            
            return None  # Limit not met
        
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                return None
            
            # Buy stop: fill if current price >= stop
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return current_price
            
            # Sell stop: fill if current price <= stop
            if order.side == OrderSide.SELL and current_price <= order.stop_price:
                return current_price
            
            return None  # Stop not triggered
        
        elif order.order_type == OrderType.STOP_LIMIT:
            if order.stop_price is None or order.limit_price is None:
                return None
            
            # Stop-limit: stop triggers, then limit order placed
            # Simplified: check both conditions
            if order.side == OrderSide.BUY:
                if current_price >= order.stop_price and current_price <= order.limit_price:
                    return order.limit_price
            else:
                if current_price <= order.stop_price and current_price >= order.limit_price:
                    return order.limit_price
            
            return None
        
        return current_price  # Default to market
    
    def _update_position(self, symbol: str, qty_change: float, price: float):
        """Update position after a fill."""
        if symbol not in self._positions:
            self._positions[symbol] = {"qty": 0, "avg_price": 0, "cost_basis": 0}
        
        pos = self._positions[symbol]
        old_qty = pos["qty"]
        new_qty = old_qty + qty_change
        
        if qty_change > 0:  # Adding to position
            # Update average price
            old_cost = pos["cost_basis"]
            new_cost = old_cost + (qty_change * price)
            pos["cost_basis"] = new_cost
            pos["qty"] = new_qty
            pos["avg_price"] = new_cost / new_qty if new_qty > 0 else 0
        
        else:  # Reducing position
            pos["qty"] = new_qty
            if new_qty <= 0:
                pos["cost_basis"] = 0
                pos["avg_price"] = 0
            else:
                # Proportionally reduce cost basis
                pos["cost_basis"] = pos["avg_price"] * new_qty
        
        # Remove if position is flat
        if abs(pos["qty"]) < 0.0001:
            del self._positions[symbol]
    
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel an open order."""
        if order_id not in self._orders:
            return ExecutionResult(
                order_id=order_id,
                broker=self.name,
                status=OrderStatus.REJECTED,
                error_message="Order not found",
            )
        
        order_data = self._orders[order_id]
        if order_data["status"] != OrderStatus.NEW:
            return ExecutionResult(
                order_id=order_id,
                broker=self.name,
                status=OrderStatus.REJECTED,
                error_message=f"Cannot cancel order in status {order_data['status']}",
            )
        
        order_data["status"] = OrderStatus.CANCELED
        
        return ExecutionResult(
            order_id=order_id,
            broker=self.name,
            status=OrderStatus.CANCELED,
            timestamp_utc=datetime.utcnow().isoformat(),
        )
    
    def get_open_orders(self) -> List[ExecutionResult]:
        """Get all open orders."""
        results = []
        for order_id, data in self._orders.items():
            if data["status"] == OrderStatus.NEW:
                results.append(ExecutionResult(
                    order_id=order_id,
                    broker=self.name,
                    status=data["status"],
                    filled_qty=data["filled_qty"],
                    avg_fill_price=data["avg_fill_price"],
                ))
        return results
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get all current positions."""
        positions = []
        for symbol, data in self._positions.items():
            qty = data["qty"]
            if abs(qty) < 0.0001:
                continue
            
            current_price = self.get_last_price(symbol)
            market_value = qty * current_price
            unrealized_pl = market_value - data["cost_basis"]
            
            positions.append(BrokerPosition(
                symbol=symbol,
                qty=qty,
                avg_entry_price=data["avg_price"],
                market_value=market_value,
                unrealized_pl=unrealized_pl,
            ))
        
        return positions
    
    def get_account_equity(self) -> float:
        """Get current account equity."""
        position_value = sum(
            data["qty"] * self.get_last_price(sym)
            for sym, data in self._positions.items()
        )
        return self._cash + position_value
    
    def get_cash(self) -> float:
        """Get current cash balance."""
        return self._cash
    
    def reset(self):
        """Reset simulator to initial state."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._last_prices.clear()
        self._order_counter = 0
