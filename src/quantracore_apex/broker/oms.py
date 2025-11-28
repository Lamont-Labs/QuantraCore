"""
Order Management System (OMS) for QuantraCore Apex.

IMPORTANT: This is a SIMULATION-ONLY implementation.
No real broker connections or live trading is enabled.
All orders are paper-only for research and backtesting.

This module provides:
- Order creation and management
- Position tracking
- Fill simulation
- Order state machine
"""

from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import hashlib
import json


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Order(BaseModel):
    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    
    risk_approved: bool = False
    risk_override_code: Optional[str] = None
    
    notes: str = ""
    simulation_mode: bool = True
    
    def hash(self) -> str:
        """Generate deterministic hash for order audit."""
        data = {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class Fill(BaseModel):
    fill_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    commission: float = 0.0


class OrderManagementSystem:
    """
    Simulation-only Order Management System.
    
    WARNING: This system does NOT connect to any real broker.
    All orders are simulated for research and backtesting purposes only.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.positions: Dict[str, float] = {}
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self._simulation_mode = True
        
    @property
    def simulation_mode(self) -> bool:
        """Always returns True - live trading is not supported."""
        return True
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        risk_approved: bool = False,
        risk_override_code: Optional[str] = None,
    ) -> Order:
        """
        Place a simulated order.
        
        Args:
            symbol: Ticker symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Market, limit, stop, or stop_limit
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            risk_approved: Whether risk engine approved this order
            risk_override_code: Override code if risk was bypassed
        
        Returns:
            Order object with pending status
        
        Note: This is simulation only. No real orders are placed.
        """
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            risk_approved=risk_approved,
            risk_override_code=risk_override_code,
            simulation_mode=True,
        )
        
        self.orders[order.order_id] = order
        return order
    
    def submit_order(self, order_id: str) -> Order:
        """Submit a pending order for execution."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            raise ValueError(f"Order {order_id} is not pending")
        
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.utcnow()
        
        return order
    
    def simulate_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: Optional[float] = None,
        commission: float = 0.0,
    ) -> Fill:
        """
        Simulate a fill for a submitted order.
        
        Args:
            order_id: Order to fill
            fill_price: Execution price
            fill_quantity: Quantity to fill (defaults to remaining)
            commission: Commission for this fill
        
        Returns:
            Fill object
        """
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL]:
            raise ValueError(f"Order {order_id} cannot be filled in status {order.status}")
        
        remaining = order.quantity - order.filled_quantity
        qty = min(fill_quantity or remaining, remaining)
        
        if qty <= 0:
            raise ValueError("Fill quantity must be positive")
        
        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=fill_price,
            commission=commission,
        )
        self.fills.append(fill)
        
        if order.avg_fill_price is None:
            order.avg_fill_price = fill_price
        else:
            total_filled = order.filled_quantity + qty
            order.avg_fill_price = (
                (order.avg_fill_price * order.filled_quantity + fill_price * qty) / total_filled
            )
        
        order.filled_quantity += qty
        order.updated_at = datetime.utcnow()
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
        else:
            order.status = OrderStatus.PARTIAL
        
        self._update_position(order.symbol, order.side, qty, fill_price, commission)
        
        return fill
    
    def cancel_order(self, order_id: str) -> Order:
        """Cancel a pending or submitted order."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"Order {order_id} cannot be cancelled in status {order.status}")
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()
        
        return order
    
    def reject_order(self, order_id: str, reason: str = "") -> Order:
        """Reject an order (used by risk engine)."""
        if order_id not in self.orders:
            raise ValueError(f"Order {order_id} not found")
        
        order = self.orders[order_id]
        order.status = OrderStatus.REJECTED
        order.notes = reason
        order.updated_at = datetime.utcnow()
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol."""
        open_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
        orders = [o for o in self.orders.values() if o.status in open_statuses]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    def get_position(self, symbol: str) -> float:
        """Get current position for a symbol."""
        return self.positions.get(symbol, 0.0)
    
    def get_all_positions(self) -> Dict[str, float]:
        """Get all positions."""
        return dict(self.positions)
    
    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        commission: float,
    ):
        """Update position and cash after a fill."""
        current_pos = self.positions.get(symbol, 0.0)
        
        if side == OrderSide.BUY:
            self.positions[symbol] = current_pos + quantity
            self.cash -= (quantity * price + commission)
        else:
            self.positions[symbol] = current_pos - quantity
            self.cash += (quantity * price - commission)
        
        if abs(self.positions[symbol]) < 0.0001:
            del self.positions[symbol]
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity given current prices."""
        position_value = sum(
            qty * prices.get(sym, 0)
            for sym, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L given current prices."""
        return self.get_equity(prices) - self.initial_cash
    
    def reset(self):
        """Reset OMS to initial state."""
        self.orders.clear()
        self.fills.clear()
        self.positions.clear()
        self.cash = self.initial_cash
