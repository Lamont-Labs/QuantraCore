"""
Null Adapter for Research Mode.

A no-op adapter that logs orders but doesn't execute them.
Used when ExecutionMode is RESEARCH.
"""

import logging
from typing import List
from datetime import datetime

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus


logger = logging.getLogger(__name__)


class NullAdapter(BrokerAdapter):
    """
    No-op adapter for research mode.
    
    All orders are logged but not executed.
    Returns simulated "success" responses for testing.
    """
    
    def __init__(self):
        self._order_counter = 0
        self._logged_orders: List[OrderTicket] = []
    
    @property
    def name(self) -> str:
        return "NULL_ADAPTER"
    
    @property
    def is_paper(self) -> bool:
        return True  # Technically no real execution
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """
        Log the order but don't execute it.
        
        Returns a simulated success response.
        """
        self._order_counter += 1
        order_id = f"NULL_{self._order_counter:08d}"
        
        self._logged_orders.append(order)
        
        logger.info(
            f"[RESEARCH MODE] Order logged (not executed): "
            f"{order.side.value} {order.qty} {order.symbol} @ {order.order_type.value}"
        )
        
        return ExecutionResult(
            order_id=order_id,
            broker=self.name,
            status=OrderStatus.NEW,
            filled_qty=0.0,
            avg_fill_price=0.0,
            timestamp_utc=datetime.utcnow().isoformat(),
            ticket_id=order.ticket_id,
            raw_broker_payload={"mode": "research", "logged_only": True},
        )
    
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Return simulated cancellation."""
        logger.info(f"[RESEARCH MODE] Cancel logged (no-op): {order_id}")
        
        return ExecutionResult(
            order_id=order_id,
            broker=self.name,
            status=OrderStatus.CANCELED,
            timestamp_utc=datetime.utcnow().isoformat(),
        )
    
    def get_open_orders(self) -> List[ExecutionResult]:
        """No open orders in research mode."""
        return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """No positions in research mode."""
        return []
    
    def get_account_equity(self) -> float:
        """Return simulated equity."""
        return 100_000.0
    
    def get_logged_orders(self) -> List[OrderTicket]:
        """Get all orders that were logged."""
        return list(self._logged_orders)
    
    def clear_logged_orders(self):
        """Clear logged orders."""
        self._logged_orders.clear()
