"""
Abstract Base Adapter for Broker Integrations.

Defines the interface that all broker adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import List

from ..models import OrderTicket, ExecutionResult, BrokerPosition


class BrokerAdapter(ABC):
    """
    Abstract base class for broker adapters.
    
    All broker implementations (Alpaca, paper sim, etc.) must
    inherit from this class and implement all abstract methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name (e.g., 'ALPACA_PAPER', 'PAPER_SIM')."""
        pass
    
    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Return True if this is a paper/simulation adapter."""
        pass
    
    @abstractmethod
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """
        Place an order with the broker.
        
        Args:
            order: OrderTicket containing order details
            
        Returns:
            ExecutionResult with order status and fill info
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """
        Cancel an open order.
        
        Args:
            order_id: Broker-assigned order ID
            
        Returns:
            ExecutionResult with cancellation status
        """
        pass
    
    @abstractmethod
    def get_open_orders(self) -> List[ExecutionResult]:
        """
        Get all open orders.
        
        Returns:
            List of ExecutionResult for open orders
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """
        Get all current positions.
        
        Returns:
            List of BrokerPosition
        """
        pass
    
    @abstractmethod
    def get_account_equity(self) -> float:
        """
        Get current account equity.
        
        Returns:
            Account equity in USD
        """
        pass
    
    def sync_state(self) -> None:
        """
        Optional: Refresh internal state from broker.
        
        Override this method if the adapter caches state that
        needs periodic synchronization.
        """
        pass
    
    def get_position(self, symbol: str) -> BrokerPosition:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            BrokerPosition (with qty=0 if no position)
        """
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        
        # Return flat position if not found
        return BrokerPosition(
            symbol=symbol,
            qty=0.0,
            avg_entry_price=0.0,
            market_value=0.0,
            unrealized_pl=0.0,
        )
    
    def get_last_price(self, symbol: str) -> float:
        """
        Get last known price for a symbol.
        
        Override this method if the adapter can provide price data.
        Returns 0.0 by default.
        """
        return 0.0
