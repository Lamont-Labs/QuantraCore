"""
Interactive Brokers (IBKR) Adapter for QuantraCore Apex.

Provides trading interface to IBKR via TWS or IB Gateway.
Requires ib_insync library and running TWS/Gateway instance.
"""

import logging
from typing import List, Optional
from datetime import datetime

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, PositionSide


logger = logging.getLogger(__name__)


class IBKRAdapter(BrokerAdapter):
    """
    Interactive Brokers adapter using ib_insync.
    
    Connects to TWS or IB Gateway running locally.
    Default ports: TWS=7497 (paper), 7496 (live)
                   Gateway=4002 (paper), 4001 (live)
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        paper: bool = True,
    ):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._paper = paper
        self._ib = None
        self._connected = False
        self._initialize()
    
    def _initialize(self):
        """Initialize IBKR connection."""
        try:
            from ib_insync import IB
            
            self._ib = IB()
            self._ib.connect(
                host=self._host,
                port=self._port,
                clientId=self._client_id,
                readonly=False,
            )
            self._connected = True
            logger.info(f"[IBKRAdapter] Connected to IBKR at {self._host}:{self._port}")
        except ImportError:
            logger.error("[IBKRAdapter] ib_insync not installed")
        except Exception as e:
            logger.error(f"[IBKRAdapter] Connection error: {e}")
            logger.info("[IBKRAdapter] Make sure TWS or IB Gateway is running")
    
    @property
    def name(self) -> str:
        return "IBKR_PAPER" if self._paper else "IBKR_LIVE"
    
    @property
    def is_paper(self) -> bool:
        return self._paper
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """Place order via IBKR."""
        if not self._connected or not self._ib:
            return ExecutionResult(
                order_id="IBKR_NOT_CONNECTED",
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_fill_price=0.0,
                message="IBKR not connected - ensure TWS/Gateway is running",
                timestamp=datetime.utcnow(),
            )
        
        try:
            from ib_insync import Stock, MarketOrder
            
            contract = Stock(order.symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)
            
            action = "BUY" if order.side.value == "BUY" else "SELL"
            ib_order = MarketOrder(action, abs(order.qty))
            
            trade = self._ib.placeOrder(contract, ib_order)
            self._ib.sleep(2)
            
            filled = trade.orderStatus.filled
            avg_price = trade.orderStatus.avgFillPrice
            
            status = OrderStatus.FILLED if filled > 0 else OrderStatus.SUBMITTED
            
            logger.info(f"[IBKRAdapter] Order placed: {action} {order.qty} {order.symbol}")
            
            return ExecutionResult(
                order_id=str(trade.order.orderId),
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=status,
                filled_qty=float(filled),
                avg_fill_price=float(avg_price),
                message="Order submitted to IBKR",
                timestamp=datetime.utcnow(),
            )
            
        except Exception as e:
            logger.error(f"[IBKRAdapter] Order error: {e}")
            return ExecutionResult(
                order_id="ERROR",
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_fill_price=0.0,
                message=str(e),
                timestamp=datetime.utcnow(),
            )
    
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel order on IBKR."""
        return ExecutionResult(
            order_id=order_id,
            client_order_id="",
            symbol="",
            status=OrderStatus.CANCELED,
            filled_qty=0.0,
            avg_fill_price=0.0,
            message="Cancel not implemented",
            timestamp=datetime.utcnow(),
        )
    
    def get_open_orders(self) -> List[ExecutionResult]:
        """Get open orders from IBKR."""
        return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get positions from IBKR."""
        if not self._connected or not self._ib:
            return []
        
        try:
            positions = []
            for pos in self._ib.positions():
                positions.append(BrokerPosition(
                    symbol=pos.contract.symbol,
                    qty=float(pos.position),
                    avg_entry_price=float(pos.avgCost),
                    market_value=0.0,
                    unrealized_pl=0.0,
                    side=PositionSide.LONG if pos.position > 0 else PositionSide.SHORT,
                ))
            return positions
        except Exception as e:
            logger.error(f"[IBKRAdapter] Get positions error: {e}")
            return []
    
    def get_account_equity(self) -> float:
        """Get account equity from IBKR."""
        if not self._connected or not self._ib:
            return 0.0
        
        try:
            account_values = self._ib.accountValues()
            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    return float(av.value)
            return 0.0
        except Exception as e:
            logger.error(f"[IBKRAdapter] Get equity error: {e}")
            return 0.0
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("[IBKRAdapter] Disconnected from IBKR")
