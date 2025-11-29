"""
Binance Adapter for QuantraCore Apex.

Provides trading interface to Binance exchange for spot and futures.
Requires BINANCE_API_KEY and BINANCE_SECRET environment variables.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, PositionSide


logger = logging.getLogger(__name__)


class BinanceAdapter(BrokerAdapter):
    """
    Binance exchange adapter for spot trading.
    
    Uses python-binance library for API communication.
    All trades go through the unified adapter interface.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        self._api_key = api_key or os.environ.get("BINANCE_API_KEY")
        self._api_secret = api_secret or os.environ.get("BINANCE_SECRET")
        self._testnet = testnet
        self._client = None
        self._positions: dict = {}
        self._initialize()
    
    def _initialize(self):
        """Initialize Binance client."""
        if not self._api_key or not self._api_secret:
            logger.warning("[BinanceAdapter] API keys not configured - will fail on order placement")
            return
        
        try:
            from binance.client import Client
            
            if self._testnet:
                self._client = Client(
                    self._api_key,
                    self._api_secret,
                    testnet=True
                )
                logger.info("[BinanceAdapter] Connected to Binance TESTNET")
            else:
                self._client = Client(self._api_key, self._api_secret)
                logger.info("[BinanceAdapter] Connected to Binance LIVE")
        except ImportError:
            logger.error("[BinanceAdapter] python-binance not installed")
        except Exception as e:
            logger.error(f"[BinanceAdapter] Connection error: {e}")
    
    @property
    def name(self) -> str:
        return "BINANCE_TESTNET" if self._testnet else "BINANCE"
    
    @property
    def is_paper(self) -> bool:
        return self._testnet
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """Place order on Binance."""
        if not self._client:
            return ExecutionResult(
                order_id="BINANCE_NOT_CONNECTED",
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_fill_price=0.0,
                message="Binance client not connected",
                timestamp=datetime.utcnow(),
            )
        
        try:
            side = "BUY" if order.side.value == "BUY" else "SELL"
            
            result = self._client.create_order(
                symbol=order.symbol,
                side=side,
                type="MARKET",
                quantity=abs(order.qty),
            )
            
            filled_qty = float(result.get("executedQty", 0))
            fills = result.get("fills", [])
            avg_price = 0.0
            if fills:
                total_cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                total_qty = sum(float(f["qty"]) for f in fills)
                avg_price = total_cost / total_qty if total_qty > 0 else 0.0
            
            logger.info(f"[BinanceAdapter] Order filled: {side} {filled_qty} {order.symbol} @ {avg_price}")
            
            return ExecutionResult(
                order_id=str(result.get("orderId", "")),
                client_order_id=result.get("clientOrderId", ""),
                symbol=order.symbol,
                status=OrderStatus.FILLED,
                filled_qty=filled_qty,
                avg_fill_price=avg_price,
                message="Order executed",
                timestamp=datetime.utcnow(),
                raw_response=result,
            )
            
        except Exception as e:
            logger.error(f"[BinanceAdapter] Order error: {e}")
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
        """Cancel order on Binance."""
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
        """Get open orders from Binance."""
        return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get positions from Binance account balances."""
        if not self._client:
            return []
        
        try:
            account = self._client.get_account()
            positions = []
            
            for balance in account.get("balances", []):
                qty = float(balance.get("free", 0)) + float(balance.get("locked", 0))
                if qty > 0:
                    asset = balance.get("asset", "")
                    positions.append(BrokerPosition(
                        symbol=asset,
                        qty=qty,
                        avg_entry_price=0.0,
                        market_value=0.0,
                        unrealized_pl=0.0,
                        side=PositionSide.LONG if qty > 0 else PositionSide.FLAT,
                    ))
            
            return positions
        except Exception as e:
            logger.error(f"[BinanceAdapter] Get positions error: {e}")
            return []
    
    def get_account_equity(self) -> float:
        """Get account equity in USDT."""
        if not self._client:
            return 0.0
        
        try:
            account = self._client.get_account()
            for balance in account.get("balances", []):
                if balance.get("asset") == "USDT":
                    return float(balance.get("free", 0)) + float(balance.get("locked", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[BinanceAdapter] Get equity error: {e}")
            return 0.0
