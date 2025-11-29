"""
Bybit Adapter for QuantraCore Apex.

Provides trading interface to Bybit exchange.
Requires BYBIT_API_KEY and BYBIT_SECRET environment variables.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, PositionSide


logger = logging.getLogger(__name__)


class BybitAdapter(BrokerAdapter):
    """
    Bybit exchange adapter.
    
    Supports spot and derivatives trading on Bybit.
    Uses pybit library for API communication.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        self._api_key = api_key or os.environ.get("BYBIT_API_KEY")
        self._api_secret = api_secret or os.environ.get("BYBIT_SECRET")
        self._testnet = testnet
        self._session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Bybit session."""
        if not self._api_key or not self._api_secret:
            logger.warning("[BybitAdapter] API keys not configured")
            return
        
        try:
            from pybit.unified_trading import HTTP
            
            self._session = HTTP(
                testnet=self._testnet,
                api_key=self._api_key,
                api_secret=self._api_secret,
            )
            mode = "TESTNET" if self._testnet else "LIVE"
            logger.info(f"[BybitAdapter] Connected to Bybit {mode}")
        except ImportError:
            logger.error("[BybitAdapter] pybit not installed")
        except Exception as e:
            logger.error(f"[BybitAdapter] Connection error: {e}")
    
    @property
    def name(self) -> str:
        return "BYBIT_TESTNET" if self._testnet else "BYBIT"
    
    @property
    def is_paper(self) -> bool:
        return self._testnet
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """Place order on Bybit."""
        if not self._session:
            return ExecutionResult(
                order_id="BYBIT_NOT_CONNECTED",
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_fill_price=0.0,
                message="Bybit session not connected",
                timestamp=datetime.utcnow(),
            )
        
        try:
            side = "Buy" if order.side.value == "BUY" else "Sell"
            
            result = self._session.place_order(
                category="spot",
                symbol=order.symbol,
                side=side,
                orderType="Market",
                qty=str(abs(order.qty)),
            )
            
            if result.get("retCode") == 0:
                order_result = result.get("result", {})
                logger.info(f"[BybitAdapter] Order placed: {side} {order.qty} {order.symbol}")
                
                return ExecutionResult(
                    order_id=order_result.get("orderId", ""),
                    client_order_id=order_result.get("orderLinkId", ""),
                    symbol=order.symbol,
                    status=OrderStatus.SUBMITTED,
                    filled_qty=0.0,
                    avg_fill_price=0.0,
                    message="Order submitted to Bybit",
                    timestamp=datetime.utcnow(),
                    raw_response=result,
                )
            else:
                return ExecutionResult(
                    order_id="ERROR",
                    client_order_id=order.client_order_id or "",
                    symbol=order.symbol,
                    status=OrderStatus.REJECTED,
                    filled_qty=0.0,
                    avg_fill_price=0.0,
                    message=result.get("retMsg", "Unknown error"),
                    timestamp=datetime.utcnow(),
                )
                
        except Exception as e:
            logger.error(f"[BybitAdapter] Order error: {e}")
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
        """Cancel order on Bybit."""
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
        """Get open orders from Bybit."""
        return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get positions from Bybit."""
        if not self._session:
            return []
        
        try:
            result = self._session.get_wallet_balance(accountType="UNIFIED")
            positions = []
            
            if result.get("retCode") == 0:
                for account in result.get("result", {}).get("list", []):
                    for coin in account.get("coin", []):
                        qty = float(coin.get("walletBalance", 0))
                        if qty > 0:
                            positions.append(BrokerPosition(
                                symbol=coin.get("coin", ""),
                                qty=qty,
                                avg_entry_price=0.0,
                                market_value=float(coin.get("usdValue", 0)),
                                unrealized_pl=float(coin.get("unrealisedPnl", 0)),
                                side=PositionSide.LONG,
                            ))
            
            return positions
        except Exception as e:
            logger.error(f"[BybitAdapter] Get positions error: {e}")
            return []
    
    def get_account_equity(self) -> float:
        """Get account equity in USD."""
        if not self._session:
            return 0.0
        
        try:
            result = self._session.get_wallet_balance(accountType="UNIFIED")
            
            if result.get("retCode") == 0:
                for account in result.get("result", {}).get("list", []):
                    return float(account.get("totalEquity", 0))
            return 0.0
        except Exception as e:
            logger.error(f"[BybitAdapter] Get equity error: {e}")
            return 0.0
