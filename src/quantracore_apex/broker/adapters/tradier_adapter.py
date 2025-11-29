"""
Tradier Adapter for QuantraCore Apex.

Provides trading interface to Tradier brokerage.
Requires TRADIER_ACCESS_TOKEN environment variable.
"""

import os
import logging
import requests
from typing import List, Optional
from datetime import datetime

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, PositionSide


logger = logging.getLogger(__name__)


class TradierAdapter(BrokerAdapter):
    """
    Tradier brokerage adapter.
    
    Uses Tradier REST API for order management.
    Supports stocks and options.
    """
    
    SANDBOX_URL = "https://sandbox.tradier.com/v1"
    LIVE_URL = "https://api.tradier.com/v1"
    
    def __init__(
        self,
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
        sandbox: bool = True,
    ):
        self._access_token = access_token or os.environ.get("TRADIER_ACCESS_TOKEN")
        self._account_id = account_id or os.environ.get("TRADIER_ACCOUNT_ID")
        self._sandbox = sandbox
        self._base_url = self.SANDBOX_URL if sandbox else self.LIVE_URL
        
        if not self._access_token:
            logger.warning("[TradierAdapter] Access token not configured")
    
    @property
    def name(self) -> str:
        return "TRADIER_SANDBOX" if self._sandbox else "TRADIER"
    
    @property
    def is_paper(self) -> bool:
        return self._sandbox
    
    def _headers(self) -> dict:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """Place order on Tradier."""
        if not self._access_token or not self._account_id:
            return ExecutionResult(
                order_id="TRADIER_NOT_CONFIGURED",
                client_order_id=order.client_order_id or "",
                symbol=order.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                avg_fill_price=0.0,
                message="Tradier not configured - set TRADIER_ACCESS_TOKEN and TRADIER_ACCOUNT_ID",
                timestamp=datetime.utcnow(),
            )
        
        try:
            url = f"{self._base_url}/accounts/{self._account_id}/orders"
            side = "buy" if order.side.value == "BUY" else "sell"
            
            data = {
                "class": "equity",
                "symbol": order.symbol,
                "side": side,
                "quantity": abs(int(order.qty)),
                "type": "market",
                "duration": "day",
            }
            
            response = requests.post(url, headers=self._headers(), data=data)
            result = response.json()
            
            if "order" in result:
                order_data = result["order"]
                logger.info(f"[TradierAdapter] Order placed: {side} {order.qty} {order.symbol}")
                
                return ExecutionResult(
                    order_id=str(order_data.get("id", "")),
                    client_order_id=order.client_order_id or "",
                    symbol=order.symbol,
                    status=OrderStatus.SUBMITTED,
                    filled_qty=0.0,
                    avg_fill_price=0.0,
                    message="Order submitted to Tradier",
                    timestamp=datetime.utcnow(),
                    raw_response=result,
                )
            else:
                error_msg = result.get("errors", {}).get("error", "Unknown error")
                return ExecutionResult(
                    order_id="ERROR",
                    client_order_id=order.client_order_id or "",
                    symbol=order.symbol,
                    status=OrderStatus.REJECTED,
                    filled_qty=0.0,
                    avg_fill_price=0.0,
                    message=str(error_msg),
                    timestamp=datetime.utcnow(),
                )
                
        except Exception as e:
            logger.error(f"[TradierAdapter] Order error: {e}")
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
        """Cancel order on Tradier."""
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
        """Get open orders from Tradier."""
        return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get positions from Tradier."""
        if not self._access_token or not self._account_id:
            return []
        
        try:
            url = f"{self._base_url}/accounts/{self._account_id}/positions"
            response = requests.get(url, headers=self._headers())
            result = response.json()
            
            positions = []
            pos_data = result.get("positions", {})
            if pos_data and "position" in pos_data:
                for pos in pos_data["position"] if isinstance(pos_data["position"], list) else [pos_data["position"]]:
                    qty = float(pos.get("quantity", 0))
                    positions.append(BrokerPosition(
                        symbol=pos.get("symbol", ""),
                        qty=qty,
                        avg_entry_price=float(pos.get("cost_basis", 0)) / qty if qty else 0,
                        market_value=float(pos.get("market_value", 0)),
                        unrealized_pl=float(pos.get("gainloss", 0)),
                        side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                    ))
            
            return positions
        except Exception as e:
            logger.error(f"[TradierAdapter] Get positions error: {e}")
            return []
    
    def get_account_equity(self) -> float:
        """Get account equity from Tradier."""
        if not self._access_token or not self._account_id:
            return 0.0
        
        try:
            url = f"{self._base_url}/accounts/{self._account_id}/balances"
            response = requests.get(url, headers=self._headers())
            result = response.json()
            
            balances = result.get("balances", {})
            return float(balances.get("total_equity", 0))
        except Exception as e:
            logger.error(f"[TradierAdapter] Get equity error: {e}")
            return 0.0
