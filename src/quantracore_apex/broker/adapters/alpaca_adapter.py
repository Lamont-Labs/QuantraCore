"""
Alpaca Paper Trading Adapter.

Implements broker integration with Alpaca's paper trading API.
LIVE trading is NOT supported - paper only.
"""

import logging
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, OrderSide, OrderType, TimeInForce

try:
    from ...investor import get_trade_journal
except ImportError:
    get_trade_journal = None


logger = logging.getLogger(__name__)


class AlpacaPaperAdapter(BrokerAdapter):
    """
    Alpaca Paper Trading Adapter.
    
    Connects to Alpaca's paper trading API for simulated order execution.
    All orders go to paper.alpaca, NOT live trading.
    
    Required environment variables:
    - ALPACA_PAPER_API_KEY
    - ALPACA_PAPER_API_SECRET
    """
    
    # Mapping from internal enums to Alpaca API values
    SIDE_MAP = {
        OrderSide.BUY: "buy",
        OrderSide.SELL: "sell",
    }
    
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop_limit",
    }
    
    TIME_IN_FORCE_MAP = {
        TimeInForce.DAY: "day",
        TimeInForce.GTC: "gtc",
        TimeInForce.IOC: "ioc",
        TimeInForce.FOK: "fok",
    }
    
    # Reverse mapping for status
    STATUS_MAP = {
        "new": OrderStatus.NEW,
        "pending_new": OrderStatus.PENDING,
        "accepted": OrderStatus.NEW,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "expired": OrderStatus.EXPIRED,
        "rejected": OrderStatus.REJECTED,
        "pending_cancel": OrderStatus.NEW,
        "stopped": OrderStatus.CANCELED,
        "suspended": OrderStatus.NEW,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: str = "https://paper-api.alpaca.markets",
        log_raw_dir: str = "logs/execution_broker_raw/",
    ):
        if requests is None:
            raise ImportError("requests library is required for Alpaca adapter")
        
        self._api_key = api_key or os.environ.get("ALPACA_PAPER_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("ALPACA_PAPER_API_SECRET", "")
        self._base_url = base_url.rstrip("/")
        self._log_raw_dir = Path(log_raw_dir)
        self._log_raw_dir.mkdir(parents=True, exist_ok=True)
        
        self._request_counter = 0
        
        if not self._api_key or not self._api_secret:
            logger.warning("Alpaca API credentials not configured")
    
    @property
    def name(self) -> str:
        return "ALPACA_PAPER"
    
    @property
    def is_paper(self) -> bool:
        return True  # This adapter is paper-only
    
    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_secret)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
            "Content-Type": "application/json",
        }
    
    def _log_raw_request(self, method: str, endpoint: str, request_body: Any, response: Any, status_code: int):
        """Log raw request/response for audit."""
        self._request_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"alpaca_{timestamp}_{self._request_counter:06d}.json"
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "endpoint": endpoint,
            "request_body": request_body,
            "status_code": status_code,
            "response": response,
        }
        
        try:
            with open(self._log_raw_dir / filename, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to log raw request: {e}")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Make an API request to Alpaca."""
        if not self.is_configured:
            raise RuntimeError("Alpaca API credentials not configured")
        
        url = f"{self._base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw_text": response.text}
            
            self._log_raw_request(method, endpoint, data, response_data, response.status_code)
            
            if response.status_code >= 400:
                logger.error(f"Alpaca API error: {response.status_code} - {response_data}")
                raise RuntimeError(f"Alpaca API error: {response_data}")
            
            return response_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpaca request failed: {e}")
            self._log_raw_request(method, endpoint, data, {"error": str(e)}, 0)
            raise
    
    def place_order(self, order: OrderTicket) -> ExecutionResult:
        """Place an order with Alpaca."""
        body = {
            "symbol": order.symbol.upper(),
            "qty": str(order.qty),
            "side": self.SIDE_MAP[order.side],
            "type": self.ORDER_TYPE_MAP[order.order_type],
            "time_in_force": self.TIME_IN_FORCE_MAP[order.time_in_force],
        }
        
        if order.limit_price is not None and order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            body["limit_price"] = str(order.limit_price)
        
        if order.stop_price is not None and order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            body["stop_price"] = str(order.stop_price)
        
        try:
            response = self._make_request("POST", "/v2/orders", body)
            
            status = self.STATUS_MAP.get(response.get("status", "new"), OrderStatus.NEW)
            filled_qty = float(response.get("filled_qty", 0) or 0)
            avg_fill_price = float(response.get("filled_avg_price", 0) or 0)
            
            logger.info(
                f"[ALPACA] Order placed: {response.get('id')} - "
                f"{order.side.value} {order.qty} {order.symbol} status={status.value}"
            )
            
            if get_trade_journal is not None and filled_qty > 0:
                try:
                    journal = get_trade_journal()
                    direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
                    journal.log_trade_entry(
                        symbol=order.symbol,
                        direction=direction,
                        entry_price=avg_fill_price,
                        quantity=filled_qty,
                        order_type=order.order_type.value,
                        broker=self.name,
                        signal_source=order.strategy_id or "ApexEngine",
                        quantrascore=order.metadata.quantra_score if order.metadata else 50.0,
                        regime=order.metadata.regime if order.metadata else "unknown",
                        risk_tier=order.metadata.risk_tier if order.metadata else "medium",
                        protocols_fired=order.metadata.protocols_fired if order.metadata else [],
                        notes=f"Order ID: {response.get('id')}",
                    )
                except Exception as e:
                    logger.warning(f"Failed to log trade to investor journal: {e}")
            
            return ExecutionResult(
                order_id=response.get("id", ""),
                broker=self.name,
                status=status,
                filled_qty=filled_qty,
                avg_fill_price=avg_fill_price,
                timestamp_utc=response.get("created_at", datetime.utcnow().isoformat()),
                ticket_id=order.ticket_id,
                raw_broker_payload=response,
            )
            
        except Exception as e:
            logger.error(f"Alpaca order placement failed: {e}")
            return ExecutionResult(
                order_id="",
                broker=self.name,
                status=OrderStatus.REJECTED,
                error_message=str(e),
                ticket_id=order.ticket_id,
            )
    
    def cancel_order(self, order_id: str) -> ExecutionResult:
        """Cancel an order with Alpaca."""
        try:
            self._make_request("DELETE", f"/v2/orders/{order_id}")
            
            logger.info(f"[ALPACA] Order cancelled: {order_id}")
            
            return ExecutionResult(
                order_id=order_id,
                broker=self.name,
                status=OrderStatus.CANCELED,
                timestamp_utc=datetime.utcnow().isoformat(),
            )
            
        except Exception as e:
            logger.error(f"Alpaca order cancellation failed: {e}")
            return ExecutionResult(
                order_id=order_id,
                broker=self.name,
                status=OrderStatus.REJECTED,
                error_message=str(e),
            )
    
    def get_open_orders(self) -> List[ExecutionResult]:
        """Get all open orders from Alpaca."""
        try:
            response = self._make_request("GET", "/v2/orders?status=open")
            
            results = []
            for order in response:
                status = self.STATUS_MAP.get(order.get("status", "new"), OrderStatus.NEW)
                results.append(ExecutionResult(
                    order_id=order.get("id", ""),
                    broker=self.name,
                    status=status,
                    filled_qty=float(order.get("filled_qty", 0) or 0),
                    avg_fill_price=float(order.get("filled_avg_price", 0) or 0),
                    timestamp_utc=order.get("created_at", ""),
                    raw_broker_payload=order,
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get all positions from Alpaca."""
        try:
            response = self._make_request("GET", "/v2/positions")
            
            positions = []
            for pos in response:
                qty = float(pos.get("qty", 0))
                positions.append(BrokerPosition(
                    symbol=pos.get("symbol", ""),
                    qty=qty,
                    avg_entry_price=float(pos.get("avg_entry_price", 0)),
                    market_value=float(pos.get("market_value", 0)),
                    unrealized_pl=float(pos.get("unrealized_pl", 0)),
                ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_account_equity(self) -> float:
        """Get account equity from Alpaca."""
        try:
            response = self._make_request("GET", "/v2/account")
            return float(response.get("equity", 0))
            
        except Exception as e:
            logger.error(f"Failed to get account equity: {e}")
            return 0.0
    
    def get_account_info(self) -> Dict:
        """Get full account information from Alpaca."""
        try:
            return self._make_request("GET", "/v2/account")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {"error": str(e)}
    
    def get_last_price(self, symbol: str) -> float:
        """Get last trade price for a symbol."""
        try:
            response = self._make_request("GET", f"/v2/stocks/{symbol}/quotes/latest")
            # Use ask price as proxy for last price
            return float(response.get("quote", {}).get("ap", 0))
        except Exception:
            return 0.0
