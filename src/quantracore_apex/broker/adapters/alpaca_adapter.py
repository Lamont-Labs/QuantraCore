"""
Alpaca Paper Trading Adapter.

Implements broker integration with Alpaca's paper trading API.
LIVE trading is NOT supported - paper only.
"""

import logging
import json
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
import os
from pathlib import Path

import requests

from .base_adapter import BrokerAdapter
from ..models import OrderTicket, ExecutionResult, BrokerPosition
from ..enums import OrderStatus, OrderSide, OrderType, TimeInForce

if TYPE_CHECKING:
    from ...investor import get_trade_journal as _get_trade_journal

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
        """Place an order with Alpaca.
        
        Extended hours trading is enabled by default to support:
        - Pre-market: 4:00 AM - 9:30 AM ET
        - Regular hours: 9:30 AM - 4:00 PM ET
        - After-hours: 4:00 PM - 8:00 PM ET
        """
        body = {
            "symbol": order.symbol.upper(),
            "qty": str(order.qty),
            "side": self.SIDE_MAP[order.side],
            "type": self.ORDER_TYPE_MAP[order.order_type],
            "time_in_force": self.TIME_IN_FORCE_MAP[order.time_in_force],
            "extended_hours": getattr(order, 'extended_hours', True),
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
                    
                    account_info = self.get_account_info()
                    equity = account_info.get("equity", 100000)
                    cash = account_info.get("cash", 100000)
                    buying_power = account_info.get("buying_power", 400000)
                    
                    positions = self.get_positions()
                    total_exposure = sum(abs(float(getattr(p, 'market_value', 0) if hasattr(p, 'market_value') else p.get("market_value", 0) if hasattr(p, 'get') else 0)) for p in positions)
                    
                    meta = order.metadata
                    journal.log_comprehensive_trade(
                        symbol=order.symbol,
                        direction=direction,
                        entry_price=avg_fill_price,
                        quantity=filled_qty,
                        order_type=order.order_type.value,
                        broker=self.name,
                        order_id=response.get("id", ""),
                        company_name=getattr(meta, 'company_name', order.symbol) if meta else order.symbol,
                        sector=getattr(meta, 'sector', "Unknown") if meta else "Unknown",
                        account_equity=float(equity),
                        account_cash=float(cash),
                        buying_power=float(buying_power),
                        portfolio_value=float(equity),
                        day_pnl=0.0,
                        total_pnl=float(account_info.get("unrealized_pl", 0)),
                        open_positions_count=len(positions),
                        total_exposure=total_exposure,
                        margin_used=float(account_info.get("initial_margin", 0)),
                        quantrascore=meta.quantra_score if meta else 50.0,
                        score_bucket=getattr(meta, 'score_bucket', "neutral") if meta else "neutral",
                        confidence=getattr(meta, 'confidence', 0.5) if meta else 0.5,
                        monster_runner_score=getattr(meta, 'monster_runner_score', 0.0) if meta else 0.0,
                        monster_runner_fired=getattr(meta, 'monster_runner_fired', False) if meta else False,
                        runner_probability=getattr(meta, 'runner_probability', 0.0) if meta else 0.0,
                        avoid_trade_probability=getattr(meta, 'avoid_trade_probability', 0.0) if meta else 0.0,
                        quality_tier=getattr(meta, 'quality_tier', "C") if meta else "C",
                        entropy_state=getattr(meta, 'entropy_state', "mid") if meta else "mid",
                        suppression_state=getattr(meta, 'suppression_state', "none") if meta else "none",
                        drift_state=getattr(meta, 'drift_state', "none") if meta else "none",
                        regime=meta.regime if meta else "unknown",
                        vix_level=getattr(meta, 'vix_level', 20.0) if meta else 20.0,
                        vix_percentile=getattr(meta, 'vix_percentile', 50.0) if meta else 50.0,
                        sector_momentum=getattr(meta, 'sector_momentum', "neutral") if meta else "neutral",
                        market_breadth=getattr(meta, 'market_breadth', 0.5) if meta else 0.5,
                        spy_change_pct=getattr(meta, 'spy_change_pct', 0.0) if meta else 0.0,
                        trading_session="regular",
                        market_phase="open",
                        risk_tier=meta.risk_tier if meta else "medium",
                        risk_approved=True,
                        risk_score=getattr(meta, 'risk_score', 50.0) if meta else 50.0,
                        max_position_size=10000.0,
                        stop_loss_price=getattr(meta, 'stop_loss_price', None) if meta else None,
                        stop_loss_pct=getattr(meta, 'stop_loss_pct', None) if meta else None,
                        take_profit_price=getattr(meta, 'take_profit_price', None) if meta else None,
                        take_profit_pct=getattr(meta, 'take_profit_pct', None) if meta else None,
                        risk_reward_ratio=getattr(meta, 'risk_reward_ratio', None) if meta else None,
                        max_drawdown_allowed=0.02,
                        volatility_adjusted=True,
                        protocols_fired=meta.protocols_fired if meta else [],
                        tier_protocols=getattr(meta, 'tier_protocols', None) if meta else None,
                        monster_runner_protocols=getattr(meta, 'monster_runner_protocols', None) if meta else None,
                        omega_alerts=getattr(meta, 'omega_alerts', []) if meta else [],
                        omega_blocked=getattr(meta, 'omega_blocked', False) if meta else False,
                        consensus_direction=getattr(meta, 'consensus_direction', direction.lower()) if meta else direction.lower(),
                        protocol_confidence=getattr(meta, 'protocol_confidence', 0.5) if meta else 0.5,
                        time_in_force=order.time_in_force.value,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        slippage=0.0,
                        commission=0.0,
                        execution_time_ms=0,
                        research_notes=f"Order ID: {response.get('id')} | Strategy: {order.strategy_id or 'ApexEngine'}",
                        compliance_notes="Paper trading - no regulatory concerns",
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
    
    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_loss_price: float,
        take_profit_price: float,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> ExecutionResult:
        """
        Place a bracket order (entry + stop-loss + take-profit in one atomic order).
        
        Bracket orders are advanced order types that include:
        1. Entry order (market or limit)
        2. Stop-loss order (triggered if price falls)
        3. Take-profit order (triggered if price rises)
        
        When one exit order fills, the other is automatically canceled.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: BUY or SELL
            stop_loss_price: Stop-loss trigger price
            take_profit_price: Take-profit limit price
            limit_price: Optional limit price for entry (None = market order)
            time_in_force: Order duration (default GTC)
        
        Returns:
            ExecutionResult with bracket order details
        """
        order_type = "limit" if limit_price else "market"
        
        body = {
            "symbol": symbol.upper(),
            "qty": str(qty),
            "side": self.SIDE_MAP[side],
            "type": order_type,
            "time_in_force": self.TIME_IN_FORCE_MAP[time_in_force],
            "order_class": "bracket",
            "stop_loss": {
                "stop_price": str(round(stop_loss_price, 2)),
            },
            "take_profit": {
                "limit_price": str(round(take_profit_price, 2)),
            },
        }
        
        if limit_price:
            body["limit_price"] = str(round(limit_price, 2))
        
        try:
            response = self._make_request("POST", "/v2/orders", body)
            
            status = self.STATUS_MAP.get(response.get("status", "new"), OrderStatus.NEW)
            filled_qty = float(response.get("filled_qty", 0) or 0)
            avg_fill_price = float(response.get("filled_avg_price", 0) or 0)
            
            legs = response.get("legs", [])
            leg_ids = [leg.get("id") for leg in legs]
            
            logger.info(
                f"[ALPACA] Bracket order placed: {response.get('id')} - "
                f"{side.value} {qty} {symbol} | "
                f"Stop: ${stop_loss_price:.2f} | Target: ${take_profit_price:.2f} | "
                f"Legs: {leg_ids}"
            )
            
            return ExecutionResult(
                order_id=response.get("id", ""),
                broker=self.name,
                status=status,
                filled_qty=filled_qty,
                avg_fill_price=avg_fill_price,
                timestamp_utc=response.get("created_at", datetime.utcnow().isoformat()),
                raw_broker_payload=response,
            )
            
        except Exception as e:
            logger.error(f"Alpaca bracket order failed: {e}")
            return ExecutionResult(
                order_id="",
                broker=self.name,
                status=OrderStatus.REJECTED,
                error_message=str(e),
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
        """Get all positions from Alpaca with real-time prices."""
        try:
            response = self._make_request("GET", "/v2/positions")
            
            positions = []
            for pos in response:
                qty = float(pos.get("qty", 0))
                positions.append(BrokerPosition(
                    symbol=pos.get("symbol", ""),
                    qty=qty,
                    avg_entry_price=float(pos.get("avg_entry_price", 0)),
                    current_price=float(pos.get("current_price", 0)),
                    market_value=float(pos.get("market_value", 0)),
                    unrealized_pl=float(pos.get("unrealized_pl", 0)),
                    unrealized_plpc=float(pos.get("unrealized_plpc", 0)) * 100,
                    change_today=float(pos.get("change_today", 0)) * 100,
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
            return float(response.get("quote", {}).get("ap", 0))
        except Exception:
            return 0.0
    
    def get_orders(self, status: str = "all", limit: int = 200) -> List[Dict]:
        """
        Get orders from Alpaca.
        
        Args:
            status: Order status filter - "open", "closed", or "all"
            limit: Maximum number of orders to return
            
        Returns:
            List of order dictionaries with symbol, side, status, filled_at, filled_avg_price, etc.
        """
        try:
            params = f"?status={status}&limit={limit}"
            response = self._make_request("GET", f"/v2/orders{params}")
            
            if not isinstance(response, list):
                return []
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
