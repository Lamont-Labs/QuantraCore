"""
Polygon.io WebSocket Stream for Real-Time Market Data.

Provides real-time trade and quote data with:
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- Data normalization to internal formats
- Graceful shutdown handling
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    """WebSocket connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class PolygonWebSocketStream:
    """
    Polygon.io WebSocket client for real-time market data.
    
    Handles:
    1. WebSocket connection lifecycle
    2. Authentication with API key
    3. Symbol subscription management
    4. Automatic reconnection with backoff
    5. Message parsing and normalization
    6. Heartbeat/health monitoring
    
    Note: Requires 'websockets' package and POLYGON_API_KEY.
    """
    
    STOCKS_URL = "wss://socket.polygon.io/stocks"
    CRYPTO_URL = "wss://socket.polygon.io/crypto"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        on_trade: Optional[Callable[[str, float, float, datetime], None]] = None,
        on_quote: Optional[Callable[[str, float, float, datetime], None]] = None,
        on_bar: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_status: Optional[Callable[[StreamState, str], None]] = None,
        reconnect_delay_base: float = 1.0,
        reconnect_delay_max: float = 60.0,
        heartbeat_interval: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.on_bar = on_bar
        self.on_status = on_status
        
        self.reconnect_delay_base = reconnect_delay_base
        self.reconnect_delay_max = reconnect_delay_max
        self.heartbeat_interval = heartbeat_interval
        
        self._state = StreamState.DISCONNECTED
        self._websocket = None
        self._subscribed_symbols: Set[str] = set()
        self._reconnect_attempt = 0
        self._should_run = False
        self._last_message_time: Optional[datetime] = None
        
        self._metrics = {
            "messages_received": 0,
            "trades_received": 0,
            "quotes_received": 0,
            "bars_received": 0,
            "reconnections": 0,
            "errors": 0,
        }
    
    @property
    def state(self) -> StreamState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == StreamState.SUBSCRIBED
    
    def _set_state(self, state: StreamState, message: str = "") -> None:
        """Update state and notify callback."""
        self._state = state
        if self.on_status:
            self.on_status(state, message)
        logger.info(f"[PolygonStream] State: {state.value} - {message}")
    
    async def connect(self, symbols: List[str]) -> None:
        """
        Connect to Polygon WebSocket and subscribe to symbols.
        
        This is the main entry point for starting the stream.
        """
        if not self.api_key:
            self._set_state(StreamState.ERROR, "POLYGON_API_KEY not configured")
            return
        
        self._should_run = True
        self._subscribed_symbols = set(symbols)
        
        while self._should_run:
            try:
                await self._connect_and_run()
            except Exception as e:
                self._metrics["errors"] += 1
                logger.error(f"[PolygonStream] Connection error: {e}")
                
                if self._should_run:
                    await self._handle_reconnect()
    
    async def _connect_and_run(self) -> None:
        """Establish connection and run message loop."""
        try:
            import websockets
        except ImportError:
            self._set_state(StreamState.ERROR, "websockets package not installed")
            self._should_run = False
            return
        
        self._set_state(StreamState.CONNECTING, "Establishing WebSocket connection")
        
        async with websockets.connect(self.STOCKS_URL) as ws:
            self._websocket = ws
            self._reconnect_attempt = 0
            
            await self._authenticate()
            
            await self._subscribe(list(self._subscribed_symbols))
            
            await self._message_loop()
    
    async def _authenticate(self) -> None:
        """Send authentication message."""
        self._set_state(StreamState.AUTHENTICATING, "Sending API key")
        
        auth_msg = {"action": "auth", "params": self.api_key}
        await self._websocket.send(json.dumps(auth_msg))
        
        response = await self._websocket.recv()
        data = json.loads(response)
        
        if isinstance(data, list) and len(data) > 0:
            status = data[0].get("status", "")
            if status == "auth_success":
                self._set_state(StreamState.CONNECTED, "Authentication successful")
            else:
                raise Exception(f"Authentication failed: {data}")
    
    async def _subscribe(self, symbols: List[str]) -> None:
        """Subscribe to trade and quote channels for symbols."""
        if not symbols:
            return
        
        trade_channels = [f"T.{s}" for s in symbols]
        quote_channels = [f"Q.{s}" for s in symbols]
        bar_channels = [f"AM.{s}" for s in symbols]
        
        all_channels = trade_channels + quote_channels + bar_channels
        
        sub_msg = {"action": "subscribe", "params": ",".join(all_channels)}
        await self._websocket.send(json.dumps(sub_msg))
        
        self._set_state(StreamState.SUBSCRIBED, f"Subscribed to {len(symbols)} symbols")
    
    async def _message_loop(self) -> None:
        """Main message receiving loop."""
        while self._should_run:
            try:
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=self.heartbeat_interval
                )
                
                self._last_message_time = datetime.utcnow()
                self._metrics["messages_received"] += 1
                
                await self._process_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("[PolygonStream] No message received, checking connection")
            except Exception as e:
                logger.error(f"[PolygonStream] Message loop error: {e}")
                raise
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if not isinstance(data, list):
                return
            
            for item in data:
                ev = item.get("ev", "")
                
                if ev == "T":
                    await self._handle_trade(item)
                elif ev == "Q":
                    await self._handle_quote(item)
                elif ev == "AM":
                    await self._handle_bar(item)
                elif ev == "status":
                    logger.debug(f"[PolygonStream] Status: {item.get('message', '')}")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"[PolygonStream] Invalid JSON: {e}")
    
    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Process trade message."""
        self._metrics["trades_received"] += 1
        
        symbol = data.get("sym", "")
        price = data.get("p", 0.0)
        size = data.get("s", 0.0)
        ts_ms = data.get("t", 0)
        
        timestamp = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.utcnow()
        
        if self.on_trade:
            try:
                self.on_trade(symbol, price, size, timestamp)
            except Exception as e:
                logger.error(f"[PolygonStream] Trade callback error: {e}")
    
    async def _handle_quote(self, data: Dict[str, Any]) -> None:
        """Process quote message."""
        self._metrics["quotes_received"] += 1
        
        symbol = data.get("sym", "")
        bid = data.get("bp", 0.0)
        ask = data.get("ap", 0.0)
        ts_ms = data.get("t", 0)
        
        timestamp = datetime.fromtimestamp(ts_ms / 1000) if ts_ms else datetime.utcnow()
        
        if self.on_quote:
            try:
                self.on_quote(symbol, bid, ask, timestamp)
            except Exception as e:
                logger.error(f"[PolygonStream] Quote callback error: {e}")
    
    async def _handle_bar(self, data: Dict[str, Any]) -> None:
        """Process aggregated bar message."""
        self._metrics["bars_received"] += 1
        
        symbol = data.get("sym", "")
        bar_data = {
            "open": data.get("o", 0.0),
            "high": data.get("h", 0.0),
            "low": data.get("l", 0.0),
            "close": data.get("c", 0.0),
            "volume": data.get("v", 0.0),
            "vwap": data.get("vw", 0.0),
            "timestamp": datetime.fromtimestamp(data.get("s", 0) / 1000),
        }
        
        if self.on_bar:
            try:
                self.on_bar(symbol, bar_data)
            except Exception as e:
                logger.error(f"[PolygonStream] Bar callback error: {e}")
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        self._reconnect_attempt += 1
        self._metrics["reconnections"] += 1
        
        delay = min(
            self.reconnect_delay_base * (2 ** self._reconnect_attempt),
            self.reconnect_delay_max
        )
        
        self._set_state(
            StreamState.RECONNECTING,
            f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempt})"
        )
        
        await asyncio.sleep(delay)
    
    async def add_symbol(self, symbol: str) -> None:
        """Add a symbol to subscription."""
        if symbol in self._subscribed_symbols:
            return
        
        self._subscribed_symbols.add(symbol)
        
        if self.is_connected and self._websocket:
            channels = f"T.{symbol},Q.{symbol},AM.{symbol}"
            sub_msg = {"action": "subscribe", "params": channels}
            await self._websocket.send(json.dumps(sub_msg))
            logger.info(f"[PolygonStream] Added symbol: {symbol}")
    
    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from subscription."""
        if symbol not in self._subscribed_symbols:
            return
        
        self._subscribed_symbols.discard(symbol)
        
        if self.is_connected and self._websocket:
            channels = f"T.{symbol},Q.{symbol},AM.{symbol}"
            unsub_msg = {"action": "unsubscribe", "params": channels}
            await self._websocket.send(json.dumps(unsub_msg))
            logger.info(f"[PolygonStream] Removed symbol: {symbol}")
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from WebSocket."""
        self._should_run = False
        self._set_state(StreamState.DISCONNECTED, "Disconnecting")
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"[PolygonStream] Error closing WebSocket: {e}")
            finally:
                self._websocket = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream metrics."""
        return {
            **self._metrics,
            "state": self._state.value,
            "subscribed_symbols": list(self._subscribed_symbols),
            "last_message_time": self._last_message_time.isoformat() if self._last_message_time else None,
        }


class SimulatedStream:
    """
    Simulated stream for testing without Polygon connection.
    
    Generates synthetic trades for subscribed symbols.
    """
    
    def __init__(
        self,
        on_trade: Optional[Callable[[str, float, float, datetime], None]] = None,
        on_quote: Optional[Callable[[str, float, float, datetime], None]] = None,
        interval_seconds: float = 1.0,
    ):
        self.on_trade = on_trade
        self.on_quote = on_quote
        self.interval_seconds = interval_seconds
        
        self._subscribed_symbols: Set[str] = set()
        self._should_run = False
        self._prices: Dict[str, float] = {}
        self._state = StreamState.DISCONNECTED
    
    @property
    def state(self) -> StreamState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == StreamState.SUBSCRIBED
    
    async def connect(self, symbols: List[str]) -> None:
        """Start simulated stream."""
        import random
        
        self._should_run = True
        self._subscribed_symbols = set(symbols)
        self._state = StreamState.SUBSCRIBED
        
        for symbol in symbols:
            self._prices[symbol] = 100 + random.uniform(0, 400)
        
        logger.info(f"[SimulatedStream] Started with {len(symbols)} symbols")
        
        while self._should_run:
            await asyncio.sleep(self.interval_seconds)
            
            for symbol in self._subscribed_symbols:
                change = random.gauss(0, 0.002)
                self._prices[symbol] *= (1 + change)
                
                price = self._prices[symbol]
                volume = random.uniform(100, 10000)
                
                if self.on_trade:
                    self.on_trade(symbol, price, volume, datetime.utcnow())
                
                if self.on_quote:
                    spread = price * 0.001
                    self.on_quote(symbol, price - spread/2, price + spread/2, datetime.utcnow())
    
    async def disconnect(self) -> None:
        """Stop simulated stream."""
        self._should_run = False
        self._state = StreamState.DISCONNECTED
        logger.info("[SimulatedStream] Stopped")
    
    async def add_symbol(self, symbol: str) -> None:
        """Add symbol to simulation."""
        import random
        self._subscribed_symbols.add(symbol)
        if symbol not in self._prices:
            self._prices[symbol] = 100 + random.uniform(0, 400)
    
    async def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from simulation."""
        self._subscribed_symbols.discard(symbol)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "subscribed_symbols": list(self._subscribed_symbols),
            "simulated": True,
        }
