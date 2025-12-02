"""
Alpaca Real-Time WebSocket Streaming Client.

Provides live market data streaming for:
- Real-time quotes (bid/ask)
- Real-time trades (price/volume)
- Real-time bars (OHLCV aggregates)

Requires Alpaca Algo Trader Plus subscription ($99/month) for SIP data.
Free tier gets IEX data only (~2% market coverage).

WebSocket Endpoints:
- Free: wss://stream.data.alpaca.markets/v2/iex
- Paid: wss://stream.data.alpaca.markets/v2/sip
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class StreamType(Enum):
    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"


class DataFeed(Enum):
    IEX = "iex"
    SIP = "sip"


@dataclass
class RealtimeQuote:
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def spread_pct(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return 0.0


@dataclass
class RealtimeTrade:
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        return self.price * self.size


@dataclass
class RealtimeBar:
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    vwap: float = 0.0
    trade_count: int = 0
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class SymbolSnapshot:
    symbol: str
    last_price: float = 0.0
    last_size: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0
    vwap: float = 0.0
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    prev_close: float = 0.0
    last_update: Optional[datetime] = None
    trade_count: int = 0
    
    @property
    def change(self) -> float:
        if self.prev_close > 0:
            return self.last_price - self.prev_close
        return 0.0
    
    @property
    def change_pct(self) -> float:
        if self.prev_close > 0:
            return ((self.last_price - self.prev_close) / self.prev_close) * 100
        return 0.0
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    def update_from_trade(self, trade: RealtimeTrade):
        self.last_price = trade.price
        self.last_size = trade.size
        self.volume += trade.size
        self.trade_count += 1
        self.last_update = trade.timestamp
        if trade.price > self.high or self.high == 0:
            self.high = trade.price
        if trade.price < self.low or self.low == 0:
            self.low = trade.price
        if self.open == 0:
            self.open = trade.price
    
    def update_from_quote(self, quote: RealtimeQuote):
        self.bid_price = quote.bid_price
        self.ask_price = quote.ask_price
        self.last_update = quote.timestamp
    
    def update_from_bar(self, bar: RealtimeBar):
        self.open = bar.open
        self.high = bar.high
        self.low = bar.low
        self.last_price = bar.close
        self.volume = bar.volume
        self.vwap = bar.vwap
        self.trade_count = bar.trade_count
        self.last_update = bar.timestamp


class AlpacaRealtimeClient:
    """
    WebSocket client for Alpaca real-time market data.
    
    Usage:
        client = AlpacaRealtimeClient()
        
        # Check subscription tier
        if client.has_realtime_subscription:
            await client.connect()
            await client.subscribe(["AAPL", "TSLA"], quotes=True, trades=True)
            
            # Get live data
            snapshot = client.get_snapshot("AAPL")
            print(f"AAPL: ${snapshot.last_price} ({snapshot.change_pct:+.2f}%)")
        else:
            # Fall back to EOD data
            pass
    
    Subscription Tiers:
        - Free (IEX): ~2% market coverage, delayed quotes
        - Algo Trader Plus ($99/mo): Full SIP data, real-time everything
        - Elite ($100K+ account): Free SIP data included
    """
    
    IEX_URL = "wss://stream.data.alpaca.markets/v2/iex"
    SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        feed: DataFeed = DataFeed.SIP,
        auto_reconnect: bool = True
    ):
        self.api_key = api_key or os.getenv("ALPACA_PAPER_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_PAPER_API_SECRET")
        self.feed = feed
        self.auto_reconnect = auto_reconnect
        
        self._ws = None
        self._connected = False
        self._authenticated = False
        self._running = False
        self._subscribed_symbols: Set[str] = set()
        self._stream_types: Set[StreamType] = set()
        
        self._snapshots: Dict[str, SymbolSnapshot] = {}
        self._quote_handlers: List[Callable[[RealtimeQuote], None]] = []
        self._trade_handlers: List[Callable[[RealtimeTrade], None]] = []
        self._bar_handlers: List[Callable[[RealtimeBar], None]] = []
        
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        self._connection_time: Optional[datetime] = None
        
        self._has_realtime = self._check_realtime_subscription()
    
    @property
    def is_connected(self) -> bool:
        return self._connected and self._authenticated
    
    @property
    def has_realtime_subscription(self) -> bool:
        return self._has_realtime
    
    @property
    def subscribed_symbols(self) -> List[str]:
        return list(self._subscribed_symbols)
    
    @property
    def message_count(self) -> int:
        return self._message_count
    
    @property
    def connection_url(self) -> str:
        return self.SIP_URL if self.feed == DataFeed.SIP else self.IEX_URL
    
    def _check_realtime_subscription(self) -> bool:
        realtime_enabled = os.getenv("ALPACA_REALTIME_ENABLED", "false").lower()
        return realtime_enabled in ("true", "1", "yes")
    
    def get_snapshot(self, symbol: str) -> Optional[SymbolSnapshot]:
        return self._snapshots.get(symbol.upper())
    
    def get_all_snapshots(self) -> Dict[str, SymbolSnapshot]:
        return dict(self._snapshots)
    
    def on_quote(self, handler: Callable[[RealtimeQuote], None]):
        self._quote_handlers.append(handler)
    
    def on_trade(self, handler: Callable[[RealtimeTrade], None]):
        self._trade_handlers.append(handler)
    
    def on_bar(self, handler: Callable[[RealtimeBar], None]):
        self._bar_handlers.append(handler)
    
    async def connect(self) -> bool:
        if not self.api_key or not self.api_secret:
            logger.error("Alpaca credentials not configured")
            return False
        
        if not self._has_realtime:
            logger.warning(
                "Real-time data not enabled. Set ALPACA_REALTIME_ENABLED=true "
                "if you have Algo Trader Plus subscription ($99/month)"
            )
            return False
        
        try:
            import websockets
            
            logger.info(f"Connecting to Alpaca WebSocket: {self.connection_url}")
            self._ws = await websockets.connect(self.connection_url)
            self._connected = True
            self._connection_time = datetime.utcnow()
            
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self._ws.send(json.dumps(auth_msg))
            
            response = await self._ws.recv()
            data = json.loads(response)
            
            if isinstance(data, list) and len(data) > 0:
                msg = data[0]
                if msg.get("T") == "success" and msg.get("msg") == "authenticated":
                    self._authenticated = True
                    logger.info("Alpaca WebSocket authenticated successfully")
                    return True
                elif msg.get("T") == "error":
                    logger.error(f"Authentication failed: {msg.get('msg')}")
                    await self.disconnect()
                    return False
            
            connected_msg = data[0] if isinstance(data, list) else data
            if connected_msg.get("T") == "success" and connected_msg.get("msg") == "connected":
                auth_response = await self._ws.recv()
                auth_data = json.loads(auth_response)
                if isinstance(auth_data, list) and len(auth_data) > 0:
                    if auth_data[0].get("T") == "success":
                        self._authenticated = True
                        logger.info("Alpaca WebSocket authenticated successfully")
                        return True
            
            logger.error(f"Unexpected authentication response: {data}")
            return False
            
        except ImportError:
            logger.error("websockets library not installed. Run: pip install websockets")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()
        self._connected = False
        self._authenticated = False
        self._subscribed_symbols.clear()
        logger.info("Disconnected from Alpaca WebSocket")
    
    async def subscribe(
        self,
        symbols: List[str],
        quotes: bool = True,
        trades: bool = True,
        bars: bool = False
    ) -> bool:
        if not self.is_connected:
            logger.error("Not connected. Call connect() first.")
            return False
        
        symbols = [s.upper() for s in symbols]
        
        for symbol in symbols:
            if symbol not in self._snapshots:
                self._snapshots[symbol] = SymbolSnapshot(symbol=symbol)
        
        subscribe_msg: Dict[str, Any] = {"action": "subscribe"}
        
        if quotes:
            subscribe_msg["quotes"] = symbols
            self._stream_types.add(StreamType.QUOTES)
        if trades:
            subscribe_msg["trades"] = symbols
            self._stream_types.add(StreamType.TRADES)
        if bars:
            subscribe_msg["bars"] = symbols
            self._stream_types.add(StreamType.BARS)
        
        try:
            await self._ws.send(json.dumps(subscribe_msg))
            self._subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to {len(symbols)} symbols: {', '.join(symbols[:5])}...")
            return True
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        if not self.is_connected:
            return False
        
        symbols = [s.upper() for s in symbols]
        
        unsubscribe_msg: Dict[str, Any] = {"action": "unsubscribe"}
        
        if StreamType.QUOTES in self._stream_types:
            unsubscribe_msg["quotes"] = symbols
        if StreamType.TRADES in self._stream_types:
            unsubscribe_msg["trades"] = symbols
        if StreamType.BARS in self._stream_types:
            unsubscribe_msg["bars"] = symbols
        
        try:
            await self._ws.send(json.dumps(unsubscribe_msg))
            self._subscribed_symbols -= set(symbols)
            for symbol in symbols:
                self._snapshots.pop(symbol, None)
            return True
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    async def run(self):
        if not self.is_connected:
            logger.error("Not connected. Call connect() first.")
            return
        
        self._running = True
        logger.info("Starting real-time data stream...")
        
        while self._running:
            try:
                message = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Stream error: {e}")
                if self.auto_reconnect and self._running:
                    logger.info("Attempting reconnection...")
                    await asyncio.sleep(5)
                    if await self.connect():
                        symbols = list(self._subscribed_symbols)
                        if symbols:
                            await self.subscribe(
                                symbols,
                                quotes=StreamType.QUOTES in self._stream_types,
                                trades=StreamType.TRADES in self._stream_types,
                                bars=StreamType.BARS in self._stream_types
                            )
                else:
                    break
    
    async def _handle_message(self, message: str):
        try:
            data = json.loads(message)
            
            if not isinstance(data, list):
                data = [data]
            
            for item in data:
                msg_type = item.get("T")
                
                if msg_type == "q":
                    quote = self._parse_quote(item)
                    if quote:
                        self._update_snapshot_quote(quote)
                        for handler in self._quote_handlers:
                            handler(quote)
                
                elif msg_type == "t":
                    trade = self._parse_trade(item)
                    if trade:
                        self._update_snapshot_trade(trade)
                        for handler in self._trade_handlers:
                            handler(trade)
                
                elif msg_type == "b":
                    bar = self._parse_bar(item)
                    if bar:
                        self._update_snapshot_bar(bar)
                        for handler in self._bar_handlers:
                            handler(bar)
                
                elif msg_type in ("success", "subscription"):
                    logger.debug(f"Control message: {item}")
                
                elif msg_type == "error":
                    logger.error(f"Stream error: {item.get('msg')}")
            
            self._message_count += len(data)
            self._last_message_time = datetime.utcnow()
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    def _parse_quote(self, data: Dict) -> Optional[RealtimeQuote]:
        try:
            return RealtimeQuote(
                symbol=data["S"],
                bid_price=float(data.get("bp", 0)),
                bid_size=int(data.get("bs", 0)),
                ask_price=float(data.get("ap", 0)),
                ask_size=int(data.get("as", 0)),
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00"))
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse quote: {e}")
            return None
    
    def _parse_trade(self, data: Dict) -> Optional[RealtimeTrade]:
        try:
            return RealtimeTrade(
                symbol=data["S"],
                price=float(data["p"]),
                size=int(data["s"]),
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                exchange=data.get("x", ""),
                conditions=data.get("c", [])
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse trade: {e}")
            return None
    
    def _parse_bar(self, data: Dict) -> Optional[RealtimeBar]:
        try:
            return RealtimeBar(
                symbol=data["S"],
                open=float(data["o"]),
                high=float(data["h"]),
                low=float(data["l"]),
                close=float(data["c"]),
                volume=int(data["v"]),
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                vwap=float(data.get("vw", 0)),
                trade_count=int(data.get("n", 0))
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse bar: {e}")
            return None
    
    def _update_snapshot_quote(self, quote: RealtimeQuote):
        snapshot = self._snapshots.get(quote.symbol)
        if snapshot:
            snapshot.update_from_quote(quote)
    
    def _update_snapshot_trade(self, trade: RealtimeTrade):
        snapshot = self._snapshots.get(trade.symbol)
        if snapshot:
            snapshot.update_from_trade(trade)
    
    def _update_snapshot_bar(self, bar: RealtimeBar):
        snapshot = self._snapshots.get(bar.symbol)
        if snapshot:
            snapshot.update_from_bar(bar)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "authenticated": self._authenticated,
            "has_realtime_subscription": self._has_realtime,
            "feed": self.feed.value,
            "subscribed_symbols": len(self._subscribed_symbols),
            "symbols": list(self._subscribed_symbols)[:10],
            "message_count": self._message_count,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
            "connection_time": self._connection_time.isoformat() if self._connection_time else None,
            "upgrade_info": {
                "current_tier": "Algo Trader Plus" if self._has_realtime else "Free (IEX)",
                "upgrade_url": "https://app.alpaca.markets/brokerage/dashboard/overview",
                "cost": "$99/month for full SIP data",
                "benefits": [
                    "100% US market coverage",
                    "Real-time quotes and trades",
                    "Unlimited API requests",
                    "WebSocket streaming",
                    "Enable day trading and scalping"
                ]
            }
        }
