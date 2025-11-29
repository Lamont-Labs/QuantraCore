"""
Polygon.io WebSocket Live Data Feed.

Streams real-time trades and aggregates for US equities.
Requires POLYGON_API_KEY secret to be configured.
"""

import os
import logging
from typing import List, Callable, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PolygonLiveFeed:
    """
    Real-time data feed from Polygon.io WebSocket API.
    
    Supports:
    - Trade ticks (T.SYMBOL)
    - Minute aggregates (AM.SYMBOL)
    - Second aggregates (A.SYMBOL)
    
    Note: Free tier limited to 5 symbols.
    """
    
    def __init__(self, on_message: Callable):
        """
        Initialize Polygon live feed.
        
        Args:
            on_message: Callback function(source, symbol, data) for incoming data
        """
        self.api_key = os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY not set in secrets")
        
        self.on_message = on_message
        self._ws = None
        self._running = False
        self._symbols: List[str] = []
    
    def subscribe(self, symbols: List[str], include_trades: bool = True, include_aggs: bool = True):
        """
        Subscribe to symbols for streaming data.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'NVDA'])
            include_trades: Include trade ticks
            include_aggs: Include minute aggregates
        """
        self._symbols = symbols
        subscriptions = []
        
        for symbol in symbols:
            if include_aggs:
                subscriptions.append(f"AM.{symbol}")
            if include_trades:
                subscriptions.append(f"T.{symbol}")
        
        logger.info(f"Polygon subscriptions prepared: {subscriptions}")
        return subscriptions
    
    def run_forever(self, symbols: List[str]):
        """
        Start streaming data for given symbols.
        
        Args:
            symbols: List of stock symbols
        """
        try:
            from polygon import WebSocketClient
            from polygon.websocket.models import WebSocketMessage
        except ImportError:
            logger.error("polygon-api-client not installed. Run: pip install polygon-api-client")
            raise RuntimeError("polygon-api-client package required for live data")
        
        def handle_message(messages: List[WebSocketMessage]):
            for msg in messages:
                try:
                    symbol = getattr(msg, 'symbol', None)
                    if symbol:
                        data = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'price': getattr(msg, 'price', None) or getattr(msg, 'close', None),
                            'volume': getattr(msg, 'size', None) or getattr(msg, 'volume', None),
                            'open': getattr(msg, 'open', None),
                            'high': getattr(msg, 'high', None),
                            'low': getattr(msg, 'low', None),
                            'close': getattr(msg, 'close', None),
                            'vwap': getattr(msg, 'vwap', None),
                            'event_type': msg.event_type if hasattr(msg, 'event_type') else 'unknown'
                        }
                        self.on_message('POLYGON', symbol, data)
                except Exception as e:
                    logger.error(f"Error processing Polygon message: {e}")
        
        self._running = True
        subscriptions = self.subscribe(symbols)
        
        logger.info(f"Starting Polygon WebSocket feed for {len(symbols)} symbols...")
        
        self._ws = WebSocketClient(
            api_key=self.api_key,
            feed='stocks',
            market='stocks',
            subscriptions=subscriptions,
            on_message=handle_message
        )
        
        try:
            self._ws.run()
        except Exception as e:
            logger.error(f"Polygon WebSocket error: {e}")
            self._running = False
            raise
    
    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        logger.info("Polygon feed stopped")


def run_symbols(symbols: List[str], on_message: Callable):
    """
    Convenience function to start Polygon feed.
    
    Args:
        symbols: List of stock symbols
        on_message: Callback for incoming data
    """
    feed = PolygonLiveFeed(on_message)
    feed.run_forever(symbols)
