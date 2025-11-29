"""
Binance WebSocket Live Data Feed.

Streams real-time klines (candlesticks) for cryptocurrency pairs.
No API key required for public market data.
"""

import json
import logging
from typing import List, Callable, Optional
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class BinanceLiveFeed:
    """
    Real-time data feed from Binance WebSocket API.
    
    Supports:
    - Kline/candlestick streams
    - Multiple timeframes (1m, 5m, 15m, 1h, etc.)
    
    Note: No API key required for public data.
    """
    
    BASE_URL = "wss://stream.binance.com:9443/stream"
    
    def __init__(self, on_message: Callable, interval: str = "1m"):
        """
        Initialize Binance live feed.
        
        Args:
            on_message: Callback function(source, symbol, dataframe) for incoming data
            interval: Kline interval (1m, 5m, 15m, 1h, etc.)
        """
        self.on_message = on_message
        self.interval = interval
        self._ws = None
        self._running = False
        self._symbols: List[str] = []
    
    def _build_url(self, symbols: List[str]) -> str:
        """Build combined stream URL for multiple symbols."""
        streams = [f"{s.lower()}@kline_{self.interval}" for s in symbols]
        return f"{self.BASE_URL}?streams={'/'.join(streams)}"
    
    def _handle_message(self, ws, message: str):
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if 'data' in data:
                stream_data = data['data']
            else:
                stream_data = data
            
            if 'k' not in stream_data:
                return
            
            kline = stream_data['k']
            symbol = kline['s']
            
            if kline['x']:
                df = pd.DataFrame([{
                    'timestamp': datetime.fromtimestamp(kline['T'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'quote_volume': float(kline['q']),
                    'trades': int(kline['n']),
                    'interval': kline['i']
                }])
                
                self.on_message('BINANCE', symbol, df)
                logger.debug(f"Binance candle closed: {symbol} @ {kline['c']}")
            
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.error(f"Binance WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"Binance WebSocket closed: {close_status_code} - {close_msg}")
        self._running = False
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        logger.info(f"Binance WebSocket connected for {len(self._symbols)} symbols")
    
    def run_forever(self, symbols: List[str]):
        """
        Start streaming data for given symbols.
        
        Args:
            symbols: List of crypto pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        """
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            raise RuntimeError("websocket-client package required for Binance feed")
        
        self._symbols = symbols
        self._running = True
        
        url = self._build_url(symbols)
        logger.info(f"Starting Binance WebSocket feed: {url}")
        
        websocket.enableTrace(False)
        
        self._ws = websocket.WebSocketApp(
            url,
            on_message=self._handle_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        try:
            self._ws.run_forever()
        except Exception as e:
            logger.error(f"Binance WebSocket error: {e}")
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
        logger.info("Binance feed stopped")


def run_symbols(symbols: List[str], callback: Callable):
    """
    Convenience function to start Binance feed.
    
    Args:
        symbols: List of crypto pairs
        callback: Callback for incoming data
    """
    feed = BinanceLiveFeed(callback)
    feed.run_forever(symbols)
