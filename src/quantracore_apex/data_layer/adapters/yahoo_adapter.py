"""
Yahoo Finance Data Adapter for QuantraCore Apex.

Secondary data provider for failover when primary (Polygon) is unavailable.
Uses yfinance library for data fetching.

Version: 9.0-A
"""

import os
from typing import List, Optional
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from src.quantracore_apex.core.schemas import OhlcvBar
from .base import DataAdapter


class YahooAdapter(DataAdapter):
    """
    Adapter for fetching market data from Yahoo Finance.
    
    Used as secondary/fallback provider when Polygon is unavailable.
    No API key required but has rate limits.
    """
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package not installed. Run: pip install yfinance")
    
    @property
    def name(self) -> str:
        return "yahoo"
    
    def is_available(self) -> bool:
        """Yahoo Finance is always available (no API key required)."""
        return YFINANCE_AVAILABLE
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe ("1d", "1h", "5m", etc.)
            
        Returns:
            List of OhlcvBar objects
        """
        interval_map = {
            "1d": "1d",
            "day": "1d",
            "1h": "1h",
            "hour": "1h",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1m": "1m",
        }
        
        interval = interval_map.get(timeframe, "1d")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
            )
            
            if df.empty:
                return []
            
            bars = []
            for idx, row in df.iterrows():
                bar = OhlcvBar(
                    timestamp=idx.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            raise RuntimeError(f"Yahoo Finance error for {symbol}: {e}")
    
    def fetch(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "1d",
        end_date: Optional[str] = None
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            timeframe: Bar timeframe
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of OhlcvBar objects
        """
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        start = end - timedelta(days=days)
        
        return self.fetch_ohlcv(symbol, start, end, timeframe)
    
    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get current quote/snapshot for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict with quote data or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "float_shares": info.get("floatShares"),
                "avg_volume": info.get("averageVolume"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            return None


def create_yahoo_adapter() -> Optional[YahooAdapter]:
    """
    Factory function to create Yahoo adapter if available.
    
    Returns:
        YahooAdapter or None if yfinance not installed
    """
    try:
        return YahooAdapter()
    except ImportError:
        return None
