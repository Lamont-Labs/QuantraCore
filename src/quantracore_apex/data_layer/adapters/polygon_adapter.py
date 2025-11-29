"""
Polygon.io Data Adapter for QuantraCore Apex.

Fetches real OHLCV data from Polygon.io API.
"""

import os
import time
from typing import List, Optional
from datetime import datetime, timedelta
from polygon import RESTClient
from src.quantracore_apex.core.schemas import OhlcvBar

DEFAULT_RATE_LIMIT_DELAY = 12.5


class PolygonAdapter:
    """
    Adapter for fetching market data from Polygon.io.
    
    Requires POLYGON_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: bool = True):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        self.client = RESTClient(api_key=self.api_key)
        self.rate_limit = rate_limit
        self.last_request_time = 0
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits (5 calls/min on free tier)."""
        if self.rate_limit:
            elapsed = time.time() - self.last_request_time
            if elapsed < DEFAULT_RATE_LIMIT_DELAY:
                time.sleep(DEFAULT_RATE_LIMIT_DELAY - elapsed)
            self.last_request_time = time.time()
    
    def fetch(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "day",
        end_date: Optional[str] = None
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars from Polygon.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            timeframe: Bar timeframe ("day", "hour", "minute")
            end_date: End date in YYYY-MM-DD format (default: today)
            
        Returns:
            List of OhlcvBar objects
        """
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        start = end - timedelta(days=days)
        
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        multiplier = 1
        span = timeframe
        
        try:
            self._rate_limit_wait()
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=span,
                from_=start_str,
                to=end_str,
                adjusted=True,
                limit=50000
            )
            
            bars = []
            for agg in aggs:
                bar = OhlcvBar(
                    timestamp=datetime.fromtimestamp(agg.timestamp / 1000),
                    open=float(agg.open),
                    high=float(agg.high),
                    low=float(agg.low),
                    close=float(agg.close),
                    volume=float(agg.volume)
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            raise RuntimeError(f"Polygon API error for {symbol}: {e}")
    
    def fetch_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "day"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars for a specific date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Bar timeframe
            
        Returns:
            List of OhlcvBar objects
        """
        multiplier = 1
        span = timeframe
        
        try:
            self._rate_limit_wait()
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=span,
                from_=start_date,
                to=end_date,
                adjusted=True,
                limit=50000
            )
            
            bars = []
            for agg in aggs:
                bar = OhlcvBar(
                    timestamp=datetime.fromtimestamp(agg.timestamp / 1000),
                    open=float(agg.open),
                    high=float(agg.high),
                    low=float(agg.low),
                    close=float(agg.close),
                    volume=float(agg.volume)
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            raise RuntimeError(f"Polygon API error for {symbol}: {e}")
    
    def get_top_symbols(
        self,
        count: int = 500,
        min_volume: int = 100000
    ) -> List[str]:
        """
        Get top symbols by volume.
        
        Args:
            count: Number of symbols to return
            min_volume: Minimum average daily volume filter
            
        Returns:
            List of ticker symbols
        """
        symbols: List[str] = []
        
        try:
            tickers = self.client.list_tickers(
                market="stocks",
                limit=2000,
                type="CS"
            )
            
            for ticker in tickers:
                if len(symbols) >= count:
                    break
                try:
                    agg = self.client.get_aggs(
                        ticker=ticker.ticker,
                        multiplier=1,
                        timespan="day",
                        from_=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                        to=datetime.now().strftime("%Y-%m-%d"),
                        limit=1
                    )
                    if agg and len(agg) > 0 and agg[0].volume >= min_volume:
                        symbols.append(ticker.ticker)
                except Exception:
                    continue
                    
        except Exception as e:
            raise RuntimeError(f"Error fetching ticker list: {e}")
        
        return symbols
