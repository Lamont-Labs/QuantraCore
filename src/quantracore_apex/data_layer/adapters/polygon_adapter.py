"""
Polygon.io Data Adapter for QuantraCore Apex.

Fetches real OHLCV and tick data from Polygon.io API.

Hybrid Setup: Polygon for market data, Alpaca for trading execution.

Subscription Tiers:
- Free: 5 calls/min, end-of-day data only
- Starter ($99/mo): 15-min delayed data
- Developer ($249/mo): Real-time tick data, NBBO pricing
- Advanced ($500/mo): Full market events, halts

Environment Variables:
- POLYGON_API_KEY: Your Polygon.io API key
- POLYGON_TIER: Subscription tier (free, starter, developer, advanced)
"""

import os
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from polygon import RESTClient
from src.quantracore_apex.core.schemas import OhlcvBar

TIER_RATE_LIMITS = {
    "free": 12.5,        # 5 calls/min = 12.5 seconds between calls
    "starter": 0.6,      # ~100 calls/min
    "developer": 0.06,   # ~1000 calls/min
    "advanced": 0.006,   # ~10000 calls/min
}

logger = logging.getLogger(__name__)


class PolygonAdapter:
    """
    Adapter for fetching market data from Polygon.io.
    
    Primary data source in hybrid setup (Alpaca for trading, Polygon for data).
    
    Features:
    - OHLCV bars (1m to 1M timeframes)
    - Tick-by-tick data (Developer tier+)
    - Real-time quotes and NBBO (Developer tier+)
    - 15+ years historical data
    - Extended hours coverage
    
    Requires POLYGON_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None, tier: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment. Get your key at https://polygon.io/")
        
        self.tier = tier or os.getenv("POLYGON_TIER", "free").lower()
        self.rate_limit_delay = TIER_RATE_LIMITS.get(self.tier, TIER_RATE_LIMITS["free"])
        self.client = RESTClient(api_key=self.api_key)
        self.last_request_time = 0
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
        
        logger.info(f"Polygon adapter initialized (tier: {self.tier}, rate limit: {1/self.rate_limit_delay:.1f} calls/min)")
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits based on subscription tier.
        
        Uses non-blocking sleep when possible.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed
            try:
                loop = asyncio.get_running_loop()
                asyncio.create_task(asyncio.sleep(wait_time))
            except RuntimeError:
                time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def is_available(self) -> bool:
        """Check if Polygon adapter is available and configured."""
        return self.api_key is not None
    
    def get_tier_info(self) -> Dict[str, Any]:
        """Get information about current subscription tier."""
        tier_features = {
            "free": {
                "real_time": False,
                "tick_data": False,
                "historical_years": 2,
                "rate_limit": "5/min"
            },
            "starter": {
                "real_time": False,
                "tick_data": False,
                "historical_years": 10,
                "rate_limit": "100/min"
            },
            "developer": {
                "real_time": True,
                "tick_data": True,
                "historical_years": 15,
                "rate_limit": "1000/min"
            },
            "advanced": {
                "real_time": True,
                "tick_data": True,
                "historical_years": 15,
                "rate_limit": "10000/min"
            }
        }
        return tier_features.get(self.tier, tier_features["free"])
    
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
