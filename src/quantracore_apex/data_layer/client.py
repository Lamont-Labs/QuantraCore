"""
Data Client with Multi-Provider Failover for QuantraCore Apex.

Provides a unified interface for fetching market data with automatic
failover between multiple data providers.

Version: 9.0-A
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.quantracore_apex.core.schemas import OhlcvBar

logger = logging.getLogger(__name__)


class DataStatus(Enum):
    """Status of data fetch operation."""
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    API_ERROR = "api_error"
    INSUFFICIENT_DATA = "insufficient_data"
    CACHED = "cached"
    UNAVAILABLE = "unavailable"


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    symbol: str
    bars: List[OhlcvBar]
    status: DataStatus
    provider: str
    cached: bool = False
    error: Optional[str] = None
    fetch_time_ms: float = 0.0


class DataClient:
    """
    Unified data client with multi-provider failover.
    
    Supports:
    - Primary provider (Polygon)
    - Secondary provider (Yahoo Finance)
    - Tertiary provider (CSV bundle/cache)
    - Automatic failover on errors
    - Caching to local storage
    """
    
    def __init__(
        self,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
        rate_limit_delay: float = 0.5,
    ):
        self.use_cache = use_cache
        self.cache_dir = cache_dir or Path("data/cache")
        self.rate_limit_delay = rate_limit_delay
        
        self._polygon = None
        self._yahoo = None
        self._csv_bundle = None
        
        self._init_adapters()
    
    def _init_adapters(self):
        """Initialize available data adapters."""
        try:
            from .adapters.polygon_adapter import PolygonAdapter
            self._polygon = PolygonAdapter(rate_limit=True)
            logger.info("Polygon adapter initialized")
        except Exception as e:
            logger.warning(f"Polygon adapter not available: {e}")
            self._polygon = None
        
        try:
            from .adapters.yahoo_adapter import YahooAdapter
            self._yahoo = YahooAdapter()
            logger.info("Yahoo adapter initialized")
        except Exception as e:
            logger.debug(f"Yahoo adapter not available: {e}")
            self._yahoo = None
        
        try:
            from .adapters.csv_bundle_adapter import CSVBundleAdapter
            self._csv_bundle = CSVBundleAdapter(cache_dir=self.cache_dir)
            logger.info("CSV bundle adapter initialized")
        except Exception as e:
            logger.debug(f"CSV bundle adapter not available: {e}")
            self._csv_bundle = None
    
    def _try_cache(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[FetchResult]:
        """Try to load from cache first."""
        if not self.use_cache or self._csv_bundle is None:
            return None
        
        try:
            if self._csv_bundle.has_symbol(symbol, timeframe):
                bars = self._csv_bundle.fetch_ohlcv(symbol, start, end, timeframe)
                if len(bars) > 0:
                    return FetchResult(
                        symbol=symbol,
                        bars=bars,
                        status=DataStatus.CACHED,
                        provider="cache",
                        cached=True,
                    )
        except Exception as e:
            logger.debug(f"Cache miss for {symbol}: {e}")
        
        return None
    
    def _try_polygon(
        self,
        symbol: str,
        days: int,
        timeframe: str
    ) -> Optional[FetchResult]:
        """Try to fetch from Polygon."""
        if self._polygon is None:
            return None
        
        start_time = time.time()
        try:
            bars = self._polygon.fetch(symbol, days=days, timeframe=timeframe)
            fetch_time = (time.time() - start_time) * 1000
            
            if len(bars) > 0:
                if self.use_cache and self._csv_bundle:
                    try:
                        self._csv_bundle.save_to_cache(symbol, bars, timeframe, "polygon")
                    except Exception as e:
                        logger.warning(f"Failed to cache {symbol}: {e}")
                
                return FetchResult(
                    symbol=symbol,
                    bars=bars,
                    status=DataStatus.SUCCESS,
                    provider="polygon",
                    fetch_time_ms=fetch_time,
                )
            else:
                return FetchResult(
                    symbol=symbol,
                    bars=[],
                    status=DataStatus.NOT_FOUND,
                    provider="polygon",
                    error="No data returned",
                    fetch_time_ms=fetch_time,
                )
                
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                return FetchResult(
                    symbol=symbol,
                    bars=[],
                    status=DataStatus.RATE_LIMITED,
                    provider="polygon",
                    error=str(e),
                )
            return FetchResult(
                symbol=symbol,
                bars=[],
                status=DataStatus.API_ERROR,
                provider="polygon",
                error=str(e),
            )
    
    def _try_yahoo(
        self,
        symbol: str,
        days: int,
        timeframe: str
    ) -> Optional[FetchResult]:
        """Try to fetch from Yahoo Finance."""
        if self._yahoo is None:
            return None
        
        start_time = time.time()
        try:
            bars = self._yahoo.fetch(symbol, days=days, timeframe=timeframe)
            fetch_time = (time.time() - start_time) * 1000
            
            if len(bars) > 0:
                if self.use_cache and self._csv_bundle:
                    try:
                        self._csv_bundle.save_to_cache(symbol, bars, timeframe, "yahoo")
                    except Exception as e:
                        logger.warning(f"Failed to cache {symbol}: {e}")
                
                return FetchResult(
                    symbol=symbol,
                    bars=bars,
                    status=DataStatus.SUCCESS,
                    provider="yahoo",
                    fetch_time_ms=fetch_time,
                )
            else:
                return FetchResult(
                    symbol=symbol,
                    bars=[],
                    status=DataStatus.NOT_FOUND,
                    provider="yahoo",
                    error="No data returned",
                    fetch_time_ms=fetch_time,
                )
                
        except Exception as e:
            return FetchResult(
                symbol=symbol,
                bars=[],
                status=DataStatus.API_ERROR,
                provider="yahoo",
                error=str(e),
            )
    
    def fetch(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "day",
        prefer_cache: bool = True,
    ) -> FetchResult:
        """
        Fetch OHLCV data with automatic failover.
        
        Provider order:
        1. Cache (if prefer_cache=True)
        2. Polygon (primary)
        3. Yahoo Finance (secondary)
        4. CSV Bundle (tertiary/fallback)
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            timeframe: Bar timeframe
            prefer_cache: Try cache first before live providers
            
        Returns:
            FetchResult with bars and status
        """
        end = datetime.now()
        start = end - timedelta(days=days)
        
        if prefer_cache:
            result = self._try_cache(symbol, start, end, timeframe)
            if result and result.status == DataStatus.CACHED:
                return result
        
        result = self._try_polygon(symbol, days, timeframe)
        if result and result.status == DataStatus.SUCCESS:
            return result
        
        if result and result.status == DataStatus.RATE_LIMITED:
            time.sleep(self.rate_limit_delay)
        
        result = self._try_yahoo(symbol, days, timeframe)
        if result and result.status == DataStatus.SUCCESS:
            return result
        
        result = self._try_cache(symbol, start, end, timeframe)
        if result and len(result.bars) > 0:
            return result
        
        return FetchResult(
            symbol=symbol,
            bars=[],
            status=DataStatus.UNAVAILABLE,
            provider="none",
            error=f"All providers failed for {symbol}",
        )
    
    def fetch_batch(
        self,
        symbols: List[str],
        days: int = 365,
        timeframe: str = "day",
        batch_delay: float = 0.1,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, FetchResult]:
        """
        Fetch data for multiple symbols with batching.
        
        Args:
            symbols: List of ticker symbols
            days: Number of days of history
            timeframe: Bar timeframe
            batch_delay: Delay between fetches (seconds)
            on_progress: Optional callback(symbol, index, total)
            
        Returns:
            Dict mapping symbol to FetchResult
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            result = self.fetch(symbol, days=days, timeframe=timeframe)
            results[symbol] = result
            
            if on_progress:
                on_progress(symbol, i + 1, total)
            
            if i < total - 1 and batch_delay > 0:
                time.sleep(batch_delay)
        
        return results
    
    def get_available_providers(self) -> List[str]:
        """Get list of available data providers."""
        providers: List[str] = []
        if self._polygon is not None:
            providers.append("polygon")
        if self._yahoo is not None:
            providers.append("yahoo")
        if self._csv_bundle is not None:
            providers.append("csv_bundle")
        return providers
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all providers."""
        health = {
            "polygon": self._polygon is not None,
            "yahoo": self._yahoo is not None,
            "csv_bundle": self._csv_bundle is not None,
            "cache_enabled": self.use_cache,
            "cache_dir": str(self.cache_dir),
        }
        
        if self._csv_bundle:
            try:
                cached_symbols = self._csv_bundle.list_available_symbols()
                health["cached_symbols_count"] = len(cached_symbols)
            except Exception:
                health["cached_symbols_count"] = 0
        
        return health


def create_data_client(
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> DataClient:
    """
    Factory function to create a configured DataClient.
    
    Args:
        use_cache: Enable caching
        cache_dir: Cache directory path
        
    Returns:
        Configured DataClient instance
    """
    return DataClient(use_cache=use_cache, cache_dir=cache_dir)
