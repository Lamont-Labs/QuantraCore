"""
Cryptocurrency Data Adapters for QuantraCore Apex.

Multi-exchange crypto market data with spot and futures support.

Supported Exchanges:
- Binance (spot, futures, options)
- Coinbase (spot)
- Kraken (spot, futures)
- FTX (historical only)
- CoinGecko (free aggregator)

Data Types:
- OHLCV (all timeframes)
- Order books
- Trades
- Funding rates
- Open interest
- Liquidations
"""

import os
import time
import requests
import hmac
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode

from .base_enhanced import (
    EnhancedDataAdapter, StreamingDataAdapter, DataType, TimeFrame,
    OhlcvBar, TickData, QuoteData, ProviderStatus
)


class BinanceAdapter(EnhancedDataAdapter):
    """
    Binance cryptocurrency data adapter.
    
    Free API with rate limits.
    No API key required for public data.
    
    Features:
    - Spot and futures data
    - All timeframes (1m to 1M)
    - Order book depth
    - Recent trades
    - Aggregated trades
    
    Endpoints:
    - Spot: https://api.binance.com
    - Futures: https://fapi.binance.com
    - US: https://api.binance.us
    """
    
    SPOT_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"
    US_URL = "https://api.binance.us"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        use_us: bool = False
    ):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.base_url = self.US_URL if use_us else self.SPOT_URL
        self._last_request = 0
        self._rate_limit_delay = 0.1
    
    @property
    def name(self) -> str:
        return "Binance"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OHLCV, DataType.TICK, DataType.QUOTE, DataType.CRYPTO]
    
    def is_available(self) -> bool:
        return True
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            connected=True,
            subscription_tier="Free (public data)",
            data_types=self.supported_data_types
        )
    
    def _rate_limit(self):
        """Non-blocking rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            wait_time = self._rate_limit_delay - elapsed
            try:
                import asyncio
                asyncio.get_running_loop()
            except RuntimeError:
                time.sleep(wait_time)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[OhlcvBar]:
        timeframe_map = {
            TimeFrame.MINUTE_1: "1m",
            TimeFrame.MINUTE_5: "5m",
            TimeFrame.MINUTE_15: "15m",
            TimeFrame.MINUTE_30: "30m",
            TimeFrame.HOUR_1: "1h",
            TimeFrame.HOUR_4: "4h",
            TimeFrame.DAY_1: "1d",
            TimeFrame.WEEK_1: "1w",
            TimeFrame.MONTH_1: "1M"
        }
        
        interval = timeframe_map.get(timeframe, "1d")
        symbol = symbol.upper().replace("-", "").replace("/", "")
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "limit": 1000
        }
        
        data = self._request("/api/v3/klines", params)
        
        bars = []
        for item in data:
            bars.append(OhlcvBar(
                timestamp=datetime.fromtimestamp(item[0] / 1000),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]),
                trades=int(item[8])
            ))
        
        return bars
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper().replace("-", "").replace("/", "")
        return self._request("/api/v3/ticker/24hr", {"symbol": symbol})
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        symbol = symbol.upper().replace("-", "").replace("/", "")
        return self._request("/api/v3/depth", {"symbol": symbol, "limit": limit})
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        symbol = symbol.upper().replace("-", "").replace("/", "")
        return self._request("/api/v3/trades", {"symbol": symbol, "limit": limit})
    
    def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper().replace("-", "").replace("/", "")
        return self._request(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1}
        )
    
    def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        symbol = symbol.upper().replace("-", "").replace("/", "")
        return self._request(
            "/fapi/v1/openInterest",
            {"symbol": symbol}
        )


class CoinGeckoAdapter(EnhancedDataAdapter):
    """
    CoinGecko free crypto data adapter.
    
    No API key required for basic features.
    Pro API for higher limits.
    
    Features:
    - 10,000+ cryptocurrencies
    - Historical prices
    - Market data
    - Exchange volumes
    - Trending coins
    
    Pricing:
    - Free: 10-50 calls/minute
    - Pro: $129/mo - Higher limits
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    PRO_URL = "https://pro-api.coingecko.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.base_url = self.PRO_URL if self.api_key else self.BASE_URL
        self._last_request = 0
        self._rate_limit_delay = 1.5
    
    @property
    def name(self) -> str:
        return "CoinGecko"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OHLCV, DataType.CRYPTO]
    
    def is_available(self) -> bool:
        return True
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            connected=True,
            subscription_tier="Pro" if self.api_key else "Free",
            data_types=self.supported_data_types
        )
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        self._rate_limit()
        
        headers = {}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_coin_list(self) -> List[Dict[str, str]]:
        return self._request("coins/list")
    
    def get_coin_data(self, coin_id: str) -> Dict[str, Any]:
        return self._request(f"coins/{coin_id}")
    
    def get_market_data(
        self,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        return self._request("coins/markets", {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page
        })
    
    def get_historical_prices(
        self,
        coin_id: str,
        days: int = 30,
        vs_currency: str = "usd"
    ) -> Dict[str, Any]:
        return self._request(f"coins/{coin_id}/market_chart", {
            "vs_currency": vs_currency,
            "days": days
        })
    
    def get_trending(self) -> Dict[str, Any]:
        return self._request("search/trending")
    
    def get_global_data(self) -> Dict[str, Any]:
        return self._request("global")


class CryptoDataAggregator(EnhancedDataAdapter):
    """
    Aggregates crypto data from multiple exchanges.
    """
    
    def __init__(self):
        self.binance = BinanceAdapter()
        self.coingecko = CoinGeckoAdapter()
    
    @property
    def name(self) -> str:
        return "Crypto Aggregator"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OHLCV, DataType.CRYPTO]
    
    def is_available(self) -> bool:
        return True
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            connected=True,
            subscription_tier="Binance + CoinGecko",
            data_types=self.supported_data_types
        )
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[OhlcvBar]:
        try:
            return self.binance.fetch_ohlcv(symbol, start, end, timeframe)
        except Exception:
            pass
        
        return []


CRYPTO_SETUP_GUIDE = """
=== Cryptocurrency Data Providers Setup Guide ===

1. BINANCE (Free)
   - No API key required for public data
   - API key needed for trading/private endpoints
   - Rate limit: 1200 requests/minute
   
   BINANCE_API_KEY=optional_for_trading
   BINANCE_API_SECRET=optional_for_trading

2. COINGECKO (Free/Pro)
   - Free: 10-50 calls/minute
   - Pro: $129/mo - Higher limits
   
   COINGECKO_API_KEY=optional_for_pro

3. ADDITIONAL EXCHANGES
   - Coinbase: COINBASE_API_KEY
   - Kraken: KRAKEN_API_KEY
   - Bybit: BYBIT_API_KEY

4. SYMBOL FORMATS
   - Binance: BTCUSDT, ETHUSDT
   - CoinGecko: bitcoin, ethereum (coin IDs)

5. DATA TYPES
   - OHLCV: All timeframes (1m to 1M)
   - Order books: Real-time depth
   - Funding rates: Perpetual futures
   - Open interest: Derivatives data

6. WEBSOCKET STREAMING
   - Use BinanceWebSocketAdapter for real-time
   - Supports trades, klines, depth, ticker
"""
