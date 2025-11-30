"""
EODHD (End of Day Historical Data) Adapter for QuantraCore Apex.

Provides global market data across 70+ exchanges.

Requirements:
- EODHD_API_KEY environment variable

Subscription Tiers:
- Free: Limited API calls, EOD data only
- All-in-One: $79.99/mo - Real-time, fundamentals, options
- Extended: $199.99/mo - Full historical, bulk data

Data Coverage:
- 70+ global stock exchanges
- ETFs, mutual funds, bonds
- Forex pairs
- Cryptocurrencies
- Economic indicators
- Fundamental data
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    OhlcvBar, ProviderStatus, FundamentalsData, NewsItem
)


class EODHDAdapter(EnhancedDataAdapter):
    """
    EODHD data adapter for global market coverage.
    
    Best for:
    - International stock data (70+ exchanges)
    - Historical EOD data
    - Fundamental analysis
    - ETF and bond data
    - Delisted/historical securities
    
    API Endpoints:
    - /eod/{symbol} - End of day prices
    - /real-time/{symbol} - Real-time quotes
    - /fundamentals/{symbol} - Fundamental data
    - /news - Market news
    - /economic-events - Economic calendar
    """
    
    BASE_URL = "https://eodhistoricaldata.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 0.5
    
    @property
    def name(self) -> str:
        return "EODHD"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.OHLCV,
            DataType.FUNDAMENTALS,
            DataType.NEWS,
            DataType.ECONOMIC
        ]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="API Key Required",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "EODHD_API_KEY not set"
        )
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        if not self.is_available():
            raise ValueError("EODHD_API_KEY not set")
        
        self._rate_limit()
        
        params = params or {}
        params["api_token"] = self.api_key
        params["fmt"] = "json"
        
        url = f"{self.BASE_URL}/{endpoint}"
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
        exchange = "US"
        if "." in symbol:
            symbol, exchange = symbol.split(".", 1)
        
        endpoint = f"eod/{symbol}.{exchange}"
        params = {
            "from": start.strftime("%Y-%m-%d"),
            "to": end.strftime("%Y-%m-%d"),
            "period": "d"
        }
        
        data = self._request(endpoint, params)
        
        bars = []
        for item in data:
            bars.append(OhlcvBar(
                timestamp=datetime.strptime(item["date"], "%Y-%m-%d"),
                open=float(item["open"]),
                high=float(item["high"]),
                low=float(item["low"]),
                close=float(item["close"]),
                volume=float(item.get("volume", 0))
            ))
        
        return bars
    
    def fetch_fundamentals(self, symbol: str) -> FundamentalsData:
        exchange = "US"
        if "." in symbol:
            symbol, exchange = symbol.split(".", 1)
        
        endpoint = f"fundamentals/{symbol}.{exchange}"
        data = self._request(endpoint)
        
        highlights = data.get("Highlights", {})
        valuation = data.get("Valuation", {})
        
        return FundamentalsData(
            symbol=symbol,
            market_cap=highlights.get("MarketCapitalization"),
            pe_ratio=highlights.get("PERatio"),
            eps=highlights.get("EarningsShare"),
            revenue=highlights.get("Revenue"),
            profit_margin=highlights.get("ProfitMargin"),
            dividend_yield=highlights.get("DividendYield"),
            beta=highlights.get("Beta")
        )
    
    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        params = {"limit": limit}
        
        if symbols:
            params["s"] = ",".join(symbols)
        if start:
            params["from"] = start.strftime("%Y-%m-%d")
        if end:
            params["to"] = end.strftime("%Y-%m-%d")
        
        data = self._request("news", params)
        
        news = []
        for item in data:
            news.append(NewsItem(
                timestamp=datetime.fromisoformat(item["date"].replace("Z", "+00:00")),
                headline=item.get("title", ""),
                summary=item.get("content", "")[:500] if item.get("content") else None,
                source=item.get("source"),
                url=item.get("link"),
                symbols=item.get("symbols", [])
            ))
        
        return news
    
    def get_exchanges(self) -> List[Dict[str, str]]:
        return self._request("exchanges-list")
    
    def get_symbols(self, exchange: str = "US") -> List[Dict[str, Any]]:
        return self._request(f"exchange-symbol-list/{exchange}")


EODHD_SETUP_GUIDE = """
=== EODHD Setup Guide ===

1. GET API KEY
   - Sign up at: https://eodhd.com/
   - Free tier: Basic EOD data, limited calls
   - Paid tiers: Real-time, fundamentals, bulk

2. PRICING
   - Free: $0/mo - 20 API calls/day
   - All-World: $19.99/mo - EOD for all exchanges
   - All-in-One: $79.99/mo - Real-time + fundamentals
   - Extended: $199.99/mo - Full historical

3. ENVIRONMENT VARIABLE
   EODHD_API_KEY=your_api_key_here

4. EXCHANGE CODES
   - US: NYSE, NASDAQ, AMEX
   - LSE: London Stock Exchange
   - TO: Toronto
   - PA: Paris
   - XETRA: Germany
   - HK: Hong Kong
   - Full list: https://eodhd.com/financial-apis/exchanges-api-list-of-tickers-and-டatus

5. SYMBOL FORMAT
   - US stocks: AAPL.US, MSFT.US
   - International: BP.LSE, SAP.XETRA
   - Crypto: BTC-USD.CC
"""
