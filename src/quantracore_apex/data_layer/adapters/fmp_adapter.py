"""
Financial Modeling Prep (FMP) Data Adapter for QuantraCore Apex.

Best for fundamental analysis and ML-ready datasets.

Requirements:
- FMP_API_KEY environment variable

Subscription Tiers:
- Free: 250 requests/day, basic data
- Starter: $19/mo - Full fundamentals
- Professional: $49/mo - Real-time, options
- Enterprise: $99/mo - Bulk data, priority

Data Coverage:
- US and international stocks
- Financial statements (10+ years)
- Key metrics and ratios
- Earnings calendars
- SEC filings
- Analyst estimates
- ESG scores
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    OhlcvBar, ProviderStatus, FundamentalsData, NewsItem, SentimentData
)


class FMPAdapter(EnhancedDataAdapter):
    """
    Financial Modeling Prep adapter for fundamental analysis.
    
    Best for:
    - Fundamental data (balance sheets, income statements)
    - Financial ratios and metrics
    - Earnings and analyst estimates
    - SEC filings and insider trading
    - ESG scores
    - ML-ready structured datasets
    
    API Endpoints:
    - /quote/{symbol} - Real-time quote
    - /historical-price-full/{symbol} - Historical prices
    - /income-statement/{symbol} - Income statements
    - /balance-sheet-statement/{symbol} - Balance sheets
    - /ratios/{symbol} - Financial ratios
    - /analyst-estimates/{symbol} - Analyst estimates
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 0.3
    
    @property
    def name(self) -> str:
        return "Financial Modeling Prep"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.OHLCV,
            DataType.FUNDAMENTALS,
            DataType.NEWS,
            DataType.SENTIMENT,
            DataType.SEC_FILINGS
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
            last_error=None if self.is_available() else "FMP_API_KEY not set"
        )
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        if not self.is_available():
            raise ValueError("FMP_API_KEY not set")
        
        self._rate_limit()
        
        params = params or {}
        params["apikey"] = self.api_key
        
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
        timeframe_map = {
            TimeFrame.MINUTE_1: "1min",
            TimeFrame.MINUTE_5: "5min",
            TimeFrame.MINUTE_15: "15min",
            TimeFrame.MINUTE_30: "30min",
            TimeFrame.HOUR_1: "1hour",
            TimeFrame.HOUR_4: "4hour",
            TimeFrame.DAY_1: "historical-price-full"
        }
        
        if timeframe in [TimeFrame.DAY_1, TimeFrame.WEEK_1, TimeFrame.MONTH_1]:
            endpoint = f"historical-price-full/{symbol}"
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d")
            }
        else:
            tf = timeframe_map.get(timeframe, "1hour")
            endpoint = f"historical-chart/{tf}/{symbol}"
            params = {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d")
            }
        
        data = self._request(endpoint, params)
        
        if isinstance(data, dict) and "historical" in data:
            data = data["historical"]
        
        bars = []
        for item in data:
            try:
                timestamp = datetime.strptime(item.get("date", ""), "%Y-%m-%d")
            except:
                timestamp = datetime.fromisoformat(item.get("date", "").replace("Z", ""))
            
            bars.append(OhlcvBar(
                timestamp=timestamp,
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=float(item.get("volume", 0)),
                vwap=item.get("vwap")
            ))
        
        return sorted(bars, key=lambda x: x.timestamp)
    
    def fetch_fundamentals(self, symbol: str) -> FundamentalsData:
        quote = self._request(f"quote/{symbol}")
        ratios = self._request(f"ratios/{symbol}")
        
        q = quote[0] if quote else {}
        r = ratios[0] if ratios else {}
        
        return FundamentalsData(
            symbol=symbol,
            market_cap=q.get("marketCap"),
            pe_ratio=q.get("pe"),
            eps=q.get("eps"),
            revenue=None,
            profit_margin=r.get("netProfitMargin"),
            dividend_yield=r.get("dividendYield"),
            beta=q.get("beta"),
            shares_outstanding=q.get("sharesOutstanding"),
            debt_to_equity=r.get("debtEquityRatio")
        )
    
    def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self._request(f"income-statement/{symbol}", {
            "period": period,
            "limit": limit
        })
    
    def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self._request(f"balance-sheet-statement/{symbol}", {
            "period": period,
            "limit": limit
        })
    
    def get_cash_flow(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        return self._request(f"cash-flow-statement/{symbol}", {
            "period": period,
            "limit": limit
        })
    
    def get_analyst_estimates(self, symbol: str) -> List[Dict[str, Any]]:
        return self._request(f"analyst-estimates/{symbol}")
    
    def get_insider_trading(self, symbol: str) -> List[Dict[str, Any]]:
        return self._request(f"insider-trading", {"symbol": symbol})
    
    def get_sec_filings(
        self,
        symbol: str,
        filing_type: str = None
    ) -> List[Dict[str, Any]]:
        params = {}
        if filing_type:
            params["type"] = filing_type
        return self._request(f"sec_filings/{symbol}", params)


FMP_SETUP_GUIDE = """
=== Financial Modeling Prep Setup Guide ===

1. GET API KEY
   - Sign up at: https://financialmodelingprep.com/
   - Verify email to activate

2. PRICING
   - Free: 250 requests/day
   - Starter: $19/mo - Full fundamentals
   - Professional: $49/mo - Real-time data
   - Enterprise: $99/mo - Bulk downloads

3. ENVIRONMENT VARIABLE
   FMP_API_KEY=your_api_key_here

4. KEY ENDPOINTS
   - /quote/{symbol} - Real-time quote
   - /income-statement/{symbol} - Income statement
   - /balance-sheet-statement/{symbol} - Balance sheet
   - /ratios/{symbol} - Financial ratios
   - /analyst-estimates/{symbol} - Analyst estimates
   - /sec_filings/{symbol} - SEC filings

5. ML USE CASES
   - Time-series forecasting with historical prices
   - Fundamental factor models
   - Earnings prediction
   - Value investing screening
"""
