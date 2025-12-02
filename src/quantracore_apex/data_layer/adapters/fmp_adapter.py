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
    
    def get_earnings_calendar(
        self,
        from_date: datetime = None,
        to_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Get earnings calendar for upcoming earnings releases.
        
        Returns list of upcoming earnings with EPS estimates.
        """
        if from_date is None:
            from_date = datetime.utcnow()
        if to_date is None:
            to_date = from_date + timedelta(days=7)
        
        if not self.is_available():
            return self._simulated_earnings_calendar()
        
        try:
            return self._request("earning_calendar", {
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            })
        except Exception:
            return self._simulated_earnings_calendar()
    
    def get_earnings_history(self, symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Get historical earnings for a symbol."""
        if not self.is_available():
            return []
        
        try:
            data = self._request(f"historical/earning_calendar/{symbol}")
            return data[:limit] if data else []
        except Exception:
            return []
    
    def get_dcf_valuation(self, symbol: str) -> Dict[str, Any]:
        """
        Get DCF (Discounted Cash Flow) valuation for a symbol.
        
        Shows if stock is undervalued/overvalued based on fundamentals.
        """
        if not self.is_available():
            return self._simulated_dcf(symbol)
        
        try:
            data = self._request(f"discounted-cash-flow/{symbol}")
            if data and len(data) > 0:
                item = data[0]
                dcf = item.get("dcf", 0)
                price = item.get("Stock Price", 0)
                
                if dcf and price:
                    upside = ((dcf - price) / price) * 100
                    signal = "UNDERVALUED" if upside > 20 else ("OVERVALUED" if upside < -20 else "FAIR")
                else:
                    upside = 0
                    signal = "UNKNOWN"
                
                return {
                    "symbol": symbol.upper(),
                    "dcf": round(dcf, 2) if dcf else None,
                    "stock_price": round(price, 2) if price else None,
                    "upside_percent": round(upside, 2),
                    "valuation_signal": signal,
                    "timestamp": datetime.utcnow().isoformat()
                }
            return self._simulated_dcf(symbol)
        except Exception:
            return self._simulated_dcf(symbol)
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile with sector, industry, and key metrics."""
        if not self.is_available():
            return {}
        
        try:
            data = self._request(f"profile/{symbol}")
            if data and len(data) > 0:
                item = data[0]
                return {
                    "symbol": symbol.upper(),
                    "name": item.get("companyName", symbol),
                    "sector": item.get("sector", "Unknown"),
                    "industry": item.get("industry", "Unknown"),
                    "market_cap": item.get("mktCap", 0),
                    "pe_ratio": item.get("peRatio"),
                    "beta": item.get("beta"),
                    "dividend_yield": item.get("lastDiv"),
                    "avg_volume": item.get("volAvg", 0),
                    "description": item.get("description", "")[:500]
                }
            return {}
        except Exception:
            return {}
    
    def get_dividend_calendar(
        self,
        from_date: datetime = None,
        to_date: datetime = None
    ) -> List[Dict[str, Any]]:
        """Get dividend calendar for upcoming ex-dividend dates."""
        if from_date is None:
            from_date = datetime.utcnow()
        if to_date is None:
            to_date = from_date + timedelta(days=14)
        
        if not self.is_available():
            return []
        
        try:
            return self._request("stock_dividend_calendar", {
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            })
        except Exception:
            return []
    
    def _simulated_earnings_calendar(self) -> List[Dict[str, Any]]:
        """Generate simulated earnings calendar data."""
        import random
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
        
        events = []
        for i in range(5):
            sym = random.choice(symbols)
            events.append({
                "symbol": sym,
                "date": (datetime.utcnow() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "epsEstimated": round(random.uniform(1.0, 5.0), 2),
                "eps": None,
                "revenueEstimated": round(random.uniform(10, 100) * 1e9),
                "time": random.choice(["bmo", "amc"]),
                "simulated": True
            })
        return events
    
    def _simulated_dcf(self, symbol: str) -> Dict[str, Any]:
        """Generate simulated DCF valuation."""
        import random
        price = round(random.uniform(50, 500), 2)
        dcf = price * random.uniform(0.7, 1.4)
        upside = ((dcf - price) / price) * 100
        
        return {
            "symbol": symbol.upper(),
            "dcf": round(dcf, 2),
            "stock_price": price,
            "upside_percent": round(upside, 2),
            "valuation_signal": "UNDERVALUED" if upside > 20 else ("OVERVALUED" if upside < -20 else "FAIR"),
            "simulated": True,
            "timestamp": datetime.utcnow().isoformat()
        }


_fmp_adapter: Optional['FMPAdapter'] = None


def get_fmp_adapter() -> 'FMPAdapter':
    """Get singleton FMP adapter instance."""
    global _fmp_adapter
    if _fmp_adapter is None:
        _fmp_adapter = FMPAdapter()
    return _fmp_adapter


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
