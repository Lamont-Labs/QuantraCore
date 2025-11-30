"""
Nasdaq Data Link (formerly Quandl) Adapter for QuantraCore Apex.

Best for alternative data, economic indicators, and unique datasets.

Requirements:
- NASDAQ_DATA_LINK_API_KEY environment variable

Subscription:
- Free tier available with rate limits
- Premium datasets vary by provider
- Some free public datasets

Data Coverage:
- Economic indicators (Fed, Treasury)
- Alternative data (sentiment, ESG)
- Commodity futures
- ETF flows
- Short interest
- Hedge fund holdings
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    OhlcvBar, ProviderStatus
)


class NasdaqDataLinkAdapter(EnhancedDataAdapter):
    """
    Nasdaq Data Link (Quandl) adapter for alternative data.
    
    Best for:
    - Economic indicators (GDP, unemployment, CPI)
    - Fed data (interest rates, balance sheet)
    - Treasury yields
    - Commodity data
    - ESG and sentiment datasets
    - Unique alternative datasets
    
    Database Codes:
    - FRED: Federal Reserve data
    - USTREASURY: Treasury yields
    - ODA: IMF data
    - MULTPL: S&P 500 ratios
    - FINRA: Short interest
    """
    
    BASE_URL = "https://data.nasdaq.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 0.5
    
    @property
    def name(self) -> str:
        return "Nasdaq Data Link"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.OHLCV,
            DataType.ECONOMIC,
            DataType.SENTIMENT
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
            last_error=None if self.is_available() else "NASDAQ_DATA_LINK_API_KEY not set"
        )
    
    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        if not self.is_available():
            raise ValueError("NASDAQ_DATA_LINK_API_KEY not set")
        
        self._rate_limit()
        
        params = params or {}
        params["api_key"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_dataset(
        self,
        database: str,
        dataset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> Dict[str, Any]:
        params = {"limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self._request(f"datasets/{database}/{dataset}.json", params)
    
    def get_fed_funds_rate(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("FRED", "FEDFUNDS", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_treasury_yield(
        self,
        maturity: str = "10Y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        maturity_map = {
            "3M": "DGS3MO",
            "1Y": "DGS1",
            "2Y": "DGS2",
            "5Y": "DGS5",
            "10Y": "DGS10",
            "30Y": "DGS30"
        }
        dataset = maturity_map.get(maturity, "DGS10")
        data = self.get_dataset("FRED", dataset, start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_vix(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("FRED", "VIXCLS", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_unemployment(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("FRED", "UNRATE", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_cpi(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("FRED", "CPIAUCSL", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_sp500_pe(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("MULTPL", "SP500_PE_RATIO_MONTH", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def get_short_interest(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        data = self.get_dataset("FINRA", f"FNYX_{symbol}", start_date, end_date)
        return data.get("dataset", {}).get("data", [])
    
    def search_datasets(self, query: str, page: int = 1) -> Dict[str, Any]:
        return self._request("datasets.json", {"query": query, "page": page})


NASDAQ_DATA_LINK_SETUP_GUIDE = """
=== Nasdaq Data Link Setup Guide ===

1. GET API KEY
   - Sign up at: https://data.nasdaq.com/
   - Free tier: 50 calls/day, public datasets

2. ENVIRONMENT VARIABLE
   NASDAQ_DATA_LINK_API_KEY=your_api_key_here
   # Legacy support:
   QUANDL_API_KEY=your_api_key_here

3. FREE DATABASES
   - FRED: Federal Reserve economic data
   - USTREASURY: Treasury rates
   - MULTPL: S&P 500 ratios
   - Wiki (limited): Stock prices

4. PREMIUM DATABASES
   - Sharadar: Core US equities (~$25-100/mo)
   - Zacks: Earnings, estimates
   - Various alternative data providers

5. COMMON DATASETS
   - FRED/FEDFUNDS: Fed funds rate
   - FRED/UNRATE: Unemployment rate
   - FRED/CPIAUCSL: CPI inflation
   - FRED/VIXCLS: VIX index
   - USTREASURY/YIELD: Treasury yields
"""
