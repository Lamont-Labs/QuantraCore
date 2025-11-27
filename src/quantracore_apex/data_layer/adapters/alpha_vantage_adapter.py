"""
Alpha Vantage Data Adapter for QuantraCore Apex.

Provides OHLCV data from Alpha Vantage API (free tier available).
"""

import os
import httpx
from typing import List, Optional
from datetime import datetime
import logging

from src.quantracore_apex.core.schemas import OhlcvBar
from .base import DataAdapter


logger = logging.getLogger(__name__)


class AlphaVantageAdapter(DataAdapter):
    """
    Alpha Vantage data adapter.
    
    Uses the free tier API for daily OHLCV data.
    Set ALPHA_VANTAGE_API_KEY environment variable.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    
    @property
    def name(self) -> str:
        return "alpha_vantage"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV data from Alpha Vantage.
        
        Note: Free tier has limitations on requests per minute.
        """
        if not self.is_available():
            logger.warning("Alpha Vantage API key not set")
            return []
        
        try:
            function = "TIME_SERIES_DAILY" if timeframe == "1d" else "TIME_SERIES_INTRADAY"
            
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json"
            }
            
            if timeframe != "1d":
                interval_map = {"1h": "60min", "30m": "30min", "15m": "15min", "5m": "5min"}
                params["interval"] = interval_map.get(timeframe, "60min")
            
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
            
            time_series_key = "Time Series (Daily)" if timeframe == "1d" else f"Time Series ({params.get('interval', '60min')})"
            
            if time_series_key not in data:
                if "Note" in data:
                    logger.warning(f"API limit reached: {data['Note']}")
                elif "Error Message" in data:
                    logger.error(f"API error: {data['Error Message']}")
                return []
            
            time_series = data[time_series_key]
            
            bars = []
            for date_str, values in time_series.items():
                try:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d") if timeframe == "1d" else datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    
                    if start <= timestamp <= end:
                        bar = OhlcvBar(
                            timestamp=timestamp,
                            open=float(values["1. open"]),
                            high=float(values["2. high"]),
                            low=float(values["3. low"]),
                            close=float(values["4. close"]),
                            volume=float(values["5. volume"]),
                        )
                        bars.append(bar)
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping bar: {e}")
                    continue
            
            bars.sort(key=lambda x: x.timestamp)
            return bars
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching data: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return []
