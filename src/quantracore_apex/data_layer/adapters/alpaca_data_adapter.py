"""
Alpaca Market Data Adapter for QuantraCore Apex.

Fetches real OHLCV data from Alpaca's Market Data API.
Free with paper trading account - no additional subscription required.

Supports:
- Historical bars (daily, hourly, minute)
- Real-time quotes
- Trade data

Rate Limits (Basic plan):
- 200 requests/minute for historical data
"""

import os
import time
import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from src.quantracore_apex.core.schemas import OhlcvBar

logger = logging.getLogger(__name__)

ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaDataAdapter:
    """
    Adapter for fetching market data from Alpaca.
    
    Uses the same credentials as paper trading:
    - ALPACA_PAPER_API_KEY
    - ALPACA_PAPER_API_SECRET
    
    Benefits:
    - Free with paper trading account
    - No separate data subscription needed
    - Higher rate limits than Polygon free tier
    - Real-time quotes available
    """
    
    TIMEFRAME_MAP = {
        "1d": "1Day",
        "day": "1Day",
        "1D": "1Day",
        "1h": "1Hour",
        "hour": "1Hour",
        "1H": "1Hour",
        "1m": "1Min",
        "minute": "1Min",
        "1Min": "1Min",
        "5m": "5Min",
        "5Min": "5Min",
        "15m": "15Min",
        "15Min": "15Min",
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit: bool = True
    ):
        self.api_key = api_key or os.getenv("ALPACA_PAPER_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_PAPER_API_SECRET")
        
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca credentials not found - adapter unavailable")
            self._available = False
        else:
            self._available = True
        
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self._rate_limit_delay = 0.3
        self._session = requests.Session()
        
        if self._available:
            self._session.headers.update({
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            })
    
    @property
    def name(self) -> str:
        return "alpaca"
    
    def is_available(self) -> bool:
        return self._available
    
    def _rate_limit_wait(self):
        if self.rate_limit:
            elapsed = time.time() - self.last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
            self.last_request_time = time.time()
    
    def _request(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        if not self._available:
            raise RuntimeError("Alpaca adapter not available - credentials missing")
        
        url = f"{ALPACA_DATA_BASE_URL}{endpoint}"
        last_error = None
        
        for attempt in range(max_retries):
            self._rate_limit_wait()
            
            try:
                response = self._session.get(url, params=params, timeout=30)
                
                if response.status_code == 429:
                    wait_time = min(60 * (2 ** attempt), 300)
                    logger.warning(f"[ALPACA] Rate limited - waiting {wait_time} seconds (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                
                if response.status_code >= 500:
                    wait_time = 5 * (2 ** attempt)
                    logger.warning(f"[ALPACA] Server error {response.status_code} - retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                wait_time = 5 * (2 ** attempt)
                logger.warning(f"[ALPACA] Timeout - retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait_time = 5 * (2 ** attempt)
                logger.warning(f"[ALPACA] Connection error - retrying in {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.error(f"[ALPACA] Request failed: {e}")
                raise RuntimeError(f"Alpaca API error: {e}")
        
        logger.error(f"[ALPACA] Max retries exceeded: {last_error}")
        raise RuntimeError(f"Alpaca API error after {max_retries} retries: {last_error}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars from Alpaca.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe ("1d", "1h", "1m", etc.)
            
        Returns:
            List of OhlcvBar objects
        """
        alpaca_tf = self.TIMEFRAME_MAP.get(timeframe, "1Day")
        
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        endpoint = f"/v2/stocks/{symbol}/bars"
        params = {
            "start": start_str,
            "end": end_str,
            "timeframe": alpaca_tf,
            "adjustment": "all",
            "limit": 10000,
            "feed": "iex",
        }
        
        all_bars = []
        next_page_token = None
        
        while True:
            if next_page_token:
                params["page_token"] = next_page_token
            
            try:
                data = self._request(endpoint, params)
            except Exception as e:
                logger.error(f"[ALPACA] Failed to fetch {symbol}: {e}")
                break
            
            bars_data = data.get("bars", [])
            
            if not bars_data:
                break
            
            for bar in bars_data:
                try:
                    timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                    timestamp = timestamp.replace(tzinfo=None)
                    
                    ohlcv = OhlcvBar(
                        timestamp=timestamp,
                        open=float(bar["o"]),
                        high=float(bar["h"]),
                        low=float(bar["l"]),
                        close=float(bar["c"]),
                        volume=float(bar["v"])
                    )
                    all_bars.append(ohlcv)
                except (KeyError, ValueError) as e:
                    logger.warning(f"[ALPACA] Skipping malformed bar: {e}")
                    continue
            
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break
        
        logger.info(f"[ALPACA] Fetched {len(all_bars)} bars for {symbol}")
        return all_bars
    
    def fetch(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "1d",
        end_date: Optional[str] = None
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars for a number of days back.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            timeframe: Bar timeframe
            end_date: End date (default: today)
            
        Returns:
            List of OhlcvBar objects
        """
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        start = end - timedelta(days=days)
        
        return self.fetch_ohlcv(symbol, start, end, timeframe)
    
    def fetch_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars for a specific date range.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Bar timeframe
            
        Returns:
            List of OhlcvBar objects
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        return self.fetch_ohlcv(symbol, start, end, timeframe)
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest quote for a symbol.
        
        Returns:
            Dict with bid_price, ask_price, bid_size, ask_size
        """
        endpoint = f"/v2/stocks/{symbol}/quotes/latest"
        
        try:
            data = self._request(endpoint)
            quote = data.get("quote", {})
            
            return {
                "symbol": symbol,
                "bid_price": float(quote.get("bp", 0)),
                "ask_price": float(quote.get("ap", 0)),
                "bid_size": int(quote.get("bs", 0)),
                "ask_size": int(quote.get("as", 0)),
                "timestamp": quote.get("t"),
            }
        except Exception as e:
            logger.error(f"[ALPACA] Failed to get quote for {symbol}: {e}")
            return None
    
    def get_latest_trade(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest trade for a symbol.
        
        Returns:
            Dict with price, size, timestamp
        """
        endpoint = f"/v2/stocks/{symbol}/trades/latest"
        
        try:
            data = self._request(endpoint)
            trade = data.get("trade", {})
            
            return {
                "symbol": symbol,
                "price": float(trade.get("p", 0)),
                "size": int(trade.get("s", 0)),
                "timestamp": trade.get("t"),
            }
        except Exception as e:
            logger.error(f"[ALPACA] Failed to get trade for {symbol}: {e}")
            return None
    
    def get_multi_bars(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> Dict[str, List[OhlcvBar]]:
        """
        Fetch bars for multiple symbols in one request.
        
        Args:
            symbols: List of ticker symbols
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            
        Returns:
            Dict mapping symbol to list of OhlcvBar
        """
        alpaca_tf = self.TIMEFRAME_MAP.get(timeframe, "1Day")
        
        start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        endpoint = "/v2/stocks/bars"
        params = {
            "symbols": ",".join(symbols),
            "start": start_str,
            "end": end_str,
            "timeframe": alpaca_tf,
            "adjustment": "all",
            "limit": 10000,
            "feed": "iex",
        }
        
        result = {sym: [] for sym in symbols}
        next_page_token = None
        
        while True:
            if next_page_token:
                params["page_token"] = next_page_token
            
            try:
                data = self._request(endpoint, params)
            except Exception as e:
                logger.error(f"[ALPACA] Multi-bar fetch failed: {e}")
                break
            
            bars_by_symbol = data.get("bars", {})
            
            for symbol, bars in bars_by_symbol.items():
                for bar in bars:
                    try:
                        timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                        timestamp = timestamp.replace(tzinfo=None)
                        
                        ohlcv = OhlcvBar(
                            timestamp=timestamp,
                            open=float(bar["o"]),
                            high=float(bar["h"]),
                            low=float(bar["l"]),
                            close=float(bar["c"]),
                            volume=float(bar["v"])
                        )
                        result[symbol].append(ohlcv)
                    except (KeyError, ValueError) as e:
                        continue
            
            next_page_token = data.get("next_page_token")
            if not next_page_token:
                break
        
        total_bars = sum(len(bars) for bars in result.values())
        logger.info(f"[ALPACA] Fetched {total_bars} bars for {len(symbols)} symbols")
        
        return result
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information from paper trading API.
        
        Returns:
            Dict with equity, buying_power, cash, etc.
        """
        if not self._available:
            return None
        
        url = f"{ALPACA_PAPER_BASE_URL}/v2/account"
        
        try:
            response = self._session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "equity": float(data.get("equity", 0)),
                "buying_power": float(data.get("buying_power", 0)),
                "cash": float(data.get("cash", 0)),
                "portfolio_value": float(data.get("portfolio_value", 0)),
                "status": data.get("status"),
            }
        except Exception as e:
            logger.error(f"[ALPACA] Failed to get account info: {e}")
            return None
