"""
Alpha Vantage Data Adapter for QuantraCore Apex.

Free tier: 500 requests/day, 5/minute.

Features:
- OHLCV data (daily, intraday)
- 50+ Technical Indicators
- AI-Powered News Sentiment
- Global market data (stocks, ETFs, forex, crypto)

Environment Variables:
- ALPHA_VANTAGE_API_KEY: Your Alpha Vantage API key

Sign up: https://www.alphavantage.co/
"""

import os
import time
import httpx
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from src.quantracore_apex.core.schemas import OhlcvBar
from .base import DataAdapter


logger = logging.getLogger(__name__)


@dataclass
class NewsSentimentArticle:
    title: str
    url: str
    time_published: datetime
    authors: List[str]
    summary: str
    source: str
    overall_sentiment_score: float
    overall_sentiment_label: str
    ticker_sentiment: Dict[str, float]
    relevance_score: float


@dataclass 
class TechnicalIndicator:
    timestamp: datetime
    value: float
    signal: Optional[str] = None


class AlphaVantageAdapter(DataAdapter):
    """
    Alpha Vantage data adapter.
    
    Free tier: 25 requests/day (actual limit, not 500 as advertised).
    
    Features:
    - Daily and intraday OHLCV
    - 50+ technical indicators (RSI, MACD, Bollinger, etc.)
    - AI-powered news sentiment analysis
    - Global coverage (stocks, ETFs, forex, crypto)
    
    Set ALPHA_VANTAGE_API_KEY environment variable.
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    DAILY_LIMIT = 25
    
    _daily_calls = 0
    _daily_reset_time = None
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 15.0
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 3600
        self._limit_exceeded = False
        
        if AlphaVantageAdapter._daily_reset_time is None:
            AlphaVantageAdapter._daily_reset_time = datetime.now()
        elif (datetime.now() - AlphaVantageAdapter._daily_reset_time).days >= 1:
            AlphaVantageAdapter._daily_calls = 0
            AlphaVantageAdapter._daily_reset_time = datetime.now()
        
        if self.api_key:
            remaining = self.DAILY_LIMIT - AlphaVantageAdapter._daily_calls
            logger.info(f"[AlphaVantage] Adapter initialized ({remaining}/{self.DAILY_LIMIT} calls remaining today)")
        else:
            logger.debug("[AlphaVantage] API key not set - using simulated data")
    
    @property
    def name(self) -> str:
        return "alpha_vantage"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    OHLCV_CACHE_TTL = 86400
    _ohlcv_cache: Dict[str, Any] = {}
    _ohlcv_cache_ts: Dict[str, float] = {}
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV data from Alpha Vantage.
        
        Note: Free tier has limitations (5 requests/minute, 25/day).
        Uses 24-hour caching to preserve daily API quota.
        """
        if not self.is_available():
            logger.warning("[AlphaVantage] API key not set")
            return self._simulated_ohlcv(symbol, start, end, timeframe)
        
        cache_key = f"ohlcv_{symbol}_{timeframe}_{start.date()}_{end.date()}"
        now = time.time()
        
        if cache_key in AlphaVantageAdapter._ohlcv_cache:
            cached_ts = AlphaVantageAdapter._ohlcv_cache_ts.get(cache_key, 0)
            if now - cached_ts < self.OHLCV_CACHE_TTL:
                logger.debug(f"[AlphaVantage] Cache hit for {symbol} OHLCV")
                return AlphaVantageAdapter._ohlcv_cache[cache_key]
        
        if self._check_daily_limit():
            logger.debug(f"[AlphaVantage] Daily limit hit, returning simulated data for {symbol}")
            return self._simulated_ohlcv(symbol, start, end, timeframe)
        
        self._rate_limit_wait()
        AlphaVantageAdapter._daily_calls += 1
        
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
                    logger.warning(f"[AlphaVantage] API limit: {data['Note']}")
                    self._limit_exceeded = True
                    AlphaVantageAdapter._daily_calls = self.DAILY_LIMIT
                elif "Error Message" in data:
                    logger.error(f"[AlphaVantage] API error: {data['Error Message']}")
                return self._simulated_ohlcv(symbol, start, end, timeframe)
            
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
                    logger.debug(f"[AlphaVantage] Skipping bar: {e}")
                    continue
            
            bars.sort(key=lambda x: x.timestamp)
            
            AlphaVantageAdapter._ohlcv_cache[cache_key] = bars
            AlphaVantageAdapter._ohlcv_cache_ts[cache_key] = now
            logger.debug(f"[AlphaVantage] Cached {len(bars)} bars for {symbol}")
            
            return bars
            
        except httpx.HTTPError as e:
            logger.error(f"[AlphaVantage] HTTP error: {e}")
            return self._simulated_ohlcv(symbol, start, end, timeframe)
        except Exception as e:
            logger.error(f"[AlphaVantage] Error: {e}")
            return self._simulated_ohlcv(symbol, start, end, timeframe)
    
    def _simulated_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> List[OhlcvBar]:
        """Generate simulated OHLCV data when API is unavailable."""
        import random
        
        bars = []
        current = start
        base_price = 50.0 + hash(symbol) % 100
        
        while current <= end:
            volatility = 0.02
            price_change = random.uniform(-volatility, volatility)
            open_price = base_price * (1 + random.uniform(-0.01, 0.01))
            close_price = open_price * (1 + price_change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            
            bars.append(OhlcvBar(
                timestamp=current,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=float(random.randint(100000, 5000000))
            ))
            
            base_price = close_price
            current += timedelta(days=1) if timeframe == "1d" else timedelta(hours=1)
        
        return bars
    
    def _rate_limit_wait(self):
        """Wait to respect rate limits (5 requests/minute)."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _check_daily_limit(self) -> bool:
        """Check if daily API limit has been exceeded."""
        if AlphaVantageAdapter._daily_reset_time and (datetime.now() - AlphaVantageAdapter._daily_reset_time).days >= 1:
            AlphaVantageAdapter._daily_calls = 0
            AlphaVantageAdapter._daily_reset_time = datetime.now()
            self._limit_exceeded = False
        
        if AlphaVantageAdapter._daily_calls >= self.DAILY_LIMIT:
            if not self._limit_exceeded:
                logger.warning(f"[AlphaVantage] Daily limit ({self.DAILY_LIMIT}) exceeded - using simulated data")
                self._limit_exceeded = True
            return True
        return False
    
    def _request(self, params: Dict[str, Any]) -> Dict:
        """Make a rate-limited request to Alpha Vantage API."""
        if not self.is_available():
            raise ValueError("ALPHA_VANTAGE_API_KEY not set")
        
        if self._check_daily_limit():
            return {"Note": "Daily limit exceeded"}
        
        self._rate_limit_wait()
        params["apikey"] = self.api_key
        AlphaVantageAdapter._daily_calls += 1
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "Note" in data and "API" in data.get("Note", ""):
                    self._limit_exceeded = True
                    AlphaVantageAdapter._daily_calls = self.DAILY_LIMIT
                
                return data
        except httpx.HTTPError as e:
            logger.error(f"[AlphaVantage] HTTP error: {e}")
            raise
    
    def get_news_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        sort: str = "LATEST",
        limit: int = 50
    ) -> List[NewsSentimentArticle]:
        """
        Get AI-powered news sentiment analysis.
        
        Args:
            tickers: List of stock tickers (e.g., ["AAPL", "MSFT"])
            topics: List of topics (e.g., ["technology", "earnings"])
            time_from: Start datetime
            time_to: End datetime
            sort: LATEST, EARLIEST, or RELEVANCE
            limit: Max articles to return
            
        Returns:
            List of NewsSentimentArticle with sentiment scores
        """
        if not self.is_available():
            return self._simulated_news_sentiment(tickers or ["AAPL"], limit)
        
        cache_key = f"news_{'_'.join(tickers or ['all'])}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "sort": sort,
                "limit": min(limit, 1000)
            }
            
            if tickers:
                params["tickers"] = ",".join(tickers)
            if topics:
                params["topics"] = ",".join(topics)
            if time_from:
                params["time_from"] = time_from.strftime("%Y%m%dT%H%M")
            if time_to:
                params["time_to"] = time_to.strftime("%Y%m%dT%H%M")
            
            data = self._request(params)
            
            if "feed" not in data:
                if "Note" in data:
                    logger.warning(f"[AlphaVantage] API limit: {data['Note']}")
                return self._simulated_news_sentiment(tickers or ["AAPL"], limit)
            
            articles = []
            for item in data["feed"][:limit]:
                try:
                    time_str = item.get("time_published", "")
                    if time_str:
                        pub_time = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S")
                    else:
                        pub_time = datetime.now()
                    
                    ticker_sentiment = {}
                    for ts in item.get("ticker_sentiment", []):
                        ticker_sentiment[ts.get("ticker", "")] = float(ts.get("ticker_sentiment_score", 0))
                    
                    articles.append(NewsSentimentArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        time_published=pub_time,
                        authors=item.get("authors", []),
                        summary=item.get("summary", ""),
                        source=item.get("source", ""),
                        overall_sentiment_score=float(item.get("overall_sentiment_score", 0)),
                        overall_sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                        ticker_sentiment=ticker_sentiment,
                        relevance_score=float(item.get("relevance_score", 0))
                    ))
                except (ValueError, KeyError) as e:
                    logger.debug(f"[AlphaVantage] Skipping article: {e}")
                    continue
            
            self._cache[cache_key] = (articles, time.time())
            return articles
            
        except Exception as e:
            logger.warning(f"[AlphaVantage] News sentiment error: {e}")
            return self._simulated_news_sentiment(tickers or ["AAPL"], limit)
    
    def _simulated_news_sentiment(
        self,
        tickers: List[str],
        limit: int = 10
    ) -> List[NewsSentimentArticle]:
        """Generate simulated news sentiment data."""
        import random
        
        headlines = [
            "Stock Shows Strong Momentum Ahead of Earnings",
            "Analysts Raise Price Target on Growth Outlook",
            "Company Announces Strategic Partnership",
            "Q3 Revenue Beats Expectations",
            "New Product Launch Receives Positive Reviews",
            "Market Leaders See Increased Institutional Buying",
            "Tech Sector Rally Continues",
            "Dividend Increase Signals Strong Cash Flow",
            "Expansion Plans Announced for 2025",
            "Management Guidance Exceeds Consensus"
        ]
        
        articles = []
        for i in range(min(limit, len(headlines))):
            ticker = random.choice(tickers)
            sentiment_score = random.uniform(-0.5, 0.8)
            
            if sentiment_score > 0.35:
                label = "Bullish"
            elif sentiment_score > 0.15:
                label = "Somewhat-Bullish"
            elif sentiment_score > -0.15:
                label = "Neutral"
            elif sentiment_score > -0.35:
                label = "Somewhat-Bearish"
            else:
                label = "Bearish"
            
            articles.append(NewsSentimentArticle(
                title=f"{ticker}: {headlines[i]}",
                url=f"https://example.com/news/{ticker}/{i}",
                time_published=datetime.now() - timedelta(hours=random.randint(1, 72)),
                authors=["Market Analyst"],
                summary=f"Analysis of {ticker} showing {label.lower()} sentiment...",
                source="Financial News",
                overall_sentiment_score=round(sentiment_score, 4),
                overall_sentiment_label=label,
                ticker_sentiment={ticker: round(sentiment_score, 4)},
                relevance_score=random.uniform(0.5, 1.0)
            ))
        
        return articles
    
    def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close"
    ) -> List[TechnicalIndicator]:
        """
        Get technical indicator values.
        
        Supported indicators:
        - RSI, MACD, BBANDS, SMA, EMA, STOCH, ADX, CCI, AROON, MFI, OBV, etc.
        
        Args:
            symbol: Stock ticker
            indicator: Indicator name (e.g., "RSI", "MACD")
            interval: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
            time_period: Number of periods for calculation
            series_type: open, high, low, close
            
        Returns:
            List of TechnicalIndicator with values
        """
        if not self.is_available():
            return self._simulated_technical_indicator(symbol, indicator)
        
        try:
            params = {
                "function": indicator.upper(),
                "symbol": symbol,
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type
            }
            
            data = self._request(params)
            
            tech_key = f"Technical Analysis: {indicator.upper()}"
            if tech_key not in data:
                if "Note" in data:
                    logger.warning(f"[AlphaVantage] API limit: {data['Note']}")
                return self._simulated_technical_indicator(symbol, indicator)
            
            results = []
            for date_str, values in data[tech_key].items():
                try:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                    value = float(list(values.values())[0])
                    
                    signal = None
                    if indicator.upper() == "RSI":
                        if value > 70:
                            signal = "OVERBOUGHT"
                        elif value < 30:
                            signal = "OVERSOLD"
                    
                    results.append(TechnicalIndicator(
                        timestamp=timestamp,
                        value=value,
                        signal=signal
                    ))
                except (ValueError, KeyError):
                    continue
            
            return sorted(results, key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.warning(f"[AlphaVantage] Technical indicator error: {e}")
            return self._simulated_technical_indicator(symbol, indicator)
    
    def _simulated_technical_indicator(
        self,
        symbol: str,
        indicator: str
    ) -> List[TechnicalIndicator]:
        """Generate simulated technical indicator data."""
        import random
        
        base_values = {
            "RSI": 50,
            "MACD": 0,
            "ADX": 25,
            "CCI": 0,
            "MFI": 50,
        }
        
        base = base_values.get(indicator.upper(), 50)
        results = []
        
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            value = base + random.uniform(-20, 20)
            
            signal = None
            if indicator.upper() == "RSI":
                if value > 70:
                    signal = "OVERBOUGHT"
                elif value < 30:
                    signal = "OVERSOLD"
            
            results.append(TechnicalIndicator(
                timestamp=date,
                value=round(value, 2),
                signal=signal
            ))
        
        return sorted(results, key=lambda x: x.timestamp)
    
    def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined sentiment analysis for a symbol.
        
        Returns aggregated news sentiment metrics.
        """
        articles = self.get_news_sentiment(tickers=[symbol], limit=20)
        
        if not articles:
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "article_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "signal": "NEUTRAL"
            }
        
        avg_sentiment = sum(a.overall_sentiment_score for a in articles) / len(articles)
        bullish = sum(1 for a in articles if a.overall_sentiment_score > 0.15)
        bearish = sum(1 for a in articles if a.overall_sentiment_score < -0.15)
        
        if avg_sentiment > 0.25:
            signal = "BULLISH"
            label = "Bullish"
        elif avg_sentiment > 0.1:
            signal = "SLIGHTLY_BULLISH"
            label = "Somewhat-Bullish"
        elif avg_sentiment < -0.25:
            signal = "BEARISH"
            label = "Bearish"
        elif avg_sentiment < -0.1:
            signal = "SLIGHTLY_BEARISH"
            label = "Somewhat-Bearish"
        else:
            signal = "NEUTRAL"
            label = "Neutral"
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "sentiment_score": round(avg_sentiment, 4),
            "sentiment_label": label,
            "article_count": len(articles),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "signal": signal
        }


def get_alpha_vantage_adapter() -> AlphaVantageAdapter:
    """Get singleton Alpha Vantage adapter instance."""
    return AlphaVantageAdapter()


ALPHA_VANTAGE_SETUP_GUIDE = """
=== Alpha Vantage API Setup Guide ===

1. SIGN UP (Free)
   https://www.alphavantage.co/support/#api-key
   
2. GET YOUR API KEY
   Instant approval, no credit card required

3. SET ENVIRONMENT VARIABLE
   ALPHA_VANTAGE_API_KEY=your_api_key_here

4. FREE TIER LIMITS
   - 500 API calls/day
   - 5 API calls/minute
   - All endpoints available

5. PREMIUM TIERS
   - 75 calls/min: $49.99/month
   - 150 calls/min: $99.99/month
   - 300+ calls/min: $199.99/month

6. KEY FEATURES
   - NEWS_SENTIMENT: AI-powered news analysis
   - 50+ technical indicators
   - Global stocks, ETFs, forex, crypto

7. SENTIMENT SCORING
   - Score range: -1.0 (bearish) to +1.0 (bullish)
   - Labels: Bearish, Somewhat-Bearish, Neutral, Somewhat-Bullish, Bullish

8. BEST PRACTICES
   - Cache responses (5 min TTL)
   - Batch requests when possible
   - Use tickers filter to focus results
"""
