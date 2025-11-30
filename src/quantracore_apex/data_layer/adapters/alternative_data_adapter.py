"""
Alternative Data Adapters for QuantraCore Apex.

Non-traditional data sources for alpha generation.

Supported Providers:
- AltIndex: Social sentiment, AI ratings
- Finnhub: News, sentiment, insider trades
- Stocktwits: Social sentiment
- Reddit: r/wallstreetbets sentiment
- Twitter/X: Social mentions

Data Types:
- Social sentiment
- News sentiment
- Insider trading
- Job postings
- Web traffic
- App downloads
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    ProviderStatus, NewsItem, SentimentData
)


class FinnhubAdapter(EnhancedDataAdapter):
    """
    Finnhub alternative data adapter.
    
    Free tier available, premium features require subscription.
    
    Features:
    - Real-time news
    - Social sentiment
    - Insider transactions
    - Earnings surprises
    - IPO calendar
    - Company filings
    
    Pricing:
    - Free: 60 calls/minute
    - Growth: $74.99/mo - More calls, real-time
    - Business: Custom - Enterprise features
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 1.0
    
    @property
    def name(self) -> str:
        return "Finnhub"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.NEWS, DataType.SENTIMENT, DataType.SEC_FILINGS]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="Free/Premium",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "FINNHUB_API_KEY not set"
        )
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        if not self.is_available():
            raise ValueError("FINNHUB_API_KEY not set")
        
        time.sleep(self._rate_limit_delay)
        
        params = params or {}
        params["token"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        if symbols:
            all_news = []
            for symbol in symbols[:5]:
                params = {"symbol": symbol}
                if start:
                    params["from"] = start.strftime("%Y-%m-%d")
                if end:
                    params["to"] = end.strftime("%Y-%m-%d")
                
                data = self._request("company-news", params)
                
                for item in data[:limit]:
                    all_news.append(NewsItem(
                        timestamp=datetime.fromtimestamp(item.get("datetime", 0)),
                        headline=item.get("headline", ""),
                        summary=item.get("summary", "")[:500],
                        source=item.get("source"),
                        url=item.get("url"),
                        symbols=[symbol],
                        categories=[item.get("category", "")]
                    ))
            return all_news
        else:
            data = self._request("news", {"category": "general"})
            return [
                NewsItem(
                    timestamp=datetime.fromtimestamp(item.get("datetime", 0)),
                    headline=item.get("headline", ""),
                    summary=item.get("summary", "")[:500],
                    source=item.get("source"),
                    url=item.get("url"),
                    categories=[item.get("category", "")]
                )
                for item in data[:limit]
            ]
    
    def fetch_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[SentimentData]:
        data = self._request("news-sentiment", {"symbol": symbol})
        
        sentiment = data.get("sentiment", {})
        buzz = data.get("buzz", {})
        
        return [SentimentData(
            timestamp=datetime.now(),
            symbol=symbol,
            score=sentiment.get("bullishPercent", 0.5),
            source="finnhub",
            volume=buzz.get("articlesInLastWeek", 0),
            positive_mentions=int(sentiment.get("bullishPercent", 0) * buzz.get("articlesInLastWeek", 0)),
            negative_mentions=int(sentiment.get("bearishPercent", 0) * buzz.get("articlesInLastWeek", 0))
        )]
    
    def get_insider_transactions(
        self,
        symbol: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        return self._request("stock/insider-transactions", {"symbol": symbol})
    
    def get_earnings_surprises(self, symbol: str) -> List[Dict[str, Any]]:
        return self._request("stock/earnings", {"symbol": symbol})
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        return self._request("stock/recommendation", {"symbol": symbol})


class AltIndexAdapter(EnhancedDataAdapter):
    """
    AltIndex alternative data adapter.
    
    AI-powered stock ratings based on alternative data.
    
    Features:
    - AI stock scores (1-100)
    - Social sentiment aggregation
    - Job postings analysis
    - Web traffic trends
    - App download metrics
    - Patent filings
    
    Pricing:
    - Free: Basic features
    - Pro: $29/mo - Full features
    - Premium: $99/mo - API access
    """
    
    BASE_URL = "https://altindex.com/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALTINDEX_API_KEY")
    
    @property
    def name(self) -> str:
        return "AltIndex"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.SENTIMENT]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="$0-99/month",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "ALTINDEX_API_KEY not set"
        )


class StocktwitsAdapter(EnhancedDataAdapter):
    """
    Stocktwits social sentiment adapter.
    
    Free API for social trading sentiment.
    
    Features:
    - Real-time sentiment
    - Message volume
    - Trending symbols
    - Sector sentiment
    """
    
    BASE_URL = "https://api.stocktwits.com/api/2"
    
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or os.getenv("STOCKTWITS_ACCESS_TOKEN")
    
    @property
    def name(self) -> str:
        return "Stocktwits"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.SENTIMENT]
    
    def is_available(self) -> bool:
        return True
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            connected=True,
            subscription_tier="Free",
            data_types=self.supported_data_types
        )
    
    def fetch_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[SentimentData]:
        url = f"{self.BASE_URL}/streams/symbol/{symbol}.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        symbol_data = data.get("symbol", {})
        sentiment = symbol_data.get("sentiment", {})
        
        bullish = sentiment.get("bullish", 0) or 0
        bearish = sentiment.get("bearish", 0) or 0
        total = bullish + bearish or 1
        
        return [SentimentData(
            timestamp=datetime.now(),
            symbol=symbol,
            score=bullish / total,
            source="stocktwits",
            volume=data.get("messages", [{}])[0].get("id", 0) if data.get("messages") else 0,
            positive_mentions=bullish,
            negative_mentions=bearish,
            social_volume=len(data.get("messages", []))
        )]
    
    def get_trending(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE_URL}/trending/symbols.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json().get("symbols", [])


class RedditSentimentAdapter(EnhancedDataAdapter):
    """
    Reddit sentiment adapter (via third-party APIs).
    
    Tracks r/wallstreetbets and other investing subreddits.
    
    Note: Direct Reddit API has rate limits.
    Consider using aggregators like:
    - Quiver Quantitative
    - SwaggyStocks
    - Ape Wisdom
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("REDDIT_SENTIMENT_API_KEY")
    
    @property
    def name(self) -> str:
        return "Reddit Sentiment"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.SENTIMENT]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="Varies",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "REDDIT_SENTIMENT_API_KEY not set"
        )


class AlternativeDataAggregator(EnhancedDataAdapter):
    """
    Aggregates alternative data from multiple sources.
    """
    
    def __init__(self):
        self.providers: List[EnhancedDataAdapter] = []
        
        if os.getenv("FINNHUB_API_KEY"):
            self.providers.append(FinnhubAdapter())
        
        self.providers.append(StocktwitsAdapter())
        
        if os.getenv("ALTINDEX_API_KEY"):
            self.providers.append(AltIndexAdapter())
    
    @property
    def name(self) -> str:
        return "Alternative Data Aggregator"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.NEWS, DataType.SENTIMENT]
    
    def is_available(self) -> bool:
        return len(self.providers) > 0
    
    def get_status(self) -> ProviderStatus:
        provider_names = [p.name for p in self.providers]
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier=f"Active: {', '.join(provider_names)}",
            data_types=self.supported_data_types
        )
    
    def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        sentiments = {}
        
        for provider in self.providers:
            try:
                if DataType.SENTIMENT in provider.supported_data_types:
                    data = provider.fetch_sentiment(symbol)
                    if data:
                        sentiments[provider.name] = data[0].score
            except Exception:
                continue
        
        if sentiments:
            avg_score = sum(sentiments.values()) / len(sentiments)
            return {
                "symbol": symbol,
                "average_score": avg_score,
                "sources": sentiments,
                "bullish": avg_score > 0.6,
                "bearish": avg_score < 0.4,
                "neutral": 0.4 <= avg_score <= 0.6
            }
        
        return {"symbol": symbol, "average_score": 0.5, "sources": {}}


ALTERNATIVE_DATA_SETUP_GUIDE = """
=== Alternative Data Providers Setup Guide ===

1. FINNHUB (Free/Premium)
   - Sign up: https://finnhub.io/
   - Free tier: 60 calls/min
   - Features: News, sentiment, insider trades
   
   FINNHUB_API_KEY=your_key_here

2. ALTINDEX ($0-99/mo)
   - Sign up: https://altindex.com/
   - Features: AI stock scores, social aggregation
   
   ALTINDEX_API_KEY=your_key_here

3. STOCKTWITS (Free)
   - No API key needed for basic features
   - Features: Real-time social sentiment
   
   STOCKTWITS_ACCESS_TOKEN=optional_for_premium

4. REDDIT SENTIMENT
   - Direct API: Limited rate
   - Third-party aggregators:
     - Quiver Quantitative
     - SwaggyStocks
     - Ape Wisdom

5. ADDITIONAL SOURCES
   - Twitter/X: $100+/mo for full API
   - Google Trends: Free (manual or SerpAPI)
   - Web traffic: SimilarWeb, Semrush
   - App downloads: Sensor Tower, App Annie

6. USE CASES
   - Retail sentiment spikes → Momentum plays
   - Congressional trades → Information edge
   - Insider buying → Bullish signal
   - Job postings → Growth indicator
"""
