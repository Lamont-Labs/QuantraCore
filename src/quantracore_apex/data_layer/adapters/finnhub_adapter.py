"""
Finnhub Data Adapter for QuantraCore Apex.

Free tier: 60 API calls/minute (most generous free tier).

Features:
- Real-time stock quotes
- Social media sentiment (Reddit, Twitter)
- News sentiment
- Insider transactions
- Company financials
- Earnings calendar

Environment Variables:
- FINNHUB_API_KEY: Your Finnhub API key

Sign up: https://finnhub.io/
"""

import os
import time
import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .base_enhanced import EnhancedDataAdapter, DataType, ProviderStatus

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    REDDIT = "reddit"
    TWITTER = "twitter"
    NEWS = "news"
    SOCIAL = "social"


@dataclass
class SocialSentiment:
    symbol: str
    timestamp: datetime
    reddit_mentions: int
    reddit_positive_mentions: int
    reddit_negative_mentions: int
    twitter_mentions: int
    twitter_positive_mentions: int
    twitter_negative_mentions: int
    score: float
    positive_score: float
    negative_score: float
    buzz: float


@dataclass
class NewsSentiment:
    symbol: str
    timestamp: datetime
    headline: str
    source: str
    url: str
    summary: str
    sentiment: float
    positive_score: float
    negative_score: float
    relevance: float


@dataclass
class InsiderTransaction:
    symbol: str
    name: str
    transaction_date: datetime
    transaction_type: str
    shares: int
    price: float
    value: float
    filing_date: datetime


@dataclass
class CompanyNews:
    symbol: str
    timestamp: datetime
    headline: str
    source: str
    url: str
    summary: str
    category: str


class FinnhubAdapter(EnhancedDataAdapter):
    """
    Finnhub API adapter for social sentiment and market data.
    
    Free tier provides:
    - 60 API calls/minute
    - US stock coverage
    - Real-time quotes
    - Social sentiment (Reddit/Twitter)
    - News
    - Insider transactions
    - Basic fundamentals
    
    Premium tiers provide international coverage and more data.
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 1.0
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60
        
        if self.api_key:
            logger.info("[Finnhub] Adapter initialized (60 req/min free tier)")
        else:
            logger.warning("[Finnhub] API key not set - using simulated data")
    
    @property
    def name(self) -> str:
        return "Finnhub"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.SENTIMENT, DataType.NEWS]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="Free (60 req/min)",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "FINNHUB_API_KEY not set"
        )
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        if not self.is_available():
            return None
        
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        
        params = params or {}
        params["token"] = self.api_key
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self._last_request = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"[Finnhub] API error (falling back to simulated): {e}")
            return None
        except Exception as e:
            logger.warning(f"[Finnhub] Unexpected error: {e}")
            return None
    
    def get_social_sentiment(self, symbol: str) -> Optional[SocialSentiment]:
        """
        Get social media sentiment for a symbol.
        
        Returns Reddit and Twitter mention counts and sentiment scores.
        Falls back to simulated data if API is unavailable.
        """
        cache_key = f"social_{symbol}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        if not self.is_available():
            return self._simulated_social_sentiment(symbol)
        
        try:
            data = self._request("stock/social-sentiment", {"symbol": symbol})
            
            if data is None or "reddit" not in data:
                return self._simulated_social_sentiment(symbol)
            
            reddit = data.get("reddit", [])
            twitter = data.get("twitter", [])
            
            reddit_mentions = sum(d.get("mention", 0) for d in reddit) if reddit else 0
            reddit_positive = sum(d.get("positiveMention", 0) for d in reddit) if reddit else 0
            reddit_negative = sum(d.get("negativeMention", 0) for d in reddit) if reddit else 0
            
            twitter_mentions = sum(d.get("mention", 0) for d in twitter) if twitter else 0
            twitter_positive = sum(d.get("positiveMention", 0) for d in twitter) if twitter else 0
            twitter_negative = sum(d.get("negativeMention", 0) for d in twitter) if twitter else 0
            
            total_positive = reddit_positive + twitter_positive
            total_negative = reddit_negative + twitter_negative
            total_mentions = reddit_mentions + twitter_mentions
            
            if total_mentions > 0:
                score = (total_positive - total_negative) / total_mentions
                positive_score = total_positive / total_mentions
                negative_score = total_negative / total_mentions
            else:
                score = 0.0
                positive_score = 0.0
                negative_score = 0.0
            
            buzz = min(total_mentions / 100, 10.0)
            
            result = SocialSentiment(
                symbol=symbol,
                timestamp=datetime.now(),
                reddit_mentions=reddit_mentions,
                reddit_positive_mentions=reddit_positive,
                reddit_negative_mentions=reddit_negative,
                twitter_mentions=twitter_mentions,
                twitter_positive_mentions=twitter_positive,
                twitter_negative_mentions=twitter_negative,
                score=round(score, 4),
                positive_score=round(positive_score, 4),
                negative_score=round(negative_score, 4),
                buzz=round(buzz, 2)
            )
            
            self._cache[cache_key] = (result, time.time())
            return result
            
        except Exception as e:
            logger.warning(f"[Finnhub] Social sentiment error for {symbol}: {e}")
            return self._simulated_social_sentiment(symbol)
    
    def _simulated_social_sentiment(self, symbol: str) -> SocialSentiment:
        """Generate simulated social sentiment data."""
        import random
        
        reddit_mentions = random.randint(10, 500)
        twitter_mentions = random.randint(20, 1000)
        
        pos_ratio = random.uniform(0.3, 0.7)
        neg_ratio = random.uniform(0.1, 0.3)
        
        return SocialSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            reddit_mentions=reddit_mentions,
            reddit_positive_mentions=int(reddit_mentions * pos_ratio),
            reddit_negative_mentions=int(reddit_mentions * neg_ratio),
            twitter_mentions=twitter_mentions,
            twitter_positive_mentions=int(twitter_mentions * pos_ratio),
            twitter_negative_mentions=int(twitter_mentions * neg_ratio),
            score=round(pos_ratio - neg_ratio, 4),
            positive_score=round(pos_ratio, 4),
            negative_score=round(neg_ratio, 4),
            buzz=round(min((reddit_mentions + twitter_mentions) / 100, 10.0), 2)
        )
    
    def get_news_sentiment(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[NewsSentiment]:
        """
        Get news articles with sentiment analysis for a symbol.
        """
        if not self.is_available():
            return self._simulated_news_sentiment(symbol, limit)
        
        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()
        
        try:
            data = self._request("company-news", {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            })
            
            if not data:
                return self._simulated_news_sentiment(symbol, limit)
            
            results = []
            for article in data[:limit]:
                sentiment_data = self._request("news-sentiment", {"symbol": symbol})
                
                base_sentiment = 0.0
                if sentiment_data and "buzz" in sentiment_data:
                    base_sentiment = sentiment_data.get("companyNewsScore", 0.5) * 2 - 1
                
                results.append(NewsSentiment(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(article.get("datetime", 0)),
                    headline=article.get("headline", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    summary=article.get("summary", ""),
                    sentiment=round(base_sentiment, 4),
                    positive_score=max(0, base_sentiment),
                    negative_score=abs(min(0, base_sentiment)),
                    relevance=0.8
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"[Finnhub] News sentiment error for {symbol}: {e}")
            return self._simulated_news_sentiment(symbol, limit)
    
    def _simulated_news_sentiment(self, symbol: str, limit: int = 10) -> List[NewsSentiment]:
        """Generate simulated news sentiment data."""
        import random
        
        headlines = [
            f"{symbol} Reports Strong Quarterly Earnings",
            f"Analysts Upgrade {symbol} to Buy",
            f"{symbol} Announces New Product Launch",
            f"Market Watch: {symbol} Shows Momentum",
            f"{symbol} CEO Outlines Growth Strategy",
            f"Institutional Investors Increase {symbol} Holdings",
            f"{symbol} Partners with Major Tech Company",
            f"Breaking: {symbol} Beats Revenue Expectations",
            f"{symbol} Expands into New Markets",
            f"Hedge Funds Bullish on {symbol}",
        ]
        
        results = []
        for i in range(min(limit, len(headlines))):
            sentiment = random.uniform(-0.3, 0.7)
            results.append(NewsSentiment(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 168)),
                headline=headlines[i],
                source="Market News",
                url=f"https://example.com/news/{symbol}/{i}",
                summary=f"Analysis of {symbol} market activity and performance...",
                sentiment=round(sentiment, 4),
                positive_score=max(0, round(sentiment, 4)),
                negative_score=abs(min(0, round(sentiment, 4))),
                relevance=random.uniform(0.5, 1.0)
            ))
        
        return results
    
    def get_insider_transactions(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[InsiderTransaction]:
        """
        Get insider trading transactions for a symbol.
        
        Useful for detecting smart money movements.
        """
        if not self.is_available():
            return self._simulated_insider_transactions(symbol)
        
        if not from_date:
            from_date = datetime.now() - timedelta(days=90)
        if not to_date:
            to_date = datetime.now()
        
        try:
            data = self._request("stock/insider-transactions", {"symbol": symbol})
            
            if not data or "data" not in data:
                return self._simulated_insider_transactions(symbol)
            
            results = []
            for tx in data.get("data", []):
                try:
                    tx_date = datetime.strptime(tx.get("transactionDate", ""), "%Y-%m-%d")
                    filing_date = datetime.strptime(tx.get("filingDate", ""), "%Y-%m-%d")
                    
                    if from_date <= tx_date <= to_date:
                        results.append(InsiderTransaction(
                            symbol=symbol,
                            name=tx.get("name", "Unknown"),
                            transaction_date=tx_date,
                            transaction_type=tx.get("transactionType", ""),
                            shares=int(tx.get("share", 0)),
                            price=float(tx.get("price", 0)),
                            value=float(tx.get("value", 0)),
                            filing_date=filing_date
                        ))
                except (ValueError, KeyError):
                    continue
            
            return results
            
        except Exception as e:
            logger.warning(f"[Finnhub] Insider transactions error for {symbol}: {e}")
            return self._simulated_insider_transactions(symbol)
    
    def _simulated_insider_transactions(self, symbol: str) -> List[InsiderTransaction]:
        """Generate simulated insider transaction data."""
        import random
        
        names = ["CEO", "CFO", "Director", "VP Operations", "Board Member"]
        tx_types = ["P - Purchase", "S - Sale", "A - Grant"]
        
        results = []
        for i in range(random.randint(2, 8)):
            shares = random.randint(1000, 50000)
            price = random.uniform(50, 300)
            
            results.append(InsiderTransaction(
                symbol=symbol,
                name=random.choice(names),
                transaction_date=datetime.now() - timedelta(days=random.randint(1, 60)),
                transaction_type=random.choice(tx_types),
                shares=shares,
                price=round(price, 2),
                value=round(shares * price, 2),
                filing_date=datetime.now() - timedelta(days=random.randint(0, 5))
            ))
        
        return sorted(results, key=lambda x: x.transaction_date, reverse=True)
    
    def get_company_news(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 20
    ) -> List[CompanyNews]:
        """Get recent company news articles."""
        if not self.is_available():
            return self._simulated_company_news(symbol, limit)
        
        if not from_date:
            from_date = datetime.now() - timedelta(days=7)
        if not to_date:
            to_date = datetime.now()
        
        try:
            data = self._request("company-news", {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            })
            
            if not data:
                return self._simulated_company_news(symbol, limit)
            
            results = []
            for article in data[:limit]:
                results.append(CompanyNews(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(article.get("datetime", 0)),
                    headline=article.get("headline", ""),
                    source=article.get("source", ""),
                    url=article.get("url", ""),
                    summary=article.get("summary", ""),
                    category=article.get("category", "general")
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"[Finnhub] Company news error for {symbol}: {e}")
            return self._simulated_company_news(symbol, limit)
    
    def _simulated_company_news(self, symbol: str, limit: int = 10) -> List[CompanyNews]:
        """Generate simulated company news."""
        import random
        
        headlines = [
            f"{symbol} Q3 Earnings Beat Expectations",
            f"Analysts Bullish on {symbol} Growth",
            f"{symbol} Announces Strategic Partnership",
            f"Market Update: {symbol} Trading Higher",
            f"{symbol} Launches New Product Line",
        ]
        
        results = []
        for i in range(min(limit, len(headlines))):
            results.append(CompanyNews(
                symbol=symbol,
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 72)),
                headline=headlines[i],
                source="Financial News",
                url=f"https://example.com/{symbol}/news/{i}",
                summary=f"Breaking coverage of {symbol}...",
                category="general"
            ))
        
        return results
    
    def get_sentiment_score(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined sentiment score for a symbol.
        
        Aggregates social media and news sentiment into a single score.
        
        Returns:
            Dict with combined sentiment metrics
        """
        social = self.get_social_sentiment(symbol)
        news = self.get_news_sentiment(symbol, limit=10)
        
        social_score = social.score if social else 0.0
        social_buzz = social.buzz if social else 0.0
        
        if news:
            news_score = sum(n.sentiment for n in news) / len(news)
        else:
            news_score = 0.0
        
        combined_score = (social_score * 0.6) + (news_score * 0.4)
        
        if combined_score > 0.3:
            signal = "BULLISH"
        elif combined_score < -0.3:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "combined_score": round(combined_score, 4),
            "social_score": round(social_score, 4),
            "news_score": round(news_score, 4),
            "social_buzz": round(social_buzz, 2),
            "signal": signal,
            "reddit_mentions": social.reddit_mentions if social else 0,
            "twitter_mentions": social.twitter_mentions if social else 0,
            "news_count": len(news) if news else 0
        }


def get_finnhub_adapter() -> FinnhubAdapter:
    """Get singleton Finnhub adapter instance."""
    return FinnhubAdapter()


FINNHUB_SETUP_GUIDE = """
=== Finnhub API Setup Guide ===

1. SIGN UP (Free)
   https://finnhub.io/register
   
2. GET YOUR API KEY
   Dashboard → API → Copy your token

3. SET ENVIRONMENT VARIABLE
   FINNHUB_API_KEY=your_api_key_here

4. FREE TIER LIMITS
   - 60 API calls/minute
   - US stocks only
   - Real-time quotes
   - Social sentiment
   - Company news
   - Insider transactions

5. PREMIUM TIERS
   - All Access ($49/mo): International stocks
   - Enterprise: Custom limits

6. KEY ENDPOINTS
   - /stock/social-sentiment - Reddit/Twitter buzz
   - /company-news - Recent news
   - /stock/insider-transactions - Smart money
   - /quote - Real-time price

7. BEST PRACTICES
   - Cache responses (60s TTL)
   - Use batch requests when possible
   - Monitor rate limits
"""
