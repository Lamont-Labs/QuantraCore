"""
Multi-Source Sentiment Aggregator for QuantraCore Apex.

Combines sentiment data from multiple free-tier sources:
- FRED: Economic regime detection
- Finnhub: Social sentiment (Reddit/Twitter)
- Alpha Vantage: AI-powered news sentiment

Provides unified sentiment analysis for trading decisions.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

from .finnhub_adapter import FinnhubAdapter, SocialSentiment
from .alpha_vantage_adapter import AlphaVantageAdapter, NewsSentimentArticle
from .economic_adapter import FredAdapter, EconomicIndicator, MacroRegime

logger = logging.getLogger(__name__)


@dataclass
class UnifiedSentiment:
    symbol: str
    timestamp: datetime
    social_score: float
    news_score: float
    combined_score: float
    signal: str
    confidence: float
    social_buzz: float
    reddit_mentions: int
    twitter_mentions: int
    news_articles: int
    bullish_articles: int
    bearish_articles: int
    economic_regime: str
    risk_appetite: str


@dataclass
class MarketSentimentSnapshot:
    timestamp: datetime
    overall_sentiment: str
    market_fear_greed: float
    economic_regime: MacroRegime
    top_bullish_symbols: List[str]
    top_bearish_symbols: List[str]
    trending_symbols: List[str]
    active_catalysts: List[str]


class MultiSourceSentimentAggregator:
    """
    Aggregates sentiment from multiple free-tier data sources.
    
    Sources:
    - Finnhub: Social sentiment (60 req/min free)
    - Alpha Vantage: News sentiment (500 req/day free)
    - FRED: Economic indicators (120 req/min free)
    
    Provides unified sentiment scoring for trading decisions.
    """
    
    def __init__(self):
        self.finnhub = FinnhubAdapter()
        self.alpha_vantage = AlphaVantageAdapter()
        self.fred = FredAdapter()
        
        self._weights = {
            "social": 0.35,
            "news": 0.45,
            "economic": 0.20
        }
        
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300
        
        available = []
        if self.finnhub.is_available():
            available.append("Finnhub")
        if self.alpha_vantage.is_available():
            available.append("AlphaVantage")
        if self.fred.is_available():
            available.append("FRED")
        
        logger.info(f"[SentimentAggregator] Initialized with sources: {', '.join(available) or 'Simulated'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all sentiment providers."""
        return {
            "finnhub": {
                "available": self.finnhub.is_available(),
                "rate_limit": "60 req/min",
                "coverage": "Social (Reddit/Twitter)"
            },
            "alpha_vantage": {
                "available": self.alpha_vantage.is_available(),
                "rate_limit": "500 req/day",
                "coverage": "News Sentiment (AI)"
            },
            "fred": {
                "available": self.fred.is_available(),
                "rate_limit": "120 req/min",
                "coverage": "Economic Indicators"
            }
        }
    
    def get_unified_sentiment(self, symbol: str) -> UnifiedSentiment:
        """
        Get unified sentiment analysis for a symbol.
        
        Combines social, news, and economic data into a single score.
        
        Args:
            symbol: Stock ticker (e.g., "AAPL")
            
        Returns:
            UnifiedSentiment with combined scores and signal
        """
        social_data = self.finnhub.get_social_sentiment(symbol)
        news_data = self.alpha_vantage.get_sentiment_score(symbol)
        
        try:
            regime = self.fred.get_current_regime()
            economic_regime = regime.regime
            risk_appetite = regime.risk_appetite
        except Exception:
            economic_regime = "NEUTRAL"
            risk_appetite = "MODERATE"
        
        social_score = social_data.score if social_data else 0.0
        social_buzz = social_data.buzz if social_data else 0.0
        reddit_mentions = social_data.reddit_mentions if social_data else 0
        twitter_mentions = social_data.twitter_mentions if social_data else 0
        
        news_score = news_data.get("sentiment_score", 0.0)
        news_articles = news_data.get("article_count", 0)
        bullish_articles = news_data.get("bullish_count", 0)
        bearish_articles = news_data.get("bearish_count", 0)
        
        economic_modifier = 0.0
        if economic_regime == "RISK_ON":
            economic_modifier = 0.1
        elif economic_regime == "RISK_OFF":
            economic_modifier = -0.1
        
        combined_score = (
            social_score * self._weights["social"] +
            news_score * self._weights["news"] +
            economic_modifier
        )
        combined_score = max(-1.0, min(1.0, combined_score))
        
        if combined_score > 0.3:
            signal = "STRONG_BUY"
            confidence = min(0.9, 0.6 + combined_score)
        elif combined_score > 0.15:
            signal = "BUY"
            confidence = 0.5 + combined_score
        elif combined_score < -0.3:
            signal = "STRONG_SELL"
            confidence = min(0.9, 0.6 + abs(combined_score))
        elif combined_score < -0.15:
            signal = "SELL"
            confidence = 0.5 + abs(combined_score)
        else:
            signal = "HOLD"
            confidence = 0.4
        
        return UnifiedSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            social_score=round(social_score, 4),
            news_score=round(news_score, 4),
            combined_score=round(combined_score, 4),
            signal=signal,
            confidence=round(confidence, 3),
            social_buzz=round(social_buzz, 2),
            reddit_mentions=reddit_mentions,
            twitter_mentions=twitter_mentions,
            news_articles=news_articles,
            bullish_articles=bullish_articles,
            bearish_articles=bearish_articles,
            economic_regime=economic_regime,
            risk_appetite=risk_appetite
        )
    
    def get_batch_sentiment(self, symbols: List[str]) -> Dict[str, UnifiedSentiment]:
        """
        Get sentiment for multiple symbols.
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dict mapping symbol to UnifiedSentiment
        """
        results = {}
        for symbol in symbols[:20]:
            try:
                results[symbol] = self.get_unified_sentiment(symbol)
            except Exception as e:
                logger.warning(f"[SentimentAggregator] Error for {symbol}: {e}")
                continue
        return results
    
    def get_market_snapshot(self) -> MarketSentimentSnapshot:
        """
        Get overall market sentiment snapshot.
        
        Combines economic regime with trending symbols.
        """
        try:
            regime = self.fred.get_current_regime()
        except Exception:
            regime = MacroRegime(
                timestamp=datetime.now(),
                regime="NEUTRAL",
                risk_appetite="MODERATE",
                yield_curve="NORMAL",
                inflation_trend="MODERATE",
                growth_trend="MODERATE",
                fed_stance="NEUTRAL",
                confidence=0.3
            )
        
        if regime.risk_appetite == "HIGH":
            fear_greed = 0.7
            overall = "BULLISH"
        elif regime.risk_appetite == "LOW":
            fear_greed = 0.3
            overall = "BEARISH"
        else:
            fear_greed = 0.5
            overall = "NEUTRAL"
        
        market_leaders = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA"]
        
        return MarketSentimentSnapshot(
            timestamp=datetime.now(),
            overall_sentiment=overall,
            market_fear_greed=fear_greed,
            economic_regime=regime,
            top_bullish_symbols=market_leaders[:3],
            top_bearish_symbols=[],
            trending_symbols=market_leaders,
            active_catalysts=self._get_active_catalysts()
        )
    
    def _get_active_catalysts(self) -> List[str]:
        """Identify active market catalysts."""
        catalysts = []
        
        try:
            regime = self.fred.get_current_regime()
            
            if regime.fed_stance == "RESTRICTIVE":
                catalysts.append("Fed hawkish - rate sensitive")
            elif regime.fed_stance == "ACCOMMODATIVE":
                catalysts.append("Fed dovish - liquidity boost")
            
            if regime.inflation_trend == "HIGH":
                catalysts.append("Elevated inflation")
            
            if regime.yield_curve == "INVERTED":
                catalysts.append("Yield curve inversion - recession risk")
            
            if regime.growth_trend == "STRONG":
                catalysts.append("Strong GDP growth")
            elif regime.growth_trend == "CONTRACTION":
                catalysts.append("Economic contraction risk")
                
        except Exception:
            pass
        
        return catalysts if catalysts else ["Normal market conditions"]
    
    def get_sentiment_for_runner_hunting(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get sentiment data formatted for RunnerHunter integration.
        
        Returns sentiment features for breakout validation.
        """
        unified = self.get_unified_sentiment(symbol)
        
        sentiment_boost = 0.0
        if unified.signal == "STRONG_BUY":
            sentiment_boost = 0.15
        elif unified.signal == "BUY":
            sentiment_boost = 0.08
        elif unified.signal == "STRONG_SELL":
            sentiment_boost = -0.15
        elif unified.signal == "SELL":
            sentiment_boost = -0.08
        
        buzz_multiplier = 1.0
        if unified.social_buzz > 5.0:
            buzz_multiplier = 1.2
        elif unified.social_buzz > 2.0:
            buzz_multiplier = 1.1
        
        regime_boost = 0.0
        if unified.economic_regime == "RISK_ON":
            regime_boost = 0.05
        elif unified.economic_regime == "RISK_OFF":
            regime_boost = -0.05
        
        return {
            "symbol": symbol,
            "sentiment_score": unified.combined_score,
            "sentiment_signal": unified.signal,
            "sentiment_boost": sentiment_boost,
            "buzz_multiplier": buzz_multiplier,
            "regime_boost": regime_boost,
            "total_adjustment": sentiment_boost + regime_boost,
            "confidence": unified.confidence,
            "social_mentions": unified.reddit_mentions + unified.twitter_mentions,
            "news_count": unified.news_articles,
            "bullish_ratio": (
                unified.bullish_articles / max(1, unified.news_articles)
            ),
            "economic_regime": unified.economic_regime,
            "risk_appetite": unified.risk_appetite
        }


def get_sentiment_aggregator() -> MultiSourceSentimentAggregator:
    """Get singleton sentiment aggregator instance."""
    return MultiSourceSentimentAggregator()


FREE_TIER_API_SUMMARY = """
=== Free-Tier Sentiment Data Sources ===

1. FRED (Federal Reserve Economic Data)
   - Cost: FREE (no limits)
   - Rate: 120 requests/minute
   - Coverage: 800,000+ economic indicators
   - Best for: Macro regime detection
   - Key indicators: Fed rates, CPI, GDP, yield curve
   - Setup: FRED_API_KEY

2. Finnhub
   - Cost: FREE tier available
   - Rate: 60 requests/minute
   - Coverage: Social sentiment (Reddit, Twitter)
   - Best for: Retail sentiment, meme stock detection
   - Key data: Mention counts, sentiment scores
   - Setup: FINNHUB_API_KEY

3. Alpha Vantage
   - Cost: FREE tier (500/day)
   - Rate: 5 requests/minute
   - Coverage: AI-powered news sentiment
   - Best for: News-driven moves, catalyst detection
   - Key data: Article sentiment, relevance scores
   - Setup: ALPHA_VANTAGE_API_KEY

=== Combined Value ===

Total free capacity per day:
- 172,800 FRED calls
- 86,400 Finnhub calls
- 500 Alpha Vantage calls

Together provides:
- Economic regime detection (macro)
- Social sentiment (retail behavior)
- News sentiment (institutional catalyst)
- Multi-factor confirmation for trades

=== Integration Points ===

1. RunnerHunter: Sentiment boost for breakout validation
2. ApexCore V4: Additional prediction features
3. Risk Management: Regime-based position sizing
4. Alert System: Sentiment-triggered notifications
"""
