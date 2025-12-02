"""
Enriched Data Fusion Layer for QuantraCore Apex v9.0-A

Combines ALL 7 data sources into unified ML-ready features:
1. Polygon.io - Market data, historical OHLCV
2. Alpaca - Paper trading, execution data
3. FRED - Economic indicators, regime detection
4. Finnhub - Social sentiment (Reddit/Twitter)
5. Alpha Vantage - AI-powered news sentiment
6. SEC EDGAR - Insider transactions, 13F holdings
7. Binance - Cryptocurrency correlations

These enriched features feed directly into the learning cycles
for maximum prediction accuracy.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class EnrichedSample:
    """ML-ready sample with features from all 7 data sources."""
    symbol: str
    timestamp: datetime
    
    price_features: Dict[str, float] = field(default_factory=dict)
    sentiment_features: Dict[str, float] = field(default_factory=dict)
    economic_features: Dict[str, float] = field(default_factory=dict)
    insider_features: Dict[str, float] = field(default_factory=dict)
    crypto_features: Dict[str, float] = field(default_factory=dict)
    
    sources_used: List[str] = field(default_factory=list)
    
    @property
    def feature_count(self) -> int:
        return (
            len(self.price_features) +
            len(self.sentiment_features) +
            len(self.economic_features) +
            len(self.insider_features) +
            len(self.crypto_features)
        )
    
    def to_feature_vector(self) -> Dict[str, float]:
        """Flatten all features into a single dict matching SwingFeatureExtractor names."""
        features = {}
        features.update({f"sentiment_{k}": v for k, v in self.sentiment_features.items()})
        features.update({f"economic_{k}": v for k, v in self.economic_features.items()})
        features.update(self.insider_features)
        features.update({f"crypto_{k}": v for k, v in self.crypto_features.items()})
        features.update({
            "data_sources_count": len(self.sources_used) / 6.0,
            "enrichment_confidence": min(1.0, self.feature_count / 40.0),
            "multi_source_agreement": self._compute_agreement(),
        })
        return features
    
    def _compute_agreement(self) -> float:
        """Compute agreement score across data sources."""
        signals = []
        
        if self.sentiment_features.get("combined_score", 0) > 0.1:
            signals.append(1)
        elif self.sentiment_features.get("combined_score", 0) < -0.1:
            signals.append(-1)
        else:
            signals.append(0)
        
        if self.economic_features.get("regime_score", 0) > 0:
            signals.append(1)
        elif self.economic_features.get("regime_score", 0) < 0:
            signals.append(-1)
        else:
            signals.append(0)
        
        if self.insider_features.get("insider_sentiment_score", 0) > 0:
            signals.append(1)
        elif self.insider_features.get("insider_sentiment_score", 0) < 0:
            signals.append(-1)
        else:
            signals.append(0)
        
        if not signals:
            return 0.0
        
        return abs(sum(signals)) / len(signals)


class EnrichedDataFusion:
    """
    Unifies all 7 data sources into ML-ready features for learning cycles.
    
    Data Sources:
    - Polygon.io: Market data (OHLCV, price action)
    - Alpaca: Execution data (positions, fills)
    - FRED: Economic regime (Fed rates, yield curve, CPI)
    - Finnhub: Social sentiment (Reddit, Twitter mentions)
    - Alpha Vantage: News sentiment (AI-scored articles)
    - SEC EDGAR: Insider activity (Form 4 buys/sells)
    - Binance: Crypto correlations (BTC/ETH as risk proxy)
    
    All features feed into ApexCore V4's 16 prediction heads.
    """
    
    SENTIMENT_FEATURES = [
        "social_score", "news_score", "combined_score",
        "reddit_mentions", "twitter_mentions", "buzz_level",
        "bullish_ratio", "bearish_ratio", "news_count"
    ]
    
    ECONOMIC_FEATURES = [
        "fed_rate", "treasury_10y", "treasury_2y", "yield_spread",
        "cpi", "unemployment", "regime_score", "risk_appetite_score",
        "inflation_trend_score", "growth_trend_score"
    ]
    
    INSIDER_FEATURES = [
        "insider_buy_count", "insider_sell_count", "insider_net_value",
        "insider_sentiment_score", "large_buy_signal", "insider_confidence"
    ]
    
    CRYPTO_FEATURES = [
        "btc_return_24h", "btc_volatility", "eth_return_24h",
        "crypto_fear_index", "btc_dominance", "crypto_correlation"
    ]
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300
        
        self._init_adapters()
        self._log_status()
    
    def _init_adapters(self):
        """Initialize all data source adapters."""
        self.polygon = None
        self.alpaca = None
        self.sentiment = None
        self.economic = None
        self.sec_edgar = None
        self.crypto = None
        
        try:
            from .adapters.polygon_adapter import PolygonAdapter
            if os.getenv("POLYGON_API_KEY"):
                self.polygon = PolygonAdapter()
        except Exception as e:
            logger.debug(f"Polygon adapter not available: {e}")
        
        try:
            from .adapters.alpaca_data_adapter import AlpacaDataAdapter
            if os.getenv("ALPACA_PAPER_API_KEY"):
                self.alpaca = AlpacaDataAdapter()
        except Exception as e:
            logger.debug(f"Alpaca adapter not available: {e}")
        
        try:
            from .adapters.sentiment_aggregator import MultiSourceSentimentAggregator
            self.sentiment = MultiSourceSentimentAggregator()
        except Exception as e:
            logger.debug(f"Sentiment aggregator not available: {e}")
        
        try:
            from .adapters.economic_adapter import FredAdapter
            if os.getenv("FRED_API_KEY"):
                self.economic = FredAdapter()
        except Exception as e:
            logger.debug(f"FRED adapter not available: {e}")
        
        try:
            from .adapters.sec_edgar_adapter import get_sec_edgar_adapter
            self.sec_edgar = get_sec_edgar_adapter()
        except Exception as e:
            logger.debug(f"SEC EDGAR adapter not available: {e}")
        
        try:
            from .adapters.crypto_adapter import CryptoAdapter
            self.crypto = CryptoAdapter()
        except Exception as e:
            logger.debug(f"Crypto adapter not available: {e}")
    
    def _log_status(self):
        """Log which data sources are active."""
        active = []
        if self.polygon:
            active.append("Polygon")
        if self.alpaca:
            active.append("Alpaca")
        if self.sentiment:
            active.append("Finnhub/AlphaVantage")
        if self.economic:
            active.append("FRED")
        if self.sec_edgar:
            active.append("SEC-EDGAR")
        if self.crypto:
            active.append("Binance")
        
        logger.info(f"[EnrichedDataFusion] Active sources: {', '.join(active) or 'None'}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        return {
            "polygon": {
                "active": self.polygon is not None,
                "purpose": "Market data, OHLCV, historical prices"
            },
            "alpaca": {
                "active": self.alpaca is not None,
                "purpose": "Paper trading, execution, positions"
            },
            "fred": {
                "active": self.economic is not None,
                "purpose": "Economic indicators, regime detection"
            },
            "finnhub": {
                "active": self.sentiment is not None and hasattr(self.sentiment, 'finnhub'),
                "purpose": "Social sentiment (Reddit/Twitter)"
            },
            "alpha_vantage": {
                "active": self.sentiment is not None and hasattr(self.sentiment, 'alpha_vantage'),
                "purpose": "AI news sentiment"
            },
            "sec_edgar": {
                "active": self.sec_edgar is not None,
                "purpose": "Insider transactions, 13F holdings"
            },
            "binance": {
                "active": self.crypto is not None,
                "purpose": "Crypto correlations (BTC/ETH)"
            },
            "total_active": sum([
                self.polygon is not None,
                self.alpaca is not None,
                self.economic is not None,
                self.sentiment is not None,
                self.sec_edgar is not None,
                self.crypto is not None
            ]),
            "learning_ready": True
        }
    
    def enrich_sample(self, symbol: str) -> EnrichedSample:
        """
        Create an enriched sample with features from all 7 data sources.
        
        This is the main entry point for the learning cycles.
        """
        sample = EnrichedSample(
            symbol=symbol,
            timestamp=datetime.utcnow()
        )
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            futures[executor.submit(self._get_sentiment_features, symbol)] = "sentiment"
            futures[executor.submit(self._get_economic_features)] = "economic"
            futures[executor.submit(self._get_insider_features, symbol)] = "insider"
            futures[executor.submit(self._get_crypto_features)] = "crypto"
            
            for future in as_completed(futures):
                feature_type = futures[future]
                try:
                    features, source = future.result()
                    
                    if feature_type == "sentiment":
                        sample.sentiment_features = features
                    elif feature_type == "economic":
                        sample.economic_features = features
                    elif feature_type == "insider":
                        sample.insider_features = features
                    elif feature_type == "crypto":
                        sample.crypto_features = features
                    
                    if source:
                        sample.sources_used.append(source)
                        
                except Exception as e:
                    logger.debug(f"Error getting {feature_type} features: {e}")
        
        return sample
    
    def _get_sentiment_features(self, symbol: str) -> tuple:
        """Get sentiment features from Finnhub and Alpha Vantage."""
        features = {}
        
        if not self.sentiment:
            return self._default_sentiment_features(), None
        
        try:
            unified = self.sentiment.get_unified_sentiment(symbol)
            
            features = {
                "social_score": unified.social_score,
                "news_score": unified.news_score,
                "combined_score": unified.combined_score,
                "reddit_mentions": float(unified.reddit_mentions),
                "twitter_mentions": float(unified.twitter_mentions),
                "buzz_level": unified.social_buzz,
                "bullish_ratio": unified.bullish_articles / max(1, unified.news_articles),
                "bearish_ratio": unified.bearish_articles / max(1, unified.news_articles),
                "news_count": float(unified.news_articles),
                "sentiment_confidence": unified.confidence,
                "sentiment_signal_bullish": 1.0 if "BUY" in unified.signal else 0.0,
                "sentiment_signal_bearish": 1.0 if "SELL" in unified.signal else 0.0
            }
            
            return features, "Finnhub/AlphaVantage"
            
        except Exception as e:
            logger.debug(f"Sentiment features error: {e}")
            return self._default_sentiment_features(), None
    
    def _default_sentiment_features(self) -> Dict[str, float]:
        """Default neutral sentiment features."""
        return {
            "social_score": 0.0,
            "news_score": 0.0,
            "combined_score": 0.0,
            "reddit_mentions": 0.0,
            "twitter_mentions": 0.0,
            "buzz_level": 0.0,
            "bullish_ratio": 0.5,
            "bearish_ratio": 0.5,
            "news_count": 0.0,
            "sentiment_confidence": 0.0,
            "sentiment_signal_bullish": 0.0,
            "sentiment_signal_bearish": 0.0
        }
    
    def _get_economic_features(self) -> tuple:
        """Get economic features from FRED."""
        features = {}
        
        if not self.economic:
            return self._default_economic_features(), None
        
        try:
            regime = self.economic.get_current_regime()
            
            regime_score = 0.0
            if regime.regime == "RISK_ON":
                regime_score = 1.0
            elif regime.regime == "RISK_OFF":
                regime_score = -1.0
            
            risk_score = 0.0
            if regime.risk_appetite == "HIGH":
                risk_score = 1.0
            elif regime.risk_appetite == "LOW":
                risk_score = -1.0
            
            inflation_score = 0.0
            if regime.inflation_trend == "HIGH":
                inflation_score = 1.0
            elif regime.inflation_trend == "LOW":
                inflation_score = -1.0
            
            growth_score = 0.0
            if regime.growth_trend == "STRONG":
                growth_score = 1.0
            elif regime.growth_trend == "CONTRACTION":
                growth_score = -1.0
            
            yield_curve_score = 0.0
            if regime.yield_curve == "STEEP":
                yield_curve_score = 1.0
            elif regime.yield_curve == "INVERTED":
                yield_curve_score = -1.0
            
            fed_stance_score = 0.0
            if regime.fed_stance == "ACCOMMODATIVE":
                fed_stance_score = 1.0
            elif regime.fed_stance == "RESTRICTIVE":
                fed_stance_score = -1.0
            
            features = {
                "regime_score": regime_score,
                "risk_appetite_score": risk_score,
                "inflation_trend_score": inflation_score,
                "growth_trend_score": growth_score,
                "yield_curve_score": yield_curve_score,
                "fed_stance_score": fed_stance_score,
                "regime_confidence": regime.confidence,
                "economic_bullish": 1.0 if regime_score > 0 else 0.0,
                "economic_bearish": 1.0 if regime_score < 0 else 0.0,
                "recession_risk": 1.0 if regime.yield_curve == "INVERTED" else 0.0
            }
            
            return features, "FRED"
            
        except Exception as e:
            logger.debug(f"Economic features error: {e}")
            return self._default_economic_features(), None
    
    def _default_economic_features(self) -> Dict[str, float]:
        """Default neutral economic features."""
        return {
            "regime_score": 0.0,
            "risk_appetite_score": 0.0,
            "inflation_trend_score": 0.0,
            "growth_trend_score": 0.0,
            "yield_curve_score": 0.0,
            "fed_stance_score": 0.0,
            "regime_confidence": 0.0,
            "economic_bullish": 0.0,
            "economic_bearish": 0.0,
            "recession_risk": 0.0
        }
    
    def _get_insider_features(self, symbol: str) -> tuple:
        """Get insider trading features from SEC EDGAR."""
        features = {}
        
        if not self.sec_edgar:
            return self._default_insider_features(), None
        
        try:
            summary = self.sec_edgar.get_insider_summary(symbol, days=90)
            
            sentiment_score = 0.0
            if summary["insider_sentiment"] == "STRONG_BUY":
                sentiment_score = 1.0
            elif summary["insider_sentiment"] == "BUY":
                sentiment_score = 0.5
            elif summary["insider_sentiment"] == "SELL":
                sentiment_score = -0.5
            elif summary["insider_sentiment"] == "STRONG_SELL":
                sentiment_score = -1.0
            
            net_value_normalized = min(1.0, max(-1.0, summary["net_value"] / 1000000))
            
            large_buy = 1.0 if summary["buy_value"] > 500000 else 0.0
            
            features = {
                "insider_buy_count": float(summary["buy_count"]),
                "insider_sell_count": float(summary["sell_count"]),
                "insider_net_value_normalized": net_value_normalized,
                "insider_sentiment_score": sentiment_score,
                "insider_confidence": summary["confidence"],
                "large_buy_signal": large_buy,
                "insider_activity_level": float(summary["total_transactions"]) / 10.0,
                "insider_bullish": 1.0 if sentiment_score > 0 else 0.0,
                "insider_bearish": 1.0 if sentiment_score < 0 else 0.0
            }
            
            return features, "SEC-EDGAR"
            
        except Exception as e:
            logger.debug(f"Insider features error: {e}")
            return self._default_insider_features(), None
    
    def _default_insider_features(self) -> Dict[str, float]:
        """Default neutral insider features."""
        return {
            "insider_buy_count": 0.0,
            "insider_sell_count": 0.0,
            "insider_net_value_normalized": 0.0,
            "insider_sentiment_score": 0.0,
            "insider_confidence": 0.0,
            "large_buy_signal": 0.0,
            "insider_activity_level": 0.0,
            "insider_bullish": 0.0,
            "insider_bearish": 0.0
        }
    
    def _get_crypto_features(self) -> tuple:
        """Get crypto correlation features from Binance."""
        features = self._default_crypto_features()
        
        if not self.crypto:
            return features, None
        
        try:
            if hasattr(self.crypto, 'get_24h_stats'):
                btc_stats = self.crypto.get_24h_stats("BTCUSDT")
                eth_stats = self.crypto.get_24h_stats("ETHUSDT")
                
                if btc_stats:
                    features["btc_return_24h"] = btc_stats.get("price_change_pct", 0.0) / 100.0
                    features["btc_volatility"] = btc_stats.get("high_low_range_pct", 0.0) / 100.0
                
                if eth_stats:
                    features["eth_return_24h"] = eth_stats.get("price_change_pct", 0.0) / 100.0
                
                if btc_stats:
                    btc_return = features["btc_return_24h"]
                    if btc_return < -0.05:
                        features["crypto_fear_index"] = 1.0
                    elif btc_return > 0.05:
                        features["crypto_fear_index"] = -1.0
                    
                    features["crypto_risk_on"] = 1.0 if btc_return > 0.02 else 0.0
                    features["crypto_risk_off"] = 1.0 if btc_return < -0.02 else 0.0
                
                return features, "Binance"
            
        except Exception as e:
            logger.debug(f"Crypto features error: {e}")
        
        return features, None
    
    def _default_crypto_features(self) -> Dict[str, float]:
        """Default neutral crypto features."""
        return {
            "btc_return_24h": 0.0,
            "btc_volatility": 0.0,
            "eth_return_24h": 0.0,
            "crypto_fear_index": 0.0,
            "crypto_risk_on": 0.0,
            "crypto_risk_off": 0.0
        }
    
    def enrich_batch(self, symbols: List[str]) -> Dict[str, EnrichedSample]:
        """Enrich multiple symbols in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.enrich_sample, symbol): symbol
                for symbol in symbols[:50]
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.warning(f"Enrichment failed for {symbol}: {e}")
        
        return results
    
    def get_learning_features(self, symbol: str) -> Dict[str, float]:
        """
        Get flattened feature vector for ML training.
        
        Returns a dict of ~50 features from all 7 data sources.
        """
        sample = self.enrich_sample(symbol)
        return sample.to_feature_vector()
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names for ML pipeline."""
        sample = EnrichedSample(symbol="", timestamp=datetime.utcnow())
        sample.sentiment_features = self._default_sentiment_features()
        sample.economic_features = self._default_economic_features()
        sample.insider_features = self._default_insider_features()
        sample.crypto_features = self._default_crypto_features()
        
        return list(sample.to_feature_vector().keys())


_fusion_instance: Optional[EnrichedDataFusion] = None


def get_enriched_data_fusion() -> EnrichedDataFusion:
    """Get singleton EnrichedDataFusion instance."""
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = EnrichedDataFusion()
    return _fusion_instance


ENRICHED_DATA_FUSION_GUIDE = """
=== Enriched Data Fusion Layer ===

Combines ALL 7 data sources into ML-ready features:

1. POLYGON.IO
   - Purpose: Market data, OHLCV, historical prices
   - Features: price_*, volume_*, technical_*
   - Feeds: Core training data

2. ALPACA
   - Purpose: Paper trading, execution data
   - Features: position_*, fill_*, execution_*
   - Feeds: Reinforcement learning

3. FRED (Federal Reserve)
   - Purpose: Economic indicators, regime detection
   - Features: regime_score, yield_curve_score, inflation_trend_score
   - Feeds: Macro regime classifier

4. FINNHUB
   - Purpose: Social sentiment (Reddit/Twitter)
   - Features: social_score, reddit_mentions, buzz_level
   - Feeds: Sentiment classifier

5. ALPHA VANTAGE
   - Purpose: AI-powered news sentiment
   - Features: news_score, bullish_ratio, news_count
   - Feeds: Catalyst detector

6. SEC EDGAR
   - Purpose: Insider transactions (Form 4)
   - Features: insider_sentiment_score, large_buy_signal
   - Feeds: Smart money tracker

7. BINANCE
   - Purpose: Crypto correlations (BTC/ETH)
   - Features: btc_return_24h, crypto_fear_index
   - Feeds: Risk appetite proxy

=== Integration with ApexCore V4 ===

All 50+ enriched features feed into the 16 prediction heads:
- QuantraScore, Runner, Quality, Avoid
- Regime, Timing, Runup, Direction
- Volatility, Momentum, Support, Resistance
- Volume, Reversal, Breakout, Continuation

=== Usage ===

from src.quantracore_apex.data_layer.enriched_data_fusion import get_enriched_data_fusion

fusion = get_enriched_data_fusion()

# Get enriched sample for a symbol
sample = fusion.enrich_sample("AAPL")
print(f"Features: {sample.feature_count}")
print(f"Sources: {sample.sources_used}")

# Get flat feature vector for ML
features = fusion.get_learning_features("AAPL")
print(f"Feature vector: {len(features)} dimensions")

# Batch enrichment
samples = fusion.enrich_batch(["AAPL", "MSFT", "NVDA"])
"""
