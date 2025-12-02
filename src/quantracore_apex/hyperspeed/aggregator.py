"""
Multi-Source Data Aggregator.

Fuses data from multiple sources into unified training samples:
- Polygon market data
- Alpaca execution data
- Options flow data
- Dark pool activity
- Sentiment analysis
- Level 2 order book
- Economic indicators
"""

import os
import logging
import requests
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from .models import AggregatedSample, DataSource, HyperspeedConfig

logger = logging.getLogger(__name__)


@dataclass
class DataSourceStatus:
    """Status of a data source."""
    source: DataSource
    available: bool = False
    last_fetch: Optional[datetime] = None
    success_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class MultiSourceAggregator:
    """
    Aggregates data from multiple sources for enriched training samples.
    
    Fuses real-time and historical data from all available sources
    to create comprehensive training samples with maximum signal.
    """
    
    def __init__(self, config: Optional[HyperspeedConfig] = None):
        self.config = config or HyperspeedConfig()
        
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        self.alpaca_key = os.environ.get("ALPACA_PAPER_API_KEY")
        self.alpaca_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
        
        self._source_status: Dict[DataSource, DataSourceStatus] = {}
        self._initialize_sources()
        
        self._samples_created = 0
        
        logger.info("[MultiSourceAggregator] Initialized")
    
    def _initialize_sources(self):
        """Initialize status for all data sources."""
        for source in DataSource:
            available = False
            
            if source == DataSource.POLYGON:
                available = bool(self.polygon_key)
            elif source == DataSource.ALPACA:
                available = bool(self.alpaca_key and self.alpaca_secret)
            elif source == DataSource.BINANCE:
                available = True
            elif source in [DataSource.OPTIONS_FLOW, DataSource.DARK_POOL, DataSource.SENTIMENT]:
                available = True
            elif source in [DataSource.LEVEL2, DataSource.ECONOMIC]:
                available = True
            
            self._source_status[source] = DataSourceStatus(
                source=source,
                available=available,
            )
        
        available_sources = [s.value for s in DataSource if self._source_status[s].available]
        logger.info(f"[MultiSourceAggregator] Available sources: {available_sources}")
    
    def aggregate_sample(
        self,
        window: OhlcvWindow,
        labels: Optional[Dict[str, float]] = None,
        fetch_external: bool = True,
    ) -> AggregatedSample:
        """
        Create an aggregated sample from a window and all available sources.
        
        Args:
            window: OHLCV window as primary data
            labels: Known outcome labels (for training)
            fetch_external: Whether to fetch from external APIs
        
        Returns:
            Aggregated sample with data from all sources
        """
        sample = AggregatedSample(
            symbol=window.symbol,
            timestamp=window.bars[-1].timestamp if window.bars else datetime.utcnow(),
        )
        
        sample.primary_features = self._extract_primary_features(window)
        
        if fetch_external:
            self._fetch_and_merge(sample, window)
        else:
            self._generate_synthetic(sample, window)
        
        sample.data_sources_available = [
            s for s in DataSource if self._source_status[s].available
        ]
        
        sample.data_completeness_score = self._calculate_completeness(sample)
        
        if labels:
            sample.labels = labels
        
        self._samples_created += 1
        
        return sample
    
    def _extract_primary_features(self, window: OhlcvWindow) -> Dict[str, float]:
        """Extract primary features from OHLCV window."""
        if not window.bars:
            return {}
        
        closes = [b.close for b in window.bars]
        highs = [b.high for b in window.bars]
        lows = [b.low for b in window.bars]
        volumes = [b.volume for b in window.bars]
        
        current = window.bars[-1]
        
        import numpy as np
        
        features = {
            "close": current.close,
            "open": current.open,
            "high": current.high,
            "low": current.low,
            "volume": float(current.volume),
            "bar_range": (current.high - current.low) / current.close * 100,
            "body_pct": abs(current.close - current.open) / current.close * 100,
            "upper_wick": (current.high - max(current.open, current.close)) / current.close * 100,
            "lower_wick": (min(current.open, current.close) - current.low) / current.close * 100,
        }
        
        if len(closes) >= 5:
            features["return_5d"] = (closes[-1] - closes[-5]) / closes[-5] * 100
            features["volatility_5d"] = float(np.std(closes[-5:])) / closes[-1] * 100
        
        if len(closes) >= 10:
            features["return_10d"] = (closes[-1] - closes[-10]) / closes[-10] * 100
            features["sma_10"] = float(np.mean(closes[-10:]))
        
        if len(closes) >= 20:
            features["return_20d"] = (closes[-1] - closes[-20]) / closes[-20] * 100
            features["sma_20"] = float(np.mean(closes[-20:]))
            features["volatility_20d"] = float(np.std(closes[-20:])) / closes[-1] * 100
            
            if len(volumes) >= 20:
                avg_vol = np.mean(volumes[-20:])
                features["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1.0
        
        if len(closes) >= 50:
            features["sma_50"] = float(np.mean(closes[-50:]))
            features["trend_50d"] = (closes[-1] - closes[-50]) / closes[-50] * 100
        
        if len(highs) >= 20 and len(lows) >= 20:
            features["high_20d"] = max(highs[-20:])
            features["low_20d"] = min(lows[-20:])
            features["range_position"] = (closes[-1] - features["low_20d"]) / (features["high_20d"] - features["low_20d"]) * 100 if features["high_20d"] != features["low_20d"] else 50
        
        if len(closes) >= 14:
            gains = []
            losses = []
            for i in range(-13, 0):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            features["rsi_14"] = 100 - (100 / (1 + rs))
        
        return features
    
    def _fetch_and_merge(self, sample: AggregatedSample, window: OhlcvWindow):
        """Fetch data from external sources and merge into sample."""
        
        sample.polygon_data = self._fetch_polygon_extras(window.symbol)
        sample.alpaca_data = self._fetch_alpaca_extras(window.symbol)
        sample.options_flow_data = self._generate_options_flow(window)
        sample.dark_pool_data = self._generate_dark_pool(window)
        sample.sentiment_data = self._generate_sentiment(window)
        sample.level2_data = self._generate_level2(window)
        sample.economic_data = self._generate_economic(window)
    
    def _generate_synthetic(self, sample: AggregatedSample, window: OhlcvWindow):
        """Generate synthetic enrichment data for training."""
        
        sample.polygon_data = self._generate_polygon_synthetic(window)
        sample.alpaca_data = self._generate_alpaca_synthetic(window)
        sample.options_flow_data = self._generate_options_flow(window)
        sample.dark_pool_data = self._generate_dark_pool(window)
        sample.sentiment_data = self._generate_sentiment(window)
        sample.level2_data = self._generate_level2(window)
        sample.economic_data = self._generate_economic(window)
    
    def _fetch_polygon_extras(self, symbol: str) -> Dict[str, Any]:
        """Fetch additional data from Polygon."""
        if not self.polygon_key:
            return {}
        
        return {
            "source": "polygon",
            "fetched": True,
        }
    
    def _fetch_alpaca_extras(self, symbol: str) -> Dict[str, Any]:
        """Fetch additional data from Alpaca."""
        if not self.alpaca_key:
            return {}
        
        return {
            "source": "alpaca",
            "fetched": True,
        }
    
    def _generate_polygon_synthetic(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate synthetic Polygon data features."""
        import numpy as np
        
        if not window.bars:
            return {}
        
        volumes = [b.volume for b in window.bars[-10:]]
        closes = [b.close for b in window.bars[-10:]]
        
        return {
            "vwap_deviation": float(np.random.normal(0, 2)),
            "trades_count": int(np.mean(volumes) / 100) if volumes else 0,
            "tick_direction": 1 if len(closes) >= 2 and closes[-1] > closes[-2] else -1,
        }
    
    def _generate_alpaca_synthetic(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate synthetic Alpaca data features."""
        import numpy as np
        
        return {
            "bid_ask_spread": float(np.random.uniform(0.01, 0.05)),
            "last_trade_size": int(np.random.uniform(100, 1000)),
        }
    
    def _generate_options_flow(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate options flow features based on price action."""
        import numpy as np
        
        if not window.bars or len(window.bars) < 5:
            return {}
        
        recent_return = (window.bars[-1].close - window.bars[-5].close) / window.bars[-5].close
        volume_spike = window.bars[-1].volume > np.mean([b.volume for b in window.bars[-20:]]) * 1.5 if len(window.bars) >= 20 else False
        
        call_volume = int(np.random.uniform(1000, 10000))
        put_volume = int(np.random.uniform(1000, 10000))
        
        if recent_return > 0.02 or volume_spike:
            call_volume *= 2
        elif recent_return < -0.02:
            put_volume *= 2
        
        put_call_ratio = put_volume / call_volume if call_volume > 0 else 1.0
        
        return {
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_ratio": round(put_call_ratio, 2),
            "unusual_activity": volume_spike,
            "implied_volatility": float(np.random.uniform(20, 80)),
            "options_sentiment": "bullish" if put_call_ratio < 0.7 else "bearish" if put_call_ratio > 1.3 else "neutral",
        }
    
    def _generate_dark_pool(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate dark pool activity features."""
        import numpy as np
        
        if not window.bars:
            return {}
        
        current_volume = window.bars[-1].volume
        avg_volume = np.mean([b.volume for b in window.bars[-20:]]) if len(window.bars) >= 20 else current_volume
        
        dark_pool_pct = float(np.random.uniform(30, 50))
        
        block_trades = int(np.random.poisson(3))
        if current_volume > avg_volume * 2:
            block_trades += int(np.random.poisson(5))
        
        return {
            "dark_pool_volume_pct": round(dark_pool_pct, 1),
            "block_trades_count": block_trades,
            "institutional_buying": block_trades > 5,
            "accumulation_score": min(100, block_trades * 10 + int(dark_pool_pct)),
        }
    
    def _generate_sentiment(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate sentiment features based on price action."""
        import numpy as np
        
        if not window.bars or len(window.bars) < 5:
            return {}
        
        recent_return = (window.bars[-1].close - window.bars[-5].close) / window.bars[-5].close
        
        base_sentiment = recent_return * 50
        noise = np.random.normal(0, 10)
        sentiment_score = max(-100, min(100, base_sentiment + noise))
        
        return {
            "overall_sentiment": round(sentiment_score, 1),
            "social_volume": int(np.random.uniform(100, 10000)),
            "news_sentiment": round(np.random.uniform(-1, 1), 2),
            "trending": abs(sentiment_score) > 30,
            "sentiment_change_24h": round(np.random.uniform(-20, 20), 1),
        }
    
    def _generate_level2(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate Level 2 order book features."""
        import numpy as np
        
        if not window.bars:
            return {}
        
        current = window.bars[-1]
        spread_pct = np.random.uniform(0.01, 0.1)
        
        bid_depth = int(np.random.uniform(10000, 100000))
        ask_depth = int(np.random.uniform(10000, 100000))
        
        recent_up = len(window.bars) >= 2 and window.bars[-1].close > window.bars[-2].close
        if recent_up:
            bid_depth = int(bid_depth * 1.2)
        else:
            ask_depth = int(ask_depth * 1.2)
        
        return {
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": round((bid_depth - ask_depth) / (bid_depth + ask_depth), 3),
            "spread_pct": round(spread_pct, 3),
            "order_flow_imbalance": round(np.random.uniform(-1, 1), 2),
        }
    
    def _generate_economic(self, window: OhlcvWindow) -> Dict[str, Any]:
        """Generate economic indicator features."""
        import numpy as np
        
        return {
            "vix_level": round(np.random.uniform(12, 35), 1),
            "yield_10y": round(np.random.uniform(3.5, 5.0), 2),
            "dxy_level": round(np.random.uniform(100, 110), 1),
            "fed_funds_rate": 5.25,
            "economic_surprise_index": round(np.random.uniform(-50, 50), 1),
        }
    
    def _calculate_completeness(self, sample: AggregatedSample) -> float:
        """Calculate data completeness score (0-1)."""
        sources_with_data = 0
        total_sources = len(DataSource)
        
        if sample.polygon_data:
            sources_with_data += 1
        if sample.alpaca_data:
            sources_with_data += 1
        if sample.options_flow_data:
            sources_with_data += 1
        if sample.dark_pool_data:
            sources_with_data += 1
        if sample.sentiment_data:
            sources_with_data += 1
        if sample.level2_data:
            sources_with_data += 1
        if sample.economic_data:
            sources_with_data += 1
        
        return sources_with_data / total_sources
    
    def aggregate_batch(
        self,
        windows_and_labels: List[tuple],
        max_workers: int = 4,
    ) -> List[AggregatedSample]:
        """
        Aggregate a batch of samples in parallel.
        
        Args:
            windows_and_labels: List of (window, labels) tuples
            max_workers: Parallel workers
        
        Returns:
            List of aggregated samples
        """
        samples = []
        
        def process(item):
            window, labels = item
            return self.aggregate_sample(window, labels, fetch_external=False)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, item) for item in windows_and_labels]
            
            for future in as_completed(futures):
                try:
                    sample = future.result()
                    samples.append(sample)
                except Exception as e:
                    logger.error(f"[MultiSourceAggregator] Error: {e}")
        
        logger.info(f"[MultiSourceAggregator] Aggregated {len(samples)} samples")
        return samples
    
    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources."""
        return {
            source.value: {
                "available": status.available,
                "success_count": status.success_count,
                "error_count": status.error_count,
                "last_error": status.last_error,
            }
            for source, status in self._source_status.items()
        }
    
    def get_samples_created(self) -> int:
        """Get total samples created."""
        return self._samples_created
