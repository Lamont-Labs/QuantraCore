"""
Fallback Data Adapters for Hyperspeed Learning System.

Provides local/synthetic data sources for testing and development
when external APIs (Polygon, Alpaca) are unavailable.
"""

import os
import json
import logging
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow
from .models import DataSource, AggregatedSample

logger = logging.getLogger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for data adapters."""
    use_cached_data: bool = True
    cache_dir: str = "data/hyperspeed_cache"
    synthetic_volatility: float = 0.02
    synthetic_trend_bias: float = 0.001
    seed: int = 42


class LocalCacheAdapter:
    """
    Adapter that reads/writes data from local cache files.
    
    Used when API keys are unavailable or for reproducible testing.
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[LocalCacheAdapter] Initialized with cache: {self.cache_dir}")
    
    def get_cache_path(self, symbol: str, start_date: date, end_date: date) -> Path:
        """Get the cache file path for a symbol and date range."""
        filename = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}.json"
        return self.cache_dir / filename
    
    def has_cached_data(self, symbol: str, start_date: date, end_date: date) -> bool:
        """Check if cached data exists."""
        return self.get_cache_path(symbol, start_date, end_date).exists()
    
    def load_cached_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[List[OhlcvBar]]:
        """Load bars from local cache."""
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            bars = []
            for item in data:
                bars.append(OhlcvBar(
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=int(item["volume"]),
                ))
            
            logger.info(f"[LocalCache] Loaded {len(bars)} bars for {symbol}")
            return bars
            
        except Exception as e:
            logger.error(f"[LocalCache] Error loading {cache_path}: {e}")
            return None
    
    def save_cached_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        bars: List[OhlcvBar],
    ) -> bool:
        """Save bars to local cache."""
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        
        try:
            data = []
            for bar in bars:
                data.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                })
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"[LocalCache] Saved {len(bars)} bars for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"[LocalCache] Error saving {cache_path}: {e}")
            return False


class SyntheticDataAdapter:
    """
    Adapter that generates realistic synthetic market data.
    
    Uses random walk with mean reversion and regime changes
    to simulate realistic price movements for testing.
    """
    
    STOCK_PROFILES = {
        "AAPL": {"base_price": 175.0, "volatility": 0.018, "trend": 0.0003},
        "MSFT": {"base_price": 375.0, "volatility": 0.016, "trend": 0.0004},
        "GOOGL": {"base_price": 140.0, "volatility": 0.020, "trend": 0.0002},
        "AMZN": {"base_price": 180.0, "volatility": 0.022, "trend": 0.0003},
        "NVDA": {"base_price": 480.0, "volatility": 0.030, "trend": 0.0008},
        "TSLA": {"base_price": 250.0, "volatility": 0.035, "trend": 0.0001},
        "META": {"base_price": 350.0, "volatility": 0.025, "trend": 0.0005},
        "AMD": {"base_price": 120.0, "volatility": 0.032, "trend": 0.0006},
        "SPY": {"base_price": 450.0, "volatility": 0.012, "trend": 0.0002},
        "QQQ": {"base_price": 380.0, "volatility": 0.015, "trend": 0.0003},
    }
    
    DEFAULT_PROFILE = {"base_price": 50.0, "volatility": 0.025, "trend": 0.0}
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        logger.info("[SyntheticDataAdapter] Initialized")
    
    def get_profile(self, symbol: str) -> Dict[str, float]:
        """Get stock profile for a symbol."""
        return self.STOCK_PROFILES.get(symbol.upper(), self.DEFAULT_PROFILE)
    
    def generate_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        include_weekends: bool = False,
    ) -> List[OhlcvBar]:
        """
        Generate realistic synthetic OHLCV bars.
        
        Uses geometric Brownian motion with:
        - Daily returns based on volatility
        - Mean reversion to prevent unrealistic drift
        - Volume clustering and seasonality
        - Realistic intraday price range
        """
        profile = self.get_profile(symbol)
        
        current_date = start_date
        bars = []
        price = profile["base_price"]
        volatility = profile["volatility"]
        trend = profile["trend"]
        
        mean_price = price
        mean_reversion = 0.02
        
        base_volume = self._get_base_volume(symbol)
        
        while current_date <= end_date:
            if not include_weekends and current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue
            
            daily_return = self.rng.normal(
                loc=trend - mean_reversion * (price - mean_price) / mean_price,
                scale=volatility,
            )
            
            new_price = price * (1 + daily_return)
            
            intraday_vol = volatility * 1.2
            high_mult = 1 + abs(self.rng.normal(0, intraday_vol))
            low_mult = 1 - abs(self.rng.normal(0, intraday_vol))
            
            high = new_price * high_mult
            low = new_price * low_mult
            
            if daily_return > 0:
                open_price = low + (new_price - low) * self.rng.uniform(0.2, 0.6)
            else:
                open_price = high - (high - new_price) * self.rng.uniform(0.2, 0.6)
            
            volume_mult = self.rng.lognormal(0, 0.4)
            if current_date.weekday() in [0, 4]:
                volume_mult *= 1.2
            volume = int(base_volume * volume_mult)
            
            bars.append(OhlcvBar(
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(new_price, 2),
                volume=volume,
            ))
            
            price = new_price
            current_date += timedelta(days=1)
        
        logger.info(f"[Synthetic] Generated {len(bars)} bars for {symbol}")
        return bars
    
    def _get_base_volume(self, symbol: str) -> int:
        """Get base daily volume for a symbol."""
        volume_map = {
            "AAPL": 80_000_000,
            "MSFT": 25_000_000,
            "GOOGL": 20_000_000,
            "AMZN": 40_000_000,
            "NVDA": 50_000_000,
            "TSLA": 100_000_000,
            "META": 15_000_000,
            "AMD": 70_000_000,
            "SPY": 90_000_000,
            "QQQ": 50_000_000,
        }
        return volume_map.get(symbol.upper(), 10_000_000)
    
    def generate_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic options flow data."""
        call_volume = int(self.rng.exponential(50000))
        put_volume = int(self.rng.exponential(40000))
        
        return {
            "symbol": symbol,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_ratio": round(put_volume / max(call_volume, 1), 2),
            "unusual_activity": self.rng.random() > 0.7,
            "large_trades": [
                {"type": "call", "strike": 100 + self.rng.integers(0, 50), "premium": int(self.rng.exponential(500000))},
                {"type": "put", "strike": 100 - self.rng.integers(0, 30), "premium": int(self.rng.exponential(300000))},
            ],
        }
    
    def generate_dark_pool(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic dark pool data."""
        total_volume = int(self.rng.exponential(3_000_000))
        
        return {
            "symbol": symbol,
            "total_volume": total_volume,
            "percentage_of_total": round(self.rng.uniform(25, 45), 1),
            "avg_trade_size": int(self.rng.uniform(200, 1000)),
            "block_trades": int(self.rng.poisson(15)),
            "net_flow": self.rng.choice(["bullish", "bearish", "neutral"]),
        }
    
    def generate_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Generate synthetic sentiment data."""
        overall = self.rng.beta(6, 4)
        
        return {
            "symbol": symbol,
            "overall_score": round(overall, 2),
            "news_sentiment": round(self.rng.beta(5, 4), 2),
            "social_sentiment": round(self.rng.beta(7, 3), 2),
            "analyst_rating": self.rng.choice(["strong_buy", "buy", "hold", "sell"]),
            "mentions_count": int(self.rng.exponential(5000)),
        }


class FallbackDataProvider:
    """
    Unified fallback data provider.
    
    Tries local cache first, then generates synthetic data.
    Automatically caches generated data for consistency.
    """
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.cache = LocalCacheAdapter(config)
        self.synthetic = SyntheticDataAdapter(config)
        
        logger.info("[FallbackDataProvider] Initialized")
    
    def get_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        prefer_cached: bool = True,
    ) -> List[OhlcvBar]:
        """
        Get OHLCV bars from cache or generate synthetically.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            prefer_cached: Whether to check cache first
        
        Returns:
            List of OHLCV bars
        """
        if prefer_cached and self.config.use_cached_data:
            cached = self.cache.load_cached_bars(symbol, start_date, end_date)
            if cached:
                return cached
        
        bars = self.synthetic.generate_bars(symbol, start_date, end_date)
        
        if self.config.use_cached_data and bars:
            self.cache.save_cached_bars(symbol, start_date, end_date, bars)
        
        return bars
    
    def get_enrichment_data(
        self,
        symbol: str,
        source: DataSource,
    ) -> Dict[str, Any]:
        """Get enrichment data for a specific source."""
        if source == DataSource.OPTIONS_FLOW:
            return self.synthetic.generate_options_flow(symbol)
        elif source == DataSource.DARK_POOL:
            return self.synthetic.generate_dark_pool(symbol)
        elif source == DataSource.SENTIMENT:
            return self.synthetic.generate_sentiment(symbol)
        else:
            return {}
    
    def is_fallback_mode(self) -> bool:
        """Check if operating in fallback mode (no live APIs)."""
        polygon_key = os.environ.get("POLYGON_API_KEY")
        alpaca_key = os.environ.get("ALPACA_PAPER_API_KEY")
        
        return not (polygon_key and alpaca_key)
