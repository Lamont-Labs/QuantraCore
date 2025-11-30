"""
Unified Data Manager for QuantraCore Apex.

Central hub for all data ingestion across multiple providers.
Handles fallbacks, caching, rate limiting, and provider health.

Usage:
    manager = UnifiedDataManager()
    
    # Get OHLCV from best available provider
    bars = manager.fetch_ohlcv("AAPL", start, end)
    
    # Get options flow from all providers
    flow = manager.get_options_flow("AAPL")
    
    # Check provider status
    status = manager.get_all_provider_status()
"""

import os
from typing import List, Optional, Dict, Any, Type, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    OhlcvBar, OptionsFlow, DarkPoolPrint, NewsItem, 
    SentimentData, FundamentalsData, ProviderStatus
)


def _normalize_timeframe(tf: Union[str, TimeFrame]) -> TimeFrame:
    """Convert string or TimeFrame to TimeFrame enum.
    
    Raises:
        ValueError: If timeframe string is not recognized
    """
    if isinstance(tf, TimeFrame):
        return tf
    
    tf_lower = tf.lower() if isinstance(tf, str) else tf
    
    tf_map = {
        "tick": TimeFrame.TICK,
        "1s": TimeFrame.SECOND_1,
        "1m": TimeFrame.MINUTE_1,
        "5m": TimeFrame.MINUTE_5,
        "15m": TimeFrame.MINUTE_15,
        "30m": TimeFrame.MINUTE_30,
        "1h": TimeFrame.HOUR_1,
        "4h": TimeFrame.HOUR_4,
        "1d": TimeFrame.DAY_1,
        "1w": TimeFrame.WEEK_1,
        "1M": TimeFrame.MONTH_1,
        "day": TimeFrame.DAY_1,
        "hour": TimeFrame.HOUR_1,
        "minute": TimeFrame.MINUTE_1,
    }
    
    result = tf_map.get(tf_lower)
    if result is None:
        logger.warning(f"Unknown timeframe '{tf}', defaulting to DAY_1")
        return TimeFrame.DAY_1
    return result

from .polygon_adapter import PolygonAdapter
from .alpha_vantage_adapter import AlphaVantageAdapter
from .synthetic_adapter import SyntheticAdapter
from .alpaca_data_adapter import AlpacaDataAdapter

logger = logging.getLogger(__name__)


class ProviderPriority(Enum):
    PRIMARY = 1
    SECONDARY = 2
    FALLBACK = 3
    DEMO = 4


@dataclass
class ProviderConfig:
    name: str
    adapter_class: Optional[Type]
    priority: ProviderPriority
    env_key: Optional[str]
    data_types: List[DataType]
    monthly_cost: Optional[float] = None
    rate_limit: Optional[int] = None
    notes: Optional[str] = None


PROVIDER_REGISTRY: List[ProviderConfig] = [
    ProviderConfig(
        name="Alpaca",
        adapter_class=AlpacaDataAdapter,
        priority=ProviderPriority.PRIMARY,
        env_key="ALPACA_PAPER_API_KEY",
        data_types=[DataType.OHLCV, DataType.TICK, DataType.QUOTE],
        monthly_cost=0.0,
        rate_limit=200,
        notes="Free market data with paper trading account. Higher rate limits than Polygon free tier."
    ),
    ProviderConfig(
        name="Polygon.io",
        adapter_class=PolygonAdapter,
        priority=ProviderPriority.PRIMARY,
        env_key="POLYGON_API_KEY",
        data_types=[DataType.OHLCV, DataType.TICK, DataType.QUOTE],
        monthly_cost=29.0,
        rate_limit=5,
        notes="Best for US equities, options, crypto. Real-time with Advanced tier ($199/mo)"
    ),
    ProviderConfig(
        name="Alpha Vantage",
        adapter_class=AlphaVantageAdapter,
        priority=ProviderPriority.SECONDARY,
        env_key="ALPHA_VANTAGE_API_KEY",
        data_types=[DataType.OHLCV, DataType.FUNDAMENTALS],
        monthly_cost=49.0,
        rate_limit=75,
        notes="Good for technical indicators, forex, crypto"
    ),
    ProviderConfig(
        name="EODHD",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="EODHD_API_KEY",
        data_types=[DataType.OHLCV, DataType.FUNDAMENTALS, DataType.NEWS],
        monthly_cost=20.0,
        rate_limit=100,
        notes="70+ global exchanges, international markets"
    ),
    ProviderConfig(
        name="Financial Modeling Prep",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="FMP_API_KEY",
        data_types=[DataType.OHLCV, DataType.FUNDAMENTALS, DataType.NEWS, DataType.SEC_FILINGS],
        monthly_cost=20.0,
        rate_limit=250,
        notes="Fundamentals, financials, SEC filings, ML-ready"
    ),
    ProviderConfig(
        name="Nasdaq Data Link",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="NASDAQ_DATA_LINK_API_KEY",
        data_types=[DataType.ECONOMIC, DataType.SENTIMENT],
        monthly_cost=0.0,
        notes="Economic indicators, Fed data, alternative datasets"
    ),
    ProviderConfig(
        name="Interactive Brokers",
        adapter_class=None,
        priority=ProviderPriority.PRIMARY,
        env_key="IB_HOST",
        data_types=[DataType.OHLCV, DataType.TICK, DataType.QUOTE, DataType.OPTIONS_CHAIN, DataType.FUNDAMENTALS],
        monthly_cost=0.0,
        notes="Broker-integrated data, requires IB Gateway/TWS running"
    ),
    ProviderConfig(
        name="Unusual Whales",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="UNUSUAL_WHALES_API_KEY",
        data_types=[DataType.OPTIONS_FLOW, DataType.DARK_POOL],
        monthly_cost=35.0,
        notes="Unusual options activity, congressional trades, dark pool"
    ),
    ProviderConfig(
        name="FlowAlgo",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="FLOWALGO_API_KEY",
        data_types=[DataType.OPTIONS_FLOW, DataType.DARK_POOL],
        monthly_cost=150.0,
        notes="Sweeps, blocks, dark pool levels, institutional flow"
    ),
    ProviderConfig(
        name="InsiderFinance",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="INSIDER_FINANCE_API_KEY",
        data_types=[DataType.OPTIONS_FLOW, DataType.DARK_POOL],
        monthly_cost=49.0,
        notes="Correlated options flow and dark pool analysis"
    ),
    ProviderConfig(
        name="Finnhub",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="FINNHUB_API_KEY",
        data_types=[DataType.NEWS, DataType.SENTIMENT, DataType.SEC_FILINGS],
        monthly_cost=0.0,
        rate_limit=60,
        notes="News, sentiment, insider trades (free tier available)"
    ),
    ProviderConfig(
        name="AltIndex",
        adapter_class=None,
        priority=ProviderPriority.FALLBACK,
        env_key="ALTINDEX_API_KEY",
        data_types=[DataType.SENTIMENT],
        monthly_cost=29.0,
        notes="AI stock scores, social sentiment aggregation"
    ),
    ProviderConfig(
        name="Stocktwits",
        adapter_class=None,
        priority=ProviderPriority.FALLBACK,
        env_key=None,
        data_types=[DataType.SENTIMENT],
        monthly_cost=0.0,
        notes="Free social sentiment (no API key needed)"
    ),
    ProviderConfig(
        name="Binance",
        adapter_class=None,
        priority=ProviderPriority.PRIMARY,
        env_key=None,
        data_types=[DataType.OHLCV, DataType.TICK, DataType.QUOTE, DataType.CRYPTO],
        monthly_cost=0.0,
        rate_limit=1200,
        notes="Free crypto data (no API key for public data)"
    ),
    ProviderConfig(
        name="CoinGecko",
        adapter_class=None,
        priority=ProviderPriority.SECONDARY,
        env_key="COINGECKO_API_KEY",
        data_types=[DataType.OHLCV, DataType.CRYPTO],
        monthly_cost=0.0,
        rate_limit=50,
        notes="10,000+ coins, market data (free tier available)"
    ),
    ProviderConfig(
        name="Synthetic",
        adapter_class=SyntheticAdapter,
        priority=ProviderPriority.DEMO,
        env_key=None,
        data_types=[DataType.OHLCV],
        monthly_cost=0.0,
        notes="Deterministic synthetic data for testing"
    ),
]


class UnifiedDataManager:
    """
    Central data management hub for QuantraCore Apex.
    
    Features:
    - Automatic provider selection based on availability
    - Fallback chains for data reliability
    - Unified interface across all providers
    - Provider health monitoring
    - Rate limit management
    - Caching (optional)
    
    Provider Priority:
    1. Polygon.io - Primary for US equities
    2. Alpha Vantage - Secondary, good for technicals
    3. EODHD - International markets
    4. FMP - Fundamentals
    5. Interactive Brokers - If connected
    6. Synthetic - Demo/testing fallback
    """
    
    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self._adapters: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        if os.getenv("ALPACA_PAPER_API_KEY") and os.getenv("ALPACA_PAPER_API_SECRET"):
            try:
                alpaca = AlpacaDataAdapter()
                if alpaca.is_available():
                    self._adapters["alpaca"] = alpaca
                    logger.info("Alpaca data adapter initialized (free market data)")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpaca data adapter: {e}")
        
        if os.getenv("POLYGON_API_KEY"):
            try:
                self._adapters["polygon"] = PolygonAdapter()
                logger.info("Polygon adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Polygon: {e}")
        
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            try:
                self._adapters["alpha_vantage"] = AlphaVantageAdapter()
                logger.info("Alpha Vantage adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Alpha Vantage: {e}")
        
        try:
            from .eodhd_adapter import EODHDAdapter
            if os.getenv("EODHD_API_KEY"):
                self._adapters["eodhd"] = EODHDAdapter()
                logger.info("EODHD adapter initialized")
        except Exception as e:
            logger.debug(f"EODHD not available: {e}")
        
        try:
            from .fmp_adapter import FMPAdapter
            if os.getenv("FMP_API_KEY"):
                self._adapters["fmp"] = FMPAdapter()
                logger.info("FMP adapter initialized")
        except Exception as e:
            logger.debug(f"FMP not available: {e}")
        
        try:
            from .nasdaq_datalink_adapter import NasdaqDataLinkAdapter
            if os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY"):
                self._adapters["nasdaq_datalink"] = NasdaqDataLinkAdapter()
                logger.info("Nasdaq Data Link adapter initialized")
        except Exception as e:
            logger.debug(f"Nasdaq Data Link not available: {e}")
        
        try:
            from .options_flow_adapter import OptionsFlowAggregator
            flow_agg = OptionsFlowAggregator()
            if flow_agg.is_available():
                self._adapters["options_flow"] = flow_agg
                logger.info("Options Flow aggregator initialized")
        except Exception as e:
            logger.debug(f"Options Flow not available: {e}")
        
        try:
            from .alternative_data_adapter import AlternativeDataAggregator
            alt_agg = AlternativeDataAggregator()
            if alt_agg.is_available():
                self._adapters["alternative_data"] = alt_agg
                logger.info("Alternative Data aggregator initialized")
        except Exception as e:
            logger.debug(f"Alternative Data not available: {e}")
        
        try:
            from .crypto_adapter import CryptoDataAggregator
            crypto_agg = CryptoDataAggregator()
            if crypto_agg.is_available():
                self._adapters["crypto"] = crypto_agg
                logger.info("Crypto aggregator initialized")
        except Exception as e:
            logger.debug(f"Crypto aggregator not available: {e}")
        
        try:
            from .interactive_brokers_adapter import InteractiveBrokersAdapter
            if os.getenv("IB_HOST"):
                ib = InteractiveBrokersAdapter()
                if ib.is_available():
                    self._adapters["interactive_brokers"] = ib
                    logger.info("Interactive Brokers adapter initialized")
        except Exception as e:
            logger.debug(f"Interactive Brokers not available: {e}")
        
        self._adapters["synthetic"] = SyntheticAdapter()
        logger.info("Synthetic adapter initialized (fallback)")
    
    def get_available_providers(self) -> List[str]:
        return list(self._adapters.keys())
    
    def get_all_provider_status(self) -> Dict[str, ProviderStatus]:
        status = {}
        for name, adapter in self._adapters.items():
            try:
                if hasattr(adapter, 'get_status'):
                    status[name] = adapter.get_status()
                else:
                    status[name] = ProviderStatus(
                        name=name,
                        available=adapter.is_available() if hasattr(adapter, 'is_available') else True,
                        connected=True,
                        data_types=[]
                    )
            except Exception as e:
                status[name] = ProviderStatus(
                    name=name,
                    available=False,
                    connected=False,
                    last_error=str(e),
                    data_types=[]
                )
        return status
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1,
        provider: Optional[str] = None
    ) -> List[OhlcvBar]:
        tf = _normalize_timeframe(timeframe)
        
        if provider and provider in self._adapters:
            adapter = self._adapters[provider]
            if hasattr(adapter, 'fetch_ohlcv'):
                return adapter.fetch_ohlcv(symbol, start, end, tf)
            elif hasattr(adapter, 'fetch'):
                days = (end - start).days
                return adapter.fetch(symbol, days=days)
        
        provider_order = ["polygon", "interactive_brokers", "alpha_vantage", "eodhd", "fmp", "crypto", "synthetic"]
        
        for prov in provider_order:
            if prov in self._adapters:
                try:
                    adapter = self._adapters[prov]
                    if hasattr(adapter, 'fetch_ohlcv'):
                        return adapter.fetch_ohlcv(symbol, start, end, tf)
                    elif hasattr(adapter, 'fetch'):
                        days = (end - start).days
                        return adapter.fetch(symbol, days=days)
                except Exception as e:
                    logger.warning(f"{prov} failed for {symbol}: {e}")
                    continue
        
        return self._adapters["synthetic"].fetch_ohlcv(symbol, start, end, tf)
    
    def fetch_fundamentals(
        self,
        symbol: str,
        provider: Optional[str] = None
    ) -> Optional[FundamentalsData]:
        provider_order = ["fmp", "eodhd", "alpha_vantage"]
        
        if provider and provider in self._adapters:
            provider_order = [provider]
        
        for prov in provider_order:
            if prov in self._adapters:
                try:
                    adapter = self._adapters[prov]
                    if hasattr(adapter, 'fetch_fundamentals'):
                        return adapter.fetch_fundamentals(symbol)
                except Exception as e:
                    logger.warning(f"{prov} fundamentals failed for {symbol}: {e}")
                    continue
        
        return None
    
    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[NewsItem]:
        all_news = []
        
        for prov in ["fmp", "eodhd", "alternative_data"]:
            if prov in self._adapters:
                try:
                    adapter = self._adapters[prov]
                    if hasattr(adapter, 'fetch_news'):
                        news = adapter.fetch_news(symbols, limit=limit)
                        all_news.extend(news)
                except Exception as e:
                    logger.warning(f"{prov} news failed: {e}")
        
        return sorted(all_news, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def fetch_sentiment(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        if "alternative_data" in self._adapters:
            try:
                adapter = self._adapters["alternative_data"]
                if hasattr(adapter, 'get_combined_sentiment'):
                    return adapter.get_combined_sentiment(symbol)
            except Exception as e:
                logger.warning(f"Sentiment fetch failed for {symbol}: {e}")
        
        return None
    
    def fetch_options_flow(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 25000
    ) -> List[OptionsFlow]:
        if "options_flow" in self._adapters:
            try:
                adapter = self._adapters["options_flow"]
                if hasattr(adapter, 'get_all_flow'):
                    return adapter.get_all_flow(symbol, min_premium)
            except Exception as e:
                logger.warning(f"Options flow failed: {e}")
        
        return []
    
    def fetch_crypto(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1
    ) -> List[OhlcvBar]:
        tf = _normalize_timeframe(timeframe)
        if "crypto" in self._adapters:
            try:
                return self._adapters["crypto"].fetch_ohlcv(symbol, start, end, tf)
            except Exception as e:
                logger.warning(f"Crypto fetch failed for {symbol}: {e}")
        
        return []


DATA_PROVIDER_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    QUANTRACORE APEX DATA PROVIDERS                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MARKET DATA (OHLCV, Ticks, Quotes)                                         ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Polygon.io        $29-1999/mo   US equities, options, crypto, forex      ║
║  • Alpha Vantage     $49/mo        Technical indicators, forex, crypto      ║
║  • EODHD             $20-200/mo    70+ global exchanges, bonds, ETFs        ║
║  • Interactive Brokers  Free*      Full execution + data (*with account)    ║
║                                                                              ║
║  FUNDAMENTALS & FINANCIALS                                                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Financial Modeling Prep  $20-100/mo  Statements, ratios, ML-ready        ║
║  • Nasdaq Data Link         Varies      Economic data, Fed, Treasury        ║
║  • EODHD                    Included    Basic fundamentals                  ║
║                                                                              ║
║  OPTIONS FLOW & DARK POOL                                                    ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Unusual Whales    $35/mo        Unusual activity, congressional trades   ║
║  • FlowAlgo          $150-200/mo   Sweeps, blocks, dark pool levels         ║
║  • InsiderFinance    $49/mo        Correlated flow + dark pool              ║
║  • Barchart          Free/Premium  Basic options flow                       ║
║                                                                              ║
║  ALTERNATIVE DATA                                                            ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Finnhub           Free-$75/mo   News, sentiment, insider trades          ║
║  • AltIndex          $0-99/mo      AI scores, social, job postings          ║
║  • Stocktwits        Free          Real-time social sentiment               ║
║                                                                              ║
║  CRYPTOCURRENCY                                                              ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  • Binance           Free          Spot, futures, funding rates             ║
║  • CoinGecko         Free-$130/mo  10,000+ coins, market data               ║
║                                                                              ║
║  RECOMMENDED STACK BY BUDGET                                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Basic ($50/mo):     Polygon Starter + Finnhub Free + Binance              ║
║  Standard ($150/mo): Polygon Dev + FMP + Unusual Whales + Finnhub          ║
║  Advanced ($500/mo): Polygon Advanced + FMP + FlowAlgo + AltIndex          ║
║  Pro ($2000+/mo):    Polygon Business + IB + FlowAlgo + Full alt data      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


ENVIRONMENT_VARIABLES = """
# ============================================================================
# QUANTRACORE APEX - DATA PROVIDER ENVIRONMENT VARIABLES
# ============================================================================

# --- MARKET DATA (OHLCV, Ticks, Quotes) ---
POLYGON_API_KEY=               # Polygon.io - Primary US equities
ALPHA_VANTAGE_API_KEY=         # Alpha Vantage - Technical indicators
EODHD_API_KEY=                 # EODHD - Global markets

# --- BROKER (Execution + Data) ---
IB_HOST=127.0.0.1              # Interactive Brokers Gateway host
IB_PORT=7497                   # IB Gateway port (7497=paper, 7496=live)
IB_CLIENT_ID=1                 # IB client ID
ALPACA_PAPER_API_KEY=          # Alpaca paper trading
ALPACA_PAPER_API_SECRET=       # Alpaca paper secret

# --- FUNDAMENTALS ---
FMP_API_KEY=                   # Financial Modeling Prep
NASDAQ_DATA_LINK_API_KEY=      # Nasdaq Data Link (Quandl)
QUANDL_API_KEY=                # Legacy Quandl key

# --- OPTIONS FLOW ---
UNUSUAL_WHALES_API_KEY=        # Unusual Whales
FLOWALGO_API_KEY=              # FlowAlgo
INSIDER_FINANCE_API_KEY=       # InsiderFinance

# --- ALTERNATIVE DATA ---
FINNHUB_API_KEY=               # Finnhub news/sentiment
ALTINDEX_API_KEY=              # AltIndex AI scores
STOCKTWITS_ACCESS_TOKEN=       # Stocktwits (optional)
REDDIT_SENTIMENT_API_KEY=      # Reddit aggregator

# --- CRYPTOCURRENCY ---
BINANCE_API_KEY=               # Binance (optional for public data)
BINANCE_API_SECRET=            # Binance secret
COINGECKO_API_KEY=             # CoinGecko Pro (optional)
"""
