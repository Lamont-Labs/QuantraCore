"""
Unified Data Provider Registry for QuantraCore Apex.

Central hub for all data providers with automatic discovery,
health monitoring, and intelligent failover.

This registry enables plug-and-play data source configuration
where any provider can be added by simply setting its API key.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base_enhanced import (
    EnhancedDataAdapter, StreamingDataAdapter, DataType,
    ProviderStatus, OhlcvBar, TickData, QuoteData,
    OptionsFlow, DarkPoolPrint, NewsItem, SentimentData,
    FundamentalsData
)

logger = logging.getLogger(__name__)


class ProviderPriority(Enum):
    PRIMARY = 1
    SECONDARY = 2
    FALLBACK = 3
    DISABLED = 99


@dataclass
class ProviderConfig:
    adapter_class: Type[EnhancedDataAdapter]
    env_key: str
    priority: ProviderPriority = ProviderPriority.SECONDARY
    data_types: List[DataType] = field(default_factory=list)
    cost_per_month: float = 0.0
    description: str = ""


@dataclass
class DataRequest:
    data_type: DataType
    symbol: Optional[str] = None
    symbols: Optional[List[str]] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: int = 100
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataResponse:
    success: bool
    data: Any
    provider: str
    latency_ms: float
    error: Optional[str] = None
    cached: bool = False


class DataProviderRegistry:
    """
    Central registry for all data providers.
    
    Features:
    - Automatic provider discovery based on environment variables
    - Health monitoring and status reporting
    - Intelligent failover between providers
    - Data type routing to appropriate providers
    - Caching layer for efficiency
    
    Usage:
        registry = DataProviderRegistry()
        status = registry.get_all_status()
        data = registry.fetch(DataRequest(DataType.OHLCV, symbol="AAPL"))
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._providers: Dict[str, EnhancedDataAdapter] = {}
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._data_type_routing: Dict[DataType, List[str]] = {}
        self._health_cache: Dict[str, ProviderStatus] = {}
        self._last_health_check = None
        
        self._register_all_providers()
        self._build_routing_table()
        self._initialized = True
        
        logger.info(f"DataProviderRegistry initialized with {len(self._providers)} active providers")
    
    def _register_all_providers(self):
        """Auto-discover and register all available providers."""
        
        from .polygon_adapter import PolygonAdapter
        from .alpaca_data_adapter import AlpacaDataAdapter
        from .options_flow_adapter import (
            UnusualWhalesAdapter, FlowAlgoAdapter, 
            InsiderFinanceAdapter, OptionsFlowAggregator
        )
        from .alternative_data_adapter import (
            FinnhubAdapter, AltIndexAdapter, StocktwitsAdapter,
            AlternativeDataAggregator
        )
        from .crypto_adapter import (
            BinanceAdapter, CoinGeckoAdapter, CryptoDataAggregator
        )
        from .level2_adapter import Level2Adapter, NasdaqTotalViewAdapter
        from .dark_pool_adapter import DarkPoolAdapter, FinraAdfAdapter
        from .economic_adapter import EconomicDataAdapter, FredAdapter
        
        provider_map = {
            "polygon": (PolygonAdapter, "POLYGON_API_KEY", ProviderPriority.PRIMARY, 
                       [DataType.OHLCV, DataType.TICK, DataType.QUOTE], 249.0,
                       "Real-time market data, ticks, OHLCV"),
            
            "alpaca_data": (AlpacaDataAdapter, "ALPACA_PAPER_API_KEY", ProviderPriority.SECONDARY,
                           [DataType.OHLCV, DataType.QUOTE], 0.0,
                           "IEX market data via Alpaca"),
            
            "unusual_whales": (UnusualWhalesAdapter, "UNUSUAL_WHALES_API_KEY", ProviderPriority.PRIMARY,
                              [DataType.OPTIONS_FLOW, DataType.DARK_POOL], 35.0,
                              "Unusual options activity, congressional trades"),
            
            "flowalgo": (FlowAlgoAdapter, "FLOWALGO_API_KEY", ProviderPriority.SECONDARY,
                        [DataType.OPTIONS_FLOW, DataType.DARK_POOL], 175.0,
                        "Institutional sweeps and block trades"),
            
            "insider_finance": (InsiderFinanceAdapter, "INSIDER_FINANCE_API_KEY", ProviderPriority.FALLBACK,
                               [DataType.OPTIONS_FLOW, DataType.DARK_POOL], 49.0,
                               "Correlated flow + dark pool"),
            
            "finnhub": (FinnhubAdapter, "FINNHUB_API_KEY", ProviderPriority.PRIMARY,
                       [DataType.NEWS, DataType.SENTIMENT, DataType.SEC_FILINGS], 0.0,
                       "News, sentiment, SEC filings"),
            
            "altindex": (AltIndexAdapter, "ALTINDEX_API_KEY", ProviderPriority.SECONDARY,
                        [DataType.SENTIMENT], 29.0,
                        "AI stock scores, alternative data"),
            
            "stocktwits": (StocktwitsAdapter, None, ProviderPriority.FALLBACK,
                          [DataType.SENTIMENT], 0.0,
                          "Social sentiment (free)"),
            
            "binance": (BinanceAdapter, None, ProviderPriority.PRIMARY,
                       [DataType.OHLCV, DataType.CRYPTO], 0.0,
                       "Crypto spot and futures data"),
            
            "coingecko": (CoinGeckoAdapter, "COINGECKO_API_KEY", ProviderPriority.FALLBACK,
                         [DataType.CRYPTO], 0.0,
                         "Crypto market data aggregator"),
            
            "nasdaq_totalview": (NasdaqTotalViewAdapter, "NASDAQ_TOTALVIEW_API_KEY", ProviderPriority.PRIMARY,
                                [DataType.QUOTE], 100.0,
                                "Level 2 order book depth"),
            
            "finra_adf": (FinraAdfAdapter, "FINRA_ADF_API_KEY", ProviderPriority.PRIMARY,
                         [DataType.DARK_POOL], 50.0,
                         "Dark pool prints via FINRA"),
            
            "fred": (FredAdapter, "FRED_API_KEY", ProviderPriority.PRIMARY,
                    [DataType.ECONOMIC], 0.0,
                    "Federal Reserve economic data"),
        }
        
        for name, (adapter_cls, env_key, priority, data_types, cost, desc) in provider_map.items():
            config = ProviderConfig(
                adapter_class=adapter_cls,
                env_key=env_key or "",
                priority=priority,
                data_types=data_types,
                cost_per_month=cost,
                description=desc
            )
            self._provider_configs[name] = config
            
            if env_key is None or os.getenv(env_key):
                try:
                    adapter = adapter_cls()
                    if adapter.is_available():
                        self._providers[name] = adapter
                        logger.info(f"Registered provider: {name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize {name}: {e}")
        
        self._providers["options_flow"] = OptionsFlowAggregator()
        self._providers["alt_data"] = AlternativeDataAggregator()
        self._providers["crypto"] = CryptoDataAggregator()
    
    def _build_routing_table(self):
        """Build data type to provider routing table."""
        for name, adapter in self._providers.items():
            config = self._provider_configs.get(name)
            data_types = getattr(adapter, 'supported_data_types', config.data_types if config else [])
            for data_type in (data_types or []):
                if data_type not in self._data_type_routing:
                    self._data_type_routing[data_type] = []
                self._data_type_routing[data_type].append(name)
        
        for data_type, providers in self._data_type_routing.items():
            providers.sort(key=lambda p: self._provider_configs.get(p, ProviderConfig(
                adapter_class=EnhancedDataAdapter, env_key="", priority=ProviderPriority.FALLBACK
            )).priority.value)
    
    def get_provider(self, name: str) -> Optional[EnhancedDataAdapter]:
        """Get a specific provider by name."""
        return self._providers.get(name)
    
    def get_providers_for_type(self, data_type: DataType) -> List[EnhancedDataAdapter]:
        """Get all providers supporting a data type, sorted by priority."""
        provider_names = self._data_type_routing.get(data_type, [])
        return [self._providers[n] for n in provider_names if n in self._providers]
    
    def get_all_status(self) -> Dict[str, ProviderStatus]:
        """Get status of all registered providers."""
        status = {}
        for name, adapter in self._providers.items():
            try:
                status[name] = adapter.get_status()
            except Exception as e:
                status[name] = ProviderStatus(
                    name=name,
                    available=False,
                    connected=False,
                    last_error=str(e)
                )
        return status
    
    def get_available_data_types(self) -> List[DataType]:
        """Get all data types available from registered providers."""
        return list(self._data_type_routing.keys())
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get monthly cost summary of active providers."""
        active_costs = {}
        potential_costs = {}
        
        for name, config in self._provider_configs.items():
            if name in self._providers:
                active_costs[name] = config.cost_per_month
            else:
                potential_costs[name] = config.cost_per_month
        
        return {
            "active_total": sum(active_costs.values()),
            "active_providers": active_costs,
            "potential_total": sum(potential_costs.values()),
            "potential_providers": potential_costs,
            "full_suite_cost": sum(active_costs.values()) + sum(potential_costs.values())
        }
    
    def fetch(self, request: DataRequest) -> DataResponse:
        """
        Fetch data with automatic provider selection and failover.
        
        Tries providers in priority order until one succeeds.
        """
        providers = self.get_providers_for_type(request.data_type)
        
        if not providers:
            return DataResponse(
                success=False,
                data=None,
                provider="none",
                latency_ms=0,
                error=f"No providers available for {request.data_type.value}"
            )
        
        for provider in providers:
            start_time = datetime.now()
            try:
                data = self._execute_fetch(provider, request)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                return DataResponse(
                    success=True,
                    data=data,
                    provider=provider.name,
                    latency_ms=latency
                )
            except Exception as e:
                logger.warning(f"{provider.name} failed: {e}")
                continue
        
        return DataResponse(
            success=False,
            data=None,
            provider="none",
            latency_ms=0,
            error="All providers failed"
        )
    
    def _execute_fetch(self, provider: EnhancedDataAdapter, request: DataRequest) -> Any:
        """Execute fetch based on data type."""
        if request.data_type == DataType.OHLCV:
            return provider.fetch_ohlcv(
                request.symbol,
                request.start or datetime.now(),
                request.end or datetime.now()
            )
        elif request.data_type == DataType.TICK:
            return provider.fetch_ticks(
                request.symbol,
                request.start or datetime.now(),
                request.end or datetime.now()
            )
        elif request.data_type == DataType.QUOTE:
            return provider.fetch_quotes(
                request.symbol,
                request.start or datetime.now(),
                request.end or datetime.now()
            )
        elif request.data_type == DataType.OPTIONS_FLOW:
            return provider.fetch_options_flow(
                request.symbol,
                request.params.get("min_premium"),
                request.start,
                request.end
            )
        elif request.data_type == DataType.DARK_POOL:
            return provider.fetch_dark_pool(
                request.symbol,
                request.params.get("min_size"),
                request.start,
                request.end
            )
        elif request.data_type == DataType.NEWS:
            return provider.fetch_news(
                request.symbols or ([request.symbol] if request.symbol else None),
                request.start,
                request.end,
                request.limit
            )
        elif request.data_type == DataType.SENTIMENT:
            return provider.fetch_sentiment(
                request.symbol,
                request.start,
                request.end
            )
        elif request.data_type == DataType.FUNDAMENTALS:
            return provider.fetch_fundamentals(request.symbol)
        else:
            raise NotImplementedError(f"Data type {request.data_type} not implemented")
    
    def get_setup_guide(self) -> str:
        """Generate setup guide for all providers."""
        guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              QUANTRACORE APEX - DATA PROVIDER SETUP GUIDE                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

This system supports plug-and-play data providers. Simply set the environment
variable for any provider and it will be automatically discovered and activated.

┌──────────────────────────────────────────────────────────────────────────────┐
│ MARKET DATA PROVIDERS                                                         │
├──────────────────────────────────────────────────────────────────────────────┤

1. POLYGON.IO (Recommended for real-time)
   Cost: $249/mo (Developer) or $500/mo (Advanced)
   Data: Real-time ticks, OHLCV, NBBO, 15+ years history
   
   POLYGON_API_KEY=your_key_here
   POLYGON_TIER=developer
   
2. ALPACA (Free with trading account)
   Cost: $0
   Data: IEX real-time, limited symbols
   
   ALPACA_PAPER_API_KEY=your_key_here
   ALPACA_PAPER_API_SECRET=your_secret_here

┌──────────────────────────────────────────────────────────────────────────────┐
│ OPTIONS FLOW PROVIDERS                                                        │
├──────────────────────────────────────────────────────────────────────────────┤

3. UNUSUAL WHALES (Best value)
   Cost: $35/mo
   Data: Unusual activity, congressional trades, dark pool
   
   UNUSUAL_WHALES_API_KEY=your_key_here

4. FLOWALGO (Institutional grade)
   Cost: $149-199/mo
   Data: Sweeps, blocks, dark pool levels
   
   FLOWALGO_API_KEY=your_key_here

5. INSIDER FINANCE (Mid-tier)
   Cost: $49/mo
   Data: Correlated flow + dark pool
   
   INSIDER_FINANCE_API_KEY=your_key_here

┌──────────────────────────────────────────────────────────────────────────────┐
│ NEWS & SENTIMENT PROVIDERS                                                    │
├──────────────────────────────────────────────────────────────────────────────┤

6. FINNHUB (Free tier available)
   Cost: $0-75/mo
   Data: News, sentiment, insider trades, earnings
   
   FINNHUB_API_KEY=your_key_here

7. ALTINDEX (AI-powered)
   Cost: $29-99/mo
   Data: AI stock scores, social aggregation
   
   ALTINDEX_API_KEY=your_key_here

8. STOCKTWITS (Free)
   Cost: $0
   Data: Real-time social sentiment
   
   (No API key required)

┌──────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2 / ORDER BOOK PROVIDERS                                                │
├──────────────────────────────────────────────────────────────────────────────┤

9. NASDAQ TOTALVIEW
   Cost: ~$100/mo
   Data: Full order book depth
   
   NASDAQ_TOTALVIEW_API_KEY=your_key_here

┌──────────────────────────────────────────────────────────────────────────────┐
│ DARK POOL PROVIDERS                                                           │
├──────────────────────────────────────────────────────────────────────────────┤

10. FINRA ADF
    Cost: ~$50/mo
    Data: Dark pool prints, block trades
    
    FINRA_ADF_API_KEY=your_key_here

┌──────────────────────────────────────────────────────────────────────────────┐
│ ECONOMIC / MACRO DATA PROVIDERS                                               │
├──────────────────────────────────────────────────────────────────────────────┤

11. FRED (Federal Reserve)
    Cost: $0
    Data: Economic indicators, Fed data
    
    FRED_API_KEY=your_key_here

┌──────────────────────────────────────────────────────────────────────────────┐
│ CRYPTOCURRENCY PROVIDERS                                                      │
├──────────────────────────────────────────────────────────────────────────────┤

12. BINANCE (Free)
    Cost: $0
    Data: All crypto pairs, futures, order books
    
    BINANCE_API_KEY=optional_for_trading
    BINANCE_API_SECRET=optional_for_trading

13. COINGECKO (Free tier)
    Cost: $0-129/mo
    Data: 10,000+ cryptos, market data
    
    COINGECKO_API_KEY=optional_for_pro

┌──────────────────────────────────────────────────────────────────────────────┐
│ COST SUMMARY                                                                  │
├──────────────────────────────────────────────────────────────────────────────┤

Minimum Setup (Free):           $0/mo
  - Alpaca, Stocktwits, Binance, FRED

Recommended Setup:              ~$350/mo
  - Polygon Developer ($249)
  - Unusual Whales ($35)
  - Finnhub Free ($0)
  - FRED ($0)

Professional Setup:             ~$800/mo
  - Polygon Developer ($249)
  - Unusual Whales ($35)
  - FlowAlgo ($175)
  - Finnhub Growth ($75)
  - Nasdaq TotalView ($100)
  - FINRA ADF ($50)
  - AltIndex Pro ($99)

Full Suite:                     ~$1,200+/mo
  - All providers enabled

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return guide


def get_registry() -> DataProviderRegistry:
    """Get the singleton data provider registry."""
    return DataProviderRegistry()


PROVIDER_ENV_KEYS = {
    "POLYGON_API_KEY": "Polygon.io real-time market data",
    "POLYGON_TIER": "Polygon subscription tier (free/starter/developer/advanced)",
    "ALPACA_PAPER_API_KEY": "Alpaca paper trading API key",
    "ALPACA_PAPER_API_SECRET": "Alpaca paper trading API secret",
    "UNUSUAL_WHALES_API_KEY": "Unusual Whales options flow",
    "FLOWALGO_API_KEY": "FlowAlgo institutional flow",
    "INSIDER_FINANCE_API_KEY": "InsiderFinance flow + dark pool",
    "FINNHUB_API_KEY": "Finnhub news and sentiment",
    "ALTINDEX_API_KEY": "AltIndex AI stock scores",
    "STOCKTWITS_ACCESS_TOKEN": "Stocktwits premium features",
    "NASDAQ_TOTALVIEW_API_KEY": "Nasdaq Level 2 order book",
    "FINRA_ADF_API_KEY": "FINRA dark pool data",
    "FRED_API_KEY": "Federal Reserve economic data",
    "BINANCE_API_KEY": "Binance trading API",
    "BINANCE_API_SECRET": "Binance trading secret",
    "COINGECKO_API_KEY": "CoinGecko Pro features",
    "QUANDL_API_KEY": "Quandl alternative data",
    "TWITTER_BEARER_TOKEN": "Twitter/X sentiment data",
}
