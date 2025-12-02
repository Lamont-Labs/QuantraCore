"""
Data Adapters for QuantraCore Apex.

Comprehensive data ingestion layer supporting multiple providers.

Free-Tier Data Sources:
- FRED: 800,000+ economic indicators (completely free)
- Finnhub: Social sentiment from Reddit/Twitter (60 req/min)
- Alpha Vantage: AI news sentiment + technical indicators (500 req/day)
- Polygon.io: OHLCV data (5 req/min free tier)
"""

from .base import DataAdapter
from .base_enhanced import (
    EnhancedDataAdapter,
    StreamingDataAdapter,
    DataType,
    TimeFrame,
    OhlcvBar,
    TickData,
    QuoteData,
    OptionsFlow,
    DarkPoolPrint,
    NewsItem,
    SentimentData,
    FundamentalsData,
    ProviderStatus
)

from .alpha_vantage_adapter import (
    AlphaVantageAdapter,
    NewsSentimentArticle,
    TechnicalIndicator,
    get_alpha_vantage_adapter,
    ALPHA_VANTAGE_SETUP_GUIDE
)
from .polygon_adapter import PolygonAdapter
from .synthetic_adapter import SyntheticAdapter

from .eodhd_adapter import EODHDAdapter
from .fmp_adapter import FMPAdapter
from .nasdaq_datalink_adapter import NasdaqDataLinkAdapter
from .interactive_brokers_adapter import InteractiveBrokersAdapter

from .options_flow_adapter import (
    UnusualWhalesAdapter,
    FlowAlgoAdapter,
    InsiderFinanceAdapter,
    OptionsFlowAggregator
)

from .finnhub_adapter import (
    FinnhubAdapter,
    SocialSentiment,
    NewsSentiment,
    InsiderTransaction,
    CompanyNews,
    get_finnhub_adapter,
    FINNHUB_SETUP_GUIDE
)

from .alternative_data_adapter import (
    AltIndexAdapter,
    StocktwitsAdapter,
    AlternativeDataAggregator
)

from .economic_adapter import (
    FredAdapter,
    EconomicIndicator,
    EconomicDataPoint,
    EconomicEvent,
    MacroRegime,
    EconomicDataAggregator,
    ECONOMIC_SETUP_GUIDE
)

from .crypto_adapter import (
    BinanceAdapter,
    CoinGeckoAdapter,
    CryptoDataAggregator
)

from .data_manager import (
    UnifiedDataManager,
    ProviderPriority,
    ProviderConfig,
    PROVIDER_REGISTRY,
    DATA_PROVIDER_SUMMARY,
    ENVIRONMENT_VARIABLES
)

__all__ = [
    "DataAdapter",
    "EnhancedDataAdapter",
    "StreamingDataAdapter",
    "DataType",
    "TimeFrame",
    "OhlcvBar",
    "TickData",
    "QuoteData",
    "OptionsFlow",
    "DarkPoolPrint",
    "NewsItem",
    "SentimentData",
    "FundamentalsData",
    "ProviderStatus",
    "AlphaVantageAdapter",
    "NewsSentimentArticle",
    "TechnicalIndicator",
    "get_alpha_vantage_adapter",
    "ALPHA_VANTAGE_SETUP_GUIDE",
    "PolygonAdapter",
    "SyntheticAdapter",
    "EODHDAdapter",
    "FMPAdapter",
    "NasdaqDataLinkAdapter",
    "InteractiveBrokersAdapter",
    "UnusualWhalesAdapter",
    "FlowAlgoAdapter",
    "InsiderFinanceAdapter",
    "OptionsFlowAggregator",
    "FinnhubAdapter",
    "SocialSentiment",
    "NewsSentiment",
    "InsiderTransaction",
    "CompanyNews",
    "get_finnhub_adapter",
    "FINNHUB_SETUP_GUIDE",
    "AltIndexAdapter",
    "StocktwitsAdapter",
    "AlternativeDataAggregator",
    "FredAdapter",
    "EconomicIndicator",
    "EconomicDataPoint",
    "EconomicEvent",
    "MacroRegime",
    "EconomicDataAggregator",
    "ECONOMIC_SETUP_GUIDE",
    "BinanceAdapter",
    "CoinGeckoAdapter",
    "CryptoDataAggregator",
    "UnifiedDataManager",
    "ProviderPriority",
    "ProviderConfig",
    "PROVIDER_REGISTRY",
    "DATA_PROVIDER_SUMMARY",
    "ENVIRONMENT_VARIABLES",
]
