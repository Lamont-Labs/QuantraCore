"""
Data Adapters for QuantraCore Apex.

Comprehensive data ingestion layer supporting multiple providers.
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

from .alpha_vantage_adapter import AlphaVantageAdapter
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

from .alternative_data_adapter import (
    FinnhubAdapter,
    AltIndexAdapter,
    StocktwitsAdapter,
    AlternativeDataAggregator
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
    "AltIndexAdapter",
    "StocktwitsAdapter",
    "AlternativeDataAggregator",
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
