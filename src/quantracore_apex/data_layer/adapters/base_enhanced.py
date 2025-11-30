"""
Enhanced Base Data Adapter for QuantraCore Apex.

Extends the base adapter with streaming, options, and alternative data support.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class DataType(Enum):
    OHLCV = "ohlcv"
    TICK = "tick"
    QUOTE = "quote"
    TRADE = "trade"
    OPTIONS_FLOW = "options_flow"
    OPTIONS_CHAIN = "options_chain"
    DARK_POOL = "dark_pool"
    NEWS = "news"
    SENTIMENT = "sentiment"
    FUNDAMENTALS = "fundamentals"
    SEC_FILINGS = "sec_filings"
    ECONOMIC = "economic"
    CRYPTO = "crypto"


class TimeFrame(Enum):
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class OhlcvBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trades: Optional[int] = None


@dataclass
class TickData:
    timestamp: datetime
    symbol: str
    price: float
    size: float
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None


@dataclass
class QuoteData:
    timestamp: datetime
    symbol: str
    bid: float
    bid_size: float
    ask: float
    ask_size: float
    exchange: Optional[str] = None


@dataclass
class OptionsFlow:
    timestamp: datetime
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str
    side: str
    premium: float
    size: int
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    is_sweep: bool = False
    is_block: bool = False
    exchange: Optional[str] = None
    sentiment: Optional[str] = None


@dataclass
class DarkPoolPrint:
    timestamp: datetime
    symbol: str
    price: float
    size: int
    value: float
    exchange: str
    is_above_ask: bool = False
    is_below_bid: bool = False


@dataclass
class NewsItem:
    timestamp: datetime
    headline: str
    summary: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    categories: List[str] = field(default_factory=list)


@dataclass
class SentimentData:
    timestamp: datetime
    symbol: str
    score: float
    source: str
    volume: Optional[int] = None
    positive_mentions: Optional[int] = None
    negative_mentions: Optional[int] = None
    social_volume: Optional[int] = None


@dataclass
class FundamentalsData:
    symbol: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    shares_outstanding: Optional[float] = None
    short_interest: Optional[float] = None
    institutional_ownership: Optional[float] = None


@dataclass
class ProviderStatus:
    name: str
    available: bool
    connected: bool
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    latency_ms: Optional[float] = None
    last_error: Optional[str] = None
    subscription_tier: Optional[str] = None
    data_types: List[DataType] = field(default_factory=list)


class EnhancedDataAdapter(ABC):
    """
    Enhanced base class for data adapters.
    
    Supports multiple data types, streaming, and async operations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name."""
        pass
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[DataType]:
        """Return list of supported data types."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is available (API key set, etc.)."""
        pass
    
    @abstractmethod
    def get_status(self) -> ProviderStatus:
        """Get detailed provider status."""
        pass
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[OhlcvBar]:
        """Fetch OHLCV data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support OHLCV data")
    
    def fetch_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """Fetch tick data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support tick data")
    
    def fetch_quotes(
        self,
        symbol: str,
        start: datetime,
        end: datetime
    ) -> List[QuoteData]:
        """Fetch quote data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support quote data")
    
    def fetch_options_flow(
        self,
        symbol: Optional[str] = None,
        min_premium: Optional[float] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[OptionsFlow]:
        """Fetch options flow data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support options flow")
    
    def fetch_dark_pool(
        self,
        symbol: Optional[str] = None,
        min_size: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[DarkPoolPrint]:
        """Fetch dark pool prints. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support dark pool data")
    
    def fetch_news(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[NewsItem]:
        """Fetch news. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support news data")
    
    def fetch_sentiment(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[SentimentData]:
        """Fetch sentiment data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support sentiment data")
    
    def fetch_fundamentals(
        self,
        symbol: str
    ) -> FundamentalsData:
        """Fetch fundamental data. Override if supported."""
        raise NotImplementedError(f"{self.name} does not support fundamentals")


class StreamingDataAdapter(EnhancedDataAdapter):
    """
    Base class for streaming data adapters with WebSocket support.
    """
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the streaming service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the streaming service."""
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        symbols: List[str],
        data_types: List[DataType],
        callback: Callable[[Any], None]
    ) -> None:
        """Subscribe to streaming data."""
        pass
    
    @abstractmethod
    async def unsubscribe(
        self,
        symbols: List[str],
        data_types: List[DataType]
    ) -> None:
        """Unsubscribe from streaming data."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected."""
        pass
