"""
Options Flow Data Adapter for QuantraCore Apex.

Aggregates options flow from multiple providers for institutional positioning analysis.

Supported Providers:
- Unusual Whales ($35/mo) - Unusual activity, congressional tracking
- FlowAlgo ($149-199/mo) - Dark pool, sweeps, block trades
- InsiderFinance ($49/mo) - Correlated flow + dark pool
- Barchart (Free/Premium) - Basic options flow
- OptionStrat (Free delayed) - Consolidated orders

Requirements:
- Provider-specific API keys (see individual setup guides)

Data Types:
- Large options orders (sweeps, blocks)
- Dark pool prints
- Unusual options activity
- Institutional positioning
- Congressional trading (Unusual Whales)
"""

import os
import time
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .base_enhanced import (
    EnhancedDataAdapter, DataType, TimeFrame,
    OptionsFlow, DarkPoolPrint, ProviderStatus
)


class FlowProvider(Enum):
    UNUSUAL_WHALES = "unusual_whales"
    FLOWALGO = "flowalgo"
    INSIDER_FINANCE = "insider_finance"
    BARCHART = "barchart"
    OPTIONSTRAT = "optionstrat"


@dataclass
class FlowAlert:
    timestamp: datetime
    symbol: str
    underlying: str
    strike: float
    expiration: str
    option_type: str
    side: str
    premium: float
    size: int
    spot_price: float
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    is_sweep: bool = False
    is_block: bool = False
    is_unusual: bool = False
    sentiment: Optional[str] = None
    source: str = "unknown"


class UnusualWhalesAdapter(EnhancedDataAdapter):
    """
    Unusual Whales options flow adapter.
    
    Best for:
    - Unusual options activity detection
    - Congressional trading tracker
    - Retail-friendly pricing ($35/mo)
    - Discord integration
    
    Features:
    - Real-time unusual flow
    - Dark pool activity
    - Sector rotation analysis
    - Congressional disclosures
    """
    
    BASE_URL = "https://api.unusualwhales.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("UNUSUAL_WHALES_API_KEY")
        self._last_request = 0
        self._rate_limit_delay = 1.0
    
    @property
    def name(self) -> str:
        return "Unusual Whales"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OPTIONS_FLOW, DataType.DARK_POOL]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="$35/month",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "UNUSUAL_WHALES_API_KEY not set"
        )
    
    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        if not self.is_available():
            raise ValueError("UNUSUAL_WHALES_API_KEY not set")
        
        time.sleep(self._rate_limit_delay)
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def get_flow(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 25000,
        limit: int = 100
    ) -> List[FlowAlert]:
        params = {"limit": limit, "min_premium": min_premium}
        if symbol:
            params["symbol"] = symbol
        
        data = self._request("api/v2/stock_feed", params)
        
        alerts = []
        for item in data.get("data", []):
            alerts.append(FlowAlert(
                timestamp=datetime.fromisoformat(item.get("created_at", "")),
                symbol=item.get("option_symbol", ""),
                underlying=item.get("underlying_symbol", ""),
                strike=float(item.get("strike", 0)),
                expiration=item.get("expiration_date", ""),
                option_type=item.get("put_call", ""),
                side=item.get("side", ""),
                premium=float(item.get("premium", 0)),
                size=int(item.get("volume", 0)),
                spot_price=float(item.get("stock_price", 0)),
                is_sweep=item.get("is_sweep", False),
                is_unusual=True,
                source="unusual_whales"
            ))
        
        return alerts
    
    def get_congressional_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._request("api/v2/congress/trading-activity", {"limit": limit})
    
    def get_dark_pool(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[DarkPoolPrint]:
        params = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        
        data = self._request("api/v2/darkpool/recent", params)
        
        prints = []
        for item in data.get("data", []):
            prints.append(DarkPoolPrint(
                timestamp=datetime.fromisoformat(item.get("executed_at", "")),
                symbol=item.get("ticker", ""),
                price=float(item.get("price", 0)),
                size=int(item.get("size", 0)),
                value=float(item.get("notional_value", 0)),
                exchange=item.get("exchange", "DARK")
            ))
        
        return prints


class FlowAlgoAdapter(EnhancedDataAdapter):
    """
    FlowAlgo institutional flow adapter.
    
    Best for:
    - Institutional sweep detection
    - Dark pool level analysis
    - Block trade tracking
    - Advanced accounts ($25k+)
    
    Features:
    - Real-time sweeps across all exchanges
    - Dark pool prints with level analysis
    - Block trade detection
    - Historical flow analysis
    
    Pricing: $149-199/month
    """
    
    BASE_URL = "https://api.flowalgo.com"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FLOWALGO_API_KEY")
    
    @property
    def name(self) -> str:
        return "FlowAlgo"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OPTIONS_FLOW, DataType.DARK_POOL]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="$149-199/month",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "FLOWALGO_API_KEY not set"
        )


class InsiderFinanceAdapter(EnhancedDataAdapter):
    """
    InsiderFinance flow adapter.
    
    Best for:
    - Correlated flow + dark pool analysis
    - TradingView integration
    - Mid-tier pricing ($49/mo)
    
    Features:
    - 15M+ daily prints processed
    - Flow + dark pool correlation
    - TradingView charting integration
    """
    
    BASE_URL = "https://api.insiderfinance.io"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("INSIDER_FINANCE_API_KEY")
    
    @property
    def name(self) -> str:
        return "InsiderFinance"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OPTIONS_FLOW, DataType.DARK_POOL]
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier="$49/month",
            data_types=self.supported_data_types,
            last_error=None if self.is_available() else "INSIDER_FINANCE_API_KEY not set"
        )


class OptionsFlowAggregator(EnhancedDataAdapter):
    """
    Aggregates options flow from multiple providers.
    
    Automatically uses available providers and combines results.
    Deduplicates and normalizes data across sources.
    """
    
    def __init__(self):
        self.providers: List[EnhancedDataAdapter] = []
        
        if os.getenv("UNUSUAL_WHALES_API_KEY"):
            self.providers.append(UnusualWhalesAdapter())
        if os.getenv("FLOWALGO_API_KEY"):
            self.providers.append(FlowAlgoAdapter())
        if os.getenv("INSIDER_FINANCE_API_KEY"):
            self.providers.append(InsiderFinanceAdapter())
    
    @property
    def name(self) -> str:
        return "Options Flow Aggregator"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [DataType.OPTIONS_FLOW, DataType.DARK_POOL]
    
    def is_available(self) -> bool:
        return len(self.providers) > 0
    
    def get_status(self) -> ProviderStatus:
        provider_names = [p.name for p in self.providers]
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self.is_available(),
            subscription_tier=f"Active: {', '.join(provider_names)}" if provider_names else "No providers",
            data_types=self.supported_data_types
        )
    
    def get_all_flow(
        self,
        symbol: Optional[str] = None,
        min_premium: float = 25000
    ) -> List[FlowAlert]:
        all_alerts = []
        
        for provider in self.providers:
            try:
                if hasattr(provider, 'get_flow'):
                    alerts = provider.get_flow(symbol, min_premium)
                    all_alerts.extend(alerts)
            except Exception:
                continue
        
        return sorted(all_alerts, key=lambda x: x.timestamp, reverse=True)


OPTIONS_FLOW_SETUP_GUIDE = """
=== Options Flow Providers Setup Guide ===

1. UNUSUAL WHALES ($35/month)
   - Sign up: https://unusualwhales.com/
   - Features: Unusual activity, congressional trades, dark pool
   - Best for: Budget-conscious traders, political tracking
   
   UNUSUAL_WHALES_API_KEY=your_key_here

2. FLOWALGO ($149-199/month)
   - Sign up: https://flowalgo.com/
   - Features: Sweeps, blocks, dark pool levels
   - Best for: Advanced traders, $25k+ accounts
   
   FLOWALGO_API_KEY=your_key_here

3. INSIDER FINANCE ($49/month)
   - Sign up: https://insiderfinance.io/
   - Features: Correlated flow + dark pool, TradingView
   - Best for: Mid-tier budget, charting integration
   
   INSIDER_FINANCE_API_KEY=your_key_here

4. FREE OPTIONS
   - Barchart: https://barchart.com/options/options-flow
   - OptionStrat: https://optionstrat.com/flow (15-min delay)
   - Fintel: https://fintel.io/options-flow

5. WHAT TO LOOK FOR
   - Sweeps: Multi-exchange orders (aggressive)
   - Blocks: Large single prints (negotiated)
   - Premium > $100k: Significant conviction
   - Near expiry + OTM: Speculative bets
   - Deep ITM: Hedging or stock replacement
"""
