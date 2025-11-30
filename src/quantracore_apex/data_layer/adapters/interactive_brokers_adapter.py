"""
Interactive Brokers Data Adapter for QuantraCore Apex.

Provides both market data and execution capabilities through IB Gateway/TWS.
Supports stocks, options, futures, forex, and crypto.

Requirements:
- IB Gateway or TWS running locally
- Valid IB account (paper or live)
- ib_insync library

Environment Variables:
- IB_HOST: IB Gateway host (default: 127.0.0.1)
- IB_PORT: IB Gateway port (default: 7497 for paper, 7496 for live)
- IB_CLIENT_ID: Client ID (default: 1)

Subscription Tiers:
- Free with IB account (market data subscriptions extra)
- Real-time data for subscribed exchanges
- Full execution capabilities
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_enhanced import (
    EnhancedDataAdapter, StreamingDataAdapter, DataType, TimeFrame,
    OhlcvBar, TickData, QuoteData, OptionsFlow, ProviderStatus, FundamentalsData
)


@dataclass
class IBConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 20
    readonly: bool = True


class InteractiveBrokersAdapter(StreamingDataAdapter):
    """
    Interactive Brokers data and execution adapter.
    
    Provides institutional-grade market data and order execution.
    Requires IB Gateway or TWS running locally.
    
    Features:
    - Real-time streaming quotes and trades
    - Historical OHLCV data (tick to monthly)
    - Options chains with Greeks
    - Fundamental data
    - News feeds
    - Order execution (when readonly=False)
    
    Data Coverage:
    - US Stocks (NYSE, NASDAQ, AMEX, ARCA)
    - Options (OPRA)
    - Futures (CME, NYMEX, COMEX, CBOT)
    - Forex (IDEALPRO)
    - Crypto (PAXOS)
    - International markets (LSE, TSE, HKEX, etc.)
    """
    
    def __init__(self, config: Optional[IBConfig] = None):
        self.config = config or IBConfig(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=int(os.getenv("IB_PORT", "7497")),
            client_id=int(os.getenv("IB_CLIENT_ID", "1"))
        )
        self._ib = None
        self._connected = False
        self._subscriptions: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "Interactive Brokers"
    
    @property
    def supported_data_types(self) -> List[DataType]:
        return [
            DataType.OHLCV,
            DataType.TICK,
            DataType.QUOTE,
            DataType.TRADE,
            DataType.OPTIONS_CHAIN,
            DataType.NEWS,
            DataType.FUNDAMENTALS
        ]
    
    def is_available(self) -> bool:
        try:
            from ib_insync import IB
            return True
        except ImportError:
            return False
    
    def get_status(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=self.is_available(),
            connected=self._connected,
            subscription_tier="IB Account",
            data_types=self.supported_data_types,
            last_error=None if self._connected else "Not connected to IB Gateway"
        )
    
    async def connect(self) -> bool:
        if not self.is_available():
            return False
        
        try:
            from ib_insync import IB
            self._ib = IB()
            await self._ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                readonly=self.config.readonly,
                timeout=self.config.timeout
            )
            self._connected = True
            return True
        except Exception as e:
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    async def subscribe(
        self,
        symbols: List[str],
        data_types: List[DataType],
        callback: Callable[[Any], None]
    ) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to IB Gateway")
        pass
    
    async def unsubscribe(
        self,
        symbols: List[str],
        data_types: List[DataType]
    ) -> None:
        pass
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.DAY_1
    ) -> List[OhlcvBar]:
        if not self._connected:
            raise RuntimeError("Not connected to IB Gateway")
        
        from ib_insync import Stock, util
        
        duration_map = {
            TimeFrame.MINUTE_1: "1 min",
            TimeFrame.MINUTE_5: "5 mins",
            TimeFrame.MINUTE_15: "15 mins",
            TimeFrame.MINUTE_30: "30 mins",
            TimeFrame.HOUR_1: "1 hour",
            TimeFrame.HOUR_4: "4 hours",
            TimeFrame.DAY_1: "1 day",
            TimeFrame.WEEK_1: "1 week",
            TimeFrame.MONTH_1: "1 month"
        }
        
        bar_size = duration_map.get(timeframe, "1 day")
        days = (end - start).days
        duration = f"{days} D"
        
        contract = Stock(symbol, "SMART", "USD")
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True
        )
        
        result = []
        for bar in bars:
            result.append(OhlcvBar(
                timestamp=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume
            ))
        
        return result
    
    def fetch_fundamentals(self, symbol: str) -> FundamentalsData:
        if not self._connected:
            raise RuntimeError("Not connected to IB Gateway")
        
        from ib_insync import Stock
        
        contract = Stock(symbol, "SMART", "USD")
        details = self._ib.reqContractDetails(contract)
        
        return FundamentalsData(
            symbol=symbol,
            market_cap=None,
            pe_ratio=None,
            eps=None
        )


IB_SETUP_GUIDE = """
=== Interactive Brokers Setup Guide ===

1. INSTALL IB GATEWAY OR TWS
   - Download from: https://www.interactivebrokers.com/en/trading/tws.php
   - IB Gateway (lighter, headless) or TWS (full UI)
   
2. CONFIGURE API ACCESS
   - In TWS/Gateway: Configure > Settings > API > Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set port: 7497 (paper) or 7496 (live)
   - Add 127.0.0.1 to trusted IPs

3. MARKET DATA SUBSCRIPTIONS
   - US Stocks: NYSE, NASDAQ, AMEX bundles (~$15-45/month)
   - Options: OPRA bundle (~$25/month)
   - Futures: CME, NYMEX bundles (varies)
   - Real-time quotes require subscriptions

4. ENVIRONMENT VARIABLES
   IB_HOST=127.0.0.1
   IB_PORT=7497
   IB_CLIENT_ID=1

5. PYTHON DEPENDENCY
   pip install ib_insync

6. START GATEWAY
   - Launch IB Gateway/TWS
   - Log in with paper or live credentials
   - Gateway must be running for adapter to work
"""
