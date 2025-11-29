"""
Rolling Window Manager for Real-Time Data.

Maintains rolling OHLCV windows per symbol for ApexEngine analysis.
Handles bar aggregation, window updates, and data integrity.
"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """Single OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0
    trade_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "trade_count": self.trade_count,
        }


@dataclass
class SymbolWindow:
    """Rolling window state for a single symbol."""
    symbol: str
    window_size: int = 100
    bars: deque = field(default_factory=lambda: deque(maxlen=100))
    current_bar: Optional[BarData] = None
    bar_interval_seconds: int = 60
    last_update: datetime = field(default_factory=datetime.utcnow)
    is_complete: bool = False
    
    def __post_init__(self):
        self.bars = deque(maxlen=self.window_size)


class RollingWindowManager:
    """
    Manages rolling OHLCV windows for multiple symbols.
    
    Responsibilities:
    1. Aggregate tick/trade data into OHLCV bars
    2. Maintain rolling window of N bars per symbol
    3. Signal when windows are ready for analysis
    4. Handle market open/close transitions
    """
    
    def __init__(
        self,
        window_size: int = 100,
        bar_interval_seconds: int = 60,
        symbols: Optional[List[str]] = None,
    ):
        self.window_size = window_size
        self.bar_interval_seconds = bar_interval_seconds
        
        self._windows: Dict[str, SymbolWindow] = {}
        
        if symbols:
            for symbol in symbols:
                self.add_symbol(symbol)
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self._windows:
            self._windows[symbol] = SymbolWindow(
                symbol=symbol,
                window_size=self.window_size,
                bar_interval_seconds=self.bar_interval_seconds,
            )
            logger.debug(f"[RollingWindowManager] Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from tracking."""
        if symbol in self._windows:
            del self._windows[symbol]
            logger.debug(f"[RollingWindowManager] Removed symbol: {symbol}")
    
    def update_from_trade(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[BarData]:
        """
        Update window with a new trade.
        
        Returns completed bar if a bar boundary was crossed.
        """
        if symbol not in self._windows:
            self.add_symbol(symbol)
        
        window = self._windows[symbol]
        ts = timestamp or datetime.utcnow()
        
        completed_bar = None
        
        if window.current_bar is None:
            bar_start = self._align_to_bar_start(ts)
            window.current_bar = BarData(
                timestamp=bar_start,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume,
                trade_count=1,
            )
        else:
            current_bar_end = window.current_bar.timestamp + timedelta(
                seconds=self.bar_interval_seconds
            )
            
            if ts >= current_bar_end:
                completed_bar = deepcopy(window.current_bar)
                window.bars.append(completed_bar)
                window.is_complete = len(window.bars) >= self.window_size
                
                bar_start = self._align_to_bar_start(ts)
                window.current_bar = BarData(
                    timestamp=bar_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                    trade_count=1,
                )
            else:
                window.current_bar.high = max(window.current_bar.high, price)
                window.current_bar.low = min(window.current_bar.low, price)
                window.current_bar.close = price
                window.current_bar.volume += volume
                window.current_bar.trade_count += 1
        
        window.last_update = ts
        
        return completed_bar
    
    def update_from_bar(self, symbol: str, bar: BarData) -> None:
        """
        Update window with a complete bar.
        
        Used when receiving aggregated bar data from feed.
        """
        if symbol not in self._windows:
            self.add_symbol(symbol)
        
        window = self._windows[symbol]
        window.bars.append(bar)
        window.current_bar = None
        window.last_update = bar.timestamp
        window.is_complete = len(window.bars) >= self.window_size
    
    def update_from_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update with quote data (for price reference).
        
        Does not create new bars, just updates current price reference.
        """
        if symbol not in self._windows:
            return
        
        mid = (bid + ask) / 2
        window = self._windows[symbol]
        
        if window.current_bar is not None:
            window.current_bar.close = mid
            window.current_bar.high = max(window.current_bar.high, mid)
            window.current_bar.low = min(window.current_bar.low, mid)
    
    def _align_to_bar_start(self, ts: datetime) -> datetime:
        """Align timestamp to bar interval boundary."""
        seconds = ts.second + ts.minute * 60 + ts.hour * 3600
        aligned_seconds = (seconds // self.bar_interval_seconds) * self.bar_interval_seconds
        
        return ts.replace(
            hour=aligned_seconds // 3600,
            minute=(aligned_seconds % 3600) // 60,
            second=aligned_seconds % 60,
            microsecond=0,
        )
    
    def is_ready(self, symbol: str) -> bool:
        """Check if symbol has complete window for analysis."""
        if symbol not in self._windows:
            return False
        return self._windows[symbol].is_complete
    
    def get_bars(self, symbol: str) -> List[BarData]:
        """Get all bars for a symbol."""
        if symbol not in self._windows:
            return []
        return list(self._windows[symbol].bars)
    
    def get_bars_as_dicts(self, symbol: str) -> List[Dict[str, Any]]:
        """Get bars as dictionaries for ApexEngine."""
        return [b.to_dict() for b in self.get_bars(symbol)]
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price (last close) for a symbol."""
        if symbol not in self._windows:
            return None
        
        window = self._windows[symbol]
        
        if window.current_bar:
            return window.current_bar.close
        
        if window.bars:
            return window.bars[-1].close
        
        return None
    
    def get_all_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        prices = {}
        for symbol in self._windows:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices
    
    def get_ready_symbols(self) -> List[str]:
        """Get list of symbols with complete windows."""
        return [s for s in self._windows if self.is_ready(s)]
    
    def get_bar_count(self, symbol: str) -> int:
        """Get number of bars for a symbol."""
        if symbol not in self._windows:
            return 0
        return len(self._windows[symbol].bars)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all windows."""
        status = {}
        for symbol, window in self._windows.items():
            status[symbol] = {
                "bars": len(window.bars),
                "is_complete": window.is_complete,
                "last_update": window.last_update.isoformat() if window.last_update else None,
                "current_price": self.get_current_price(symbol),
            }
        return status
    
    def clear(self) -> None:
        """Clear all windows."""
        self._windows.clear()
    
    def clear_symbol(self, symbol: str) -> None:
        """Clear window for a specific symbol."""
        if symbol in self._windows:
            self._windows[symbol] = SymbolWindow(
                symbol=symbol,
                window_size=self.window_size,
                bar_interval_seconds=self.bar_interval_seconds,
            )
    
    def force_complete_current_bars(self) -> Dict[str, BarData]:
        """
        Force completion of all current bars.
        
        Used at market close or for immediate analysis.
        """
        completed = {}
        
        for symbol, window in self._windows.items():
            if window.current_bar is not None:
                completed_bar = deepcopy(window.current_bar)
                window.bars.append(completed_bar)
                window.current_bar = None
                window.is_complete = len(window.bars) >= self.window_size
                completed[symbol] = completed_bar
        
        return completed
    
    def load_historical(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """
        Load historical bars to initialize window.
        
        Used to bootstrap window at startup.
        """
        if symbol not in self._windows:
            self.add_symbol(symbol)
        
        window = self._windows[symbol]
        
        for bar_dict in bars[-self.window_size:]:
            ts = bar_dict.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            
            bar = BarData(
                timestamp=ts,
                open=bar_dict.get("open", 0),
                high=bar_dict.get("high", 0),
                low=bar_dict.get("low", 0),
                close=bar_dict.get("close", 0),
                volume=bar_dict.get("volume", 0),
                vwap=bar_dict.get("vwap", 0),
                trade_count=bar_dict.get("trade_count", 0),
            )
            window.bars.append(bar)
        
        window.is_complete = len(window.bars) >= self.window_size
        window.last_update = datetime.utcnow()
        
        logger.info(
            f"[RollingWindowManager] Loaded {len(window.bars)} historical bars for {symbol}"
        )
