"""
Base Data Adapter for QuantraCore Apex.

Defines the standard interface for all data providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from src.quantracore_apex.core.schemas import OhlcvBar


class DataAdapter(ABC):
    """
    Abstract base class for data adapters.
    
    All data providers must implement this interface.
    """
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe ("1d", "1h", etc.)
            
        Returns:
            List of OhlcvBar objects
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the adapter is available (API key set, etc.)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name."""
        pass
