"""
Synthetic Data Adapter for QuantraCore Apex.

Generates synthetic OHLCV data for testing and demo purposes.
No API key required.
"""

import numpy as np
from typing import List
from datetime import datetime, timedelta
import hashlib

from src.quantracore_apex.core.schemas import OhlcvBar
from .base import DataAdapter


class SyntheticAdapter(DataAdapter):
    """
    Synthetic data adapter for testing.
    
    Generates deterministic synthetic OHLCV data based on symbol and dates.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    @property
    def name(self) -> str:
        return "synthetic"
    
    def is_available(self) -> bool:
        return True
    
    def _get_symbol_seed(self, symbol: str) -> int:
        """Generate deterministic seed from symbol."""
        hash_val = hashlib.md5(symbol.encode()).hexdigest()
        return int(hash_val[:8], 16)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Generate synthetic OHLCV data.
        
        Data is deterministic based on symbol and dates.
        """
        symbol_seed = self._get_symbol_seed(symbol)
        np.random.seed(self.seed + symbol_seed)
        
        if timeframe == "1d":
            delta = timedelta(days=1)
        elif timeframe == "1h":
            delta = timedelta(hours=1)
        else:
            delta = timedelta(days=1)
        
        bars = []
        current = start
        
        base_price = 100 + (symbol_seed % 400)
        base_volume = 1000000 + (symbol_seed % 9000000)
        
        price = float(base_price)
        
        while current <= end:
            if timeframe == "1d" and current.weekday() >= 5:
                current += delta
                continue
            
            daily_return = np.random.normal(0.0003, 0.02)
            volatility = np.random.uniform(0.01, 0.03)
            
            open_price = price
            
            intraday_moves = np.random.normal(0, volatility, 4)
            prices = open_price * (1 + np.cumsum(intraday_moves))
            
            high_price = max(open_price, max(prices))
            low_price = min(open_price, min(prices))
            close_price = open_price * (1 + daily_return)
            
            close_price = max(low_price, min(high_price, close_price))
            
            volume = base_volume * np.random.uniform(0.5, 2.0)
            
            bar = OhlcvBar(
                timestamp=current,
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=round(volume, 0),
            )
            bars.append(bar)
            
            price = close_price
            current += delta
        
        return bars
