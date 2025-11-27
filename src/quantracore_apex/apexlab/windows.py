"""
Window Builder for ApexLab.

Builds 100-bar OHLCV windows from normalized data.
"""

from typing import List, Optional
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow
from src.quantracore_apex.data_layer.normalization import build_windows


class WindowBuilder:
    """
    Builds training windows from OHLCV data.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        step: int = 1,
        min_bars: int = 100
    ):
        self.window_size = window_size
        self.step = step
        self.min_bars = min_bars
    
    def build(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        timeframe: str = "1d"
    ) -> List[OhlcvWindow]:
        """
        Build windows from OHLCV bars.
        
        Args:
            bars: Normalized OHLCV bars
            symbol: Ticker symbol
            timeframe: Bar timeframe
            
        Returns:
            List of OhlcvWindow objects
        """
        if len(bars) < self.min_bars:
            return []
        
        return build_windows(
            bars=bars,
            symbol=symbol,
            timeframe=timeframe,
            window_size=self.window_size,
            step=self.step
        )
    
    def build_single(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        timeframe: str = "1d"
    ) -> Optional[OhlcvWindow]:
        """
        Build a single window from the last N bars.
        """
        if len(bars) < self.window_size:
            return None
        
        return OhlcvWindow(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars[-self.window_size:]
        )
