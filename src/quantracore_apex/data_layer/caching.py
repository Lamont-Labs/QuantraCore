"""
Data Caching module for QuantraCore Apex.

Provides disk-based caching for OHLCV data.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.quantracore_apex.core.schemas import OhlcvBar
from .hashing import compute_data_hash


logger = logging.getLogger(__name__)


class OhlcvCache:
    """
    Disk-based cache for OHLCV data.
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        historical_dir: str = "data/historical"
    ):
        self.cache_dir = Path(cache_dir)
        self.historical_dir = Path(historical_dir)
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.historical_dir.mkdir(parents=True, exist_ok=True)
        
        self._metadata: Dict[str, Any] = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)
    
    def _get_cache_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> str:
        """Generate cache key for a data request."""
        return f"{symbol}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for a cache key."""
        return self.historical_dir / f"{cache_key}.json"
    
    def get(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[List[OhlcvBar]]:
        """
        Get cached data if available and valid.
        """
        cache_key = self._get_cache_key(symbol, start, end, timeframe)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        if cache_key not in self._metadata:
            return None
        
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            
            stored_hash = self._metadata[cache_key].get("hash")
            current_hash = compute_data_hash(json.dumps(data["bars"], sort_keys=True))
            
            if stored_hash != current_hash:
                logger.warning(f"Cache hash mismatch for {cache_key}")
                return None
            
            bars = []
            for bar_data in data["bars"]:
                bar = OhlcvBar(
                    timestamp=datetime.fromisoformat(bar_data["timestamp"]),
                    open=bar_data["open"],
                    high=bar_data["high"],
                    low=bar_data["low"],
                    close=bar_data["close"],
                    volume=bar_data["volume"],
                )
                bars.append(bar)
            
            logger.debug(f"Cache hit for {cache_key}")
            return bars
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            return None
    
    def put(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        bars: List[OhlcvBar]
    ) -> str:
        """
        Store data in cache.
        
        Returns:
            Data hash
        """
        cache_key = self._get_cache_key(symbol, start, end, timeframe)
        cache_path = self._get_cache_path(cache_key)
        
        bars_data = [
            {
                "timestamp": bar.timestamp.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        
        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bar_count": len(bars),
            "bars": bars_data,
        }
        
        data_hash = compute_data_hash(json.dumps(bars_data, sort_keys=True))
        
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        
        self._metadata[cache_key] = {
            "hash": data_hash,
            "created": datetime.utcnow().isoformat(),
            "bar_count": len(bars),
        }
        self._save_metadata()
        
        logger.debug(f"Cached {len(bars)} bars for {cache_key}")
        return data_hash
    
    def invalidate(self, symbol: str) -> None:
        """Invalidate all cache entries for a symbol."""
        keys_to_remove = [k for k in self._metadata if k.startswith(f"{symbol}_")]
        
        for key in keys_to_remove:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            del self._metadata[key]
        
        self._save_metadata()
    
    def clear(self) -> None:
        """Clear all cached data."""
        for cache_path in self.historical_dir.glob("*.json"):
            cache_path.unlink()
        
        self._metadata = {}
        self._save_metadata()
