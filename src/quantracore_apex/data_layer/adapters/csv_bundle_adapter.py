"""
CSV Bundle Data Adapter for QuantraCore Apex.

Provides offline data access from pre-downloaded CSV/Parquet bundles.
Used as fallback when live data providers are unavailable.

Version: 9.0-A
"""

from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import json

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.quantracore_apex.core.schemas import OhlcvBar
from .base import DataAdapter


DEFAULT_BUNDLE_DIR = Path("data/bundles")
DEFAULT_CACHE_DIR = Path("data/cache")


class CSVBundleAdapter(DataAdapter):
    """
    Adapter for loading market data from local CSV/Parquet bundles.
    
    Supports both raw CSV files and Parquet format for efficiency.
    Used for offline analysis and as fallback when APIs fail.
    """
    
    def __init__(
        self,
        bundle_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas package required for CSV bundle adapter")
        
        self.bundle_dir = bundle_dir or DEFAULT_BUNDLE_DIR
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._manifest: Optional[Dict] = None
    
    @property
    def name(self) -> str:
        return "csv_bundle"
    
    def is_available(self) -> bool:
        """Check if bundle directory exists with data."""
        return (
            self.bundle_dir.exists() or 
            self.cache_dir.exists()
        )
    
    def _load_manifest(self) -> Dict:
        """Load the cache manifest."""
        if self._manifest is not None:
            return self._manifest
        
        manifest_path = self.cache_dir / "MANIFEST.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {"symbols": {}}
        
        return self._manifest
    
    def _find_data_file(
        self,
        symbol: str,
        timeframe: str = "1d"
    ) -> Optional[Path]:
        """
        Find the data file for a symbol.
        
        Searches in order:
        1. Cache directory (Parquet)
        2. Cache directory (CSV)
        3. Bundle directory (Parquet)
        4. Bundle directory (CSV)
        """
        search_paths = [
            self.cache_dir / "polygon" / timeframe / f"{symbol}.parquet",
            self.cache_dir / "polygon" / timeframe / f"{symbol}.csv",
            self.cache_dir / timeframe / f"{symbol}.parquet",
            self.cache_dir / timeframe / f"{symbol}.csv",
            self.bundle_dir / timeframe / f"{symbol}.parquet",
            self.bundle_dir / timeframe / f"{symbol}.csv",
            self.bundle_dir / f"{symbol}.parquet",
            self.bundle_dir / f"{symbol}.csv",
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_dataframe(self, path: Path) -> pd.DataFrame:
        """Load a dataframe from file."""
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path, parse_dates=["timestamp"])
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d"
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV data from local bundle.
        
        Args:
            symbol: Stock ticker symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            
        Returns:
            List of OhlcvBar objects
        """
        data_file = self._find_data_file(symbol, timeframe)
        
        if data_file is None:
            raise RuntimeError(f"No local data found for {symbol}")
        
        try:
            df = self._load_dataframe(data_file)
            
            if "timestamp" not in df.columns and df.index.name == "timestamp":
                df = df.reset_index()
            
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
            
            bars = []
            for _, row in df.iterrows():
                bar = OhlcvBar(
                    timestamp=row.get("timestamp", datetime.now()),
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0))),
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            raise RuntimeError(f"Error loading bundle data for {symbol}: {e}")
    
    def fetch(
        self,
        symbol: str,
        days: int = 365,
        timeframe: str = "1d",
        end_date: Optional[str] = None
    ) -> List[OhlcvBar]:
        """
        Fetch OHLCV bars from local bundle.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of history
            timeframe: Bar timeframe
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of OhlcvBar objects
        """
        from datetime import timedelta
        
        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.now()
        
        start = end - timedelta(days=days)
        
        return self.fetch_ohlcv(symbol, start, end, timeframe)
    
    def has_symbol(self, symbol: str, timeframe: str = "1d") -> bool:
        """Check if we have data for a symbol."""
        return self._find_data_file(symbol, timeframe) is not None
    
    def list_available_symbols(self, timeframe: str = "1d") -> List[str]:
        """List all symbols available in the bundle."""
        symbols = set()
        
        search_dirs = [
            self.cache_dir / "polygon" / timeframe,
            self.cache_dir / timeframe,
            self.bundle_dir / timeframe,
            self.bundle_dir,
        ]
        
        for dir_path in search_dirs:
            if not dir_path.exists():
                continue
            
            for file in dir_path.iterdir():
                if file.suffix in [".parquet", ".csv"]:
                    symbols.add(file.stem)
        
        return sorted(symbols)
    
    def save_to_cache(
        self,
        symbol: str,
        bars: List[OhlcvBar],
        timeframe: str = "1d",
        provider: str = "polygon"
    ) -> Path:
        """
        Save bars to cache for future use.
        
        Args:
            symbol: Stock ticker symbol
            bars: List of OhlcvBar objects
            timeframe: Bar timeframe
            provider: Data provider name
            
        Returns:
            Path to saved file
        """
        cache_path = self.cache_dir / provider / timeframe
        cache_path.mkdir(parents=True, exist_ok=True)
        
        file_path = cache_path / f"{symbol}.parquet"
        
        data = [
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
        
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
        
        self._update_manifest(symbol, timeframe, provider, len(bars))
        
        return file_path
    
    def _update_manifest(
        self,
        symbol: str,
        timeframe: str,
        provider: str,
        row_count: int
    ):
        """Update the cache manifest."""
        manifest = self._load_manifest()
        
        key = f"{provider}/{timeframe}/{symbol}"
        manifest["symbols"][key] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "provider": provider,
            "row_count": row_count,
            "updated": datetime.now().isoformat(),
        }
        
        manifest_path = self.cache_dir / "MANIFEST.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        self._manifest = manifest


def create_csv_bundle_adapter(
    bundle_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None
) -> Optional[CSVBundleAdapter]:
    """
    Factory function to create CSV bundle adapter.
    
    Returns:
        CSVBundleAdapter or None if pandas not available
    """
    try:
        return CSVBundleAdapter(bundle_dir, cache_dir)
    except ImportError:
        return None
