"""
1-Minute Intraday Data Pipeline for ML Training

This module provides tools to:
1. Download and process Kaggle S&P 500 1-min dataset (2008-2021)
2. Fetch recent 1-min data from Alpha Vantage (2022-present)
3. Merge datasets for comprehensive training

The 1-minute data provides ~390x more training samples than EOD data,
enabling much more accurate pattern detection and timing signals.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
import requests
import time
import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow

logger = logging.getLogger(__name__)


@dataclass
class IntradayConfig:
    """Configuration for intraday data pipeline."""
    kaggle_data_dir: str = "data/intraday_1min/kaggle"
    alphavantage_data_dir: str = "data/intraday_1min/alphavantage"
    merged_data_dir: str = "data/intraday_1min/merged"
    cache_dir: str = "data/intraday_1min/cache"
    window_size: int = 100
    step_size: int = 10
    symbols: List[str] = field(default_factory=list)


class KaggleDataProcessor:
    """
    Processes Kaggle S&P 500 1-minute historical data.
    
    Dataset: https://www.kaggle.com/datasets/gratefuldata/intraday-stock-data-1-min-sp-500-200821
    Coverage: 2008-2021, ~500 S&P 500 stocks
    """
    
    def __init__(self, data_dir: str = "data/intraday_1min/kaggle"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def load_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load 1-minute data for a single symbol from Kaggle dataset."""
        if symbol in self._cache:
            return self._cache[symbol]
        
        possible_files = [
            self.data_dir / f"{symbol}.csv",
            self.data_dir / f"{symbol.lower()}.csv",
            self.data_dir / f"{symbol}.csv.gz",
            self.data_dir / f"{symbol.lower()}.csv.gz",
        ]
        
        for filepath in possible_files:
            if filepath.exists():
                try:
                    if str(filepath).endswith('.gz'):
                        df = pd.read_csv(filepath, compression='gzip')
                    else:
                        df = pd.read_csv(filepath)
                    
                    df = self._normalize_columns(df)
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    self._cache[symbol] = df
                    logger.info(f"Loaded {len(df):,} 1-min bars for {symbol}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
        
        return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format."""
        column_map = {
            'Date': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'Datetime': 'timestamp',
            'Time': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
        }
        
        df = df.rename(columns=column_map)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        result: pd.DataFrame = df[required_cols].copy()
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols available in Kaggle dataset."""
        symbols = []
        for f in self.data_dir.glob("*.csv*"):
            symbol = f.stem.replace('.csv', '').upper()
            symbols.append(symbol)
        return sorted(set(symbols))
    
    def to_ohlcv_bars(self, df: pd.DataFrame) -> List[OhlcvBar]:
        """Convert DataFrame to list of OhlcvBar objects."""
        bars = []
        for _, row in df.iterrows():
            ts = row['timestamp']
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            bar = OhlcvBar(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            )
            bars.append(bar)
        return bars
    
    def generate_windows(
        self,
        symbol: str,
        window_size: int = 100,
        step_size: int = 10,
    ) -> Generator[OhlcvWindow, None, None]:
        """Generate sliding windows from 1-minute data."""
        df = self.load_symbol(symbol)
        if df is None or len(df) < window_size:
            return
        
        bars = self.to_ohlcv_bars(df)
        
        for i in range(0, len(bars) - window_size + 1, step_size):
            window_bars = bars[i:i + window_size]
            yield OhlcvWindow(
                symbol=symbol,
                timeframe="1min",
                bars=window_bars,
            )


class AlphaVantageIntradayFetcher:
    """
    Fetches 1-minute intraday data from Alpha Vantage.
    
    Free tier: 500 calls/day, 5 calls/minute
    Data: Real-time and historical intraday
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.data_dir = Path("data/intraday_1min/alphavantage")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0
        self._rate_delay = 12.5
    
    def _rate_limit(self):
        """Enforce rate limiting (5 calls/min = 12 sec between calls)."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_delay:
            time.sleep(self._rate_delay - elapsed)
        self._last_request = time.time()
    
    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "1min",
        outputsize: str = "full",
        month: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch intraday data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: 1min, 5min, 15min, 30min, 60min
            outputsize: 'compact' (last 100) or 'full' (full month)
            month: YYYY-MM format for historical months (premium feature)
        
        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.api_key:
            logger.error("ALPHA_VANTAGE_API_KEY not set")
            return None
        
        self._rate_limit()
        
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": outputsize,
            "datatype": "json",
        }
        
        if month:
            params["month"] = month
        
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if "Error Message" in data:
                logger.error(f"API error for {symbol}: {data['Error Message']}")
                return None
            
            if "Note" in data:
                logger.warning(f"Rate limit warning: {data['Note']}")
                return None
            
            time_series_key = f"Time Series ({interval})"
            if time_series_key not in data:
                logger.warning(f"No data for {symbol}")
                return None
            
            time_series = data[time_series_key]
            
            records = []
            for timestamp_str, values in time_series.items():
                records.append({
                    'timestamp': pd.to_datetime(timestamp_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': float(values['5. volume']),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            cache_file = self.data_dir / f"{symbol}_{interval}.csv"
            df.to_csv(cache_file, index=False)
            
            logger.info(f"Fetched {len(df):,} bars for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def load_cached(self, symbol: str, interval: str = "1min") -> Optional[pd.DataFrame]:
        """Load cached data if available."""
        cache_file = self.data_dir / f"{symbol}_{interval}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return None
    
    def fetch_extended_history(
        self,
        symbol: str,
        months: List[str],
        interval: str = "1min",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch extended history across multiple months.
        
        Args:
            symbol: Stock symbol
            months: List of 'YYYY-MM' strings
            interval: Time interval
        
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for month in months:
            logger.info(f"Fetching {symbol} for {month}...")
            df = self.fetch_intraday(symbol, interval, outputsize="full", month=month)
            if df is not None and len(df) > 0:
                all_data.append(df)
            time.sleep(1)
        
        if not all_data:
            return None
        
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['timestamp'])
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        return combined


class IntradayDataMerger:
    """
    Merges Kaggle historical data with Alpha Vantage recent data.
    
    Creates a unified dataset with:
    - Kaggle: 2008-2021 (bulk historical)
    - Alpha Vantage: 2022-present (recent data)
    """
    
    def __init__(self, output_dir: str = "data/intraday_1min/merged"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.kaggle_processor = KaggleDataProcessor()
        self.av_fetcher = AlphaVantageIntradayFetcher()
    
    def merge_symbol(
        self,
        symbol: str,
        fetch_recent: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Merge Kaggle and Alpha Vantage data for a symbol.
        
        Args:
            symbol: Stock symbol
            fetch_recent: Whether to fetch recent data from Alpha Vantage
        
        Returns:
            Merged DataFrame
        """
        kaggle_df = self.kaggle_processor.load_symbol(symbol)
        
        av_df = None
        if fetch_recent:
            av_df = self.av_fetcher.load_cached(symbol)
            if av_df is None:
                av_df = self.av_fetcher.fetch_intraday(symbol)
        
        if kaggle_df is None and av_df is None:
            logger.warning(f"No data available for {symbol}")
            return None
        
        if kaggle_df is None:
            return av_df
        if av_df is None:
            return kaggle_df
        
        kaggle_end = kaggle_df['timestamp'].max()
        av_new = av_df[av_df['timestamp'] > kaggle_end]
        
        merged = pd.concat([kaggle_df, av_new], ignore_index=True)
        merged = merged.drop_duplicates(subset=['timestamp'])
        merged = merged.sort_values('timestamp').reset_index(drop=True)
        
        output_file = self.output_dir / f"{symbol}_1min.csv"
        merged.to_csv(output_file, index=False)
        
        logger.info(f"Merged {len(merged):,} bars for {symbol}")
        return merged
    
    def generate_training_windows(
        self,
        symbol: str,
        window_size: int = 100,
        step_size: int = 10,
        fetch_recent: bool = False,
    ) -> Generator[OhlcvWindow, None, None]:
        """Generate training windows from merged data."""
        df = self.merge_symbol(symbol, fetch_recent=fetch_recent)
        if df is None or len(df) < window_size:
            return
        
        bars = []
        for _, row in df.iterrows():
            ts = row['timestamp']
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            bar = OhlcvBar(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            )
            bars.append(bar)
        
        for i in range(0, len(bars) - window_size + 1, step_size):
            window_bars = bars[i:i + window_size]
            yield OhlcvWindow(
                symbol=symbol,
                timeframe="1min",
                bars=window_bars,
            )


class IntradayTrainingPipeline:
    """
    Complete pipeline for training on 1-minute data.
    
    Handles:
    1. Data loading from Kaggle + Alpha Vantage
    2. Window generation with configurable parameters
    3. Feature extraction optimized for intraday
    4. Label generation for various prediction horizons
    """
    
    def __init__(self, config: Optional[IntradayConfig] = None):
        self.config = config or IntradayConfig()
        self.merger = IntradayDataMerger(self.config.merged_data_dir)
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure all data directories exist."""
        for dir_path in [
            self.config.kaggle_data_dir,
            self.config.alphavantage_data_dir,
            self.config.merged_data_dir,
            self.config.cache_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_training_stats(self) -> Dict[str, any]:
        """Get statistics about available training data."""
        kaggle_symbols = self.merger.kaggle_processor.get_available_symbols()
        
        stats = {
            "kaggle_symbols": len(kaggle_symbols),
            "total_symbols": len(kaggle_symbols),
            "estimated_bars_per_symbol": 390 * 252 * 13,
            "bars_per_year": 390 * 252,
            "window_size": self.config.window_size,
            "step_size": self.config.step_size,
        }
        
        estimated_windows = (
            stats["estimated_bars_per_symbol"] // self.config.step_size
        ) * len(kaggle_symbols)
        stats["estimated_windows"] = estimated_windows
        
        return stats
    
    def generate_all_windows(
        self,
        symbols: Optional[List[str]] = None,
        max_symbols: Optional[int] = None,
        fetch_recent: bool = False,
    ) -> Generator[OhlcvWindow, None, None]:
        """
        Generate training windows for all specified symbols.
        
        Args:
            symbols: List of symbols (defaults to all available)
            max_symbols: Limit number of symbols processed
            fetch_recent: Whether to fetch recent Alpha Vantage data
        
        Yields:
            OhlcvWindow objects for training
        """
        if symbols is None:
            symbols = self.merger.kaggle_processor.get_available_symbols()
        
        if max_symbols:
            symbols = symbols[:max_symbols]
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            try:
                for window in self.merger.generate_training_windows(
                    symbol,
                    window_size=self.config.window_size,
                    step_size=self.config.step_size,
                    fetch_recent=fetch_recent,
                ):
                    yield window
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
    
    def save_windows_to_disk(
        self,
        symbols: Optional[List[str]] = None,
        output_file: str = "data/intraday_1min/training_windows.jsonl.gz",
        max_symbols: Optional[int] = None,
    ) -> int:
        """
        Save all windows to disk for later training.
        
        Args:
            symbols: List of symbols
            output_file: Output file path
            max_symbols: Limit symbols
        
        Returns:
            Number of windows saved
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with gzip.open(output_path, 'wt') as f:
            for window in self.generate_all_windows(symbols, max_symbols):
                window_dict = {
                    "symbol": window.symbol,
                    "timeframe": window.timeframe,
                    "bars": [
                        {
                            "timestamp": bar.timestamp.isoformat(),
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                        }
                        for bar in window.bars
                    ],
                }
                f.write(json.dumps(window_dict) + "\n")
                count += 1
                
                if count % 10000 == 0:
                    logger.info(f"Saved {count:,} windows...")
        
        logger.info(f"Saved {count:,} windows to {output_file}")
        return count
    
    def load_windows_from_disk(
        self,
        input_file: str = "data/intraday_1min/training_windows.jsonl.gz",
        limit: Optional[int] = None,
    ) -> Generator[OhlcvWindow, None, None]:
        """
        Load windows from disk for training.
        
        Args:
            input_file: Input file path
            limit: Maximum windows to load
        
        Yields:
            OhlcvWindow objects
        """
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"File not found: {input_file}")
            return
        
        count = 0
        with gzip.open(input_path, 'rt') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                data = json.loads(line)
                bars = [
                    OhlcvBar(
                        timestamp=datetime.fromisoformat(b["timestamp"]),
                        open=b["open"],
                        high=b["high"],
                        low=b["low"],
                        close=b["close"],
                        volume=b["volume"],
                    )
                    for b in data["bars"]
                ]
                
                yield OhlcvWindow(
                    symbol=data["symbol"],
                    timeframe=data["timeframe"],
                    bars=bars,
                )
                count += 1


def download_kaggle_dataset_instructions() -> str:
    """Return instructions for downloading Kaggle dataset."""
    return """
    ============================================================
    HOW TO DOWNLOAD KAGGLE S&P 500 1-MINUTE DATA
    ============================================================
    
    1. Go to: https://www.kaggle.com/datasets/gratefuldata/intraday-stock-data-1-min-sp-500-200821
    
    2. Click "Download" (requires free Kaggle account)
    
    3. Extract the ZIP file
    
    4. Copy CSV files to: data/intraday_1min/kaggle/
    
    Dataset contains:
    - ~500 S&P 500 stocks
    - 2008-2021 (12+ years)
    - 1-minute OHLCV bars
    - ~50 million+ data points per symbol
    
    Alternative: Use Kaggle API
    
    pip install kaggle
    kaggle datasets download -d gratefuldata/intraday-stock-data-1-min-sp-500-200821
    unzip intraday-stock-data-1-min-sp-500-200821.zip -d data/intraday_1min/kaggle/
    
    ============================================================
    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print(download_kaggle_dataset_instructions())
    
    pipeline = IntradayTrainingPipeline()
    stats = pipeline.get_training_stats()
    print(f"\nTraining Data Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
