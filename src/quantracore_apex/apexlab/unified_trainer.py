"""
Unified Multi-Source Training Pipeline.

Uses both Alpaca and Polygon data sources in parallel for faster training.
Symbols are distributed across providers to maximize throughput.
"""

import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar, ApexContext
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.prediction.apexcore_v3 import ApexCoreV3Model

logger = logging.getLogger(__name__)


@dataclass
class UnifiedTrainingConfig:
    """Configuration for multi-source training."""
    symbols: List[str] = field(default_factory=lambda: [
        "UBER", "LYFT", "ABNB", "DASH", "RBLX",
        "SNAP", "PINS", "TWLO", "DDOG", "NET",
        "CRWD", "ZS", "PANW", "FTNT", "OKTA",
        "SHOP", "SQ", "COIN", "HOOD", "SOFI",
        "PLTR", "SNOW", "MDB", "TEAM", "ZM",
        "ROKU", "TTD", "UNITY", "U", "BILL",
        "ARM", "SMCI", "MRVL", "AVGO", "QCOM",
        "MU", "LRCX", "KLAC", "AMAT", "ASML",
        "NOW", "WDAY", "HUBS", "VEEV", "CDNS",
        "SNPS", "ANSS", "PTC", "ADSK", "TYL",
        "ENPH", "SEDG", "FSLR", "RUN", "NOVA",
        "RIVN", "LCID", "NIO", "XPEV", "LI",
    ])
    lookback_days: int = 365
    window_size: int = 100
    step_size: int = 2
    future_bars: int = 10
    runner_threshold: float = 0.05
    model_output_dir: str = "models/apexcore_v3"
    max_workers: int = 4


class PolygonFetcher:
    """Fetches data from Polygon.io."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        self.api_key = os.environ.get("POLYGON_API_KEY")
        self._available = bool(self.api_key)
        self._last_request = 0
        self._rate_delay = 12.5
        self._lock = Lock()
    
    @property
    def name(self) -> str:
        return "polygon"
    
    def is_available(self) -> bool:
        return self._available
    
    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_delay:
                time.sleep(self._rate_delay - elapsed)
            self._last_request = time.time()
    
    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        max_retries: int = 3,
    ) -> List[OhlcvBar]:
        """Fetch daily bars from Polygon."""
        if not self._available:
            return []
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        for attempt in range(max_retries):
            self._rate_limit()
            
            try:
                resp = requests.get(url, params=params, timeout=30)
                
                if resp.status_code == 429:
                    wait = (attempt + 1) * 15
                    logger.warning(f"[Polygon] Rate limited on {symbol}, waiting {wait}s")
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                results = data.get("results", [])
                if not results:
                    return []
                
                bars = []
                for r in results:
                    bar = OhlcvBar(
                        timestamp=datetime.fromtimestamp(r["t"] / 1000),
                        open=float(r["o"]),
                        high=float(r["h"]),
                        low=float(r["l"]),
                        close=float(r["c"]),
                        volume=float(r["v"]),
                    )
                    bars.append(bar)
                
                return bars
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                logger.error(f"[Polygon] Error fetching {symbol}: {e}")
                return []
        
        return []


class AlpacaFetcher:
    """Fetches data from Alpaca Markets."""
    
    BASE_URL = "https://data.alpaca.markets"
    
    def __init__(self):
        self.api_key = os.environ.get("ALPACA_PAPER_API_KEY")
        self.api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
        self._available = bool(self.api_key and self.api_secret)
        self._last_request = 0
        self._rate_delay = 0.3
        self._lock = Lock()
        self._session = requests.Session()
        
        if self._available:
            self._session.headers["APCA-API-KEY-ID"] = self.api_key or ""
            self._session.headers["APCA-API-SECRET-KEY"] = self.api_secret or ""
    
    @property
    def name(self) -> str:
        return "alpaca"
    
    def is_available(self) -> bool:
        return self._available
    
    def _rate_limit(self):
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_delay:
                time.sleep(self._rate_delay - elapsed)
            self._last_request = time.time()
    
    def fetch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        max_retries: int = 3,
    ) -> List[OhlcvBar]:
        """Fetch daily bars from Alpaca."""
        if not self._available:
            return []
        
        endpoint = f"/v2/stocks/{symbol}/bars"
        params = {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeframe": "1Day",
            "adjustment": "all",
            "limit": 10000,
            "feed": "iex",
        }
        
        all_bars = []
        
        for attempt in range(max_retries):
            self._rate_limit()
            
            try:
                url = f"{self.BASE_URL}{endpoint}"
                resp = self._session.get(url, params=params, timeout=30)
                
                if resp.status_code == 403:
                    logger.debug(f"[Alpaca] 403 for {symbol} - data subscription may be required")
                    return []
                
                if resp.status_code == 429:
                    wait = min(60 * (2 ** attempt), 300)
                    logger.warning(f"[Alpaca] Rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                data = resp.json()
                
                bars_data = data.get("bars", [])
                for bar in bars_data:
                    try:
                        timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                        timestamp = timestamp.replace(tzinfo=None)
                        
                        ohlcv = OhlcvBar(
                            timestamp=timestamp,
                            open=float(bar["o"]),
                            high=float(bar["h"]),
                            low=float(bar["l"]),
                            close=float(bar["c"]),
                            volume=float(bar["v"])
                        )
                        all_bars.append(ohlcv)
                    except (KeyError, ValueError):
                        continue
                
                next_token = data.get("next_page_token")
                if next_token:
                    params["page_token"] = next_token
                    continue
                
                return all_bars
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                logger.error(f"[Alpaca] Error fetching {symbol}: {e}")
                return []
        
        return all_bars


class WindowGenerator:
    """Generates training windows from bar data."""
    
    def __init__(self, window_size: int = 100, step_size: int = 5):
        self.window_size = window_size
        self.step_size = step_size
    
    def generate(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        future_bars: int = 10,
    ) -> List[Tuple[OhlcvWindow, np.ndarray]]:
        """Generate windows with future price data."""
        if len(bars) < self.window_size + future_bars:
            return []
        
        windows = []
        for i in range(0, len(bars) - self.window_size - future_bars, self.step_size):
            window_bars = bars[i:i + self.window_size]
            future_closes = np.array([
                b.close for b in bars[i + self.window_size:i + self.window_size + future_bars]
            ])
            
            window = OhlcvWindow(
                symbol=symbol,
                bars=window_bars,
                timeframe="1d",
            )
            windows.append((window, future_closes))
        
        return windows


class OutcomeLabelGenerator:
    """Generates labels from actual price outcomes."""
    
    def __init__(self, runner_threshold: float = 0.05):
        self.runner_threshold = runner_threshold
    
    def generate(
        self,
        entry_price: float,
        future_closes: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate labels from what actually happened."""
        if len(future_closes) == 0:
            return self._default()
        
        returns = (future_closes - entry_price) / entry_price
        
        final_return = returns[-1]
        max_return = np.max(returns)
        min_return = np.min(returns)
        max_drawdown = -min_return if min_return < 0 else 0
        
        if final_return > 0.02:
            regime_label = "trending_up"
        elif final_return < -0.02:
            regime_label = "trending_down"
        else:
            regime_label = "range_bound"
        
        avoid_trade = 1 if (max_drawdown > 0.08 or final_return < -0.05) else 0
        
        if max_return >= 0.10:
            quality_tier = "A+"
        elif max_return >= 0.05:
            quality_tier = "A"
        elif max_return >= 0.02:
            quality_tier = "B"
        elif max_return >= 0:
            quality_tier = "C"
        else:
            quality_tier = "D"
        
        reward_risk = max_return / (max_drawdown + 0.001)
        base = 50 + (final_return * 500)
        penalty = max_drawdown * 300
        quantra_score = np.clip(base - penalty + (reward_risk * 5), 0, 100)
        
        if final_return < -0.03 or max_drawdown > 0.08:
            quantra_score = min(quantra_score, 30)
        
        return {
            "quantra_score": float(quantra_score),
            "hit_runner_threshold": 1 if max_return >= self.runner_threshold else 0,
            "future_quality_tier": quality_tier,
            "avoid_trade": avoid_trade,
            "regime_label": regime_label,
            "ret_1d": float(returns[0]) if len(returns) > 0 else 0.0,
            "ret_3d": float(returns[2]) if len(returns) > 2 else 0.0,
            "ret_5d": float(returns[4]) if len(returns) > 4 else 0.0,
            "max_runup_5d": float(np.max(returns[:5])) if len(returns) >= 5 else float(max_return),
            "max_drawdown_5d": float(np.min(returns[:5])) if len(returns) >= 5 else float(min_return),
        }
    
    def _default(self) -> Dict[str, Any]:
        return {
            "quantra_score": 50.0,
            "hit_runner_threshold": 0,
            "future_quality_tier": "C",
            "avoid_trade": 0,
            "regime_label": "range_bound",
            "ret_1d": 0.0, "ret_3d": 0.0, "ret_5d": 0.0,
            "max_runup_5d": 0.0, "max_drawdown_5d": 0.0,
        }


class UnifiedTrainer:
    """
    Multi-source training pipeline using both Alpaca and Polygon.
    
    Distributes symbols across available providers for parallel fetching,
    effectively doubling training speed when both sources are available.
    """
    
    def __init__(self, config: Optional[UnifiedTrainingConfig] = None):
        self.config = config or UnifiedTrainingConfig()
        
        self.polygon = PolygonFetcher()
        self.alpaca = AlpacaFetcher()
        
        self.window_gen = WindowGenerator(
            window_size=self.config.window_size,
            step_size=self.config.step_size,
        )
        self.label_gen = OutcomeLabelGenerator(
            runner_threshold=self.config.runner_threshold,
        )
        self.feature_extractor = FeatureExtractor()
        self.engine = ApexEngine(enable_logging=False)
        
        self.training_rows: List[Dict[str, Any]] = []
        self.stats = {
            "polygon_symbols": 0,
            "polygon_bars": 0,
            "alpaca_symbols": 0,
            "alpaca_bars": 0,
            "total_samples": 0,
        }
        self._lock = Lock()
    
    def _get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = []
        if self.polygon.is_available():
            sources.append("polygon")
        if self.alpaca.is_available():
            sources.append("alpaca")
        return sources
    
    def _fetch_symbol(
        self,
        symbol: str,
        source: str,
        start: datetime,
        end: datetime,
    ) -> Tuple[str, str, List[OhlcvBar]]:
        """Fetch a single symbol from specified source."""
        if source == "polygon":
            bars = self.polygon.fetch(symbol, start, end)
        elif source == "alpaca":
            bars = self.alpaca.fetch(symbol, start, end)
        else:
            bars = []
        
        return (symbol, source, bars)
    
    def _process_symbol_data(
        self,
        symbol: str,
        bars: List[OhlcvBar],
        source: str,
    ) -> int:
        """Process bars into training samples."""
        if not bars:
            return 0
        
        windows = self.window_gen.generate(
            bars=bars,
            symbol=symbol,
            future_bars=self.config.future_bars,
        )
        
        samples = 0
        for window, future_closes in windows:
            entry_price = window.bars[-1].close
            labels = self.label_gen.generate(entry_price, future_closes)
            
            try:
                context = ApexContext(seed=42, compliance_mode=True)
                apex_result = self.engine.run(window, context)
                features = self.feature_extractor.extract(window)
                
                row = {
                    "symbol": symbol,
                    "source": source,
                    "timestamp": window.bars[-1].timestamp.isoformat(),
                    "entry_price": entry_price,
                    "features": features.tolist() if hasattr(features, 'tolist') else list(features),
                    "quantra_score": labels["quantra_score"],
                    "hit_runner_threshold": labels["hit_runner_threshold"],
                    "future_quality_tier": labels["future_quality_tier"],
                    "avoid_trade": labels["avoid_trade"],
                    "regime_label": labels["regime_label"],
                    "ret_1d": labels["ret_1d"],
                    "ret_3d": labels["ret_3d"],
                    "ret_5d": labels["ret_5d"],
                    "max_runup_5d": labels["max_runup_5d"],
                    "max_drawdown_5d": labels["max_drawdown_5d"],
                    "engine_score": apex_result.quantrascore,
                    "engine_regime": apex_result.regime.value,
                    "vix_level": 20.0,
                    "vix_percentile": 50.0,
                    "sector_momentum": 0.0,
                    "market_breadth": 0.5,
                }
                
                with self._lock:
                    self.training_rows.append(row)
                    samples += 1
                    
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        return samples
    
    def fetch_and_process_all(self) -> Dict[str, int]:
        """Fetch data from all sources and generate training samples."""
        sources = self._get_available_sources()
        
        if not sources:
            raise RuntimeError("No data sources available - check API credentials")
        
        logger.info(f"Available sources: {sources}")
        
        end_date = datetime.now() - timedelta(minutes=20)
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        symbols = self.config.symbols
        
        if len(sources) == 2:
            mid = len(symbols) // 2
            assignments = [
                (s, "polygon") for s in symbols[:mid]
            ] + [
                (s, "alpaca") for s in symbols[mid:]
            ]
        elif "polygon" in sources:
            assignments = [(s, "polygon") for s in symbols]
        else:
            assignments = [(s, "alpaca") for s in symbols]
        
        logger.info(f"Fetching {len(symbols)} symbols using {len(sources)} source(s)...")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_symbol, sym, src, start_date, end_date
                ): (sym, src)
                for sym, src in assignments
            }
            
            for i, future in enumerate(as_completed(futures)):
                sym, src = futures[future]
                try:
                    symbol, source, bars = future.result()
                    
                    if bars:
                        samples = self._process_symbol_data(symbol, bars, source)
                        
                        with self._lock:
                            if source == "polygon":
                                self.stats["polygon_symbols"] += 1
                                self.stats["polygon_bars"] += len(bars)
                            else:
                                self.stats["alpaca_symbols"] += 1
                                self.stats["alpaca_bars"] += len(bars)
                            self.stats["total_samples"] += samples
                        
                        logger.info(f"[{i+1}/{len(assignments)}] {symbol} ({source}): {len(bars)} bars → {samples} samples")
                    else:
                        logger.warning(f"[{i+1}/{len(assignments)}] {symbol} ({source}): No data")
                        
                except Exception as e:
                    logger.error(f"Error processing {sym}: {e}")
        
        return self.stats
    
    def train_model(self, model_size: str = "big") -> Dict[str, Any]:
        """Train ApexCore V3 on collected samples."""
        if not self.training_rows:
            raise ValueError("No training samples - run fetch_and_process_all first")
        
        logger.info(f"Training on {len(self.training_rows)} samples...")
        
        model = ApexCoreV3Model(model_size=model_size)
        metrics = model.fit(self.training_rows)
        
        model_dir = Path(self.config.model_output_dir) / model_size
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(model_dir))
        
        manifest = {
            "version": "3.0.0",
            "model_size": model_size,
            "trained_at": datetime.utcnow().isoformat(),
            "training_samples": len(self.training_rows),
            "data_sources": {
                "polygon": {
                    "symbols": self.stats["polygon_symbols"],
                    "bars": self.stats["polygon_bars"],
                },
                "alpaca": {
                    "symbols": self.stats["alpaca_symbols"],
                    "bars": self.stats["alpaca_bars"],
                },
            },
            "lookback_days": self.config.lookback_days,
            "metrics": metrics,
            "note": "Trained on real market data with actual outcome labels",
        }
        
        import json
        with open(model_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
        
        return manifest
    
    def run_full_pipeline(self, model_size: str = "big") -> Dict[str, Any]:
        """Run complete multi-source training pipeline."""
        logger.info("=" * 60)
        logger.info("UNIFIED MULTI-SOURCE TRAINING PIPELINE")
        logger.info("=" * 60)
        
        sources = self._get_available_sources()
        logger.info(f"Data sources: {sources}")
        logger.info(f"Symbols: {len(self.config.symbols)}")
        logger.info(f"Lookback: {self.config.lookback_days} days")
        
        logger.info("\n[1/2] Fetching and processing data...")
        stats = self.fetch_and_process_all()
        
        logger.info(f"\nData collected:")
        logger.info(f"  Polygon: {stats['polygon_symbols']} symbols, {stats['polygon_bars']} bars")
        logger.info(f"  Alpaca: {stats['alpaca_symbols']} symbols, {stats['alpaca_bars']} bars")
        logger.info(f"  Total samples: {stats['total_samples']}")
        
        logger.info("\n[2/2] Training model...")
        manifest = self.train_model(model_size)
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Total samples: {manifest['training_samples']}")
        logger.info(f"  Model saved to: {self.config.model_output_dir}/{model_size}")
        logger.info("=" * 60)
        
        return manifest


def run_unified_training(
    symbols: Optional[List[str]] = None,
    lookback_days: int = 365,
    model_size: str = "big",
) -> Dict[str, Any]:
    """
    Run multi-source training using all available data providers.
    
    Args:
        symbols: List of symbols (default: 30 major stocks)
        lookback_days: Historical data period (default: 365)
        model_size: Model variant ("big" or "mini")
    
    Returns:
        Training manifest with metrics
    """
    config = UnifiedTrainingConfig(lookback_days=lookback_days)
    if symbols:
        config.symbols = symbols
    
    trainer = UnifiedTrainer(config)
    return trainer.run_full_pipeline(model_size)
