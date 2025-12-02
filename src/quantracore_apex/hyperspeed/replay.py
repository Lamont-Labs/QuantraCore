"""
Historical Replay Engine.

Replays years of market data at hyperspeed through the prediction pipeline.
Generates training samples at 1000x real-time speed.
"""

import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from threading import Lock
import numpy as np

from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from src.quantracore_apex.apexlab.features import FeatureExtractor
from .models import ReplaySession, ReplaySpeed, HyperspeedConfig
from .adapters import FallbackDataProvider, AdapterConfig

logger = logging.getLogger(__name__)


class HistoricalReplayEngine:
    """
    Replays historical market data at hyperspeed.
    
    Fetches years of data and streams it through the prediction
    pipeline as fast as possible, generating training samples
    with known outcomes.
    """
    
    POLYGON_BASE_URL = "https://api.polygon.io"
    
    def __init__(self, config: Optional[HyperspeedConfig] = None):
        self.config = config or HyperspeedConfig()
        self.polygon_key = os.environ.get("POLYGON_API_KEY")
        self.alpaca_key = os.environ.get("ALPACA_PAPER_API_KEY")
        self.alpaca_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
        
        self._rate_limiter = Lock()
        self._last_request = 0.0
        self._rate_delay = 0.25
        
        self._sessions: List[ReplaySession] = []
        self._current_session: Optional[ReplaySession] = None
        
        self._feature_extractor = FeatureExtractor()
        
        self._bar_cache: Dict[str, List[OhlcvBar]] = {}
        
        self._fallback_provider = FallbackDataProvider(AdapterConfig())
        self._use_fallback = True
        
        logger.info("[HistoricalReplay] Initialized")
        logger.info(f"[HistoricalReplay] Fallback provider enabled: {self._use_fallback}")
    
    def _rate_limit(self):
        """Enforce rate limiting for API calls."""
        with self._rate_limiter:
            elapsed = time.time() - self._last_request
            if elapsed < self._rate_delay:
                time.sleep(self._rate_delay - elapsed)
            self._last_request = time.time()
    
    def fetch_historical_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> List[OhlcvBar]:
        """Fetch historical OHLCV bars for a symbol with fallback support."""
        cache_key = f"{symbol}:{start_date}:{end_date}"
        
        if use_cache and cache_key in self._bar_cache:
            return self._bar_cache[cache_key]
        
        bars = []
        
        if self.polygon_key:
            bars = self._fetch_from_polygon(symbol, start_date, end_date)
        elif self.alpaca_key and self.alpaca_secret:
            bars = self._fetch_from_alpaca(symbol, start_date, end_date)
        
        if not bars and self._use_fallback:
            logger.info(f"[HistoricalReplay] Using fallback for {symbol}")
            bars = self._fallback_provider.get_historical_bars(symbol, start_date, end_date)
        
        if not bars:
            logger.warning(f"[HistoricalReplay] No data available for {symbol}")
            return []
        
        if use_cache and bars:
            self._bar_cache[cache_key] = bars
        
        return bars
    
    def _fetch_from_polygon(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[OhlcvBar]:
        """Fetch bars from Polygon.io."""
        url = f"{self.POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.isoformat()}/{end_date.isoformat()}"
        
        params = {
            "apiKey": self.polygon_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        try:
            self._rate_limit()
            resp = requests.get(url, params=params, timeout=30)
            
            if resp.status_code == 429:
                time.sleep(15)
                return self._fetch_from_polygon(symbol, start_date, end_date)
            
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("results", [])
            bars = []
            
            for r in results:
                bar = OhlcvBar(
                    timestamp=datetime.fromtimestamp(r["t"] / 1000),
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=int(r.get("v", 0)),
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            logger.error(f"[HistoricalReplay] Polygon error for {symbol}: {e}")
            return []
    
    def _fetch_from_alpaca(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[OhlcvBar]:
        """Fetch bars from Alpaca."""
        url = "https://data.alpaca.markets/v2/stocks/bars"
        
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }
        
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "limit": 10000,
            "adjustment": "split",
        }
        
        try:
            self._rate_limit()
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            symbol_bars = data.get("bars", {}).get(symbol, [])
            bars = []
            
            for r in symbol_bars:
                bar = OhlcvBar(
                    timestamp=datetime.fromisoformat(r["t"].replace("Z", "+00:00")),
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=int(r.get("v", 0)),
                )
                bars.append(bar)
            
            return bars
            
        except Exception as e:
            logger.error(f"[HistoricalReplay] Alpaca error for {symbol}: {e}")
            return []
    
    def generate_windows(
        self,
        bars: List[OhlcvBar],
        symbol: str,
        window_size: int = 100,
        step_size: int = 2,
        future_bars: int = 10,
    ) -> Generator[Tuple[OhlcvWindow, Dict[str, float]], None, None]:
        """
        Generate sliding windows with future outcomes.
        
        Yields (window, labels) tuples for training.
        """
        if len(bars) < window_size + future_bars:
            return
        
        for i in range(0, len(bars) - window_size - future_bars, step_size):
            window_bars = bars[i:i + window_size]
            future_closes = [b.close for b in bars[i + window_size:i + window_size + future_bars]]
            
            current_close = window_bars[-1].close
            max_future = max(future_closes)
            min_future = min(future_closes)
            final_future = future_closes[-1]
            
            max_runup = (max_future - current_close) / current_close
            max_drawdown = (current_close - min_future) / current_close
            final_return = (final_future - current_close) / current_close
            
            is_runner = max_runup >= 0.05
            direction = 1 if final_return > 0 else 0
            
            quality_tier = 0
            if max_runup >= 0.10:
                quality_tier = 4
            elif max_runup >= 0.07:
                quality_tier = 3
            elif max_runup >= 0.05:
                quality_tier = 2
            elif max_runup >= 0.03:
                quality_tier = 1
            
            window = OhlcvWindow(
                symbol=symbol,
                timeframe="1D",
                bars=window_bars,
            )
            
            labels = {
                "quantrascore": min(100, max(0, max_runup * 500 + 30)),
                "runner": float(is_runner),
                "quality": float(quality_tier),
                "avoid": float(max_drawdown > 0.10),
                "regime": float(self._detect_regime(window_bars)),
                "timing": float(self._detect_timing(future_closes)),
                "runup": max_runup * 100,
                "direction": float(direction),
                "volatility": float(np.std([b.close for b in window_bars[-20:]]) / current_close * 100),
                "momentum": float((current_close - window_bars[-20].close) / window_bars[-20].close * 100),
                "support": float(self._detect_support_proximity(window_bars)),
                "resistance": float(self._detect_resistance_proximity(window_bars)),
                "volume": float(window_bars[-1].volume / np.mean([b.volume for b in window_bars[-20:]]) if np.mean([b.volume for b in window_bars[-20:]]) > 0 else 1.0),
                "reversal": float(self._detect_reversal(window_bars, future_closes)),
                "breakout": float(self._detect_breakout(window_bars, future_closes)),
                "continuation": float(self._detect_continuation(window_bars, future_closes)),
            }
            
            yield window, labels
    
    def _detect_regime(self, bars: List[OhlcvBar]) -> int:
        """Detect market regime from price action."""
        if len(bars) < 50:
            return 2
        
        closes = [b.close for b in bars]
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-50:])
        current = closes[-1]
        
        if current > sma20 > sma50:
            return 0
        elif current > sma20 and sma20 < sma50:
            return 1
        elif current < sma20 and sma20 > sma50:
            return 3
        elif current < sma20 < sma50:
            return 4
        else:
            return 2
    
    def _detect_timing(self, future_closes: List[float]) -> int:
        """Detect when the move happens."""
        if not future_closes:
            return 2
        
        base = future_closes[0]
        max_idx = 0
        max_val = future_closes[0]
        
        for i, c in enumerate(future_closes):
            if c > max_val:
                max_val = c
                max_idx = i
        
        if max_idx <= 2:
            return 0
        elif max_idx <= 4:
            return 1
        elif max_idx <= 6:
            return 2
        elif max_idx <= 8:
            return 3
        else:
            return 4
    
    def _detect_support_proximity(self, bars: List[OhlcvBar]) -> float:
        """Detect proximity to support level."""
        if len(bars) < 20:
            return 0.5
        
        lows = [b.low for b in bars[-20:]]
        current = bars[-1].close
        support = min(lows)
        
        if support <= 0:
            return 0.5
        
        proximity = (current - support) / support
        return min(1.0, max(0.0, 1.0 - proximity * 10))
    
    def _detect_resistance_proximity(self, bars: List[OhlcvBar]) -> float:
        """Detect proximity to resistance level."""
        if len(bars) < 20:
            return 0.5
        
        highs = [b.high for b in bars[-20:]]
        current = bars[-1].close
        resistance = max(highs)
        
        if resistance <= 0:
            return 0.5
        
        proximity = (resistance - current) / resistance
        return min(1.0, max(0.0, 1.0 - proximity * 10))
    
    def _detect_reversal(self, bars: List[OhlcvBar], future_closes: List[float]) -> float:
        """Detect reversal pattern."""
        if len(bars) < 10 or len(future_closes) < 5:
            return 0.0
        
        recent_trend = (bars[-1].close - bars[-10].close) / bars[-10].close
        future_trend = (future_closes[-1] - future_closes[0]) / future_closes[0] if future_closes[0] > 0 else 0
        
        if recent_trend > 0.02 and future_trend < -0.02:
            return 1.0
        elif recent_trend < -0.02 and future_trend > 0.02:
            return 1.0
        
        return 0.0
    
    def _detect_breakout(self, bars: List[OhlcvBar], future_closes: List[float]) -> float:
        """Detect breakout pattern."""
        if len(bars) < 20 or len(future_closes) < 5:
            return 0.0
        
        highs = [b.high for b in bars[-20:-1]]
        if not highs:
            return 0.0
        
        resistance = max(highs)
        current = bars[-1].close
        future_high = max(future_closes)
        
        if current > resistance * 0.99 and future_high > resistance * 1.03:
            return 1.0
        
        return 0.0
    
    def _detect_continuation(self, bars: List[OhlcvBar], future_closes: List[float]) -> float:
        """Detect trend continuation probability."""
        if len(bars) < 20 or len(future_closes) < 5:
            return 0.5
        
        recent_trend = (bars[-1].close - bars[-20].close) / bars[-20].close
        future_trend = (future_closes[-1] - future_closes[0]) / future_closes[0] if future_closes[0] > 0 else 0
        
        if recent_trend > 0 and future_trend > 0:
            return min(1.0, 0.5 + abs(future_trend) * 5)
        elif recent_trend < 0 and future_trend < 0:
            return min(1.0, 0.5 + abs(future_trend) * 5)
        else:
            return max(0.0, 0.5 - abs(future_trend) * 5)
    
    def start_replay_session(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> ReplaySession:
        """Start a new historical replay session."""
        session = ReplaySession(
            replay_start_date=start_date or (date.today() - timedelta(days=365 * self.config.replay_years)),
            replay_end_date=end_date or date.today(),
        )
        
        self._current_session = session
        self._sessions.append(session)
        
        symbols = symbols or self.config.replay_symbols
        
        logger.info(f"[HistoricalReplay] Starting session {session.session_id}")
        logger.info(f"[HistoricalReplay] Replaying {len(symbols)} symbols from {session.replay_start_date} to {session.replay_end_date}")
        
        return session
    
    def run_replay(
        self,
        session: Optional[ReplaySession] = None,
        symbols: Optional[List[str]] = None,
        max_workers: int = 4,
        callback: Optional[callable] = None,
    ) -> Generator[Tuple[OhlcvWindow, Dict[str, float]], None, None]:
        """
        Run historical replay and yield training samples.
        
        Args:
            session: Replay session to use
            symbols: Symbols to replay
            max_workers: Parallel workers for fetching
            callback: Optional callback for progress updates
        
        Yields:
            (window, labels) tuples
        """
        session = session or self._current_session or self.start_replay_session(symbols)
        symbols = symbols or self.config.replay_symbols
        
        start_time = time.time()
        
        def process_symbol(symbol: str) -> List[Tuple[OhlcvWindow, Dict[str, float]]]:
            bars = self.fetch_historical_bars(
                symbol,
                session.replay_start_date,
                session.replay_end_date,
            )
            
            if not bars:
                return []
            
            samples = list(self.generate_windows(bars, symbol))
            return samples
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_symbol, s): s for s in symbols}
            
            for future in as_completed(futures):
                symbol = futures[future]
                
                try:
                    samples = future.result()
                    session.symbols_processed += 1
                    
                    for window, labels in samples:
                        session.windows_generated += 1
                        session.samples_created += 1
                        session.bars_replayed += len(window.bars)
                        
                        yield window, labels
                    
                    if callback:
                        callback(session)
                    
                except Exception as e:
                    session.errors.append(f"{symbol}: {str(e)}")
                    logger.error(f"[HistoricalReplay] Error processing {symbol}: {e}")
        
        elapsed = time.time() - start_time
        session.completed_at = datetime.utcnow()
        
        days_replayed = (session.replay_end_date - session.replay_start_date).days * len(symbols)
        session.effective_days_per_minute = (days_replayed / elapsed * 60) if elapsed > 0 else 0
        session.speed_multiplier = session.effective_days_per_minute / (1 / 1440)
        
        logger.info(f"[HistoricalReplay] Session {session.session_id} complete")
        logger.info(f"[HistoricalReplay] Processed {session.symbols_processed} symbols, {session.samples_created} samples")
        logger.info(f"[HistoricalReplay] Effective speed: {session.effective_days_per_minute:.1f} days/min")
    
    def get_session(self, session_id: str) -> Optional[ReplaySession]:
        """Get a replay session by ID."""
        for session in self._sessions:
            if session.session_id == session_id:
                return session
        return None
    
    def get_all_sessions(self) -> List[ReplaySession]:
        """Get all replay sessions."""
        return list(self._sessions)
    
    def get_current_session(self) -> Optional[ReplaySession]:
        """Get the current active session."""
        return self._current_session
    
    def clear_cache(self):
        """Clear the bar cache."""
        self._bar_cache.clear()
        logger.info("[HistoricalReplay] Cache cleared")
