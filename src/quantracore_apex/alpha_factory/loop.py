"""
Alpha Factory Main Loop.

24/7 live research loop that:
1. Receives real-time data from Polygon (equities) and Binance (crypto)
2. Runs ApexEngine scans on each tick
3. Generates alpha signals and portfolio rebalancing
4. Tracks equity curve and performance
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable

import pandas as pd

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from src.quantracore_apex.portfolio.core import (
    Portfolio, EQUITY_UNIVERSE, CRYPTO_UNIVERSE, FULL_UNIVERSE
)

logger = logging.getLogger(__name__)


class AlphaFactoryLoop:
    """
    Main 24/7 alpha factory loop.
    
    Integrates:
    - Polygon WebSocket for US equities
    - Binance WebSocket for crypto
    - ApexEngine for signal generation
    - Portfolio engine for position management
    
    Note: Research mode only - all positions are simulated.
    """
    
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        equity_symbols: Optional[list] = None,
        crypto_symbols: Optional[list] = None,
        min_score: int = 60,
        on_signal: Optional[Callable] = None
    ):
        """
        Initialize alpha factory.
        
        Args:
            initial_cash: Starting portfolio value
            equity_symbols: Equity symbols to track (default: EQUITY_UNIVERSE)
            crypto_symbols: Crypto pairs to track (default: CRYPTO_UNIVERSE)
            min_score: Minimum QuantraScore to log (default: 60)
            on_signal: Optional callback for signals
        """
        self.equity_symbols = equity_symbols or ["AAPL", "NVDA", "TSLA", "SPY"]
        self.crypto_symbols = crypto_symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.min_score = min_score
        self.on_signal = on_signal
        
        all_symbols = self.equity_symbols + self.crypto_symbols
        
        self.engine = ApexEngine()
        self.portfolio = Portfolio(initial_cash=initial_cash, universe=all_symbols)
        
        self._running = False
        self._threads: list = []
        self._tick_count = 0
        self._signal_count = 0
        self._start_time: datetime = datetime.now()
        
        self._price_buffer: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"Alpha Factory initialized: ${initial_cash:,.0f} | "
                   f"Equities: {len(self.equity_symbols)} | Crypto: {len(self.crypto_symbols)}")
    
    def _scan_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run ApexEngine scan on a symbol's data.
        
        Args:
            symbol: Symbol to analyze
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with scan results including quantra_score, regime, alerts
        """
        try:
            bars = []
            for _, row in df.iterrows():
                bar = OhlcvBar(
                    timestamp=row.get('timestamp', datetime.now()),
                    open=float(row.get('open', 0)),
                    high=float(row.get('high', 0)),
                    low=float(row.get('low', 0)),
                    close=float(row.get('close', 0)),
                    volume=float(row.get('volume', 0))
                )
                bars.append(bar)
            
            if len(bars) < 20:
                return {'quantra_score': 0, 'regime': 'UNKNOWN', 'omega_alerts': []}
            
            result = self.engine.run_scan(bars, symbol)
            
            return {
                'quantra_score': result.quantrascore,
                'regime': result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
                'monster_alerts': [],
                'omega_alerts': result.omega_overrides if hasattr(result, 'omega_overrides') else [],
                'risk_tier': result.risk_tier if hasattr(result, 'risk_tier') else 'UNKNOWN',
                'atr_pct': 0.02
            }
        except Exception as e:
            logger.error(f"Scan error for {symbol}: {e}")
            return {'quantra_score': 0, 'regime': 'ERROR', 'omega_alerts': []}
    
    def _on_tick(self, source: str, symbol: str, data: Any):
        """
        Handle incoming tick data.
        
        Args:
            source: Data source ('POLYGON' or 'BINANCE')
            symbol: Symbol
            data: Tick data (dict or DataFrame)
        """
        self._tick_count += 1
        
        try:
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                df = pd.DataFrame([{
                    'timestamp': data.get('timestamp', datetime.now()),
                    'open': data.get('open', data.get('price', 0)),
                    'high': data.get('high', data.get('price', 0)),
                    'low': data.get('low', data.get('price', 0)),
                    'close': data.get('close', data.get('price', 0)),
                    'volume': data.get('volume', 0)
                }])
            
            if symbol not in self._price_buffer:
                self._price_buffer[symbol] = df
            else:
                self._price_buffer[symbol] = pd.concat([
                    self._price_buffer[symbol], df
                ]).tail(100)
            
            if len(self._price_buffer[symbol]) < 5:
                return
            
            result = self._scan_symbol(symbol, self._price_buffer[symbol])
            
            score = result.get('quantra_score', 0)
            
            if score >= self.min_score:
                self._signal_count += 1
                
                regime = result.get('regime', 'UNKNOWN')
                monsters = result.get('monster_alerts', [])
                omega = result.get('omega_alerts', [])
                
                logger.info(
                    f"ALPHA [{source}] {symbol}: Score {score} | "
                    f"Regime: {regime} | Monsters: {len(monsters)}"
                )
                
                if self.on_signal:
                    self.on_signal({
                        'source': source,
                        'symbol': symbol,
                        'score': score,
                        'result': result,
                        'timestamp': datetime.now()
                    })
            
            close_price = df['close'].iloc[-1] if 'close' in df.columns else 0
            signal_data = {
                symbol: {
                    'quantra_score': score,
                    'close': close_price,
                    'atr_pct': result.get('atr_pct', 0.02),
                    'omega_alert': len(result.get('omega_alerts', [])) > 0
                }
            }
            self.portfolio.rebalance(signal_data)
            
            if self._tick_count % 100 == 0:
                self.portfolio.save_equity_curve('equity_curve.csv')
                logger.info(f"Ticks: {self._tick_count} | Signals: {self._signal_count} | "
                           f"NAV: ${self.portfolio.nav:,.2f}")
        
        except Exception as e:
            logger.error(f"Error processing tick {symbol}: {e}")
    
    def _start_polygon_feed(self):
        """Start Polygon WebSocket feed in thread."""
        if not self.equity_symbols:
            return
        
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            logger.warning("POLYGON_API_KEY not set - skipping equity feed")
            return
        
        try:
            from src.quantracore_apex.data_layer.live.polygon_ws import PolygonLiveFeed
            
            feed = PolygonLiveFeed(on_message=self._on_tick)
            feed.run_forever(self.equity_symbols)
        except Exception as e:
            logger.error(f"Polygon feed error: {e}")
    
    def _start_binance_feed(self):
        """Start Binance WebSocket feed in thread."""
        if not self.crypto_symbols:
            return
        
        try:
            from src.quantracore_apex.data_layer.live.binance_ws import BinanceLiveFeed
            
            feed = BinanceLiveFeed(on_message=self._on_tick)
            feed.run_forever(self.crypto_symbols)
        except Exception as e:
            logger.error(f"Binance feed error: {e}")
    
    def start(self):
        """Start the alpha factory loop."""
        if self._running:
            logger.warning("Alpha factory already running")
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info("ALPHA FACTORY STARTING")
        logger.info(f"Equities: {self.equity_symbols}")
        logger.info(f"Crypto: {self.crypto_symbols}")
        logger.info(f"Min Score: {self.min_score}")
        logger.info("=" * 60)
        
        polygon_thread = threading.Thread(
            target=self._start_polygon_feed,
            daemon=True,
            name="PolygonFeed"
        )
        self._threads.append(polygon_thread)
        
        binance_thread = threading.Thread(
            target=self._start_binance_feed,
            daemon=True,
            name="BinanceFeed"
        )
        self._threads.append(binance_thread)
        
        for t in self._threads:
            t.start()
        
        logger.info("All feeds started - Alpha factory is LIVE")
    
    def run_forever(self):
        """Start and run forever (blocking)."""
        self.start()
        
        try:
            while self._running:
                time.sleep(60)
                
                uptime = datetime.now() - self._start_time
                logger.info(
                    f"[Heartbeat] Uptime: {uptime} | "
                    f"Ticks: {self._tick_count} | Signals: {self._signal_count} | "
                    f"NAV: ${self.portfolio.nav:,.2f} ({self.portfolio.total_return:+.2f}%)"
                )
                
                self.portfolio.save_equity_curve('equity_curve.csv')
        
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
            self.stop()
    
    def stop(self):
        """Stop the alpha factory."""
        self._running = False
        
        self.portfolio.save_equity_curve('equity_curve.csv')
        
        logger.info("=" * 60)
        logger.info("ALPHA FACTORY STOPPED")
        logger.info(f"Total ticks: {self._tick_count}")
        logger.info(f"Total signals: {self._signal_count}")
        logger.info(f"Final NAV: ${self.portfolio.nav:,.2f}")
        logger.info(f"Total return: {self.portfolio.total_return:+.2f}%")
        logger.info("=" * 60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        uptime = None
        if self._start_time:
            uptime = str(datetime.now() - self._start_time)
        
        return {
            "running": self._running,
            "uptime": uptime,
            "tick_count": self._tick_count,
            "signal_count": self._signal_count,
            "portfolio": self.portfolio.get_summary(),
            "active_positions": self.portfolio.get_active_positions()
        }
