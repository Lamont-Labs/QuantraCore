"""
Low Float Runner Screener Service.

Real-time screener for low-float penny stock runners.
Fetches live market data and identifies potential runners based on:
- Volume surge (relative volume > 3x)
- Price momentum (gap up, breakout)
- Float characteristics (low float = explosive potential)
- Compression patterns (Monster Runner detection)

Version: 9.0-A
"""

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import os

logger = logging.getLogger(__name__)


@dataclass
class LowFloatAlert:
    """Alert for a low-float runner candidate."""
    symbol: str
    alert_type: str
    trigger_time: str
    current_price: float
    change_percent: float
    volume: int
    relative_volume: float
    float_millions: float
    market_cap_bucket: str
    runner_score: float
    timing_bucket: str
    expected_runup_pct: float
    alert_message: str
    priority: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScreenerConfig:
    """Configuration for the low-float screener."""
    min_relative_volume: float = 3.0
    min_change_percent: float = 5.0
    max_float_millions: float = 50.0
    min_price: float = 0.10
    max_price: float = 20.0
    scan_interval_seconds: int = 60
    max_alerts_per_scan: int = 20
    alert_cooldown_minutes: int = 15
    sectors_focus: List[str] = field(default_factory=lambda: [
        "Healthcare", "Technology", "Consumer Discretionary", "Energy"
    ])


class LowFloatScreener:
    """
    Real-time screener for low-float penny stock runners.
    
    Uses Alpaca's market data API to fetch real-time quotes and
    identify potential runners based on volume surge, momentum, and
    float characteristics.
    """
    
    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or ScreenerConfig()
        self._alerts: List[LowFloatAlert] = []
        self._alert_history: Dict[str, datetime] = {}
        self._is_running = False
        self._last_scan: Optional[datetime] = None
        self._scan_count = 0
        self._symbols_scanned = 0
        self._runners_detected = 0
        
    async def scan_universe(
        self, 
        symbols: Optional[List[str]] = None,
        include_prediction: bool = True
    ) -> List[LowFloatAlert]:
        """
        Scan the low-float universe for runner candidates.
        
        Args:
            symbols: Optional list of symbols to scan (defaults to universe)
            include_prediction: Whether to include ML predictions
            
        Returns:
            List of LowFloatAlert objects for detected runners
        """
        from src.quantracore_apex.config.symbol_universe import get_symbols_by_bucket
        
        if symbols is None:
            symbols = get_symbols_by_bucket(["penny", "nano", "micro"])
        
        self._last_scan = datetime.utcnow()
        self._scan_count += 1
        self._symbols_scanned = len(symbols)
        
        alerts = []
        
        try:
            quotes = await self._fetch_realtime_quotes(symbols)
            
            for symbol, quote in quotes.items():
                if self._should_skip_symbol(symbol):
                    continue
                
                alert = await self._analyze_for_runner(symbol, quote, include_prediction)
                if alert:
                    alerts.append(alert)
                    self._alert_history[symbol] = datetime.utcnow()
                    self._runners_detected += 1
            
            alerts.sort(key=lambda x: x.priority, reverse=True)
            alerts = alerts[:self.config.max_alerts_per_scan]
            
            self._alerts = alerts
            
            logger.info(f"[LowFloatScreener] Scan complete: {len(alerts)} runners from {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"[LowFloatScreener] Scan error: {e}")
        
        return alerts
    
    async def _fetch_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch real-time quotes from Alpaca."""
        quotes = {}
        
        try:
            api_key = os.environ.get("ALPACA_PAPER_API_KEY")
            api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
            
            if not api_key or not api_secret:
                logger.warning("[LowFloatScreener] Alpaca credentials not configured")
                return self._generate_synthetic_quotes(symbols)
            
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            client = StockHistoricalDataClient(api_key, api_secret)
            
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                
                try:
                    request = StockLatestQuoteRequest(symbol_or_symbols=batch)
                    latest_quotes = client.get_stock_latest_quote(request)
                    
                    bars_request = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame.Day,
                        start=datetime.now() - timedelta(days=5),
                    )
                    bars_data = client.get_stock_bars(bars_request)
                    
                    for sym in batch:
                        if sym in latest_quotes:
                            quote = latest_quotes[sym]
                            
                            avg_volume = 0
                            prev_close = quote.ask_price * 0.95
                            
                            if sym in bars_data and len(bars_data[sym]) > 0:
                                recent_bars = bars_data[sym][-5:]
                                if recent_bars:
                                    avg_volume = sum(b.volume for b in recent_bars) / len(recent_bars)
                                    if len(recent_bars) > 1:
                                        prev_close = recent_bars[-2].close
                            
                            current_price = (quote.ask_price + quote.bid_price) / 2 if quote.bid_price else quote.ask_price
                            
                            quotes[sym] = {
                                "symbol": sym,
                                "price": current_price,
                                "prev_close": prev_close,
                                "change_pct": ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0,
                                "volume": quote.ask_size + quote.bid_size,
                                "avg_volume": avg_volume,
                                "relative_volume": (quote.ask_size + quote.bid_size) / avg_volume if avg_volume > 0 else 1.0,
                                "timestamp": quote.timestamp.isoformat() if hasattr(quote, 'timestamp') else datetime.utcnow().isoformat(),
                            }
                            
                except Exception as batch_error:
                    logger.warning(f"[LowFloatScreener] Batch fetch error: {batch_error}")
                    continue
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"[LowFloatScreener] Quote fetch error: {e}")
            return self._generate_synthetic_quotes(symbols)
        
        return quotes
    
    def _generate_synthetic_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Generate synthetic quotes for testing when API unavailable."""
        import random
        
        quotes = {}
        for sym in symbols[:50]:
            base_price = random.uniform(0.5, 15.0)
            change = random.uniform(-10, 25)
            volume = random.randint(50000, 5000000)
            avg_vol = random.randint(100000, 1000000)
            
            quotes[sym] = {
                "symbol": sym,
                "price": round(base_price, 2),
                "prev_close": round(base_price / (1 + change/100), 2),
                "change_pct": round(change, 2),
                "volume": volume,
                "avg_volume": avg_vol,
                "relative_volume": round(volume / avg_vol, 2) if avg_vol > 0 else 1.0,
                "timestamp": datetime.utcnow().isoformat(),
                "synthetic": True,
            }
        
        return quotes
    
    def _should_skip_symbol(self, symbol: str) -> bool:
        """Check if symbol should be skipped due to cooldown."""
        if symbol in self._alert_history:
            last_alert = self._alert_history[symbol]
            cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
            if datetime.utcnow() - last_alert < cooldown:
                return True
        return False
    
    async def _analyze_for_runner(
        self, 
        symbol: str, 
        quote: Dict, 
        include_prediction: bool
    ) -> Optional[LowFloatAlert]:
        """Analyze a symbol for runner potential."""
        from src.quantracore_apex.config.symbol_universe import get_symbol_info
        
        price = quote.get("price", 0)
        change_pct = quote.get("change_pct", 0)
        relative_volume = quote.get("relative_volume", 1.0)
        volume = quote.get("volume", 0)
        
        if price < self.config.min_price or price > self.config.max_price:
            return None
        
        if relative_volume < self.config.min_relative_volume:
            return None
        
        if abs(change_pct) < self.config.min_change_percent:
            return None
        
        try:
            symbol_info = get_symbol_info(symbol)
            if symbol_info:
                float_millions = symbol_info.float_millions if symbol_info.float_millions > 0 else 100
                market_cap_bucket = symbol_info.market_cap_bucket or "unknown"
            else:
                float_millions = 100
                market_cap_bucket = "unknown"
        except Exception as e:
            logger.debug(f"[LowFloatScreener] Could not get symbol info for {symbol}: {e}")
            float_millions = 100
            market_cap_bucket = "unknown"
        
        if float_millions > self.config.max_float_millions:
            return None
        
        runner_score = 0.0
        timing_bucket = "unknown"
        expected_runup_pct = 0.0
        
        if include_prediction:
            try:
                from src.quantracore_apex.signals.signal_service import ApexSignalService
                signal_service = ApexSignalService()
                signal = signal_service.generate_signal(symbol)
                if signal:
                    runner_score = signal.runner_probability
                    timing_bucket = signal.timing_bucket
                    expected_runup_pct = signal.expected_runup_pct
            except Exception as e:
                logger.debug(f"[LowFloatScreener] Prediction unavailable for {symbol}: {e}")
        
        priority = self._calculate_priority(
            change_pct, relative_volume, float_millions, runner_score
        )
        
        alert_type = self._determine_alert_type(change_pct, relative_volume)
        alert_message = self._generate_alert_message(
            symbol, alert_type, price, change_pct, relative_volume, float_millions
        )
        
        return LowFloatAlert(
            symbol=symbol,
            alert_type=alert_type,
            trigger_time=datetime.utcnow().isoformat(),
            current_price=price,
            change_percent=change_pct,
            volume=volume,
            relative_volume=relative_volume,
            float_millions=float_millions,
            market_cap_bucket=market_cap_bucket,
            runner_score=runner_score,
            timing_bucket=timing_bucket,
            expected_runup_pct=expected_runup_pct,
            alert_message=alert_message,
            priority=priority,
        )
    
    def _calculate_priority(
        self, 
        change_pct: float, 
        relative_volume: float, 
        float_millions: float,
        runner_score: float
    ) -> int:
        """Calculate alert priority (higher = more urgent)."""
        priority = 0
        
        if change_pct > 20:
            priority += 50
        elif change_pct > 10:
            priority += 30
        elif change_pct > 5:
            priority += 15
        
        if relative_volume > 10:
            priority += 40
        elif relative_volume > 5:
            priority += 25
        elif relative_volume > 3:
            priority += 10
        
        if float_millions < 5:
            priority += 30
        elif float_millions < 15:
            priority += 20
        elif float_millions < 30:
            priority += 10
        
        priority += int(runner_score * 30)
        
        return priority
    
    def _determine_alert_type(self, change_pct: float, relative_volume: float) -> str:
        """Determine the type of alert based on conditions."""
        if change_pct > 20 and relative_volume > 5:
            return "BREAKOUT_RUNNER"
        elif change_pct > 10 and relative_volume > 3:
            return "MOMENTUM_SURGE"
        elif relative_volume > 8:
            return "VOLUME_EXPLOSION"
        elif change_pct > 15:
            return "GAP_UP"
        elif change_pct < -15:
            return "GAP_DOWN_REVERSAL"
        else:
            return "EARLY_MOVER"
    
    def _generate_alert_message(
        self,
        symbol: str,
        alert_type: str,
        price: float,
        change_pct: float,
        relative_volume: float,
        float_millions: float
    ) -> str:
        """Generate human-readable alert message."""
        direction = "UP" if change_pct > 0 else "DOWN"
        
        messages = {
            "BREAKOUT_RUNNER": f"RUNNER ALERT: {symbol} breaking out! {direction} {abs(change_pct):.1f}% on {relative_volume:.1f}x volume. Float: {float_millions:.0f}M shares. Price: ${price:.2f}",
            "MOMENTUM_SURGE": f"MOMENTUM: {symbol} surging {direction} {abs(change_pct):.1f}% with {relative_volume:.1f}x normal volume. Low float ({float_millions:.0f}M). Watch for continuation.",
            "VOLUME_EXPLOSION": f"VOLUME ALERT: {symbol} seeing {relative_volume:.1f}x normal volume. Price {direction} {abs(change_pct):.1f}%. Float: {float_millions:.0f}M. Institutions may be loading.",
            "GAP_UP": f"GAP ALERT: {symbol} gapped {direction} {abs(change_pct):.1f}%. Float: {float_millions:.0f}M. Volume: {relative_volume:.1f}x avg.",
            "GAP_DOWN_REVERSAL": f"REVERSAL WATCH: {symbol} gap down {abs(change_pct):.1f}% may reverse. Float: {float_millions:.0f}M. Watch for dip buy setup.",
            "EARLY_MOVER": f"EARLY MOVER: {symbol} showing early momentum. {direction} {abs(change_pct):.1f}% on {relative_volume:.1f}x volume. Float: {float_millions:.0f}M.",
        }
        
        return messages.get(alert_type, f"ALERT: {symbol} {direction} {abs(change_pct):.1f}%")
    
    def get_status(self) -> Dict[str, Any]:
        """Get screener status."""
        return {
            "is_running": self._is_running,
            "last_scan": self._last_scan.isoformat() if self._last_scan else None,
            "scan_count": self._scan_count,
            "symbols_scanned": self._symbols_scanned,
            "runners_detected": self._runners_detected,
            "active_alerts": len(self._alerts),
            "cooldown_symbols": len(self._alert_history),
            "config": {
                "min_relative_volume": self.config.min_relative_volume,
                "min_change_percent": self.config.min_change_percent,
                "max_float_millions": self.config.max_float_millions,
                "scan_interval_seconds": self.config.scan_interval_seconds,
            }
        }
    
    def get_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get current alerts."""
        return [alert.to_dict() for alert in self._alerts[:limit]]
    
    def clear_alerts(self):
        """Clear all alerts and reset cooldowns."""
        self._alerts = []
        self._alert_history = {}


_screener_instance: Optional[LowFloatScreener] = None


def get_screener() -> LowFloatScreener:
    """Get or create the global screener instance."""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = LowFloatScreener()
    return _screener_instance
