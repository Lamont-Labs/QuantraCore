"""
Scheduled Autonomous Trading System.

Runs the unified autotrader on a schedule (3x daily during market hours).
Uses Alpaca market data exclusively - no Alpha Vantage API calls.

PAPER TRADING ONLY - No real money at risk.
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import pytz

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of scheduled scans."""
    MORNING = "morning"
    MIDDAY = "midday" 
    CLOSE = "close"
    MANUAL = "manual"


@dataclass
class ScanResult:
    """Result of a scheduled scan."""
    scan_id: str
    scan_type: ScanType
    timestamp: datetime
    symbols_scanned: int
    candidates_found: int
    trades_executed: int
    trades_skipped: int
    errors: List[str] = field(default_factory=list)
    trade_details: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "scan_type": self.scan_type.value,
            "timestamp": self.timestamp.isoformat(),
            "symbols_scanned": self.symbols_scanned,
            "candidates_found": self.candidates_found,
            "trades_executed": self.trades_executed,
            "trades_skipped": self.trades_skipped,
            "errors": self.errors,
            "trade_details": self.trade_details,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ScheduleConfig:
    """Configuration for scheduled scans."""
    morning_enabled: bool = True
    morning_time: str = "09:35"
    midday_enabled: bool = True
    midday_time: str = "12:00"
    close_enabled: bool = True
    close_time: str = "15:30"
    timezone: str = "America/New_York"
    max_trades_per_scan: int = 2
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.50
    require_high_conviction: bool = False
    quick_scan: bool = True
    dry_run: bool = False


class ScheduledAutomation:
    """
    Scheduled Autonomous Trading System.
    
    Runs unified autotrader scans at configured times during market hours.
    All trading is PAPER ONLY via Alpaca.
    
    Schedule (Eastern Time):
    - Morning scan: 9:35 AM (5 minutes after open)
    - Midday scan: 12:00 PM
    - Close scan: 3:30 PM (30 minutes before close)
    """
    
    SCAN_LOG_DIR = Path("logs/scheduled_scans/")
    
    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._scan_history: List[ScanResult] = []
        self._last_scans: Dict[ScanType, Optional[datetime]] = {
            ScanType.MORNING: None,
            ScanType.MIDDAY: None,
            ScanType.CLOSE: None,
        }
        self._tz = pytz.timezone(self.config.timezone)
        
        self.SCAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[ScheduledAutomation] Initialized with config: %s", self.config)
    
    def _get_current_time(self) -> datetime:
        """Get current time in configured timezone."""
        return datetime.now(self._tz)
    
    def _parse_time(self, time_str: str) -> tuple:
        """Parse HH:MM to (hour, minute)."""
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])
    
    def _should_run_scan(self, scan_type: ScanType) -> bool:
        """Check if a scan should run now."""
        now = self._get_current_time()
        
        if scan_type == ScanType.MORNING:
            if not self.config.morning_enabled:
                return False
            hour, minute = self._parse_time(self.config.morning_time)
        elif scan_type == ScanType.MIDDAY:
            if not self.config.midday_enabled:
                return False
            hour, minute = self._parse_time(self.config.midday_time)
        elif scan_type == ScanType.CLOSE:
            if not self.config.close_enabled:
                return False
            hour, minute = self._parse_time(self.config.close_time)
        else:
            return False
        
        scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        window_start = scheduled_time - timedelta(minutes=2)
        window_end = scheduled_time + timedelta(minutes=5)
        
        if not (window_start <= now <= window_end):
            return False
        
        last_run = self._last_scans.get(scan_type)
        if last_run is not None:
            if (now - last_run).total_seconds() < 3600:
                return False
        
        if now.weekday() >= 5:
            return False
        
        return True
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = self._get_current_time()
        
        if now.weekday() >= 5:
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        return market_open <= now <= market_close
    
    def run_scan(self, scan_type: ScanType = ScanType.MANUAL) -> ScanResult:
        """
        Execute a trading scan using the unified autotrader.
        
        Uses Alpaca market data exclusively.
        """
        start_time = time.time()
        scan_id = f"{scan_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"[ScheduledAutomation] Starting {scan_type.value} scan (id: {scan_id})")
        
        try:
            from src.quantracore_apex.trading.unified_auto_trader import UnifiedAutoTrader, QUICK_SCAN_UNIVERSE
            from src.quantracore_apex.server.ml_scanner import MOONSHOT_UNIVERSE
            
            trader = UnifiedAutoTrader(
                max_new_positions=self.config.max_trades_per_scan,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct,
                require_high_conviction=self.config.require_high_conviction,
            )
            
            universe = QUICK_SCAN_UNIVERSE if self.config.quick_scan else MOONSHOT_UNIVERSE
            
            result = trader.scan_analyze_trade(
                symbols=universe,
                max_trades=self.config.max_trades_per_scan,
                include_eod=True,
                include_intraday=True,
                dry_run=self.config.dry_run,
            )
            
            duration = time.time() - start_time
            
            scan_result = ScanResult(
                scan_id=scan_id,
                scan_type=scan_type,
                timestamp=datetime.utcnow(),
                symbols_scanned=result.get("symbols_scanned", len(universe)),
                candidates_found=result.get("candidates_found", 0),
                trades_executed=result.get("trades_executed", 0),
                trades_skipped=result.get("trades_skipped", 0),
                errors=result.get("errors", []),
                trade_details=result.get("trade_details", []),
                duration_seconds=duration,
            )
            
            with self._lock:
                self._scan_history.append(scan_result)
                if scan_type != ScanType.MANUAL:
                    self._last_scans[scan_type] = self._get_current_time()
                if len(self._scan_history) > 100:
                    self._scan_history = self._scan_history[-100:]
            
            self._log_scan_result(scan_result)
            
            logger.info(
                f"[ScheduledAutomation] {scan_type.value} scan complete: "
                f"{scan_result.candidates_found} candidates, "
                f"{scan_result.trades_executed} trades in {duration:.1f}s"
            )
            
            return scan_result
            
        except Exception as e:
            logger.error(f"[ScheduledAutomation] Scan error: {e}")
            import traceback
            return ScanResult(
                scan_id=scan_id,
                scan_type=scan_type,
                timestamp=datetime.utcnow(),
                symbols_scanned=0,
                candidates_found=0,
                trades_executed=0,
                trades_skipped=0,
                errors=[str(e), traceback.format_exc()],
                duration_seconds=time.time() - start_time,
            )
    
    def _log_scan_result(self, result: ScanResult):
        """Log scan result to file."""
        try:
            log_file = self.SCAN_LOG_DIR / f"{result.scan_id}.json"
            with open(log_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[ScheduledAutomation] Failed to log scan: {e}")
    
    def _scheduler_loop(self):
        """Main scheduler loop - runs in background thread."""
        logger.info("[ScheduledAutomation] Scheduler started")
        
        while self._running:
            try:
                if not self._is_market_open():
                    time.sleep(60)
                    continue
                
                for scan_type in [ScanType.MORNING, ScanType.MIDDAY, ScanType.CLOSE]:
                    if self._should_run_scan(scan_type):
                        logger.info(f"[ScheduledAutomation] Triggering {scan_type.value} scan")
                        self.run_scan(scan_type)
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"[ScheduledAutomation] Scheduler error: {e}")
                time.sleep(60)
        
        logger.info("[ScheduledAutomation] Scheduler stopped")
    
    def start(self):
        """Start the scheduler."""
        with self._lock:
            if self._running:
                logger.warning("[ScheduledAutomation] Already running")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._thread.start()
            logger.info("[ScheduledAutomation] Started background scheduler")
    
    def stop(self):
        """Stop the scheduler."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._thread:
                self._thread.join(timeout=5)
                self._thread = None
            logger.info("[ScheduledAutomation] Stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and configuration."""
        now = self._get_current_time()
        
        next_scans = {}
        for scan_type, enabled, time_str in [
            (ScanType.MORNING, self.config.morning_enabled, self.config.morning_time),
            (ScanType.MIDDAY, self.config.midday_enabled, self.config.midday_time),
            (ScanType.CLOSE, self.config.close_enabled, self.config.close_time),
        ]:
            if enabled:
                hour, minute = self._parse_time(time_str)
                scheduled = now.replace(hour=hour, minute=minute, second=0)
                if scheduled < now:
                    scheduled += timedelta(days=1)
                while scheduled.weekday() >= 5:
                    scheduled += timedelta(days=1)
                next_scans[scan_type.value] = scheduled.isoformat()
        
        return {
            "running": self._running,
            "timezone": self.config.timezone,
            "current_time": now.isoformat(),
            "market_open": self._is_market_open(),
            "config": {
                "morning": {"enabled": self.config.morning_enabled, "time": self.config.morning_time},
                "midday": {"enabled": self.config.midday_enabled, "time": self.config.midday_time},
                "close": {"enabled": self.config.close_enabled, "time": self.config.close_time},
                "max_trades_per_scan": self.config.max_trades_per_scan,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct,
                "require_high_conviction": self.config.require_high_conviction,
                "quick_scan": self.config.quick_scan,
                "dry_run": self.config.dry_run,
            },
            "next_scans": next_scans,
            "last_scans": {
                k.value: v.isoformat() if v else None 
                for k, v in self._last_scans.items()
            },
            "recent_scans": [s.to_dict() for s in self._scan_history[-5:]],
            "total_scans": len(self._scan_history),
        }
    
    def get_scan_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scan history."""
        return [s.to_dict() for s in self._scan_history[-limit:]]
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update scheduler configuration."""
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"[ScheduledAutomation] Updated {key} = {value}")
        
        return self.get_status()


_scheduler_instance: Optional[ScheduledAutomation] = None


def get_scheduled_automation() -> ScheduledAutomation:
    """Get or create the singleton scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ScheduledAutomation()
    return _scheduler_instance


def start_scheduled_automation():
    """Start the scheduled automation system."""
    scheduler = get_scheduled_automation()
    scheduler.start()
    return scheduler


def stop_scheduled_automation():
    """Stop the scheduled automation system."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop()
