"""
Scheduled Autonomous Trading System.

Runs the unified autotrader on a schedule during extended market hours.
Uses Alpaca market data exclusively - no Alpha Vantage API calls.

Extended Hours (Eastern Time):
- Pre-market: 4:00 AM - 9:30 AM
- Regular: 9:30 AM - 4:00 PM
- After-hours: 4:00 PM - 8:00 PM

Default: Scans every 30 minutes = 32 scans/day

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
    INTERVAL = "interval"
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
    interval_minutes: int = 30
    extended_hours_start: str = "04:00"
    extended_hours_end: str = "20:00"
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
    
    Runs unified autotrader scans at regular intervals during extended market hours.
    All trading is PAPER ONLY via Alpaca.
    
    Extended Hours (Eastern Time):
    - Pre-market: 4:00 AM - 9:30 AM
    - Regular market: 9:30 AM - 4:00 PM
    - After-hours: 4:00 PM - 8:00 PM
    
    Default: Every 30 minutes = 32 scans/day
    """
    
    SCAN_LOG_DIR = Path("logs/scheduled_scans/")
    
    def __init__(self, config: Optional[ScheduleConfig] = None):
        self.config = config or ScheduleConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._scan_history: List[ScanResult] = []
        self._last_scan_time: Optional[datetime] = None
        self._tz = pytz.timezone(self.config.timezone)
        
        self.SCAN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("[ScheduledAutomation] Initialized - scanning every %d minutes during extended hours", 
                   self.config.interval_minutes)
    
    def _get_current_time(self) -> datetime:
        """Get current time in configured timezone."""
        return datetime.now(self._tz)
    
    def _parse_time(self, time_str: str) -> tuple:
        """Parse HH:MM to (hour, minute)."""
        parts = time_str.split(":")
        return int(parts[0]), int(parts[1])
    
    def _is_extended_hours(self) -> bool:
        """Check if we're within extended market hours (4 AM - 8 PM ET on weekdays)."""
        now = self._get_current_time()
        
        if now.weekday() >= 5:
            return False
        
        start_hour, start_min = self._parse_time(self.config.extended_hours_start)
        end_hour, end_min = self._parse_time(self.config.extended_hours_end)
        
        market_start = now.replace(hour=start_hour, minute=start_min, second=0)
        market_end = now.replace(hour=end_hour, minute=end_min, second=0)
        
        return market_start <= now <= market_end
    
    def _should_run_interval_scan(self) -> bool:
        """Check if enough time has passed since last scan."""
        now = self._get_current_time()
        
        if self._last_scan_time is None:
            return True
        
        elapsed = (now - self._last_scan_time).total_seconds()
        interval_seconds = self.config.interval_minutes * 60
        
        return elapsed >= interval_seconds
    
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
                self._last_scan_time = self._get_current_time()
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
        logger.info("[ScheduledAutomation] Scheduler started - %d min intervals during extended hours (%s - %s ET)",
                   self.config.interval_minutes,
                   self.config.extended_hours_start,
                   self.config.extended_hours_end)
        
        while self._running:
            try:
                if not self._is_extended_hours():
                    time.sleep(60)
                    continue
                
                if self._should_run_interval_scan():
                    logger.info("[ScheduledAutomation] Triggering interval scan")
                    self.run_scan(ScanType.INTERVAL)
                
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
    
    def _calculate_next_scan(self) -> Optional[str]:
        """Calculate when the next scan will occur."""
        now = self._get_current_time()
        
        if self._is_extended_hours():
            if self._last_scan_time is None:
                return now.isoformat()
            else:
                next_scan = self._last_scan_time + timedelta(minutes=self.config.interval_minutes)
                return next_scan.isoformat()
        
        start_hour, start_min = self._parse_time(self.config.extended_hours_start)
        next_open = now.replace(hour=start_hour, minute=start_min, second=0)
        
        if now.hour >= 20:
            next_open += timedelta(days=1)
        
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        
        return next_open.isoformat()
    
    def _calculate_scans_today(self) -> int:
        """Calculate expected number of scans for today."""
        start_hour, start_min = self._parse_time(self.config.extended_hours_start)
        end_hour, end_min = self._parse_time(self.config.extended_hours_end)
        
        total_minutes = (end_hour * 60 + end_min) - (start_hour * 60 + start_min)
        return total_minutes // self.config.interval_minutes
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and configuration."""
        now = self._get_current_time()
        
        scans_today = len([s for s in self._scan_history 
                          if s.timestamp.date() == datetime.utcnow().date()])
        
        return {
            "running": self._running,
            "timezone": self.config.timezone,
            "current_time": now.isoformat(),
            "extended_hours_active": self._is_extended_hours(),
            "config": {
                "interval_minutes": self.config.interval_minutes,
                "extended_hours": f"{self.config.extended_hours_start} - {self.config.extended_hours_end} ET",
                "max_trades_per_scan": self.config.max_trades_per_scan,
                "stop_loss_pct": self.config.stop_loss_pct,
                "take_profit_pct": self.config.take_profit_pct,
                "require_high_conviction": self.config.require_high_conviction,
                "quick_scan": self.config.quick_scan,
                "dry_run": self.config.dry_run,
            },
            "next_scan": self._calculate_next_scan(),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "scans_today": scans_today,
            "expected_scans_today": self._calculate_scans_today(),
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
