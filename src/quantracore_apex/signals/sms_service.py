"""
SMS Notification Service for Trading Signals.

Sends formatted trading signals via Twilio SMS with:
- QuantraScore and conviction tier
- Predicted top price (runup target)
- Entry/exit levels
- Timing guidance

Uses Replit's Twilio integration for secure credential management.
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class SMSConfig:
    """Configuration for SMS alert service."""
    
    enabled: bool = True
    recipient_phone: str = ""
    min_quantrascore: float = 0.65
    min_runner_probability: float = 0.5
    max_avoid_probability: float = 0.5
    min_timing_confidence: float = 0.4
    only_immediate_timing: bool = False
    max_alerts_per_hour: int = 10
    cooldown_minutes: int = 5
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SMSAlertRecord:
    """Record of sent SMS alert."""
    symbol: str
    timestamp: datetime
    message: str
    twilio_sid: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


class TwilioClient:
    """Async Twilio client using Replit's connector integration."""
    
    def __init__(self):
        self._credentials: Optional[Dict] = None
        self._lock = asyncio.Lock()
    
    async def _get_credentials(self) -> Dict:
        """Fetch Twilio credentials from Replit connector."""
        if self._credentials:
            return self._credentials
        
        async with self._lock:
            if self._credentials:
                return self._credentials
            
            hostname = os.environ.get("REPLIT_CONNECTORS_HOSTNAME")
            repl_identity = os.environ.get("REPL_IDENTITY")
            web_renewal = os.environ.get("WEB_REPL_RENEWAL")
            
            if repl_identity:
                x_replit_token = f"repl {repl_identity}"
            elif web_renewal:
                x_replit_token = f"depl {web_renewal}"
            else:
                raise RuntimeError("Replit identity token not found")
            
            if not hostname:
                raise RuntimeError("REPLIT_CONNECTORS_HOSTNAME not set")
            
            url = f"https://{hostname}/api/v2/connection?include_secrets=true&connector_names=twilio"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "X_REPLIT_TOKEN": x_replit_token,
                    }
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Failed to fetch Twilio credentials: {resp.status}")
                    
                    data = await resp.json()
                    items = data.get("items", [])
                    
                    if not items:
                        raise RuntimeError("Twilio connection not configured")
                    
                    settings = items[0].get("settings", {})
                    
                    self._credentials = {
                        "account_sid": settings.get("account_sid"),
                        "api_key": settings.get("api_key"),
                        "api_key_secret": settings.get("api_key_secret"),
                        "phone_number": settings.get("phone_number"),
                    }
                    
                    if not all([
                        self._credentials["account_sid"],
                        self._credentials["api_key"],
                        self._credentials["api_key_secret"],
                    ]):
                        raise RuntimeError("Incomplete Twilio credentials")
                    
                    return self._credentials
    
    async def send_sms(self, to_phone: str, message: str) -> Dict:
        """Send SMS via Twilio API."""
        creds = await self._get_credentials()
        
        account_sid = creds["account_sid"]
        api_key = creds["api_key"]
        api_key_secret = creds["api_key_secret"]
        from_phone = creds.get("phone_number")
        
        if not from_phone:
            raise RuntimeError("No Twilio phone number configured")
        
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        
        auth = aiohttp.BasicAuth(api_key, api_key_secret)
        
        data = {
            "To": to_phone,
            "From": from_phone,
            "Body": message,
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=auth, data=data) as resp:
                result = await resp.json()
                
                if resp.status >= 400:
                    error_msg = result.get("message", "Unknown error")
                    raise RuntimeError(f"Twilio API error: {error_msg}")
                
                return {
                    "sid": result.get("sid"),
                    "status": result.get("status"),
                    "to": result.get("to"),
                }


class SMSAlertService:
    """
    Sends SMS alerts for high-quality trading signals.
    
    Features:
    - Configurable alert thresholds
    - Rate limiting to prevent spam
    - Signal formatting with all key information
    - Alert history tracking
    """
    
    def __init__(self, config: Optional[SMSConfig] = None):
        self.config = config or SMSConfig()
        self.twilio = TwilioClient()
        
        self._alert_history: List[SMSAlertRecord] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._alerts_this_hour: int = 0
        self._hour_start: datetime = datetime.now()
        self._lock = threading.Lock()
        
        self._config_file = Path("config/sms_alerts.json")
        self._load_config()
    
    def _load_config(self):
        """Load configuration from disk."""
        if self._config_file.exists():
            try:
                with open(self._config_file) as f:
                    data = json.load(f)
                    self.config = SMSConfig(**data)
                    logger.info(f"Loaded SMS config: recipient={self.config.recipient_phone}")
            except Exception as e:
                logger.warning(f"Failed to load SMS config: {e}")
    
    def _save_config(self):
        """Save configuration to disk."""
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save SMS config: {e}")
    
    def update_config(self, **kwargs) -> Dict:
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._save_config()
        return self.config.to_dict()
    
    def _check_rate_limit(self, symbol: str) -> bool:
        """Check if we can send another alert."""
        now = datetime.now()
        
        with self._lock:
            if (now - self._hour_start).total_seconds() > 3600:
                self._alerts_this_hour = 0
                self._hour_start = now
            
            if self._alerts_this_hour >= self.config.max_alerts_per_hour:
                logger.debug(f"Rate limit reached: {self._alerts_this_hour} alerts this hour")
                return False
            
            last_alert = self._last_alert_time.get(symbol)
            if last_alert:
                minutes_since = (now - last_alert).total_seconds() / 60
                if minutes_since < self.config.cooldown_minutes:
                    logger.debug(f"Cooldown active for {symbol}: {minutes_since:.1f}m since last alert")
                    return False
            
            return True
    
    def _format_signal_message(self, signal: Dict) -> str:
        """Format signal data into SMS message."""
        symbol = signal.get("symbol", "???")
        qs = signal.get("quantrascore_calibrated", 0) * 100
        runner_prob = signal.get("runner_probability", 0) * 100
        conviction = signal.get("conviction_tier", "unknown").upper()
        
        current_price = signal.get("current_price", 0)
        predicted_top = signal.get("predicted_top_price", 0)
        expected_runup = signal.get("expected_runup_pct", 0) * 100
        
        stop_loss = signal.get("stop_loss", 0)
        target_1 = signal.get("target_level_1", 0)
        
        timing = signal.get("timing_bucket", "none")
        direction = signal.get("direction", "neutral").upper()
        
        timing_labels = {
            "immediate": "NOW",
            "very_soon": "2-3 bars",
            "soon": "4-6 bars",
            "late": "7-10 bars",
            "none": "unclear",
        }
        timing_text = timing_labels.get(timing, timing)
        
        lines = [
            f"APEX {symbol} | {conviction}",
            f"",
            f"QS: {qs:.0f} | Runner: {runner_prob:.0f}%",
            f"Direction: {direction}",
            f"Timing: {timing_text}",
            f"",
            f"Price: ${current_price:.2f}",
        ]
        
        if expected_runup > 0 and predicted_top > current_price:
            lines.append(f"Target Top: ${predicted_top:.2f} (+{expected_runup:.1f}%)")
        
        lines.extend([
            f"",
            f"Stop: ${stop_loss:.2f}",
            f"T1: ${target_1:.2f}",
            f"",
            f"Structural analysis only",
        ])
        
        return "\n".join(lines)
    
    def _passes_threshold(self, signal: Dict) -> bool:
        """Check if signal passes alert thresholds."""
        qs = signal.get("quantrascore_calibrated", 0)
        runner = signal.get("runner_probability", 0)
        avoid = signal.get("avoid_probability", 1)
        timing_conf = signal.get("timing_confidence", 0)
        timing_bucket = signal.get("timing_bucket", "none")
        
        if qs < self.config.min_quantrascore:
            return False
        
        if runner < self.config.min_runner_probability:
            return False
        
        if avoid > self.config.max_avoid_probability:
            return False
        
        if timing_conf < self.config.min_timing_confidence:
            return False
        
        if self.config.only_immediate_timing and timing_bucket != "immediate":
            return False
        
        return True
    
    async def send_signal_alert(self, signal: Dict) -> Optional[SMSAlertRecord]:
        """Send SMS alert for a trading signal."""
        if not self.config.enabled:
            logger.debug("SMS alerts disabled")
            return None
        
        if not self.config.recipient_phone:
            logger.warning("No recipient phone configured for SMS alerts")
            return None
        
        symbol = signal.get("symbol", "???")
        
        if not self._passes_threshold(signal):
            logger.debug(f"Signal {symbol} doesn't meet alert thresholds")
            return None
        
        if not self._check_rate_limit(symbol):
            return None
        
        message = self._format_signal_message(signal)
        
        record = SMSAlertRecord(
            symbol=symbol,
            timestamp=datetime.now(),
            message=message,
        )
        
        try:
            result = await self.twilio.send_sms(
                to_phone=self.config.recipient_phone,
                message=message,
            )
            
            record.twilio_sid = result.get("sid")
            record.success = True
            
            with self._lock:
                self._alerts_this_hour += 1
                self._last_alert_time[symbol] = datetime.now()
                self._alert_history.append(record)
            
            logger.info(f"SMS alert sent for {symbol}: {result.get('sid')}")
            
        except Exception as e:
            record.error = str(e)
            logger.error(f"Failed to send SMS alert for {symbol}: {e}")
            
            with self._lock:
                self._alert_history.append(record)
        
        return record
    
    def send_signal_alert_sync(self, signal: Dict) -> Optional[SMSAlertRecord]:
        """Synchronous wrapper for send_signal_alert."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.send_signal_alert(signal))
    
    async def send_batch_alerts(self, signals: List[Dict]) -> List[SMSAlertRecord]:
        """Send alerts for multiple signals."""
        results = []
        
        for signal in signals:
            record = await self.send_signal_alert(signal)
            if record:
                results.append(record)
        
        return results
    
    def get_status(self) -> Dict:
        """Get service status."""
        return {
            "enabled": self.config.enabled,
            "recipient_configured": bool(self.config.recipient_phone),
            "alerts_this_hour": self._alerts_this_hour,
            "max_alerts_per_hour": self.config.max_alerts_per_hour,
            "total_alerts_sent": len([r for r in self._alert_history if r.success]),
            "total_alerts_failed": len([r for r in self._alert_history if not r.success]),
            "config": self.config.to_dict(),
        }
    
    def get_recent_alerts(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history."""
        with self._lock:
            recent = self._alert_history[-limit:]
        
        return [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp.isoformat(),
                "success": r.success,
                "twilio_sid": r.twilio_sid,
                "error": r.error,
            }
            for r in reversed(recent)
        ]


_sms_service: Optional[SMSAlertService] = None


def get_sms_service() -> SMSAlertService:
    """Get or create the global SMS service instance."""
    global _sms_service
    if _sms_service is None:
        _sms_service = SMSAlertService()
    return _sms_service
