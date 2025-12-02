"""
Push Notification Service for Trading Signals.

Sends browser push notifications with:
- QuantraScore and conviction tier
- Predicted top price (runup target)
- Entry/exit levels
- Timing guidance

Uses VAPID keys for secure web push authentication.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading

from pywebpush import webpush, WebPushException
from py_vapid import Vapid

logger = logging.getLogger(__name__)

VAPID_KEYS_PATH = Path("config/vapid_keys.json")
SUBSCRIPTIONS_PATH = Path("config/push_subscriptions.json")


@dataclass
class PushConfig:
    """Configuration for push notification service."""
    
    enabled: bool = True
    min_quantrascore: float = 0.65
    min_runner_probability: float = 0.5
    max_avoid_probability: float = 0.5
    min_timing_confidence: float = 0.4
    only_immediate_timing: bool = False
    max_alerts_per_hour: int = 20
    cooldown_minutes: int = 3
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PushAlertRecord:
    """Record of sent push notification."""
    symbol: str
    timestamp: datetime
    title: str
    body: str
    success: bool = False
    error: Optional[str] = None
    subscribers_notified: int = 0


class VAPIDKeyManager:
    """Manages VAPID keys for web push authentication."""
    
    def __init__(self):
        self._keys: Optional[Dict] = None
        self._lock = threading.Lock()
    
    def get_keys(self) -> Dict[str, str]:
        """Get or generate VAPID keys."""
        if self._keys:
            return self._keys
        
        with self._lock:
            if self._keys:
                return self._keys
            
            if VAPID_KEYS_PATH.exists():
                try:
                    with open(VAPID_KEYS_PATH, "r") as f:
                        self._keys = json.load(f)
                    logger.info("Loaded existing VAPID keys")
                    return self._keys
                except Exception as e:
                    logger.warning(f"Failed to load VAPID keys: {e}")
            
            self._keys = self._generate_keys()
            return self._keys
    
    def _generate_keys(self) -> Dict[str, str]:
        """Generate new VAPID keys."""
        vapid = Vapid()
        vapid.generate_keys()
        
        public_key = vapid.public_key
        private_key = vapid.private_key
        
        if public_key is None or private_key is None:
            raise RuntimeError("Failed to generate VAPID keys")
        
        keys = {
            "public_key": public_key.public_bytes_raw().hex(),
            "private_key": private_key.private_bytes_raw().hex(),
            "public_key_urlsafe": self._to_urlsafe_base64(public_key.public_bytes_raw()),
        }
        
        VAPID_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VAPID_KEYS_PATH, "w") as f:
            json.dump(keys, f, indent=2)
        
        logger.info("Generated new VAPID keys")
        return keys
    
    def _to_urlsafe_base64(self, data: bytes) -> str:
        """Convert bytes to URL-safe base64."""
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')
    
    def get_public_key(self) -> str:
        """Get public key for client subscription."""
        keys = self.get_keys()
        return keys.get("public_key_urlsafe", "")


class SubscriptionManager:
    """Manages push notification subscriptions."""
    
    def __init__(self):
        self._subscriptions: List[Dict] = []
        self._lock = threading.Lock()
        self._load_subscriptions()
    
    def _load_subscriptions(self):
        """Load subscriptions from disk."""
        if SUBSCRIPTIONS_PATH.exists():
            try:
                with open(SUBSCRIPTIONS_PATH, "r") as f:
                    self._subscriptions = json.load(f)
                logger.info(f"Loaded {len(self._subscriptions)} push subscriptions")
            except Exception as e:
                logger.warning(f"Failed to load subscriptions: {e}")
                self._subscriptions = []
    
    def _save_subscriptions(self):
        """Save subscriptions to disk."""
        try:
            SUBSCRIPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SUBSCRIPTIONS_PATH, "w") as f:
                json.dump(self._subscriptions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save subscriptions: {e}")
    
    def add_subscription(self, subscription: Dict) -> bool:
        """Add a new subscription."""
        with self._lock:
            endpoint = subscription.get("endpoint", "")
            
            for existing in self._subscriptions:
                if existing.get("endpoint") == endpoint:
                    existing.update(subscription)
                    self._save_subscriptions()
                    logger.info("Updated existing push subscription")
                    return True
            
            subscription["created_at"] = datetime.now().isoformat()
            self._subscriptions.append(subscription)
            self._save_subscriptions()
            logger.info(f"Added new push subscription (total: {len(self._subscriptions)})")
            return True
    
    def remove_subscription(self, endpoint: str) -> bool:
        """Remove a subscription by endpoint."""
        with self._lock:
            initial_count = len(self._subscriptions)
            self._subscriptions = [
                s for s in self._subscriptions 
                if s.get("endpoint") != endpoint
            ]
            
            if len(self._subscriptions) < initial_count:
                self._save_subscriptions()
                logger.info("Removed push subscription")
                return True
            return False
    
    def get_all_subscriptions(self) -> List[Dict]:
        """Get all active subscriptions."""
        with self._lock:
            return self._subscriptions.copy()
    
    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        with self._lock:
            return len(self._subscriptions)


class PushNotificationService:
    """
    Sends push notifications for high-quality trading signals.
    
    Features:
    - VAPID-authenticated web push
    - Configurable alert thresholds
    - Rate limiting to prevent spam
    - Signal formatting with all key information
    - Alert history tracking
    """
    
    def __init__(self, config: Optional[PushConfig] = None):
        self.config = config or PushConfig()
        self.vapid_manager = VAPIDKeyManager()
        self.subscription_manager = SubscriptionManager()
        
        self._alert_history: List[PushAlertRecord] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._alerts_this_hour: int = 0
        self._hour_start: datetime = datetime.now()
        self._lock = threading.Lock()
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from disk."""
        config_path = Path("config/push_config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                logger.info("Loaded push notification config")
            except Exception as e:
                logger.warning(f"Failed to load push config: {e}")
    
    def _save_config(self):
        """Save configuration to disk."""
        config_path = Path("config/push_config.json")
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save push config: {e}")
    
    def get_public_key(self) -> str:
        """Get VAPID public key for client subscription."""
        return self.vapid_manager.get_public_key()
    
    def get_config(self) -> Dict:
        """Get current configuration."""
        return self.config.to_dict()
    
    def update_config(self, **kwargs) -> Dict:
        """Update configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._save_config()
        return self.config.to_dict()
    
    def add_subscription(self, subscription: Dict) -> bool:
        """Add a push subscription."""
        return self.subscription_manager.add_subscription(subscription)
    
    def remove_subscription(self, endpoint: str) -> bool:
        """Remove a push subscription."""
        return self.subscription_manager.remove_subscription(endpoint)
    
    def get_status(self) -> Dict:
        """Get service status."""
        return {
            "enabled": self.config.enabled,
            "subscribers": self.subscription_manager.get_subscription_count(),
            "alerts_this_hour": self._alerts_this_hour,
            "max_alerts_per_hour": self.config.max_alerts_per_hour,
            "thresholds": {
                "min_quantrascore": self.config.min_quantrascore,
                "min_runner_probability": self.config.min_runner_probability,
                "max_avoid_probability": self.config.max_avoid_probability,
            },
            "recent_alerts": len(self._alert_history),
        }
    
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
    
    def _format_notification(self, signal: Dict) -> Dict[str, Any]:
        """Format signal data into push notification."""
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
        timing_emoji = {
            "immediate": "NOW",
            "near_term": "SOON",
            "medium_term": "WAIT",
            "extended": "HOLD",
        }.get(timing, "?")
        
        title = f"APEX: {symbol} | {conviction} | QS:{qs:.0f}%"
        
        body_lines = [
            f"Runner: {runner_prob:.0f}% | Timing: {timing_emoji}",
            f"Entry: ${current_price:.2f} -> Target: ${predicted_top:.2f} (+{expected_runup:.0f}%)",
            f"Stop: ${stop_loss:.2f} | T1: ${target_1:.2f}",
        ]
        
        return {
            "title": title,
            "body": "\n".join(body_lines),
            "icon": "/assets/apex-icon.png",
            "badge": "/assets/apex-badge.png",
            "tag": f"signal-{symbol}",
            "data": {
                "symbol": symbol,
                "url": f"/?symbol={symbol}",
                "timestamp": datetime.now().isoformat(),
            }
        }
    
    def _passes_threshold(self, signal: Dict) -> bool:
        """Check if signal passes alert thresholds."""
        qs = signal.get("quantrascore_calibrated", 0)
        runner_prob = signal.get("runner_probability", 0)
        avoid_prob = signal.get("avoid_probability", 1.0)
        timing_confidence = signal.get("timing_confidence", 0)
        timing_bucket = signal.get("timing_bucket", "")
        
        if qs < self.config.min_quantrascore:
            return False
        
        if runner_prob < self.config.min_runner_probability:
            return False
        
        if avoid_prob > self.config.max_avoid_probability:
            return False
        
        if timing_confidence < self.config.min_timing_confidence:
            return False
        
        if self.config.only_immediate_timing and timing_bucket != "immediate":
            return False
        
        return True
    
    async def send_notification(self, title: str, body: str, data: Optional[Dict] = None) -> Dict:
        """Send a push notification to all subscribers."""
        subscriptions = self.subscription_manager.get_all_subscriptions()
        
        if not subscriptions:
            return {"success": False, "error": "No subscribers", "sent": 0}
        
        vapid_keys = self.vapid_manager.get_keys()
        
        payload = json.dumps({
            "title": title,
            "body": body,
            "icon": "/assets/apex-icon.png",
            "badge": "/assets/apex-badge.png",
            "data": data or {},
        })
        
        success_count = 0
        failed_endpoints = []
        
        for subscription in subscriptions:
            try:
                webpush(
                    subscription_info=subscription,
                    data=payload,
                    vapid_private_key=vapid_keys["private_key"],
                    vapid_claims={
                        "sub": "mailto:alerts@quantracore.apex",
                    }
                )
                success_count += 1
            except WebPushException as e:
                logger.warning(f"Push failed: {e}")
                if e.response and e.response.status_code in [404, 410]:
                    failed_endpoints.append(subscription.get("endpoint"))
            except Exception as e:
                logger.error(f"Push error: {e}")
        
        for endpoint in failed_endpoints:
            self.subscription_manager.remove_subscription(endpoint)
        
        return {
            "success": success_count > 0,
            "sent": success_count,
            "failed": len(subscriptions) - success_count,
            "removed_stale": len(failed_endpoints),
        }
    
    async def send_signal_alert(self, signal: Dict) -> Optional[PushAlertRecord]:
        """Send push notification for a trading signal if it meets thresholds."""
        if not self.config.enabled:
            logger.debug("Push notifications disabled")
            return None
        
        symbol = signal.get("symbol", "???")
        
        if not self._passes_threshold(signal):
            logger.debug(f"Signal {symbol} did not pass threshold")
            return None
        
        if not self._check_rate_limit(symbol):
            logger.debug(f"Rate limit prevents alert for {symbol}")
            return None
        
        notification = self._format_notification(signal)
        
        record = PushAlertRecord(
            symbol=symbol,
            timestamp=datetime.now(),
            title=notification["title"],
            body=notification["body"],
        )
        
        try:
            notification_data = notification.get("data")
            if isinstance(notification_data, str):
                notification_data = {"raw": notification_data}
            
            result = await self.send_notification(
                title=notification["title"],
                body=notification["body"],
                data=notification_data,
            )
            
            record.success = result.get("success", False)
            record.subscribers_notified = result.get("sent", 0)
            
            if record.success:
                with self._lock:
                    self._alerts_this_hour += 1
                    self._last_alert_time[symbol] = datetime.now()
                
                logger.info(f"Push notification sent for {symbol} to {record.subscribers_notified} subscribers")
            else:
                record.error = result.get("error", "Unknown error")
                
        except Exception as e:
            record.success = False
            record.error = str(e)
            logger.error(f"Failed to send push for {symbol}: {e}")
        
        self._alert_history.append(record)
        if len(self._alert_history) > 100:
            self._alert_history = self._alert_history[-100:]
        
        return record
    
    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history."""
        records = self._alert_history[-limit:]
        return [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp.isoformat(),
                "title": r.title,
                "body": r.body,
                "success": r.success,
                "subscribers_notified": r.subscribers_notified,
                "error": r.error,
            }
            for r in reversed(records)
        ]


_push_service: Optional[PushNotificationService] = None
_push_lock = threading.Lock()


def get_push_service() -> PushNotificationService:
    """Get singleton push notification service."""
    global _push_service
    
    if _push_service is None:
        with _push_lock:
            if _push_service is None:
                _push_service = PushNotificationService()
    
    return _push_service
