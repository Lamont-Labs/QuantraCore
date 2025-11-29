"""
Kill Switch System.

Implements manual and automatic kill switches for emergency halts.
Once engaged, no new orders are allowed until explicitly reset.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class KillSwitchReason(Enum):
    """Reasons for kill switch activation."""
    MANUAL = "MANUAL"
    DAILY_DRAWDOWN_EXCEEDED = "DAILY_DRAWDOWN_EXCEEDED"
    BROKER_ERROR_RATE_HIGH = "BROKER_ERROR_RATE_HIGH"
    RISK_VIOLATIONS_SPIKE = "RISK_VIOLATIONS_SPIKE"
    DATA_FEED_FAILURE = "DATA_FEED_FAILURE"
    MODEL_FAILURE = "MODEL_FAILURE"
    DETERMINISM_FAILURE = "DETERMINISM_FAILURE"
    EXTERNAL_SIGNAL = "EXTERNAL_SIGNAL"


@dataclass
class KillSwitchState:
    """Current state of kill switch."""
    engaged: bool = False
    reason: KillSwitchReason | None = None
    engaged_at: str | None = None
    engaged_by: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    auto_flatten_positions: bool = False
    reset_at: str | None = None
    reset_by: str = ""


@dataclass
class KillSwitchManager:
    """
    Manages kill switch state and enforcement.
    
    Kill switch types:
        - manual_kill_switch: operator sets flag to halt orders
        - auto_kill_switch: triggered by thresholds
    
    Once engaged:
        - No new orders allowed
        - Open positions optionally flattened
        - System remains in SAFE mode until explicitly reset
    """
    
    state: KillSwitchState = field(default_factory=KillSwitchState)
    state_path: Path = field(default_factory=lambda: Path("config/kill_switch_state.json"))
    log_path: Path = field(default_factory=lambda: Path("logs/incidents/kill_switch.jsonl"))
    
    daily_drawdown_threshold_pct: float = 5.0
    broker_error_rate_threshold_pct: float = 20.0
    risk_violations_threshold_count: int = 10
    
    _daily_drawdown_pct: float = 0.0
    _broker_error_rate_pct: float = 0.0
    _risk_violations_count: int = 0
    
    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state if exists."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self.state = KillSwitchState(
                    engaged=data.get("engaged", False),
                    reason=KillSwitchReason(data["reason"]) if data.get("reason") else None,
                    engaged_at=data.get("engaged_at"),
                    engaged_by=data.get("engaged_by", ""),
                    context=data.get("context", {}),
                    auto_flatten_positions=data.get("auto_flatten_positions", False),
                    reset_at=data.get("reset_at"),
                    reset_by=data.get("reset_by", ""),
                )
            except Exception:
                self.state = KillSwitchState()
    
    def _save_state(self) -> None:
        """Persist current state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "engaged": self.state.engaged,
            "reason": self.state.reason.value if self.state.reason else None,
            "engaged_at": self.state.engaged_at,
            "engaged_by": self.state.engaged_by,
            "context": self.state.context,
            "auto_flatten_positions": self.state.auto_flatten_positions,
            "reset_at": self.state.reset_at,
            "reset_by": self.state.reset_by,
        }
        with open(self.state_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _log_event(self, event_type: str, details: dict) -> None:
        """Log kill switch event."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **details,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def engage(
        self,
        reason: KillSwitchReason,
        engaged_by: str = "system",
        context: dict[str, Any] | None = None,
        flatten_positions: bool = False,
    ) -> None:
        """
        Engage the kill switch.
        
        Args:
            reason: Why the kill switch is being engaged
            engaged_by: Who/what engaged it (operator name or "auto")
            context: Additional context about the trigger
            flatten_positions: Whether to flatten open positions
        """
        if self.state.engaged:
            return
        
        self.state = KillSwitchState(
            engaged=True,
            reason=reason,
            engaged_at=datetime.now(timezone.utc).isoformat(),
            engaged_by=engaged_by,
            context=context or {},
            auto_flatten_positions=flatten_positions,
        )
        
        self._save_state()
        self._log_event("ENGAGED", {
            "reason": reason.value,
            "engaged_by": engaged_by,
            "context": context or {},
            "flatten_positions": flatten_positions,
        })
    
    def reset(self, reset_by: str = "operator") -> None:
        """
        Reset the kill switch, allowing orders to resume.
        
        Args:
            reset_by: Who is resetting the switch
        """
        if not self.state.engaged:
            return
        
        old_state = self.state
        self.state = KillSwitchState(
            engaged=False,
            reset_at=datetime.now(timezone.utc).isoformat(),
            reset_by=reset_by,
        )
        
        self._save_state()
        self._log_event("RESET", {
            "reset_by": reset_by,
            "was_engaged_at": old_state.engaged_at,
            "was_reason": old_state.reason.value if old_state.reason else None,
        })
        
        self._daily_drawdown_pct = 0.0
        self._broker_error_rate_pct = 0.0
        self._risk_violations_count = 0
    
    def is_engaged(self) -> bool:
        """Check if kill switch is currently engaged."""
        return self.state.engaged
    
    def check_order_allowed(self) -> tuple[bool, str]:
        """
        Check if new orders are allowed.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if self.state.engaged:
            return (
                False,
                f"Kill switch engaged: {self.state.reason.value if self.state.reason else 'unknown'}"
            )
        return (True, "")
    
    def update_daily_drawdown(self, drawdown_pct: float) -> None:
        """Update daily drawdown and check threshold."""
        self._daily_drawdown_pct = drawdown_pct
        if drawdown_pct > self.daily_drawdown_threshold_pct:
            self.engage(
                reason=KillSwitchReason.DAILY_DRAWDOWN_EXCEEDED,
                engaged_by="auto",
                context={"drawdown_pct": drawdown_pct, "threshold": self.daily_drawdown_threshold_pct},
            )
    
    def update_broker_error_rate(self, error_rate_pct: float) -> None:
        """Update broker error rate and check threshold."""
        self._broker_error_rate_pct = error_rate_pct
        if error_rate_pct > self.broker_error_rate_threshold_pct:
            self.engage(
                reason=KillSwitchReason.BROKER_ERROR_RATE_HIGH,
                engaged_by="auto",
                context={"error_rate_pct": error_rate_pct, "threshold": self.broker_error_rate_threshold_pct},
            )
    
    def increment_risk_violations(self) -> None:
        """Increment risk violation count and check threshold."""
        self._risk_violations_count += 1
        if self._risk_violations_count > self.risk_violations_threshold_count:
            self.engage(
                reason=KillSwitchReason.RISK_VIOLATIONS_SPIKE,
                engaged_by="auto",
                context={"violations": self._risk_violations_count, "threshold": self.risk_violations_threshold_count},
            )
    
    def get_status(self) -> dict[str, Any]:
        """Get current kill switch status."""
        return {
            "engaged": self.state.engaged,
            "reason": self.state.reason.value if self.state.reason else None,
            "engaged_at": self.state.engaged_at,
            "engaged_by": self.state.engaged_by,
            "context": self.state.context,
            "auto_flatten_positions": self.state.auto_flatten_positions,
            "reset_at": self.state.reset_at,
            "reset_by": self.state.reset_by,
            "metrics": {
                "daily_drawdown_pct": self._daily_drawdown_pct,
                "broker_error_rate_pct": self._broker_error_rate_pct,
                "risk_violations_count": self._risk_violations_count,
            },
            "thresholds": {
                "daily_drawdown_threshold_pct": self.daily_drawdown_threshold_pct,
                "broker_error_rate_threshold_pct": self.broker_error_rate_threshold_pct,
                "risk_violations_threshold_count": self.risk_violations_threshold_count,
            },
        }


_kill_switch_manager: KillSwitchManager | None = None


def get_kill_switch_manager() -> KillSwitchManager:
    """Get or create the global kill switch manager singleton."""
    global _kill_switch_manager
    if _kill_switch_manager is None:
        _kill_switch_manager = KillSwitchManager()
    return _kill_switch_manager
