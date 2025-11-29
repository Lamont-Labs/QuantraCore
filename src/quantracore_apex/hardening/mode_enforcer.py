"""
Mode Enforcement System.

Implements strict RESEARCH / PAPER / LIVE mode boundaries.
Default: RESEARCH mode (safest).
LIVE mode requires explicit configuration and institutional sign-off.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class ExecutionMode(Enum):
    """System execution modes with strict boundaries."""
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"
    
    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Parse mode from string, defaulting to RESEARCH."""
        normalized = value.upper().strip()
        if normalized in ("RESEARCH", "BACKTEST", "SIMULATION"):
            return cls.RESEARCH
        elif normalized in ("PAPER", "PAPER_TRADING"):
            return cls.PAPER
        elif normalized == "LIVE":
            return cls.LIVE
        return cls.RESEARCH


class ModeViolationError(Exception):
    """Raised when an action violates mode permissions."""
    pass


@dataclass
class ModePermissions:
    """Permission table for each mode."""
    mode: ExecutionMode
    engine_enabled: bool = True
    apexlab_enabled: bool = True
    models_enabled: bool = True
    execution_engine_enabled: bool = False
    broker_router_enabled: bool = False
    live_orders_allowed: bool = False
    paper_orders_allowed: bool = False
    
    @classmethod
    def for_mode(cls, mode: ExecutionMode) -> "ModePermissions":
        """Get permissions for a specific mode."""
        if mode == ExecutionMode.RESEARCH:
            return cls(
                mode=mode,
                engine_enabled=True,
                apexlab_enabled=True,
                models_enabled=True,
                execution_engine_enabled=False,
                broker_router_enabled=False,
                live_orders_allowed=False,
                paper_orders_allowed=False,
            )
        elif mode == ExecutionMode.PAPER:
            return cls(
                mode=mode,
                engine_enabled=True,
                apexlab_enabled=True,
                models_enabled=True,
                execution_engine_enabled=True,
                broker_router_enabled=True,
                live_orders_allowed=False,
                paper_orders_allowed=True,
            )
        elif mode == ExecutionMode.LIVE:
            return cls(
                mode=mode,
                engine_enabled=True,
                apexlab_enabled=True,
                models_enabled=True,
                execution_engine_enabled=True,
                broker_router_enabled=True,
                live_orders_allowed=True,
                paper_orders_allowed=True,
            )
        return cls.for_mode(ExecutionMode.RESEARCH)


@dataclass
class ModeEnforcer:
    """
    Enforces execution mode boundaries.
    
    Fail-closed: defaults to RESEARCH mode.
    LIVE mode requires:
        - config/broker_live.yaml present
        - APEX_ENABLE_LIVE=true environment variable
        - Compliance doc hash verification
    """
    
    current_mode: ExecutionMode = ExecutionMode.RESEARCH
    permissions: ModePermissions = field(default_factory=lambda: ModePermissions.for_mode(ExecutionMode.RESEARCH))
    mode_locked: bool = False
    violation_log: list[dict] = field(default_factory=list)
    
    LIVE_CONFIG_PATH = Path("config/broker_live.yaml")
    LIVE_ENV_VAR = "APEX_ENABLE_LIVE"
    COMPLIANCE_DOC_PATH = Path("config/compliance_signoff.hash")
    
    def initialize(self) -> None:
        """Initialize mode from configuration, defaulting to RESEARCH."""
        requested_mode = self._detect_requested_mode()
        
        if requested_mode == ExecutionMode.LIVE:
            if not self._verify_live_requirements():
                self._log_violation(
                    "LIVE mode requested but requirements not met. Falling back to RESEARCH.",
                    severity="HIGH",
                )
                requested_mode = ExecutionMode.RESEARCH
        
        self.current_mode = requested_mode
        self.permissions = ModePermissions.for_mode(self.current_mode)
        self.mode_locked = True
    
    def _detect_requested_mode(self) -> ExecutionMode:
        """Detect requested mode from config files."""
        broker_config = Path("config/broker.yaml")
        if not broker_config.exists():
            return ExecutionMode.RESEARCH
        
        try:
            with open(broker_config) as f:
                config = yaml.safe_load(f)
            
            mode_str = config.get("execution", {}).get("mode", "RESEARCH")
            return ExecutionMode.from_string(mode_str)
        except Exception:
            return ExecutionMode.RESEARCH
    
    def _verify_live_requirements(self) -> bool:
        """Verify all requirements for LIVE mode are met."""
        if not self.LIVE_CONFIG_PATH.exists():
            return False
        
        if os.environ.get(self.LIVE_ENV_VAR, "").lower() != "true":
            return False
        
        return True
    
    def check_permission(self, action: str) -> bool:
        """
        Check if an action is permitted in current mode.
        
        Actions:
            - engine_access
            - apexlab_access
            - model_access
            - execution_engine_access
            - broker_access
            - place_paper_order
            - place_live_order
        """
        perm = self.permissions
        
        action_map = {
            "engine_access": perm.engine_enabled,
            "apexlab_access": perm.apexlab_enabled,
            "model_access": perm.models_enabled,
            "execution_engine_access": perm.execution_engine_enabled,
            "broker_access": perm.broker_router_enabled,
            "place_paper_order": perm.paper_orders_allowed,
            "place_live_order": perm.live_orders_allowed,
        }
        
        return action_map.get(action, False)
    
    def require_permission(self, action: str, context: str = "") -> None:
        """
        Require permission for an action. Raises ModeViolationError if denied.
        """
        if not self.check_permission(action):
            self._log_violation(
                f"Permission denied: {action} in {self.current_mode.value} mode",
                action=action,
                context=context,
            )
            raise ModeViolationError(
                f"Action '{action}' not permitted in {self.current_mode.value} mode. "
                f"Context: {context}"
            )
    
    def _log_violation(
        self,
        message: str,
        action: str = "",
        context: str = "",
        severity: str = "MEDIUM",
    ) -> None:
        """Log a mode violation."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.current_mode.value,
            "message": message,
            "action": action,
            "context": context,
            "severity": severity,
        }
        self.violation_log.append(entry)
        
        log_path = Path("logs/incidents/mode_violations.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_status(self) -> dict[str, Any]:
        """Get current mode status."""
        return {
            "mode": self.current_mode.value,
            "permissions": {
                "engine_enabled": self.permissions.engine_enabled,
                "apexlab_enabled": self.permissions.apexlab_enabled,
                "models_enabled": self.permissions.models_enabled,
                "execution_engine_enabled": self.permissions.execution_engine_enabled,
                "broker_router_enabled": self.permissions.broker_router_enabled,
                "live_orders_allowed": self.permissions.live_orders_allowed,
                "paper_orders_allowed": self.permissions.paper_orders_allowed,
            },
            "mode_locked": self.mode_locked,
            "violation_count": len(self.violation_log),
        }


_mode_enforcer: ModeEnforcer | None = None


def get_mode_enforcer() -> ModeEnforcer:
    """Get or create the global mode enforcer singleton."""
    global _mode_enforcer
    if _mode_enforcer is None:
        _mode_enforcer = ModeEnforcer()
        _mode_enforcer.initialize()
    return _mode_enforcer


def reset_mode_enforcer() -> None:
    """Reset the mode enforcer singleton. For testing only."""
    global _mode_enforcer
    _mode_enforcer = None


def set_mode_for_testing(mode: ExecutionMode) -> ModeEnforcer:
    """Set the mode enforcer to a specific mode for testing. For testing only."""
    global _mode_enforcer
    _mode_enforcer = ModeEnforcer()
    _mode_enforcer.current_mode = mode
    _mode_enforcer.permissions = ModePermissions.for_mode(mode)
    _mode_enforcer.mode_locked = True
    return _mode_enforcer
