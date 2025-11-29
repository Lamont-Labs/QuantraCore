"""
QuantraCore Apex Hardening Infrastructure.

Global hardening components for determinism, safety, security, and compliance.
Implements fail-closed behavior across all critical paths.
"""

from .manifest import ProtocolManifest, ManifestValidator
from .config_validator import ConfigValidator, ConfigValidationError
from .mode_enforcer import ModeEnforcer, ExecutionMode, ModeViolationError
from .incident_logger import IncidentLogger, IncidentClass, IncidentSeverity
from .kill_switch import KillSwitchManager, KillSwitchReason

__all__ = [
    "ProtocolManifest",
    "ManifestValidator",
    "ConfigValidator",
    "ConfigValidationError",
    "ModeEnforcer",
    "ExecutionMode",
    "ModeViolationError",
    "IncidentLogger",
    "IncidentClass",
    "IncidentSeverity",
    "KillSwitchManager",
    "KillSwitchReason",
]
