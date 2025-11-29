"""
Incident Logging System.

Structured incident logging for all critical system events.
Supports incident classification, severity levels, and auto-escalation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class IncidentClass(Enum):
    """Classification of incidents for structured handling."""
    DATA_FEED_DIVERGENCE = "DATA_FEED_DIVERGENCE"
    MODEL_MANIFEST_FAILURE = "MODEL_MANIFEST_FAILURE"
    RISK_REJECT_SPIKE = "RISK_REJECT_SPIKE"
    BROKER_ERROR_RATE_SPIKE = "BROKER_ERROR_RATE_SPIKE"
    NUCLEAR_DETERMINISM_FAILURE = "NUCLEAR_DETERMINISM_FAILURE"
    UNEXPECTED_PNL_DRAWDOWN = "UNEXPECTED_PNL_DRAWDOWN"
    CONFIG_VALIDATION_FAILURE = "CONFIG_VALIDATION_FAILURE"
    MODE_VIOLATION = "MODE_VIOLATION"
    KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
    MODEL_NUMERICAL_INSTABILITY = "MODEL_NUMERICAL_INSTABILITY"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNKNOWN = "UNKNOWN"


class IncidentSeverity(Enum):
    """Severity levels for incidents."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    
    def should_trigger_kill_switch(self) -> bool:
        """Check if severity warrants kill switch."""
        return self in (IncidentSeverity.HIGH, IncidentSeverity.CRITICAL)


@dataclass
class Incident:
    """Single incident record."""
    incident_id: str
    incident_class: IncidentClass
    severity: IncidentSeverity
    message: str
    context: dict[str, Any]
    timestamp: str
    resolved: bool = False
    resolution_notes: str = ""
    triggered_kill_switch: bool = False


@dataclass
class IncidentLogger:
    """
    Central incident logging system.
    
    For any incident:
        - Log to incident_log with timestamp, class, and context
        - If severity >= HIGH, can trigger auto_kill_switch
    """
    
    incidents: list[Incident] = field(default_factory=list)
    log_path: Path = field(default_factory=lambda: Path("logs/incidents/incident_log.jsonl"))
    auto_escalate: bool = True
    _counter: int = 0
    
    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_incident(
        self,
        incident_class: IncidentClass,
        severity: IncidentSeverity,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> Incident:
        """
        Log an incident.
        
        Returns the created Incident object.
        High/Critical severity incidents may trigger kill switch.
        """
        self._counter += 1
        incident_id = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._counter:04d}"
        
        incident = Incident(
            incident_id=incident_id,
            incident_class=incident_class,
            severity=severity,
            message=message,
            context=context or {},
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        self.incidents.append(incident)
        self._write_to_log(incident)
        
        if self.auto_escalate and severity.should_trigger_kill_switch():
            incident.triggered_kill_switch = True
            self._trigger_escalation(incident)
        
        return incident
    
    def _write_to_log(self, incident: Incident) -> None:
        """Write incident to log file."""
        record = {
            "incident_id": incident.incident_id,
            "class": incident.incident_class.value,
            "severity": incident.severity.value,
            "message": incident.message,
            "context": incident.context,
            "timestamp": incident.timestamp,
            "resolved": incident.resolved,
            "triggered_kill_switch": incident.triggered_kill_switch,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def _trigger_escalation(self, incident: Incident) -> None:
        """Trigger escalation for high-severity incidents."""
        pass
    
    def log_data_feed_divergence(
        self,
        symbol: str,
        feed1: str,
        feed2: str,
        deviation_pct: float,
    ) -> Incident:
        """Log a data feed divergence incident."""
        return self.log_incident(
            incident_class=IncidentClass.DATA_FEED_DIVERGENCE,
            severity=IncidentSeverity.MEDIUM if deviation_pct < 5 else IncidentSeverity.HIGH,
            message=f"Data feed divergence detected for {symbol}",
            context={
                "symbol": symbol,
                "feed1": feed1,
                "feed2": feed2,
                "deviation_pct": deviation_pct,
            },
        )
    
    def log_model_manifest_failure(
        self,
        model_path: str,
        reason: str,
    ) -> Incident:
        """Log a model manifest failure incident."""
        return self.log_incident(
            incident_class=IncidentClass.MODEL_MANIFEST_FAILURE,
            severity=IncidentSeverity.HIGH,
            message=f"Model manifest validation failed: {model_path}",
            context={
                "model_path": model_path,
                "reason": reason,
            },
        )
    
    def log_risk_rejection(
        self,
        order_id: str,
        symbol: str,
        reason: str,
        rejection_count_recent: int,
    ) -> Incident:
        """Log risk rejection, escalating if spike detected."""
        severity = (
            IncidentSeverity.HIGH if rejection_count_recent > 10
            else IncidentSeverity.MEDIUM if rejection_count_recent > 5
            else IncidentSeverity.LOW
        )
        return self.log_incident(
            incident_class=IncidentClass.RISK_REJECT_SPIKE if rejection_count_recent > 5 else IncidentClass.EXECUTION_FAILURE,
            severity=severity,
            message=f"Risk engine rejected order for {symbol}",
            context={
                "order_id": order_id,
                "symbol": symbol,
                "reason": reason,
                "recent_rejection_count": rejection_count_recent,
            },
        )
    
    def log_broker_error(
        self,
        broker: str,
        error_type: str,
        error_message: str,
        error_rate_pct: float,
    ) -> Incident:
        """Log broker error, escalating if rate spike detected."""
        severity = (
            IncidentSeverity.CRITICAL if error_rate_pct > 50
            else IncidentSeverity.HIGH if error_rate_pct > 20
            else IncidentSeverity.MEDIUM
        )
        return self.log_incident(
            incident_class=IncidentClass.BROKER_ERROR_RATE_SPIKE if error_rate_pct > 10 else IncidentClass.EXECUTION_FAILURE,
            severity=severity,
            message=f"Broker error from {broker}: {error_type}",
            context={
                "broker": broker,
                "error_type": error_type,
                "error_message": error_message,
                "error_rate_pct": error_rate_pct,
            },
        )
    
    def log_numerical_instability(
        self,
        component: str,
        field: str,
        value: Any,
    ) -> Incident:
        """Log model numerical instability (NaN/Inf)."""
        return self.log_incident(
            incident_class=IncidentClass.MODEL_NUMERICAL_INSTABILITY,
            severity=IncidentSeverity.MEDIUM,
            message=f"Numerical instability in {component}.{field}",
            context={
                "component": component,
                "field": field,
                "value": str(value),
            },
        )
    
    def get_recent_incidents(
        self,
        limit: int = 100,
        severity_filter: IncidentSeverity | None = None,
        class_filter: IncidentClass | None = None,
    ) -> list[Incident]:
        """Get recent incidents with optional filtering."""
        filtered = self.incidents
        
        if severity_filter:
            filtered = [i for i in filtered if i.severity == severity_filter]
        
        if class_filter:
            filtered = [i for i in filtered if i.incident_class == class_filter]
        
        return filtered[-limit:]
    
    def get_incident_counts(self) -> dict[str, int]:
        """Get counts by incident class."""
        counts: dict[str, int] = {}
        for incident in self.incidents:
            key = incident.incident_class.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    def get_severity_counts(self) -> dict[str, int]:
        """Get counts by severity."""
        counts: dict[str, int] = {}
        for incident in self.incidents:
            key = incident.severity.value
            counts[key] = counts.get(key, 0) + 1
        return counts


_incident_logger: IncidentLogger | None = None


def get_incident_logger() -> IncidentLogger:
    """Get or create the global incident logger singleton."""
    global _incident_logger
    if _incident_logger is None:
        _incident_logger = IncidentLogger()
    return _incident_logger
