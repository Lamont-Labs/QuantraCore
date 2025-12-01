"""
Investor Due Diligence Logging System.

Comprehensive logging infrastructure for institutional-grade due diligence:
- Compliance attestations and control checks
- Enhanced incident lifecycle tracking
- Policy manifest with version control
- Broker reconciliation records
- Consent and document access logs

All data is stored in investor_logs/ for audit and regulatory review.
"""

import json
import hashlib
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

INVESTOR_LOGS_DIR = Path("investor_logs")
COMPLIANCE_DIR = INVESTOR_LOGS_DIR / "compliance"
AUDIT_DIR = INVESTOR_LOGS_DIR / "audit"
LEGAL_DIR = INVESTOR_LOGS_DIR / "legal"


class AttestationType(str, Enum):
    DAILY_RECONCILIATION = "DAILY_RECONCILIATION"
    RISK_LIMIT_COMPLIANCE = "RISK_LIMIT_COMPLIANCE"
    POLICY_ACKNOWLEDGMENT = "POLICY_ACKNOWLEDGMENT"
    CONTROL_EFFECTIVENESS = "CONTROL_EFFECTIVENESS"
    INCIDENT_REVIEW = "INCIDENT_REVIEW"
    MODEL_VALIDATION = "MODEL_VALIDATION"
    DATA_QUALITY = "DATA_QUALITY"


class AttestationStatus(str, Enum):
    PENDING = "PENDING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    WAIVED = "WAIVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class IncidentLifecycleStatus(str, Enum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    ROOT_CAUSE_IDENTIFIED = "ROOT_CAUSE_IDENTIFIED"
    REMEDIATION_IN_PROGRESS = "REMEDIATION_IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class ReconciliationStatus(str, Enum):
    MATCHED = "MATCHED"
    UNMATCHED_INTERNAL = "UNMATCHED_INTERNAL"
    UNMATCHED_BROKER = "UNMATCHED_BROKER"
    DISCREPANCY = "DISCREPANCY"
    PENDING_REVIEW = "PENDING_REVIEW"


class ConsentType(str, Enum):
    SMS_ALERTS = "SMS_ALERTS"
    EMAIL_COMMUNICATIONS = "EMAIL_COMMUNICATIONS"
    DATA_SHARING = "DATA_SHARING"
    MARKETING = "MARKETING"
    TERMS_OF_SERVICE = "TERMS_OF_SERVICE"
    PRIVACY_POLICY = "PRIVACY_POLICY"


class DocumentAccessAction(str, Enum):
    VIEW = "VIEW"
    DOWNLOAD = "DOWNLOAD"
    SHARE = "SHARE"
    PRINT = "PRINT"


@dataclass
class ComplianceAttestation:
    """Record of a compliance attestation or control check."""
    attestation_id: str
    attestation_type: AttestationType
    control_id: str
    control_name: str
    status: AttestationStatus
    attestor: str
    attestor_role: str
    timestamp: str
    evidence_path: Optional[str] = None
    notes: str = ""
    exceptions: List[str] = field(default_factory=list)
    next_review_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "attestation_type": self.attestation_type.value if isinstance(self.attestation_type, Enum) else self.attestation_type,
            "control_id": self.control_id,
            "control_name": self.control_name,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "attestor": self.attestor,
            "attestor_role": self.attestor_role,
            "timestamp": self.timestamp,
            "evidence_path": self.evidence_path,
            "notes": self.notes,
            "exceptions": self.exceptions,
            "next_review_date": self.next_review_date,
        }


@dataclass
class IncidentLifecycle:
    """Enhanced incident record with full lifecycle tracking."""
    incident_id: str
    original_incident_class: str
    severity: str
    title: str
    description: str
    status: IncidentLifecycleStatus
    created_at: str
    updated_at: str
    root_cause: Optional[str] = None
    root_cause_category: Optional[str] = None
    impact_assessment: Optional[str] = None
    remediation_steps: List[str] = field(default_factory=list)
    remediation_owner: Optional[str] = None
    remediation_deadline: Optional[str] = None
    prevention_measures: List[str] = field(default_factory=list)
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    approval_status: Optional[str] = None
    approved_by: Optional[str] = None
    lessons_learned: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "original_incident_class": self.original_incident_class,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "root_cause": self.root_cause,
            "root_cause_category": self.root_cause_category,
            "impact_assessment": self.impact_assessment,
            "remediation_steps": self.remediation_steps,
            "remediation_owner": self.remediation_owner,
            "remediation_deadline": self.remediation_deadline,
            "prevention_measures": self.prevention_measures,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "approval_status": self.approval_status,
            "approved_by": self.approved_by,
            "lessons_learned": self.lessons_learned,
        }


@dataclass
class PolicyManifestEntry:
    """Record of a policy document with version tracking."""
    policy_id: str
    policy_name: str
    version: str
    effective_date: str
    review_date: str
    next_review_date: str
    approver: str
    approver_role: str
    document_path: str
    checksum: str
    status: str
    category: str
    supersedes: Optional[str] = None
    changelog: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReconciliationRecord:
    """Record of trade reconciliation between internal logs and broker."""
    reconciliation_id: str
    reconciliation_date: str
    internal_trade_id: str
    broker_order_id: Optional[str]
    symbol: str
    side: str
    internal_quantity: float
    broker_quantity: Optional[float]
    internal_price: float
    broker_price: Optional[float]
    price_variance: Optional[float]
    quantity_variance: Optional[float]
    internal_timestamp: str
    broker_timestamp: Optional[str]
    time_variance_seconds: Optional[float]
    status: ReconciliationStatus
    discrepancy_type: Optional[str] = None
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reconciliation_id": self.reconciliation_id,
            "reconciliation_date": self.reconciliation_date,
            "internal_trade_id": self.internal_trade_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "internal_quantity": self.internal_quantity,
            "broker_quantity": self.broker_quantity,
            "internal_price": self.internal_price,
            "broker_price": self.broker_price,
            "price_variance": self.price_variance,
            "quantity_variance": self.quantity_variance,
            "internal_timestamp": self.internal_timestamp,
            "broker_timestamp": self.broker_timestamp,
            "time_variance_seconds": self.time_variance_seconds,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "discrepancy_type": self.discrepancy_type,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "notes": self.notes,
        }


@dataclass
class ConsentRecord:
    """Record of user consent for communications and data use."""
    consent_id: str
    consent_type: ConsentType
    user_id: str
    user_identifier: str
    granted: bool
    granted_at: str
    source: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    revoked_at: Optional[str] = None
    revocation_reason: Optional[str] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "consent_type": self.consent_type.value if isinstance(self.consent_type, Enum) else self.consent_type,
            "user_id": self.user_id,
            "user_identifier": self.user_identifier,
            "granted": self.granted,
            "granted_at": self.granted_at,
            "source": self.source,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "revoked_at": self.revoked_at,
            "revocation_reason": self.revocation_reason,
            "version": self.version,
        }


@dataclass
class DocumentAccessLog:
    """Record of document access for audit trail."""
    access_id: str
    document_id: str
    document_name: str
    document_category: str
    actor_id: str
    actor_name: str
    actor_role: str
    action: DocumentAccessAction
    timestamp: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    failure_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_id": self.access_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_category": self.document_category,
            "actor_id": self.actor_id,
            "actor_name": self.actor_name,
            "actor_role": self.actor_role,
            "action": self.action.value if isinstance(self.action, Enum) else self.action,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


class DueDiligenceLogger:
    """
    Central logging system for all investor due diligence data.
    
    Manages:
    - Compliance attestations
    - Incident lifecycle tracking
    - Policy manifests
    - Broker reconciliation
    - Consent records
    - Document access logs
    """
    
    def __init__(self):
        self._ensure_directories()
        self._counters = {
            "att": 0,
            "inc": 0,
            "rec": 0,
            "con": 0,
            "acc": 0,
            "pol": 0,
        }
    
    def _ensure_directories(self):
        """Create all required logging directories."""
        dirs = [
            COMPLIANCE_DIR,
            AUDIT_DIR,
            LEGAL_DIR,
            COMPLIANCE_DIR / "attestations",
            COMPLIANCE_DIR / "incidents",
            COMPLIANCE_DIR / "policies",
            AUDIT_DIR / "reconciliation",
            LEGAL_DIR / "consents",
            LEGAL_DIR / "access_logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with timestamp."""
        self._counters[prefix.lower()] += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix.upper()}-{ts}-{self._counters[prefix.lower()]:04d}"
    
    def _get_date_str(self) -> str:
        """Get current date string for file naming."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    
    def _append_jsonl(self, path: Path, record: Dict[str, Any]):
        """Append record to JSONL file."""
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    
    def log_attestation(
        self,
        attestation_type: AttestationType,
        control_id: str,
        control_name: str,
        status: AttestationStatus,
        attestor: str,
        attestor_role: str = "system",
        evidence_path: Optional[str] = None,
        notes: str = "",
        exceptions: Optional[List[str]] = None,
        next_review_date: Optional[str] = None,
    ) -> ComplianceAttestation:
        """
        Log a compliance attestation or control check.
        
        Args:
            attestation_type: Type of attestation
            control_id: Unique control identifier
            control_name: Human-readable control name
            status: Pass/Fail status
            attestor: Who performed the attestation
            attestor_role: Role of attestor
            evidence_path: Path to supporting evidence
            notes: Additional notes
            exceptions: List of exceptions noted
            next_review_date: When to review again
            
        Returns:
            ComplianceAttestation record
        """
        attestation = ComplianceAttestation(
            attestation_id=self._generate_id("ATT"),
            attestation_type=attestation_type,
            control_id=control_id,
            control_name=control_name,
            status=status,
            attestor=attestor,
            attestor_role=attestor_role,
            timestamp=datetime.now(timezone.utc).isoformat(),
            evidence_path=evidence_path,
            notes=notes,
            exceptions=exceptions or [],
            next_review_date=next_review_date,
        )
        
        log_file = COMPLIANCE_DIR / "attestations" / f"attestations_{self._get_date_str()}.jsonl"
        self._append_jsonl(log_file, attestation.to_dict())
        
        logger.info(f"Logged attestation {attestation.attestation_id}: {control_name} = {status.value}")
        return attestation
    
    def log_incident_lifecycle(
        self,
        incident_id: str,
        original_class: str,
        severity: str,
        title: str,
        description: str,
        status: IncidentLifecycleStatus = IncidentLifecycleStatus.OPEN,
        root_cause: Optional[str] = None,
        root_cause_category: Optional[str] = None,
        impact_assessment: Optional[str] = None,
        remediation_steps: Optional[List[str]] = None,
        remediation_owner: Optional[str] = None,
        prevention_measures: Optional[List[str]] = None,
        resolved_by: Optional[str] = None,
        lessons_learned: str = "",
    ) -> IncidentLifecycle:
        """
        Log or update incident lifecycle record with full tracking.
        
        Args:
            incident_id: Original incident ID from IncidentLogger
            original_class: Original incident classification
            severity: Incident severity
            title: Short incident title
            description: Full description
            status: Current lifecycle status
            root_cause: Identified root cause
            root_cause_category: Category (human_error, system, external, etc.)
            impact_assessment: Assessment of impact
            remediation_steps: Steps taken to fix
            remediation_owner: Person responsible for fix
            prevention_measures: Steps to prevent recurrence
            resolved_by: Who resolved it
            lessons_learned: Key learnings
            
        Returns:
            IncidentLifecycle record
        """
        now = datetime.now(timezone.utc).isoformat()
        
        incident = IncidentLifecycle(
            incident_id=incident_id,
            original_incident_class=original_class,
            severity=severity,
            title=title,
            description=description,
            status=status,
            created_at=now,
            updated_at=now,
            root_cause=root_cause,
            root_cause_category=root_cause_category,
            impact_assessment=impact_assessment,
            remediation_steps=remediation_steps or [],
            remediation_owner=remediation_owner,
            prevention_measures=prevention_measures or [],
            resolved_at=now if status == IncidentLifecycleStatus.RESOLVED else None,
            resolved_by=resolved_by,
            lessons_learned=lessons_learned,
        )
        
        log_file = COMPLIANCE_DIR / "incidents" / f"incidents_lifecycle_{self._get_date_str()}.jsonl"
        self._append_jsonl(log_file, incident.to_dict())
        
        logger.info(f"Logged incident lifecycle {incident_id}: {status.value}")
        return incident
    
    def log_policy_manifest(
        self,
        policy_name: str,
        version: str,
        document_path: str,
        approver: str,
        approver_role: str,
        category: str,
        effective_date: Optional[str] = None,
        review_date: Optional[str] = None,
        next_review_date: Optional[str] = None,
        supersedes: Optional[str] = None,
        changelog: str = "",
    ) -> PolicyManifestEntry:
        """
        Log policy document to manifest with version tracking.
        
        Args:
            policy_name: Name of the policy
            version: Version string (e.g., "1.0", "2.1")
            document_path: Path to policy document
            approver: Who approved the policy
            approver_role: Role of approver
            category: Policy category (trading, risk, compliance, etc.)
            effective_date: When policy takes effect
            review_date: When policy was last reviewed
            next_review_date: When policy should be reviewed
            supersedes: Previous policy ID this replaces
            changelog: Description of changes
            
        Returns:
            PolicyManifestEntry record
        """
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        
        checksum = self._calculate_file_checksum(document_path)
        
        policy = PolicyManifestEntry(
            policy_id=f"POL-{policy_name.upper().replace(' ', '_')}-{version.replace('.', '_')}",
            policy_name=policy_name,
            version=version,
            effective_date=effective_date or now_str,
            review_date=review_date or now_str,
            next_review_date=next_review_date or (now.replace(year=now.year + 1)).isoformat(),
            approver=approver,
            approver_role=approver_role,
            document_path=document_path,
            checksum=checksum,
            status="ACTIVE",
            category=category,
            supersedes=supersedes,
            changelog=changelog,
        )
        
        manifest_file = COMPLIANCE_DIR / "policies" / "policy_manifest.jsonl"
        self._append_jsonl(manifest_file, policy.to_dict())
        
        logger.info(f"Logged policy {policy.policy_id}: {policy_name} v{version}")
        return policy
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return "FILE_NOT_FOUND"
        except Exception as e:
            return f"ERROR:{str(e)}"
    
    def log_reconciliation(
        self,
        internal_trade_id: str,
        symbol: str,
        side: str,
        internal_quantity: float,
        internal_price: float,
        internal_timestamp: str,
        broker_order_id: Optional[str] = None,
        broker_quantity: Optional[float] = None,
        broker_price: Optional[float] = None,
        broker_timestamp: Optional[str] = None,
        status: Optional[ReconciliationStatus] = None,
        notes: str = "",
    ) -> ReconciliationRecord:
        """
        Log trade reconciliation between internal logs and broker confirms.
        
        Args:
            internal_trade_id: Trade ID from investor_logs
            symbol: Trading symbol
            side: BUY or SELL
            internal_quantity: Quantity from internal logs
            internal_price: Price from internal logs
            internal_timestamp: Timestamp from internal logs
            broker_order_id: Order ID from broker (Alpaca)
            broker_quantity: Quantity from broker
            broker_price: Price from broker
            broker_timestamp: Timestamp from broker
            status: Reconciliation status (auto-calculated if not provided)
            notes: Additional notes
            
        Returns:
            ReconciliationRecord
        """
        price_variance = None
        quantity_variance = None
        time_variance = None
        
        if broker_price is not None:
            price_variance = abs(internal_price - broker_price)
        
        if broker_quantity is not None:
            quantity_variance = abs(internal_quantity - broker_quantity)
        
        if broker_timestamp is not None:
            try:
                internal_dt = datetime.fromisoformat(internal_timestamp.replace('Z', '+00:00'))
                broker_dt = datetime.fromisoformat(broker_timestamp.replace('Z', '+00:00'))
                time_variance = abs((internal_dt - broker_dt).total_seconds())
            except:
                pass
        
        if status is None:
            if broker_order_id is None:
                status = ReconciliationStatus.UNMATCHED_BROKER
            elif price_variance is not None and price_variance > 0.01:
                status = ReconciliationStatus.DISCREPANCY
            elif quantity_variance is not None and quantity_variance > 0:
                status = ReconciliationStatus.DISCREPANCY
            else:
                status = ReconciliationStatus.MATCHED
        
        record = ReconciliationRecord(
            reconciliation_id=self._generate_id("REC"),
            reconciliation_date=self._get_date_str(),
            internal_trade_id=internal_trade_id,
            broker_order_id=broker_order_id,
            symbol=symbol,
            side=side,
            internal_quantity=internal_quantity,
            broker_quantity=broker_quantity,
            internal_price=internal_price,
            broker_price=broker_price,
            price_variance=price_variance,
            quantity_variance=quantity_variance,
            internal_timestamp=internal_timestamp,
            broker_timestamp=broker_timestamp,
            time_variance_seconds=time_variance,
            status=status,
            notes=notes,
        )
        
        log_file = AUDIT_DIR / "reconciliation" / f"reconciliation_{self._get_date_str()}.jsonl"
        self._append_jsonl(log_file, record.to_dict())
        
        logger.info(f"Logged reconciliation {record.reconciliation_id}: {internal_trade_id} = {status.value}")
        return record
    
    def log_consent(
        self,
        consent_type: ConsentType,
        user_id: str,
        user_identifier: str,
        granted: bool,
        source: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        version: str = "1.0",
    ) -> ConsentRecord:
        """
        Log user consent for communications or data use.
        
        Args:
            consent_type: Type of consent (SMS, email, etc.)
            user_id: Internal user ID
            user_identifier: Email or phone number
            granted: Whether consent was granted
            source: Where consent was obtained (web, sms, email)
            ip_address: User's IP address
            user_agent: User's browser/device
            version: Terms version
            
        Returns:
            ConsentRecord
        """
        record = ConsentRecord(
            consent_id=self._generate_id("CON"),
            consent_type=consent_type,
            user_id=user_id,
            user_identifier=user_identifier,
            granted=granted,
            granted_at=datetime.now(timezone.utc).isoformat(),
            source=source,
            ip_address=ip_address,
            user_agent=user_agent,
            version=version,
        )
        
        log_file = LEGAL_DIR / "consents" / "consents.jsonl"
        self._append_jsonl(log_file, record.to_dict())
        
        logger.info(f"Logged consent {record.consent_id}: {consent_type.value} = {granted}")
        return record
    
    def log_document_access(
        self,
        document_id: str,
        document_name: str,
        document_category: str,
        actor_id: str,
        actor_name: str,
        actor_role: str,
        action: DocumentAccessAction,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        failure_reason: Optional[str] = None,
    ) -> DocumentAccessLog:
        """
        Log document access for audit trail.
        
        Args:
            document_id: Unique document identifier
            document_name: Human-readable document name
            document_category: Category (formation, performance, risk, etc.)
            actor_id: Who accessed the document
            actor_name: Name of accessor
            actor_role: Role of accessor
            action: What action was taken
            ip_address: Accessor's IP
            user_agent: Accessor's browser/device
            success: Whether access succeeded
            failure_reason: Reason for failure if any
            
        Returns:
            DocumentAccessLog record
        """
        record = DocumentAccessLog(
            access_id=self._generate_id("ACC"),
            document_id=document_id,
            document_name=document_name,
            document_category=document_category,
            actor_id=actor_id,
            actor_name=actor_name,
            actor_role=actor_role,
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            failure_reason=failure_reason,
        )
        
        log_file = LEGAL_DIR / "access_logs" / f"access_{self._get_date_str()}.jsonl"
        self._append_jsonl(log_file, record.to_dict())
        
        logger.info(f"Logged document access {record.access_id}: {document_name} - {action.value}")
        return record
    
    def generate_daily_summary(self) -> Dict[str, Any]:
        """
        Generate a daily summary of all due diligence logging activity.
        
        Returns summary with counts and status of all log types.
        """
        date_str = self._get_date_str()
        
        summary = {
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "attestations": self._count_jsonl_records(
                COMPLIANCE_DIR / "attestations" / f"attestations_{date_str}.jsonl"
            ),
            "incidents": self._count_jsonl_records(
                COMPLIANCE_DIR / "incidents" / f"incidents_lifecycle_{date_str}.jsonl"
            ),
            "reconciliations": self._count_jsonl_records(
                AUDIT_DIR / "reconciliation" / f"reconciliation_{date_str}.jsonl"
            ),
            "document_accesses": self._count_jsonl_records(
                LEGAL_DIR / "access_logs" / f"access_{date_str}.jsonl"
            ),
            "consents_total": self._count_jsonl_records(
                LEGAL_DIR / "consents" / "consents.jsonl"
            ),
            "policies_total": self._count_jsonl_records(
                COMPLIANCE_DIR / "policies" / "policy_manifest.jsonl"
            ),
        }
        
        summary_file = INVESTOR_LOGS_DIR / "summaries" / f"dd_summary_{date_str}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _count_jsonl_records(self, path: Path) -> int:
        """Count records in a JSONL file."""
        try:
            with open(path) as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0


_logger_instance: Optional[DueDiligenceLogger] = None


def get_due_diligence_logger() -> DueDiligenceLogger:
    """Get singleton instance of DueDiligenceLogger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DueDiligenceLogger()
    return _logger_instance
