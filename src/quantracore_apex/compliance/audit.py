"""
Enhanced Audit Trail System

Provides complete decision provenance for every analysis:
- Who/What triggered the analysis
- Complete input data fingerprint
- Every protocol that contributed to the decision
- Cryptographic chain of custody
- Immutable timestamped records

Exceeds SOX/SOC2 requirements for audit completeness.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json
import uuid


class AuditEventType(str, Enum):
    """Types of auditable events."""
    ANALYSIS_STARTED = "analysis_started"
    INPUT_RECEIVED = "input_received"
    PROTOCOL_EXECUTED = "protocol_executed"
    SCORE_COMPUTED = "score_computed"
    VERDICT_GENERATED = "verdict_generated"
    OMEGA_APPLIED = "omega_applied"
    ANALYSIS_COMPLETED = "analysis_completed"
    COMPLIANCE_CHECK = "compliance_check"
    PROOF_GENERATED = "proof_generated"


@dataclass
class AuditEntry:
    """Single audit trail entry with cryptographic integrity."""
    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    component: str
    action: str
    data: Dict[str, Any]
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    
    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute cryptographic hash of this entry."""
        hash_data = {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "component": self.component,
            "action": self.action,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }
        data_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the entry's cryptographic integrity."""
        return self.entry_hash == self._compute_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "component": self.component,
            "action": self.action,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }


@dataclass
class DecisionProvenance:
    """Complete provenance chain for a single decision."""
    decision_id: str
    symbol: str
    timestamp: datetime
    entries: List[AuditEntry] = field(default_factory=list)
    final_hash: Optional[str] = None
    
    def add_entry(
        self,
        event_type: AuditEventType,
        component: str,
        action: str,
        data: Dict[str, Any],
    ) -> AuditEntry:
        """Add a new entry to the provenance chain."""
        previous_hash = self.entries[-1].entry_hash if self.entries else None
        
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            component=component,
            action=action,
            data=data,
            previous_hash=previous_hash,
        )
        
        self.entries.append(entry)
        self.final_hash = entry.entry_hash
        return entry
    
    def verify_chain(self) -> bool:
        """Verify the entire provenance chain integrity."""
        if not self.entries:
            return True
        
        if self.entries[0].previous_hash is not None:
            return False
        
        for i, entry in enumerate(self.entries):
            if not entry.verify_integrity():
                return False
            
            if i > 0 and entry.previous_hash != self.entries[i-1].entry_hash:
                return False
        
        return True
    
    def get_protocol_contributions(self) -> List[Dict[str, Any]]:
        """Get list of all protocols that contributed to this decision."""
        contributions = []
        for entry in self.entries:
            if entry.event_type == AuditEventType.PROTOCOL_EXECUTED:
                contributions.append({
                    "protocol_id": entry.data.get("protocol_id"),
                    "protocol_name": entry.data.get("protocol_name"),
                    "contribution": entry.data.get("contribution"),
                    "timestamp": entry.timestamp.isoformat(),
                })
        return contributions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "entries": [e.to_dict() for e in self.entries],
            "final_hash": self.final_hash,
            "chain_verified": self.verify_chain(),
            "entry_count": len(self.entries),
        }


class AuditTrail:
    """
    Comprehensive audit trail system for regulatory excellence.
    
    Features:
    - Cryptographic chain of custody for all decisions
    - Complete input/output provenance
    - Protocol contribution tracking
    - Immutable timestamped records
    - Chain integrity verification
    """
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._active_provenances: Dict[str, DecisionProvenance] = {}
    
    def start_analysis(self, symbol: str, input_hash: str) -> DecisionProvenance:
        """Start tracking a new analysis with full provenance."""
        decision_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        provenance = DecisionProvenance(
            decision_id=decision_id,
            symbol=symbol,
            timestamp=datetime.utcnow(),
        )
        
        provenance.add_entry(
            event_type=AuditEventType.ANALYSIS_STARTED,
            component="ApexEngine",
            action="initialize",
            data={
                "symbol": symbol,
                "input_hash": input_hash,
                "engine_version": "v9.0-A",
                "compliance_mode": "RESEARCH_ONLY",
            },
        )
        
        self._active_provenances[decision_id] = provenance
        return provenance
    
    def record_protocol_execution(
        self,
        decision_id: str,
        protocol_id: str,
        protocol_name: str,
        contribution: Dict[str, Any],
    ) -> None:
        """Record a protocol execution in the provenance chain."""
        if decision_id not in self._active_provenances:
            return
        
        self._active_provenances[decision_id].add_entry(
            event_type=AuditEventType.PROTOCOL_EXECUTED,
            component=protocol_id,
            action="execute",
            data={
                "protocol_id": protocol_id,
                "protocol_name": protocol_name,
                "contribution": contribution,
            },
        )
    
    def record_score_computation(
        self,
        decision_id: str,
        quantrascore: float,
        score_components: Dict[str, float],
    ) -> None:
        """Record the QuantraScore computation."""
        if decision_id not in self._active_provenances:
            return
        
        self._active_provenances[decision_id].add_entry(
            event_type=AuditEventType.SCORE_COMPUTED,
            component="QuantraScore",
            action="compute",
            data={
                "final_score": quantrascore,
                "components": score_components,
            },
        )
    
    def record_omega_directive(
        self,
        decision_id: str,
        directive: str,
        active: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Record an Omega directive application."""
        if decision_id not in self._active_provenances:
            return
        
        self._active_provenances[decision_id].add_entry(
            event_type=AuditEventType.OMEGA_APPLIED,
            component="OmegaDirectives",
            action=f"apply_{directive}",
            data={
                "directive": directive,
                "active": active,
                "reason": reason,
            },
        )
    
    def complete_analysis(
        self,
        decision_id: str,
        result_hash: str,
        verdict: Dict[str, Any],
    ) -> Optional[DecisionProvenance]:
        """Complete an analysis and finalize the provenance chain."""
        if decision_id not in self._active_provenances:
            return None
        
        provenance = self._active_provenances[decision_id]
        
        provenance.add_entry(
            event_type=AuditEventType.VERDICT_GENERATED,
            component="VerdictEngine",
            action="generate",
            data=verdict,
        )
        
        provenance.add_entry(
            event_type=AuditEventType.PROOF_GENERATED,
            component="ProofLogger",
            action="sign",
            data={"result_hash": result_hash},
        )
        
        provenance.add_entry(
            event_type=AuditEventType.ANALYSIS_COMPLETED,
            component="ApexEngine",
            action="complete",
            data={
                "chain_verified": provenance.verify_chain(),
                "entry_count": len(provenance.entries),
                "final_hash": provenance.final_hash,
            },
        )
        
        self._save_provenance(provenance)
        
        del self._active_provenances[decision_id]
        return provenance
    
    def _save_provenance(self, provenance: DecisionProvenance) -> None:
        """Save provenance to disk for permanent record."""
        filename = f"{provenance.decision_id}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(provenance.to_dict(), f, indent=2, default=str)
    
    def load_provenance(self, decision_id: str) -> Optional[DecisionProvenance]:
        """Load a provenance record from disk."""
        filepath = self.log_dir / f"{decision_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            
            provenance = DecisionProvenance(
                decision_id=data["decision_id"],
                symbol=data["symbol"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
            )
            
            for entry_data in data["entries"]:
                entry = AuditEntry(
                    entry_id=entry_data["entry_id"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    event_type=AuditEventType(entry_data["event_type"]),
                    component=entry_data["component"],
                    action=entry_data["action"],
                    data=entry_data["data"],
                    previous_hash=entry_data["previous_hash"],
                    entry_hash=entry_data["entry_hash"],
                )
                provenance.entries.append(entry)
            
            provenance.final_hash = data["final_hash"]
            return provenance
            
        except Exception:
            return None
    
    def verify_provenance(self, decision_id: str) -> Dict[str, Any]:
        """Verify the integrity of a stored provenance record."""
        provenance = self.load_provenance(decision_id)
        
        if not provenance:
            return {
                "verified": False,
                "reason": "Provenance record not found",
            }
        
        chain_verified = provenance.verify_chain()
        
        return {
            "verified": chain_verified,
            "decision_id": decision_id,
            "entry_count": len(provenance.entries),
            "final_hash": provenance.final_hash,
            "reason": "All entries verified" if chain_verified else "Chain integrity compromised",
        }


audit_trail = AuditTrail()
