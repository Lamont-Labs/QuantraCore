"""
Protocol Manifest System for Deterministic Kernel Verification.

Ensures protocol execution order is deterministic and verifiable.
On startup, computes hash of protocol order manifest and compares to stored reference.
If mismatch → refuses to run engine until reconciled.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("config/protocol_manifest.yaml")
MANIFEST_HASH_PATH = Path("config/protocol_manifest.hash")


@dataclass
class ProtocolEntry:
    """Single protocol entry in the manifest."""
    protocol_id: str
    category: str
    execution_order: int
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ProtocolManifest:
    """Complete protocol manifest with execution order and hashing."""
    
    version: str
    engine_snapshot_id: str
    created_at: str
    protocols: list[ProtocolEntry]
    hash: str = ""
    
    TIER_PROTOCOLS = [f"T{i:02d}" for i in range(1, 81)]
    LEARNING_PROTOCOLS = [f"LP{i:02d}" for i in range(1, 26)]
    MONSTER_RUNNER_PROTOCOLS = [f"MR{i:02d}" for i in range(1, 6)]
    OMEGA_PROTOCOLS = [f"Ω{i}" for i in range(1, 6)]
    
    @classmethod
    def generate_default(cls) -> "ProtocolManifest":
        """Generate default manifest with all protocols in standard order."""
        protocols = []
        order = 0
        
        for proto_id in cls.TIER_PROTOCOLS:
            order += 1
            protocols.append(ProtocolEntry(
                protocol_id=proto_id,
                category="tier",
                execution_order=order,
                inputs=["ohlcv_bars", "regime_context", "prior_protocol_results"],
                outputs=["protocol_score", "flags", "trace_info"],
                failure_modes=["missing_bars", "nan_values", "insufficient_history"],
                assumptions=["bars_sorted_by_time", "no_future_data_leak"],
            ))
        
        for proto_id in cls.LEARNING_PROTOCOLS:
            order += 1
            protocols.append(ProtocolEntry(
                protocol_id=proto_id,
                category="learning",
                execution_order=order,
                inputs=["tier_outputs", "historical_labels"],
                outputs=["learning_signal", "confidence"],
                failure_modes=["missing_labels", "stale_model"],
                assumptions=["labels_from_past_only"],
            ))
        
        for proto_id in cls.MONSTER_RUNNER_PROTOCOLS:
            order += 1
            protocols.append(ProtocolEntry(
                protocol_id=proto_id,
                category="monster_runner",
                execution_order=order,
                inputs=["tier_outputs", "learning_outputs", "volatility_context"],
                outputs=["monster_runner_score", "runner_probability"],
                failure_modes=["insufficient_volatility_data"],
                assumptions=["volatility_regime_current"],
            ))
        
        for proto_id in cls.OMEGA_PROTOCOLS:
            order += 1
            protocols.append(ProtocolEntry(
                protocol_id=proto_id,
                category="omega",
                execution_order=order,
                inputs=["all_prior_outputs", "risk_context", "compliance_state"],
                outputs=["omega_override", "safety_flag"],
                failure_modes=["compliance_state_unknown"],
                assumptions=["omega_always_evaluated_last"],
            ))
        
        manifest = cls(
            version="9.0-A",
            engine_snapshot_id=f"apex-v9.0A-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
            created_at=datetime.now(timezone.utc).isoformat(),
            protocols=protocols,
        )
        manifest.hash = manifest.compute_hash()
        return manifest
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of protocol order and positions."""
        protocol_positions = [
            (p.protocol_id, p.execution_order, p.category)
            for p in self.protocols
        ]
        content = json.dumps({
            "version": self.version,
            "protocol_positions": protocol_positions,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "version": self.version,
            "engine_snapshot_id": self.engine_snapshot_id,
            "created_at": self.created_at,
            "hash": self.hash,
            "protocol_count": len(self.protocols),
            "protocols": [
                {
                    "id": p.protocol_id,
                    "category": p.category,
                    "order": p.execution_order,
                    "inputs": p.inputs,
                    "outputs": p.outputs,
                    "failure_modes": p.failure_modes,
                    "assumptions": p.assumptions,
                }
                for p in self.protocols
            ],
        }
    
    def save(self, path: Path | None = None) -> None:
        """Save manifest to YAML file."""
        import yaml
        target = path or MANIFEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        hash_path = target.with_suffix(".hash")
        with open(hash_path, "w") as f:
            f.write(self.hash)
    
    @classmethod
    def load(cls, path: Path | None = None) -> "ProtocolManifest":
        """Load manifest from YAML file."""
        import yaml
        target = path or MANIFEST_PATH
        if not target.exists():
            raise FileNotFoundError(f"Protocol manifest not found: {target}")
        
        with open(target) as f:
            data = yaml.safe_load(f)
        
        protocols = [
            ProtocolEntry(
                protocol_id=p["id"],
                category=p["category"],
                execution_order=p["order"],
                inputs=p.get("inputs", []),
                outputs=p.get("outputs", []),
                failure_modes=p.get("failure_modes", []),
                assumptions=p.get("assumptions", []),
            )
            for p in data.get("protocols", [])
        ]
        
        return cls(
            version=data["version"],
            engine_snapshot_id=data["engine_snapshot_id"],
            created_at=data["created_at"],
            protocols=protocols,
            hash=data.get("hash", ""),
        )


class ManifestValidationError(Exception):
    """Raised when manifest validation fails."""
    pass


@dataclass
class ManifestValidator:
    """Validates protocol manifest integrity on startup."""
    
    manifest: ProtocolManifest | None = None
    validated: bool = False
    validation_error: str | None = None
    
    def validate(self, manifest_path: Path | None = None) -> bool:
        """
        Validate manifest exists and hash matches.
        Returns True if valid, raises ManifestValidationError if invalid.
        """
        try:
            self.manifest = ProtocolManifest.load(manifest_path)
            
            computed_hash = self.manifest.compute_hash()
            if computed_hash != self.manifest.hash:
                self.validation_error = (
                    f"Manifest hash mismatch: stored={self.manifest.hash}, "
                    f"computed={computed_hash}. Protocol order may have changed."
                )
                raise ManifestValidationError(self.validation_error)
            
            expected_count = (
                len(ProtocolManifest.TIER_PROTOCOLS) +
                len(ProtocolManifest.LEARNING_PROTOCOLS) +
                len(ProtocolManifest.MONSTER_RUNNER_PROTOCOLS) +
                len(ProtocolManifest.OMEGA_PROTOCOLS)
            )
            if len(self.manifest.protocols) != expected_count:
                self.validation_error = (
                    f"Protocol count mismatch: expected={expected_count}, "
                    f"found={len(self.manifest.protocols)}"
                )
                raise ManifestValidationError(self.validation_error)
            
            self.validated = True
            return True
            
        except FileNotFoundError:
            self.validation_error = "Protocol manifest file not found"
            raise ManifestValidationError(self.validation_error)
    
    def ensure_manifest_exists(self) -> None:
        """Create default manifest if it doesn't exist."""
        if not MANIFEST_PATH.exists():
            manifest = ProtocolManifest.generate_default()
            manifest.save()
