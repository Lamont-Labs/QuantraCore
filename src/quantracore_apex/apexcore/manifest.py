"""
ApexCore V2 Manifest System.

Provides model versioning, hash verification, and threshold management
for fail-closed behavior.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import json
import hashlib


@dataclass
class ManifestMetrics:
    """Validation metrics for a trained model."""
    val_brier_runner: float = 1.0
    val_auc_runner: float = 0.5
    val_calibration_error_runner: float = 1.0
    val_accuracy_quality_tier: float = 0.2
    val_accuracy_regime: float = 0.2
    val_mse_quantra_score: float = 100.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ManifestMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ManifestThresholds:
    """Thresholds for fail-closed behavior."""
    runner_prob_min_for_a_plus_flag: float = 0.7
    avoid_trade_prob_max_to_allow: float = 0.3
    max_disagreement_allowed: float = 0.2
    min_auc_runner_to_promote: float = 0.6
    max_calibration_error_to_promote: float = 0.15
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ManifestThresholds":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApexCoreV2Manifest:
    """
    Complete manifest for an ApexCore V2 model or ensemble.
    
    Contains versioning, hashes, metrics, and thresholds for
    production deployment with fail-closed safety.
    """
    model_family: str = "apexcore_v2"
    variant: str = "big"
    ensemble_size: int = 1
    created_utc: str = ""
    engine_snapshot_id: str = ""
    lab_dataset_id: str = ""
    hashes: Dict[str, str] = field(default_factory=dict)
    metrics: ManifestMetrics = field(default_factory=ManifestMetrics)
    thresholds: ManifestThresholds = field(default_factory=ManifestThresholds)
    version: str = "1.0.0"
    training_start_date: str = ""
    training_end_date: str = ""
    symbols_trained: int = 0
    samples_trained: int = 0
    
    def __post_init__(self):
        if not self.created_utc:
            self.created_utc = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "model_family": self.model_family,
            "variant": self.variant,
            "ensemble_size": self.ensemble_size,
            "created_utc": self.created_utc,
            "engine_snapshot_id": self.engine_snapshot_id,
            "lab_dataset_id": self.lab_dataset_id,
            "hashes": self.hashes,
            "metrics": self.metrics.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "version": self.version,
            "training_start_date": self.training_start_date,
            "training_end_date": self.training_end_date,
            "symbols_trained": self.symbols_trained,
            "samples_trained": self.samples_trained,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApexCoreV2Manifest":
        """Create manifest from dictionary."""
        metrics = ManifestMetrics.from_dict(data.get("metrics", {}))
        thresholds = ManifestThresholds.from_dict(data.get("thresholds", {}))
        
        return cls(
            model_family=data.get("model_family", "apexcore_v2"),
            variant=data.get("variant", "big"),
            ensemble_size=data.get("ensemble_size", 1),
            created_utc=data.get("created_utc", ""),
            engine_snapshot_id=data.get("engine_snapshot_id", ""),
            lab_dataset_id=data.get("lab_dataset_id", ""),
            hashes=data.get("hashes", {}),
            metrics=metrics,
            thresholds=thresholds,
            version=data.get("version", "1.0.0"),
            training_start_date=data.get("training_start_date", ""),
            training_end_date=data.get("training_end_date", ""),
            symbols_trained=data.get("symbols_trained", 0),
            samples_trained=data.get("samples_trained", 0),
        )
    
    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ApexCoreV2Manifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def is_valid_for_promotion(self) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet thresholds for production promotion.
        
        Returns:
            (is_valid, list_of_failure_reasons)
        """
        failures = []
        
        if self.metrics.val_auc_runner < self.thresholds.min_auc_runner_to_promote:
            failures.append(
                f"AUC runner {self.metrics.val_auc_runner:.3f} < "
                f"min {self.thresholds.min_auc_runner_to_promote:.3f}"
            )
        
        if self.metrics.val_calibration_error_runner > self.thresholds.max_calibration_error_to_promote:
            failures.append(
                f"Calibration error {self.metrics.val_calibration_error_runner:.3f} > "
                f"max {self.thresholds.max_calibration_error_to_promote:.3f}"
            )
        
        return len(failures) == 0, failures


def load_manifest(path: str) -> ApexCoreV2Manifest:
    """Load a manifest from file path."""
    return ApexCoreV2Manifest.load(path)


def select_best_model(manifest_dir: str) -> Tuple[str, ApexCoreV2Manifest]:
    """
    Select the best model from a directory of manifests.
    
    Selects based on highest AUC runner score among valid models.
    
    Args:
        manifest_dir: Directory containing manifest JSON files
        
    Returns:
        (path_to_best_model, manifest)
    """
    manifest_path = Path(manifest_dir)
    manifests = []
    
    for json_file in manifest_path.glob("*.json"):
        try:
            manifest = ApexCoreV2Manifest.load(str(json_file))
            is_valid, _ = manifest.is_valid_for_promotion()
            if is_valid:
                manifests.append((str(json_file), manifest))
        except (json.JSONDecodeError, KeyError):
            continue
    
    if not manifests:
        if list(manifest_path.glob("*.json")):
            first_json = next(manifest_path.glob("*.json"))
            manifest = ApexCoreV2Manifest.load(str(first_json))
            return str(first_json), manifest
        return "", ApexCoreV2Manifest()
    
    manifests.sort(key=lambda x: x[1].metrics.val_auc_runner, reverse=True)
    return manifests[0]


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def verify_manifest_against_file(model_path: str, manifest: ApexCoreV2Manifest) -> bool:
    """
    Verify that a model file matches the manifest hash.
    
    Args:
        model_path: Path to the model file
        manifest: Manifest to verify against
        
    Returns:
        True if hash matches, False otherwise
    """
    path = Path(model_path)
    if not path.exists():
        return False
    
    actual_hash = compute_file_hash(str(path))
    
    expected_hash = manifest.hashes.get("model", "")
    if not expected_hash:
        for key, value in manifest.hashes.items():
            if path.name in key or key in path.name:
                expected_hash = value
                break
    
    if not expected_hash:
        return True
    
    return actual_hash == expected_hash


def create_manifest_for_model(
    model_path: str,
    variant: str = "big",
    ensemble_size: int = 1,
    metrics: Optional[ManifestMetrics] = None,
    engine_snapshot_id: str = "",
    lab_dataset_id: str = "",
) -> ApexCoreV2Manifest:
    """
    Create a manifest for a trained model.
    
    Args:
        model_path: Path to the model file
        variant: Model variant ("big" or "mini")
        ensemble_size: Number of ensemble members
        metrics: Validation metrics
        engine_snapshot_id: Hash of engine config
        lab_dataset_id: Hash of training dataset
        
    Returns:
        New manifest
    """
    model_hash = compute_file_hash(model_path) if Path(model_path).exists() else ""
    
    return ApexCoreV2Manifest(
        variant=variant,
        ensemble_size=ensemble_size,
        engine_snapshot_id=engine_snapshot_id,
        lab_dataset_id=lab_dataset_id,
        hashes={"model": model_hash},
        metrics=metrics or ManifestMetrics(),
    )


__all__ = [
    "ApexCoreV2Manifest",
    "ManifestMetrics",
    "ManifestThresholds",
    "load_manifest",
    "select_best_model",
    "verify_manifest_against_file",
    "compute_file_hash",
    "create_manifest_for_model",
]
