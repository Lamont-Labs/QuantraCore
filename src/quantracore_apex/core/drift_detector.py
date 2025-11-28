"""
Drift Detection Framework for v9.0-A
Monitors statistical properties and detects distribution shifts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


class DriftMode(str, Enum):
    NORMAL = "normal"
    DRIFT_GUARDED = "drift_guarded"


class DriftSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    SEVERE = "severe"


@dataclass
class DriftBaseline:
    """Baseline statistics for drift detection."""
    metric_name: str
    mean: float
    std_dev: float
    quantiles: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
    last_updated: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "quantiles": self.quantiles,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftBaseline":
        return cls(
            metric_name=data["metric_name"],
            mean=data["mean"],
            std_dev=data["std_dev"],
            quantiles=data.get("quantiles", {}),
            sample_count=data.get("sample_count", 0),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class DriftEvent:
    """Record of a drift detection event."""
    timestamp: str
    metric_name: str
    severity: DriftSeverity
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "current_value": self.current_value,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "z_score": self.z_score,
            "details": self.details,
        }


class DriftDetector:
    """
    Drift detection framework for QuantraCore Apex.
    
    Monitors key metrics and detects distribution shifts relative to baselines.
    """
    
    MILD_DRIFT_Z_THRESHOLD = 2.0
    SEVERE_DRIFT_Z_THRESHOLD = 3.0
    
    def __init__(self, baselines_dir: Optional[str] = None):
        self.baselines: Dict[str, DriftBaseline] = {}
        self.current_mode = DriftMode.NORMAL
        self.drift_events: List[DriftEvent] = []
        self.rolling_values: Dict[str, List[float]] = {}
        self.baselines_dir = Path(baselines_dir) if baselines_dir else Path("provenance/drift_baselines")
    
    def load_baselines(self, baselines_dir: Optional[str] = None) -> int:
        """Load baseline files from directory. Returns count loaded."""
        dir_path = Path(baselines_dir) if baselines_dir else self.baselines_dir
        
        if not dir_path.exists():
            logger.warning(f"Baselines directory not found: {dir_path}")
            return 0
        
        loaded = 0
        for baseline_file in dir_path.glob("*.json"):
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
                baseline = DriftBaseline.from_dict(data)
                self.baselines[baseline.metric_name] = baseline
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load baseline {baseline_file}: {e}")
        
        logger.info(f"Loaded {loaded} drift baselines")
        return loaded
    
    def save_baseline(self, baseline: DriftBaseline):
        """Save a baseline to disk."""
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.baselines_dir / f"{baseline.metric_name}.json"
        
        baseline.last_updated = datetime.utcnow().isoformat()
        
        with open(filepath, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2)
        
        self.baselines[baseline.metric_name] = baseline
        logger.info(f"Saved baseline: {baseline.metric_name}")
    
    def update_rolling(self, metric_name: str, value: float, max_window: int = 100):
        """Update rolling values for a metric."""
        if metric_name not in self.rolling_values:
            self.rolling_values[metric_name] = []
        
        self.rolling_values[metric_name].append(value)
        
        if len(self.rolling_values[metric_name]) > max_window:
            self.rolling_values[metric_name] = self.rolling_values[metric_name][-max_window:]
    
    def compute_baseline_from_rolling(self, metric_name: str) -> Optional[DriftBaseline]:
        """Compute a baseline from current rolling values."""
        if metric_name not in self.rolling_values:
            return None
        
        values = self.rolling_values[metric_name]
        if len(values) < 10:
            return None
        
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.001
        
        sorted_values = sorted(values)
        quantiles = {
            "p25": sorted_values[int(n * 0.25)],
            "p50": sorted_values[int(n * 0.50)],
            "p75": sorted_values[int(n * 0.75)],
        }
        
        return DriftBaseline(
            metric_name=metric_name,
            mean=mean,
            std_dev=std_dev,
            quantiles=quantiles,
            sample_count=n,
            last_updated=datetime.utcnow().isoformat(),
        )
    
    def check_drift(self, metric_name: str, current_value: float) -> DriftEvent:
        """
        Check if a metric has drifted from its baseline.
        
        Returns a DriftEvent with severity classification.
        """
        if metric_name not in self.baselines:
            return DriftEvent(
                timestamp=datetime.utcnow().isoformat(),
                metric_name=metric_name,
                severity=DriftSeverity.NONE,
                current_value=current_value,
                baseline_mean=0.0,
                baseline_std=0.0,
                z_score=0.0,
                details={"reason": "no_baseline"}
            )
        
        baseline = self.baselines[metric_name]
        
        std = baseline.std_dev if baseline.std_dev > 0 else 0.001
        z_score = abs(current_value - baseline.mean) / std
        
        if z_score >= self.SEVERE_DRIFT_Z_THRESHOLD:
            severity = DriftSeverity.SEVERE
        elif z_score >= self.MILD_DRIFT_Z_THRESHOLD:
            severity = DriftSeverity.MILD
        else:
            severity = DriftSeverity.NONE
        
        event = DriftEvent(
            timestamp=datetime.utcnow().isoformat(),
            metric_name=metric_name,
            severity=severity,
            current_value=current_value,
            baseline_mean=baseline.mean,
            baseline_std=baseline.std_dev,
            z_score=z_score,
            details={
                "sample_count": baseline.sample_count,
                "baseline_updated": baseline.last_updated,
            }
        )
        
        if severity != DriftSeverity.NONE:
            self.drift_events.append(event)
            logger.warning(
                f"Drift detected: {metric_name} z={z_score:.2f} "
                f"current={current_value:.2f} baseline={baseline.mean:.2f}"
            )
        
        return event
    
    def check_universe_metrics(
        self,
        quantrascore_values: List[float],
        regime_counts: Dict[str, int],
        runner_primed_count: int,
        runner_idle_count: int,
        consistency_fail_rate: float
    ) -> Dict[str, DriftEvent]:
        """
        Check multiple universe metrics for drift.
        
        Returns dict of metric name -> DriftEvent.
        """
        results = {}
        
        if quantrascore_values:
            mean_score = sum(quantrascore_values) / len(quantrascore_values)
            self.update_rolling("quantrascore_mean", mean_score)
            results["quantrascore_mean"] = self.check_drift("quantrascore_mean", mean_score)
        
        total_regime = sum(regime_counts.values())
        if total_regime > 0:
            trending_ratio = regime_counts.get("trending_up", 0) / total_regime
            self.update_rolling("trending_up_ratio", trending_ratio)
            results["trending_up_ratio"] = self.check_drift("trending_up_ratio", trending_ratio)
        
        total_runner = runner_primed_count + runner_idle_count
        if total_runner > 0:
            primed_ratio = runner_primed_count / total_runner
            self.update_rolling("runner_primed_ratio", primed_ratio)
            results["runner_primed_ratio"] = self.check_drift("runner_primed_ratio", primed_ratio)
        
        self.update_rolling("consistency_fail_rate", consistency_fail_rate)
        results["consistency_fail_rate"] = self.check_drift("consistency_fail_rate", consistency_fail_rate)
        
        return results
    
    def evaluate_mode(self) -> DriftMode:
        """
        Evaluate current drift events and determine appropriate mode.
        """
        recent_events = self.drift_events[-20:] if len(self.drift_events) >= 20 else self.drift_events
        
        severe_count = sum(1 for e in recent_events if e.severity == DriftSeverity.SEVERE)
        mild_count = sum(1 for e in recent_events if e.severity == DriftSeverity.MILD)
        
        if severe_count >= 3 or (severe_count >= 1 and mild_count >= 5):
            self.current_mode = DriftMode.DRIFT_GUARDED
        elif severe_count == 0 and mild_count <= 2:
            self.current_mode = DriftMode.NORMAL
        
        return self.current_mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get current drift detection status."""
        recent_events = self.drift_events[-10:]
        
        return {
            "mode": self.current_mode.value,
            "baselines_loaded": len(self.baselines),
            "total_drift_events": len(self.drift_events),
            "recent_events": [e.to_dict() for e in recent_events],
            "rolling_metrics": {
                name: len(values) for name, values in self.rolling_values.items()
            }
        }
    
    def reset(self):
        """Reset drift detection state (keeps baselines)."""
        self.current_mode = DriftMode.NORMAL
        self.drift_events = []
        self.rolling_values = {}
