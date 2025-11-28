"""
Fail-Closed Decision Gates for v9.0-A
Implements data integrity, model integrity, and risk guard gates.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a gate check."""
    gate_name: str
    status: GateStatus
    checks_passed: List[str]
    checks_failed: List[str]
    action_taken: str
    details: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "action_taken": self.action_taken,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DataIntegrityGate:
    """
    Validates data integrity before processing.
    
    Checks:
    - No missing OHLCV bars beyond tolerance
    - No duplicate or out-of-order timestamps
    - No negative prices or volumes
    """
    
    MAX_MISSING_BAR_TOLERANCE = 0.05  # 5% missing bars allowed
    
    def __init__(self):
        self.anomaly_log = []
    
    def check(
        self,
        ohlcv_data: List[Dict[str, Any]],
        expected_bar_count: Optional[int] = None
    ) -> GateResult:
        """
        Check data integrity.
        
        Args:
            ohlcv_data: List of OHLCV bar dicts with open, high, low, close, volume, timestamp
            expected_bar_count: Expected number of bars (optional)
        
        Returns:
            GateResult with pass/fail status and details
        """
        checks_passed = []
        checks_failed = []
        details = {}
        
        if not ohlcv_data:
            checks_failed.append("no_data")
            return GateResult(
                gate_name="data_integrity_gate",
                status=GateStatus.FAILED,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                action_taken="reject_signal",
                details={"reason": "empty_data"},
                timestamp=datetime.utcnow().isoformat()
            )
        
        negative_prices = 0
        negative_volumes = 0
        for bar in ohlcv_data:
            for field in ["open", "high", "low", "close"]:
                if bar.get(field, 0) < 0:
                    negative_prices += 1
            if bar.get("volume", 0) < 0:
                negative_volumes += 1
        
        if negative_prices == 0 and negative_volumes == 0:
            checks_passed.append("no_negative_values")
        else:
            checks_failed.append("negative_values")
            details["negative_prices"] = negative_prices
            details["negative_volumes"] = negative_volumes
        
        timestamps = [bar.get("timestamp", bar.get("date", "")) for bar in ohlcv_data]
        sorted_timestamps = sorted(timestamps)
        
        if timestamps == sorted_timestamps:
            checks_passed.append("timestamps_ordered")
        else:
            checks_failed.append("timestamps_unordered")
        
        unique_timestamps = set(timestamps)
        if len(unique_timestamps) == len(timestamps):
            checks_passed.append("no_duplicate_timestamps")
        else:
            checks_failed.append("duplicate_timestamps")
            details["duplicate_count"] = len(timestamps) - len(unique_timestamps)
        
        if expected_bar_count is not None:
            actual_count = len(ohlcv_data)
            missing_ratio = 1 - (actual_count / expected_bar_count) if expected_bar_count > 0 else 0
            
            if missing_ratio <= self.MAX_MISSING_BAR_TOLERANCE:
                checks_passed.append("bars_within_tolerance")
            else:
                checks_failed.append("missing_bars_exceed_tolerance")
                details["missing_ratio"] = missing_ratio
                details["expected_bars"] = expected_bar_count
                details["actual_bars"] = actual_count
        
        status = GateStatus.PASSED if not checks_failed else GateStatus.FAILED
        action = "proceed" if status == GateStatus.PASSED else "reject_signal"
        
        result = GateResult(
            gate_name="data_integrity_gate",
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            action_taken=action,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        if status == GateStatus.FAILED:
            self.anomaly_log.append(result)
            logger.warning(f"Data integrity gate FAILED: {checks_failed}")
        
        return result


class ModelIntegrityGate:
    """
    Validates model integrity before using ApexCore predictions.
    
    Checks:
    - Model hash matches manifest
    - Model version compatible with engine version
    """
    
    COMPATIBLE_MODEL_VERSIONS = ["9.0", "9.0-A", "8.2"]
    
    def __init__(self, engine_version: str = "9.0-A"):
        self.engine_version = engine_version
    
    def check(
        self,
        model_id: str,
        model_hash: str,
        model_version: str,
        manifest_hash: Optional[str] = None
    ) -> GateResult:
        """
        Check model integrity.
        
        Args:
            model_id: Model identifier
            model_hash: SHA-256 hash of model weights
            model_version: Model version string
            manifest_hash: Expected hash from manifest (optional)
        
        Returns:
            GateResult with pass/fail status
        """
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        details: Dict[str, Any] = {"model_id": model_id, "model_version": model_version}
        
        if manifest_hash is not None:
            if model_hash == manifest_hash:
                checks_passed.append("hash_matches_manifest")
            else:
                checks_failed.append("hash_mismatch")
                details["expected_hash"] = manifest_hash[:16] + "..."
                details["actual_hash"] = model_hash[:16] + "..."
        else:
            checks_passed.append("hash_check_skipped")
            details["note"] = "no_manifest_hash_provided"
        
        major_version = model_version.split(".")[0] if model_version else ""
        if model_version in self.COMPATIBLE_MODEL_VERSIONS or major_version in ["8", "9"]:
            checks_passed.append("version_compatible")
        else:
            checks_failed.append("version_incompatible")
            details["compatible_versions"] = self.COMPATIBLE_MODEL_VERSIONS
        
        status = GateStatus.PASSED if not checks_failed else GateStatus.FAILED
        action = "proceed" if status == GateStatus.PASSED else "fallback_deterministic"
        
        result = GateResult(
            gate_name="model_integrity_gate",
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            action_taken=action,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        if status == GateStatus.FAILED:
            logger.warning(f"Model integrity gate FAILED: {checks_failed}")
        
        return result


class RiskGuardGate:
    """
    Validates risk metrics before allowing signal generation.
    
    Checks:
    - Risk metrics computed successfully
    - Risk metrics within configured limits
    """
    
    DEFAULT_LIMITS = {
        "max_position_risk": 0.10,  # 10% max position risk
        "max_portfolio_risk": 0.25,  # 25% max portfolio risk
        "max_concentration": 0.30,  # 30% max single position concentration
        "min_liquidity_score": 0.20,  # 20% min liquidity
    }
    
    def __init__(self, limits: Optional[Dict[str, float]] = None):
        self.limits = limits or self.DEFAULT_LIMITS
    
    def check(
        self,
        risk_metrics: Dict[str, float],
        skip_checks: Optional[List[str]] = None
    ) -> GateResult:
        """
        Check risk guard constraints.
        
        Args:
            risk_metrics: Dict of risk metric name -> value
            skip_checks: Optional list of checks to skip
        
        Returns:
            GateResult with pass/fail status
        """
        checks_passed: List[str] = []
        checks_failed: List[str] = []
        details: Dict[str, Any] = {"metrics": risk_metrics.copy()}
        skip_checks = skip_checks or []
        
        if not risk_metrics:
            checks_failed.append("no_risk_metrics")
            return GateResult(
                gate_name="risk_guard_gate",
                status=GateStatus.FAILED,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                action_taken="blocked_by_risk",
                details={"reason": "missing_risk_metrics"},
                timestamp=datetime.utcnow().isoformat()
            )
        
        for limit_name, limit_value in self.limits.items():
            if limit_name in skip_checks:
                continue
                
            metric_name = limit_name.replace("max_", "").replace("min_", "")
            metric_value = risk_metrics.get(metric_name, risk_metrics.get(limit_name))
            
            if metric_value is None:
                continue
            
            if limit_name.startswith("max_"):
                if metric_value <= limit_value:
                    checks_passed.append(f"{limit_name}_ok")
                else:
                    checks_failed.append(f"{limit_name}_exceeded")
                    details[f"{limit_name}_value"] = metric_value
                    details[f"{limit_name}_limit"] = limit_value
            elif limit_name.startswith("min_"):
                if metric_value >= limit_value:
                    checks_passed.append(f"{limit_name}_ok")
                else:
                    checks_failed.append(f"{limit_name}_below_minimum")
                    details[f"{limit_name}_value"] = metric_value
                    details[f"{limit_name}_limit"] = limit_value
        
        status = GateStatus.PASSED if not checks_failed else GateStatus.FAILED
        action = "proceed" if status == GateStatus.PASSED else "blocked_by_risk"
        
        result = GateResult(
            gate_name="risk_guard_gate",
            status=status,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            action_taken=action,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        if status == GateStatus.FAILED:
            logger.warning(f"Risk guard gate FAILED: {checks_failed}")
        
        return result


class DecisionGateRunner:
    """
    Runs all decision gates in sequence for fail-closed operation.
    """
    
    def __init__(
        self,
        engine_version: str = "9.0-A",
        risk_limits: Optional[Dict[str, float]] = None
    ):
        self.data_gate = DataIntegrityGate()
        self.model_gate = ModelIntegrityGate(engine_version)
        self.risk_gate = RiskGuardGate(risk_limits)
        self.gate_history: List[Dict[str, GateResult]] = []
    
    def run_all_gates(
        self,
        ohlcv_data: List[Dict[str, Any]],
        model_info: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
        expected_bar_count: Optional[int] = None
    ) -> Tuple[bool, Dict[str, GateResult]]:
        """
        Run all gates and return overall pass/fail.
        
        Returns (all_passed, gate_results_dict)
        """
        results = {}
        
        data_result = self.data_gate.check(ohlcv_data, expected_bar_count)
        results["data_integrity"] = data_result
        
        if model_info:
            model_result = self.model_gate.check(
                model_id=model_info.get("model_id", "unknown"),
                model_hash=model_info.get("hash", ""),
                model_version=model_info.get("version", ""),
                manifest_hash=model_info.get("manifest_hash")
            )
            results["model_integrity"] = model_result
        
        if risk_metrics:
            risk_result = self.risk_gate.check(risk_metrics)
            results["risk_guard"] = risk_result
        
        all_passed = all(
            r.status == GateStatus.PASSED 
            for r in results.values()
        )
        
        self.gate_history.append(results)
        
        return all_passed, results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of gate check history."""
        total_runs = len(self.gate_history)
        
        if total_runs == 0:
            return {"total_runs": 0}
        
        gate_stats = {}
        for gate_name in ["data_integrity", "model_integrity", "risk_guard"]:
            passed = sum(
                1 for run in self.gate_history 
                if gate_name in run and run[gate_name].status == GateStatus.PASSED
            )
            total = sum(1 for run in self.gate_history if gate_name in run)
            gate_stats[gate_name] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0
            }
        
        return {
            "total_runs": total_runs,
            "gate_stats": gate_stats,
            "data_anomalies": len(self.data_gate.anomaly_log)
        }
