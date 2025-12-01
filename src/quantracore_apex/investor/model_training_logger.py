"""
Model Training History Logger.

Captures all ML model training events for investor due diligence:
- Training runs with hyperparameters
- Validation results and accuracy metrics
- Model versioning and deployment history
- Drift detection events

Provides complete transparency into AI/ML model lifecycle.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)

MODEL_LOGS_DIR = Path("investor_logs/models")


@dataclass
class ModelTrainingRun:
    """Record of a model training run."""
    run_id: str
    model_name: str
    model_version: str
    started_at: str
    completed_at: Optional[str]
    status: str
    
    training_samples: int
    validation_samples: int
    symbols_count: int
    date_range_start: str
    date_range_end: str
    
    hyperparameters: Dict[str, Any]
    feature_count: int
    target_heads: List[str]
    
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    training_duration_seconds: Optional[float] = None
    model_file_path: Optional[str] = None
    model_checksum: Optional[str] = None
    
    notes: str = ""
    triggered_by: str = "manual"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelValidationResult:
    """Record of model validation/backtesting."""
    validation_id: str
    model_name: str
    model_version: str
    validated_at: str
    
    validation_type: str
    dataset_name: str
    samples_evaluated: int
    
    accuracy_metrics: Dict[str, float]
    confusion_matrix: Optional[Dict[str, Any]] = None
    
    passed_threshold: bool = True
    threshold_criteria: Dict[str, float] = field(default_factory=dict)
    
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelDeploymentEvent:
    """Record of model deployment to production."""
    deployment_id: str
    model_name: str
    model_version: str
    deployed_at: str
    
    previous_version: Optional[str]
    deployment_type: str
    
    deployed_by: str
    approval_status: str
    approved_by: Optional[str] = None
    
    rollback_plan: str = ""
    monitoring_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftDetectionEvent:
    """Record of model drift detection."""
    event_id: str
    model_name: str
    detected_at: str
    
    drift_type: str
    severity: str
    
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    prediction_drift: Optional[float] = None
    
    recommended_action: str = ""
    action_taken: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelTrainingLogger:
    """
    Logger for all ML model training and lifecycle events.
    
    Provides institutional investors with complete transparency
    into model development, validation, and deployment.
    """
    
    def __init__(self):
        MODEL_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0
    
    def log_training_run(
        self,
        model_name: str,
        model_version: str,
        training_samples: int,
        validation_samples: int,
        symbols_count: int,
        date_range_start: str,
        date_range_end: str,
        hyperparameters: Dict[str, Any],
        feature_count: int,
        target_heads: List[str],
        training_metrics: Optional[Dict[str, float]] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
        training_duration_seconds: Optional[float] = None,
        model_file_path: Optional[str] = None,
        model_checksum: Optional[str] = None,
        notes: str = "",
        triggered_by: str = "manual",
        status: str = "completed",
    ) -> ModelTrainingRun:
        """Log a model training run."""
        self._run_counter += 1
        now = datetime.now(timezone.utc)
        
        run = ModelTrainingRun(
            run_id=f"TRAIN-{now.strftime('%Y%m%d%H%M%S')}-{self._run_counter:04d}",
            model_name=model_name,
            model_version=model_version,
            started_at=now.isoformat(),
            completed_at=now.isoformat() if status == "completed" else None,
            status=status,
            training_samples=training_samples,
            validation_samples=validation_samples,
            symbols_count=symbols_count,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            hyperparameters=hyperparameters,
            feature_count=feature_count,
            target_heads=target_heads,
            training_metrics=training_metrics or {},
            validation_metrics=validation_metrics or {},
            training_duration_seconds=training_duration_seconds,
            model_file_path=model_file_path,
            model_checksum=model_checksum,
            notes=notes,
            triggered_by=triggered_by,
        )
        
        self._save_training_run(run)
        logger.info(f"Logged training run {run.run_id}: {model_name} v{model_version}")
        
        return run
    
    def log_validation_result(
        self,
        model_name: str,
        model_version: str,
        validation_type: str,
        dataset_name: str,
        samples_evaluated: int,
        accuracy_metrics: Dict[str, float],
        confusion_matrix: Optional[Dict[str, Any]] = None,
        passed_threshold: bool = True,
        threshold_criteria: Optional[Dict[str, float]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> ModelValidationResult:
        """Log model validation results."""
        now = datetime.now(timezone.utc)
        
        result = ModelValidationResult(
            validation_id=f"VAL-{now.strftime('%Y%m%d%H%M%S')}-{self._run_counter:04d}",
            model_name=model_name,
            model_version=model_version,
            validated_at=now.isoformat(),
            validation_type=validation_type,
            dataset_name=dataset_name,
            samples_evaluated=samples_evaluated,
            accuracy_metrics=accuracy_metrics,
            confusion_matrix=confusion_matrix,
            passed_threshold=passed_threshold,
            threshold_criteria=threshold_criteria or {},
            recommendations=recommendations or [],
        )
        
        self._save_validation_result(result)
        logger.info(f"Logged validation {result.validation_id}: passed={passed_threshold}")
        
        return result
    
    def log_deployment_event(
        self,
        model_name: str,
        model_version: str,
        deployment_type: str,
        deployed_by: str,
        approval_status: str,
        previous_version: Optional[str] = None,
        approved_by: Optional[str] = None,
        rollback_plan: str = "",
        monitoring_enabled: bool = True,
    ) -> ModelDeploymentEvent:
        """Log model deployment event."""
        now = datetime.now(timezone.utc)
        
        event = ModelDeploymentEvent(
            deployment_id=f"DEPLOY-{now.strftime('%Y%m%d%H%M%S')}",
            model_name=model_name,
            model_version=model_version,
            deployed_at=now.isoformat(),
            previous_version=previous_version,
            deployment_type=deployment_type,
            deployed_by=deployed_by,
            approval_status=approval_status,
            approved_by=approved_by,
            rollback_plan=rollback_plan,
            monitoring_enabled=monitoring_enabled,
        )
        
        self._save_deployment_event(event)
        logger.info(f"Logged deployment {event.deployment_id}: {model_name} v{model_version}")
        
        return event
    
    def log_drift_detection(
        self,
        model_name: str,
        drift_type: str,
        severity: str,
        feature_drifts: Optional[Dict[str, float]] = None,
        prediction_drift: Optional[float] = None,
        recommended_action: str = "",
        action_taken: Optional[str] = None,
    ) -> DriftDetectionEvent:
        """Log drift detection event."""
        now = datetime.now(timezone.utc)
        
        event = DriftDetectionEvent(
            event_id=f"DRIFT-{now.strftime('%Y%m%d%H%M%S')}",
            model_name=model_name,
            detected_at=now.isoformat(),
            drift_type=drift_type,
            severity=severity,
            feature_drifts=feature_drifts or {},
            prediction_drift=prediction_drift,
            recommended_action=recommended_action,
            action_taken=action_taken,
        )
        
        self._save_drift_event(event)
        logger.warning(f"Logged drift event {event.event_id}: {drift_type} ({severity})")
        
        return event
    
    def _save_training_run(self, run: ModelTrainingRun):
        """Save training run to log file."""
        log_file = MODEL_LOGS_DIR / "training_runs.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(run.to_dict()) + "\n")
    
    def _save_validation_result(self, result: ModelValidationResult):
        """Save validation result to log file."""
        log_file = MODEL_LOGS_DIR / "validation_results.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
    
    def _save_deployment_event(self, event: ModelDeploymentEvent):
        """Save deployment event to log file."""
        log_file = MODEL_LOGS_DIR / "deployments.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def _save_drift_event(self, event: DriftDetectionEvent):
        """Save drift event to log file."""
        log_file = MODEL_LOGS_DIR / "drift_events.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
    
    def get_model_history(self, model_name: str) -> Dict[str, Any]:
        """Get complete history for a model."""
        history = {
            "model_name": model_name,
            "training_runs": [],
            "validations": [],
            "deployments": [],
            "drift_events": [],
        }
        
        for log_type, file_name, key in [
            ("training_runs", "training_runs.jsonl", "training_runs"),
            ("validations", "validation_results.jsonl", "validations"),
            ("deployments", "deployments.jsonl", "deployments"),
            ("drift_events", "drift_events.jsonl", "drift_events"),
        ]:
            log_file = MODEL_LOGS_DIR / file_name
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("model_name") == model_name:
                                history[key].append(record)
        
        return history
    
    def generate_model_card(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        Generate a model card for investor due diligence.
        
        Model cards provide standardized documentation of ML models.
        """
        history = self.get_model_history(model_name)
        
        latest_training = None
        latest_validation = None
        
        for run in reversed(history["training_runs"]):
            if run.get("model_version") == model_version:
                latest_training = run
                break
        
        for val in reversed(history["validations"]):
            if val.get("model_version") == model_version:
                latest_validation = val
                break
        
        model_card = {
            "model_name": model_name,
            "model_version": model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_details": {
                "developer": "QuantraCore",
                "model_type": "Gradient Boosting Ensemble",
                "primary_use": "Trading signal quality prediction",
                "out_of_scope_use": "Live trading decisions without human review",
            },
            "training_data": {
                "samples": latest_training.get("training_samples") if latest_training else None,
                "symbols": latest_training.get("symbols_count") if latest_training else None,
                "date_range": {
                    "start": latest_training.get("date_range_start") if latest_training else None,
                    "end": latest_training.get("date_range_end") if latest_training else None,
                },
                "features": latest_training.get("feature_count") if latest_training else None,
            },
            "performance_metrics": latest_validation.get("accuracy_metrics") if latest_validation else {},
            "validation_passed": latest_validation.get("passed_threshold") if latest_validation else None,
            "ethical_considerations": {
                "fairness": "Model trained on market data without demographic features",
                "transparency": "All predictions include confidence scores",
                "accountability": "Human oversight required for all trading decisions",
            },
            "limitations": [
                "Model trained on historical data that may not reflect future conditions",
                "Performance may degrade during unprecedented market events",
                "Predictions are probabilistic and not guarantees",
            ],
            "caveats": [
                "Not investment advice",
                "Past performance does not guarantee future results",
                "Use for research and informational purposes only",
            ],
        }
        
        model_card_file = MODEL_LOGS_DIR / f"model_card_{model_name}_{model_version}.json"
        with open(model_card_file, "w") as f:
            json.dump(model_card, f, indent=2)
        
        return model_card


_model_logger = None


def get_model_training_logger() -> ModelTrainingLogger:
    """Get singleton model training logger instance."""
    global _model_logger
    if _model_logger is None:
        _model_logger = ModelTrainingLogger()
    return _model_logger
