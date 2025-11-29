"""
Automatic Retraining Trigger.

Monitors training data quality and model performance to trigger
retraining when conditions warrant improved model.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .quality_scorer import TrainingQualityScorer, QualityReport

logger = logging.getLogger(__name__)


@dataclass 
class RetrainDecision:
    """Decision about whether to retrain."""
    should_retrain: bool
    reason: str
    priority: str  # "low", "medium", "high", "critical"
    metrics: Dict[str, Any]
    timestamp: str


class AutoRetrainTrigger:
    """
    Monitors conditions and triggers retraining automatically.
    
    Trigger Conditions:
    1. Sample threshold reached (new samples since last train)
    2. Quality score improved significantly
    3. Time-based (periodic retraining)
    4. Performance degradation detected
    5. New scenario types added
    
    The trigger is part of the self-improving loop - it decides
    when retraining will actually improve the model.
    """
    
    def __init__(
        self,
        state_path: str = "data/apexlab/retrain_state.json",
        sample_threshold: int = 100,
        quality_improvement_threshold: float = 0.1,
        max_days_between_retrain: int = 7,
    ):
        self.state_path = Path(state_path)
        self.sample_threshold = sample_threshold
        self.quality_improvement_threshold = quality_improvement_threshold
        self.max_days = max_days_between_retrain
        
        self.quality_scorer = TrainingQualityScorer()
        self._state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load persisted state."""
        if self.state_path.exists():
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {
            "last_retrain_time": None,
            "last_sample_count": 0,
            "last_quality_score": 0.0,
            "retrain_history": [],
        }
    
    def _save_state(self) -> None:
        """Persist current state."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._state, f, indent=2)
    
    def check(self) -> RetrainDecision:
        """
        Check if retraining should be triggered.
        
        Returns:
            RetrainDecision with recommendation
        """
        quality_report = self.quality_scorer.score()
        
        current_samples = quality_report.total_samples
        current_quality = quality_report.overall_quality_score
        
        last_samples = self._state.get("last_sample_count", 0)
        last_quality = self._state.get("last_quality_score", 0.0)
        last_retrain = self._state.get("last_retrain_time")
        
        new_samples = current_samples - last_samples
        quality_improvement = current_quality - last_quality
        
        days_since_retrain = None
        if last_retrain:
            last_dt = datetime.fromisoformat(last_retrain)
            days_since_retrain = (datetime.utcnow() - last_dt).days
        
        should_retrain = False
        reason = ""
        priority = "low"
        
        if new_samples >= self.sample_threshold:
            should_retrain = True
            reason = f"Sample threshold reached: {new_samples} new samples"
            priority = "medium" if new_samples >= self.sample_threshold * 2 else "low"
        
        if quality_improvement >= self.quality_improvement_threshold:
            should_retrain = True
            reason = f"Quality improved by {quality_improvement:.2f}"
            priority = "high"
        
        if days_since_retrain and days_since_retrain >= self.max_days:
            should_retrain = True
            reason = f"Time-based: {days_since_retrain} days since last retrain"
            priority = "medium"
        
        if current_samples >= 500 and current_quality >= 0.7:
            if not last_retrain:
                should_retrain = True
                reason = "Initial training conditions met"
                priority = "high"
        
        if current_samples < 200:
            should_retrain = False
            reason = f"Insufficient samples: {current_samples}/200 minimum"
            priority = "low"
        
        return RetrainDecision(
            should_retrain=should_retrain,
            reason=reason,
            priority=priority,
            metrics={
                "current_samples": current_samples,
                "new_samples": new_samples,
                "current_quality": current_quality,
                "quality_improvement": quality_improvement,
                "days_since_retrain": days_since_retrain,
            },
            timestamp=datetime.utcnow().isoformat(),
        )
    
    def record_retrain(self, success: bool, model_path: str, metrics: Dict[str, Any]) -> None:
        """Record that retraining occurred."""
        quality_report = self.quality_scorer.score()
        
        self._state["last_retrain_time"] = datetime.utcnow().isoformat()
        self._state["last_sample_count"] = quality_report.total_samples
        self._state["last_quality_score"] = quality_report.overall_quality_score
        
        self._state["retrain_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "model_path": model_path,
            "sample_count": quality_report.total_samples,
            "quality_score": quality_report.overall_quality_score,
            "metrics": metrics,
        })
        
        if len(self._state["retrain_history"]) > 50:
            self._state["retrain_history"] = self._state["retrain_history"][-50:]
        
        self._save_state()
        logger.info(f"Recorded retrain: success={success}, samples={quality_report.total_samples}")
    
    def trigger_retrain(self, train_function: Callable) -> Dict[str, Any]:
        """
        Trigger retraining if conditions are met.
        
        Args:
            train_function: Function to call for training
            
        Returns:
            Result of training or skip message
        """
        decision = self.check()
        
        if not decision.should_retrain:
            return {
                "action": "skipped",
                "reason": decision.reason,
                "metrics": decision.metrics,
            }
        
        logger.info(f"Triggering retrain: {decision.reason} (priority: {decision.priority})")
        
        try:
            result = train_function()
            
            self.record_retrain(
                success=True,
                model_path=result.get("model_path", ""),
                metrics=result.get("metrics", {}),
            )
            
            return {
                "action": "retrained",
                "reason": decision.reason,
                "priority": decision.priority,
                "result": result,
            }
            
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            
            self.record_retrain(
                success=False,
                model_path="",
                metrics={"error": str(e)},
            )
            
            return {
                "action": "failed",
                "reason": decision.reason,
                "error": str(e),
            }
    
    def get_history(self) -> list:
        """Get retrain history."""
        return self._state.get("retrain_history", [])
    
    def reset_state(self) -> None:
        """Reset state (use carefully)."""
        self._state = {
            "last_retrain_time": None,
            "last_sample_count": 0,
            "last_quality_score": 0.0,
            "retrain_history": [],
        }
        self._save_state()
