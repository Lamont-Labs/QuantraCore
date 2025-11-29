"""
Validation module for ApexLab.

Compares ApexCore model outputs against Apex engine teacher labels.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.apexcore.interface import ApexCoreModel
from .features import FeatureExtractor


logger = logging.getLogger(__name__)


class AlignmentValidator:
    """
    Validates ApexCore model alignment with Apex engine.
    """
    
    def __init__(self, model: ApexCoreModel, log_dir: str = "logs/proof_logs"):
        self.model = model
        self.engine = ApexEngine(enable_logging=False)
        self.feature_extractor = FeatureExtractor()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self, windows: List[OhlcvWindow]) -> Dict[str, Any]:
        """
        Validate model predictions against teacher labels.
        """
        if not self.model.is_loaded:
            raise RuntimeError("Model not loaded")
        
        regime_correct = 0
        risk_correct = 0
        score_errors = []
        
        detailed_results = []
        
        for window in windows:
            teacher_result = self.engine.run(window)
            
            model_output = self.model.predict_window(window)
            
            teacher_regime = self._regime_to_class(teacher_result.regime.value)
            teacher_risk = self._risk_to_class(teacher_result.risk_tier.value)
            teacher_score = teacher_result.quantrascore
            
            if model_output.regime_prediction == teacher_regime:
                regime_correct += 1
            
            if model_output.risk_prediction == teacher_risk:
                risk_correct += 1
            
            score_error = abs(model_output.quantrascore_prediction - teacher_score)
            score_errors.append(score_error)
            
            detailed_results.append({
                "symbol": window.symbol,
                "window_hash": window.get_hash(),
                "teacher_regime": teacher_regime,
                "model_regime": model_output.regime_prediction,
                "regime_match": model_output.regime_prediction == teacher_regime,
                "teacher_risk": teacher_risk,
                "model_risk": model_output.risk_prediction,
                "risk_match": model_output.risk_prediction == teacher_risk,
                "teacher_score": teacher_score,
                "model_score": model_output.quantrascore_prediction,
                "score_error": score_error,
            })
        
        n_samples = len(windows)
        
        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_samples": n_samples,
            "metrics": {
                "regime_accuracy": regime_correct / n_samples if n_samples > 0 else 0,
                "risk_accuracy": risk_correct / n_samples if n_samples > 0 else 0,
                "score_mae": float(np.mean(score_errors)) if score_errors else 0,
                "score_rmse": float(np.sqrt(np.mean(np.array(score_errors) ** 2))) if score_errors else 0,
            },
            "thresholds": {
                "regime_accuracy_min": 0.6,
                "risk_accuracy_min": 0.6,
                "score_mae_max": 15.0,
            },
            "passed": True,
            "detailed_results": detailed_results,
        }
        
        if validation_report["metrics"]["regime_accuracy"] < validation_report["thresholds"]["regime_accuracy_min"]:
            validation_report["passed"] = False
        if validation_report["metrics"]["risk_accuracy"] < validation_report["thresholds"]["risk_accuracy_min"]:
            validation_report["passed"] = False
        if validation_report["metrics"]["score_mae"] > validation_report["thresholds"]["score_mae_max"]:
            validation_report["passed"] = False
        
        return validation_report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save validation report to disk."""
        if filename is None:
            filename = f"apexcore_alignment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        report_copy = report.copy()
        if "detailed_results" in report_copy and len(report_copy["detailed_results"]) > 100:
            report_copy["detailed_results"] = report_copy["detailed_results"][:100]
            report_copy["detailed_results_truncated"] = True
        
        with open(filepath, "w") as f:
            json.dump(report_copy, f, indent=2)
        
        logger.info(f"Validation report saved to {filepath}")
        return str(filepath)
    
    def _regime_to_class(self, regime: str) -> int:
        mapping = {
            "trending_up": 0, "trending_down": 1, "range_bound": 2,
            "volatile": 3, "compressed": 4, "unknown": 5
        }
        return mapping.get(regime, 5)
    
    def _risk_to_class(self, risk: str) -> int:
        mapping = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        return mapping.get(risk, 1)
