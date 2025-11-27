"""
Label Generator for ApexLab.

Generates deterministic teacher labels from Apex engine.
"""

from typing import Dict, Any, List
import numpy as np

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexContext
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.protocols.learning import LearningProtocolRunner


class LabelGenerator:
    """
    Generates training labels using Apex engine as teacher.
    """
    
    def __init__(self, enable_logging: bool = False):
        self.engine = ApexEngine(enable_logging=enable_logging)
        self.learning_runner = LearningProtocolRunner()
    
    def generate(self, window: OhlcvWindow) -> Dict[str, Any]:
        """
        Generate all labels for a window.
        
        Returns:
            Dictionary of label names to values
        """
        context = ApexContext(seed=42, compliance_mode=True)
        
        apex_result = self.engine.run(window, context)
        
        labels = self.learning_runner.generate_all_labels(window, apex_result)
        label_dict = self.learning_runner.to_label_dict(labels)
        
        label_dict["regime_class"] = self._regime_to_class(apex_result.regime.value)
        label_dict["risk_tier"] = self._risk_to_class(apex_result.risk_tier.value)
        label_dict["score_bucket"] = self._bucket_to_class(apex_result.score_bucket.value)
        label_dict["quantrascore_numeric"] = apex_result.quantrascore
        label_dict["entropy_state"] = self._entropy_to_class(apex_result.entropy_state.value)
        label_dict["suppression_state"] = self._suppression_to_class(apex_result.suppression_state.value)
        label_dict["drift_state"] = self._drift_to_class(apex_result.drift_state.value)
        
        return label_dict
    
    def generate_batch(self, windows: List[OhlcvWindow]) -> Dict[str, np.ndarray]:
        """
        Generate labels for multiple windows.
        
        Returns:
            Dictionary of label names to numpy arrays
        """
        all_labels = [self.generate(w) for w in windows]
        
        label_names = all_labels[0].keys() if all_labels else []
        
        result = {}
        for name in label_names:
            values = [labels[name] for labels in all_labels]
            result[name] = np.array(values)
        
        return result
    
    def _regime_to_class(self, regime: str) -> int:
        mapping = {
            "trending_up": 0, "trending_down": 1, "range_bound": 2,
            "volatile": 3, "compressed": 4, "unknown": 5
        }
        return mapping.get(regime, 5)
    
    def _risk_to_class(self, risk: str) -> int:
        mapping = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
        return mapping.get(risk, 1)
    
    def _bucket_to_class(self, bucket: str) -> int:
        mapping = {"very_low": 0, "low": 1, "neutral": 2, "high": 3, "very_high": 4}
        return mapping.get(bucket, 2)
    
    def _entropy_to_class(self, state: str) -> int:
        mapping = {"stable": 0, "elevated": 1, "chaotic": 2}
        return mapping.get(state, 0)
    
    def _suppression_to_class(self, state: str) -> int:
        mapping = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}
        return mapping.get(state, 0)
    
    def _drift_to_class(self, state: str) -> int:
        mapping = {"none": 0, "mild": 1, "significant": 2, "critical": 3}
        return mapping.get(state, 0)
