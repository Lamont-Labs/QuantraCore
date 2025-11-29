"""
MonsterRunner Protocol Loader

Discovers and executes all MRxx protocols in deterministic order.
Aggregates results into a unified MonsterRunnerResult.

Version: 8.1
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import importlib

from ...core.schemas import OhlcvBar


@dataclass
class MonsterRunnerResult:
    """
    Aggregated result from all MonsterRunner protocols.
    
    Contains firing status, combined score, and individual
    protocol results for rare-event detection.
    """
    any_fired: bool = False
    monster_score: float = 0.0
    protocols_fired: List[str] = field(default_factory=list)
    individual_results: Dict[str, Any] = field(default_factory=dict)
    dominant_signal: Optional[str] = None
    confidence: float = 0.0
    compliance_note: str = "MonsterRunner output is structural probability, not trading advice"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "any_fired": self.any_fired,
            "monster_score": self.monster_score,
            "protocols_fired": self.protocols_fired,
            "individual_results": {
                k: v.__dict__ if hasattr(v, '__dict__') else v
                for k, v in self.individual_results.items()
            },
            "dominant_signal": self.dominant_signal,
            "confidence": self.confidence,
            "compliance_note": self.compliance_note,
        }


class MonsterRunnerLoader:
    """
    Loads and executes MonsterRunner protocols.
    
    Protocols MR01-MR05 are executed in deterministic order.
    Results are aggregated into a single MonsterRunnerResult.
    
    All analysis is deterministic and research-only.
    """
    
    PROTOCOL_IDS = ["MR01", "MR02", "MR03", "MR04", "MR05"]
    
    def __init__(self):
        """Initialize the loader and discover protocols."""
        self.protocols: Dict[str, Any] = {}
        self._load_protocols()
    
    def _load_protocols(self) -> None:
        """Load all MRxx protocol modules."""
        for protocol_id in self.PROTOCOL_IDS:
            try:
                module = importlib.import_module(
                    f".{protocol_id}",
                    package="src.quantracore_apex.protocols.monster_runner"
                )
                run_fn = getattr(module, f"run_{protocol_id}", None)
                if run_fn:
                    self.protocols[protocol_id] = run_fn
            except ImportError:
                pass
    
    def run_all(self, bars: List[OhlcvBar]) -> MonsterRunnerResult:
        """
        Execute all MonsterRunner protocols.
        
        Args:
            bars: OHLCV price bars for analysis
            
        Returns:
            MonsterRunnerResult with aggregated analysis
        """
        result = MonsterRunnerResult()
        
        scores = []
        confidences = []
        
        for protocol_id in self.PROTOCOL_IDS:
            if protocol_id not in self.protocols:
                continue
            
            try:
                run_fn = self.protocols[protocol_id]
                protocol_result = run_fn(bars)
                
                result.individual_results[protocol_id] = protocol_result
                
                if hasattr(protocol_result, 'fired') and protocol_result.fired:
                    result.protocols_fired.append(protocol_id)
                    result.any_fired = True
                
                if hasattr(protocol_result, 'confidence'):
                    confidences.append(protocol_result.confidence)
                
                score_attr = None
                for attr in ['compression_score', 'volume_anomaly_score', 
                           'regime_shift_score', 'institutional_score', 'alignment_score']:
                    if hasattr(protocol_result, attr):
                        score_attr = getattr(protocol_result, attr)
                        break
                
                if score_attr is not None:
                    scores.append(score_attr)
                    
            except Exception:
                result.individual_results[protocol_id] = {"error": "Protocol execution failed"}
        
        if scores:
            result.monster_score = round(float(max(scores)), 4)
        
        if confidences:
            result.confidence = round(float(max(confidences)), 2)
        
        if result.protocols_fired:
            highest_score = 0.0
            dominant = None
            
            for pid in result.protocols_fired:
                pr = result.individual_results.get(pid)
                if pr and hasattr(pr, 'confidence') and pr.confidence > highest_score:
                    highest_score = pr.confidence
                    dominant = pid
            
            result.dominant_signal = dominant
        
        return result
    
    def get_loaded_protocols(self) -> List[str]:
        """Get list of successfully loaded protocol IDs."""
        return list(self.protocols.keys())
    
    def run_single(self, protocol_id: str, bars: List[OhlcvBar]) -> Any:
        """
        Run a single protocol by ID.
        
        Args:
            protocol_id: Protocol identifier (e.g., "MR01")
            bars: OHLCV price bars
            
        Returns:
            Protocol-specific result or None if not found
        """
        if protocol_id not in self.protocols:
            return None
        
        return self.protocols[protocol_id](bars)
