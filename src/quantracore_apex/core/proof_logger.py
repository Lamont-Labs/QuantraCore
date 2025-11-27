"""
Proof logging module for QuantraCore Apex.

Provides deterministic, auditable logging of all engine executions.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from .schemas import ApexResult


class ProofLogger:
    """
    Logs Apex engine executions with cryptographic proof.
    """
    
    def __init__(self, log_dir: str = "logs/proof_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_proof_hash(self, result: ApexResult) -> str:
        """
        Compute deterministic hash of result for proof.
        """
        proof_data = {
            "symbol": result.symbol,
            "window_hash": result.window_hash,
            "quantrascore": result.quantrascore,
            "regime": result.regime.value,
            "timestamp": result.timestamp.isoformat(),
        }
        
        data_str = json.dumps(proof_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_execution(
        self,
        result: ApexResult,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an Apex engine execution with proof.
        
        Returns:
            Path to the log file.
        """
        proof_hash = self.compute_proof_hash(result)
        result.proof_hash = proof_hash
        
        log_entry = {
            "proof_hash": proof_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result.model_dump(mode="json"),
            "context": context or {},
        }
        
        filename = f"{result.symbol}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}_{proof_hash[:8]}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        return str(filepath)
    
    def verify_proof(self, result: ApexResult) -> bool:
        """
        Verify that a result's proof hash is valid.
        """
        if not result.proof_hash:
            return False
        
        computed_hash = self.compute_proof_hash(result)
        return computed_hash == result.proof_hash
    
    def load_execution(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a logged execution from file.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None


proof_logger = ProofLogger()
