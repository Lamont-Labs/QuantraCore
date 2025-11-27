"""
Tier Protocol Loader for QuantraCore Apex.

Auto-discovers and executes T01-T80 protocols in order.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import importlib
import logging

from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


logger = logging.getLogger(__name__)


class TierProtocolRunner:
    """
    Runs all tier protocols in sequence.
    """
    
    def __init__(self):
        self.protocols: Dict[str, Any] = {}
        self._load_protocols()
    
    def _load_protocols(self) -> None:
        """
        Auto-discover and load all T01-T80 protocol modules.
        """
        protocol_dir = Path(__file__).parent
        
        for i in range(1, 81):
            protocol_id = f"T{i:02d}"
            module_name = f"src.quantracore_apex.protocols.tier.{protocol_id}"
            
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "run"):
                    self.protocols[protocol_id] = module.run
                    logger.debug(f"Loaded protocol {protocol_id}")
            except ImportError:
                pass
    
    def run_single(
        self,
        protocol_id: str,
        window: OhlcvWindow,
        microtraits: Microtraits
    ) -> Optional[ProtocolResult]:
        """
        Run a single protocol.
        """
        if protocol_id not in self.protocols:
            return None
        
        try:
            result = self.protocols[protocol_id](window, microtraits)
            return result
        except Exception as e:
            logger.error(f"Error running protocol {protocol_id}: {e}")
            return ProtocolResult(
                protocol_id=protocol_id,
                fired=False,
                confidence=0.0,
                details={"error": str(e)}
            )
    
    def run_all(
        self,
        window: OhlcvWindow,
        microtraits: Microtraits
    ) -> List[ProtocolResult]:
        """
        Run all protocols in order.
        """
        results = []
        
        for i in range(1, 81):
            protocol_id = f"T{i:02d}"
            result = self.run_single(protocol_id, window, microtraits)
            if result:
                results.append(result)
        
        return results
    
    def run_range(
        self,
        window: OhlcvWindow,
        microtraits: Microtraits,
        start: int = 1,
        end: int = 80
    ) -> List[ProtocolResult]:
        """
        Run a range of protocols.
        """
        results = []
        
        for i in range(start, end + 1):
            protocol_id = f"T{i:02d}"
            result = self.run_single(protocol_id, window, microtraits)
            if result:
                results.append(result)
        
        return results
