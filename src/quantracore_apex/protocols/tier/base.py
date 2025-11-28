"""
Base classes and utilities for Tier Protocols.
"""

from typing import Dict, Any, Optional
from src.quantracore_apex.core.schemas import OhlcvWindow, Microtraits, ProtocolResult


class TierProtocol:
    """
    Base class for tier protocols.
    """
    
    protocol_id: str = "T00"
    name: str = "Base Protocol"
    category: str = "base"
    description: str = "Base protocol class"
    
    @classmethod
    def run(cls, window: OhlcvWindow, microtraits: Microtraits) -> ProtocolResult:
        """
        Run the protocol analysis.
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    @classmethod
    def create_result(
        cls,
        fired: bool,
        confidence: float = 0.0,
        signal_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ProtocolResult:
        """
        Create a standardized protocol result.
        """
        return ProtocolResult(
            protocol_id=cls.protocol_id,
            fired=fired,
            confidence=confidence,
            signal_type=signal_type,
            details=details or {}
        )


