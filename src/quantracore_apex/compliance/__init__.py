"""
QuantraCore Apex - Regulatory Excellence Module

This module implements institutional-grade compliance that EXCEEDS regulatory 
requirements by significant margins. Our approach:

1. FINRA 15-09 Compliance: 3x stricter than required
2. SEC 15c3-5 Pre-Trade Risk: 4x sensitivity levels
3. MiFID II RTS 6: 5x latency margins
4. Basel Committee: 20+ historical stress scenarios
5. SOX/SOC2: Full audit trail with cryptographic proof

All outputs are structural probability analyses, NOT trading recommendations.
"""

from .excellence import (
    RegulatoryExcellenceEngine,
    ComplianceScore,
    ExcellenceLevel,
    ComplianceMetrics,
)
from .audit import (
    AuditTrail,
    DecisionProvenance,
    AuditEntry,
)
from .reporter import (
    RegulatoryReporter,
    ComplianceReport,
)

__all__ = [
    "RegulatoryExcellenceEngine",
    "ComplianceScore",
    "ExcellenceLevel",
    "ComplianceMetrics",
    "AuditTrail",
    "DecisionProvenance",
    "AuditEntry",
    "RegulatoryReporter",
    "ComplianceReport",
]
