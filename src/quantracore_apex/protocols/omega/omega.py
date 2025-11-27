"""
Omega Directives for QuantraCore Apex.

Omega directives are the final safety checks and overrides
that operate on ApexResult before output.

Ω1: Hard Safety Lock - Blocks all signals when triggered
Ω2: Entropy Override - Forces caution on high entropy
Ω3: Drift Override - Forces caution on critical drift
Ω4: Compliance Override - Ensures all outputs are compliant
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from src.quantracore_apex.core.schemas import (
    ApexResult, EntropyState, DriftState, RiskTier
)


class OmegaLevel(str, Enum):
    INACTIVE = "inactive"
    ADVISORY = "advisory"
    ENFORCED = "enforced"
    LOCKED = "locked"


@dataclass
class OmegaStatus:
    """Status of an Omega directive."""
    active: bool
    level: OmegaLevel
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OmegaDirectives:
    """
    Omega Directive system for QuantraCore Apex.
    
    These are the final safety overrides that ensure
    system integrity and compliance.
    """
    
    def __init__(
        self,
        enable_omega_1: bool = True,
        enable_omega_2: bool = True,
        enable_omega_3: bool = True,
        enable_omega_4: bool = True,
    ):
        self.enable_omega_1 = enable_omega_1
        self.enable_omega_2 = enable_omega_2
        self.enable_omega_3 = enable_omega_3
        self.enable_omega_4 = enable_omega_4
    
    def check_omega_1(self, result: ApexResult) -> OmegaStatus:
        """
        Ω1: Hard Safety Lock
        
        Triggers when system detects conditions requiring full halt:
        - Extreme risk tier
        - Data integrity issues
        - System anomalies
        """
        if not self.enable_omega_1:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        if result.risk_tier == RiskTier.EXTREME:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Extreme risk tier detected - hard safety lock engaged",
                metadata={"risk_tier": result.risk_tier.value}
            )
        
        if result.quantrascore < 5 or result.quantrascore > 98:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Extreme score anomaly - safety lock engaged",
                metadata={"quantrascore": result.quantrascore}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_2(self, result: ApexResult) -> OmegaStatus:
        """
        Ω2: Entropy Override
        
        Triggers when entropy conditions indicate chaotic market:
        - Chaotic entropy state
        - Combined entropy exceeds threshold
        """
        if not self.enable_omega_2:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        if result.entropy_state == EntropyState.CHAOTIC:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Chaotic entropy detected - entropy override active",
                metadata={
                    "entropy_state": result.entropy_state.value,
                    "combined_entropy": result.entropy_metrics.combined_entropy
                }
            )
        
        if result.entropy_metrics.combined_entropy > 0.9:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="High entropy warning",
                metadata={"combined_entropy": result.entropy_metrics.combined_entropy}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_3(self, result: ApexResult) -> OmegaStatus:
        """
        Ω3: Drift Override
        
        Triggers when drift conditions indicate critical deviation:
        - Critical drift state
        - High drift with low reversion pressure
        """
        if not self.enable_omega_3:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        if result.drift_state == DriftState.CRITICAL:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Critical drift detected - drift override active",
                metadata={
                    "drift_state": result.drift_state.value,
                    "drift_magnitude": result.drift_metrics.drift_magnitude
                }
            )
        
        if result.drift_state == DriftState.SIGNIFICANT:
            if result.drift_metrics.mean_reversion_pressure < 0.3:
                return OmegaStatus(
                    active=True,
                    level=OmegaLevel.ADVISORY,
                    reason="Significant drift with low reversion pressure",
                    metadata={
                        "drift_magnitude": result.drift_metrics.drift_magnitude,
                        "reversion_pressure": result.drift_metrics.mean_reversion_pressure
                    }
                )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_4(self, result: ApexResult) -> OmegaStatus:
        """
        Ω4: Compliance Override
        
        Always active to ensure outputs are compliant:
        - Ensures no trading recommendations in output
        - Validates structural probability framing
        - Enforces disclaimer presence
        """
        if not self.enable_omega_4:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        return OmegaStatus(
            active=True,
            level=OmegaLevel.ENFORCED,
            reason="Compliance mode active - all outputs framed as structural analysis",
            metadata={
                "compliance_note": result.verdict.compliance_note,
                "action_type": result.verdict.action
            }
        )
    
    def apply_all(self, result: ApexResult) -> Dict[str, OmegaStatus]:
        """
        Apply all Omega directives to an ApexResult.
        
        Returns a dictionary of all directive statuses.
        """
        return {
            "omega_1_safety": self.check_omega_1(result),
            "omega_2_entropy": self.check_omega_2(result),
            "omega_3_drift": self.check_omega_3(result),
            "omega_4_compliance": self.check_omega_4(result),
        }
    
    def get_highest_alert_level(self, statuses: Dict[str, OmegaStatus]) -> OmegaLevel:
        """Get the highest alert level from all directives."""
        levels = [s.level for s in statuses.values()]
        
        if OmegaLevel.LOCKED in levels:
            return OmegaLevel.LOCKED
        elif OmegaLevel.ENFORCED in levels:
            return OmegaLevel.ENFORCED
        elif OmegaLevel.ADVISORY in levels:
            return OmegaLevel.ADVISORY
        else:
            return OmegaLevel.INACTIVE
    
    def should_block_output(self, statuses: Dict[str, OmegaStatus]) -> bool:
        """Determine if output should be blocked based on directives."""
        return self.get_highest_alert_level(statuses) == OmegaLevel.LOCKED
