"""
Risk Engine for QuantraCore Apex.

Implements comprehensive risk checks including:
- Volatility assessment
- Spread analysis
- Regime mismatch detection
- Entropy/drift/suppression state checks
- Fundamental context (earnings proximity)

All outputs are structural probabilities for research only.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RiskPermission(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    RESTRICTED = "restricted"


class RiskAssessment(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    volatility_risk: float = Field(ge=0.0, le=1.0, description="Volatility risk score 0-1")
    spread_risk: float = Field(ge=0.0, le=1.0, description="Spread/liquidity risk 0-1")
    regime_risk: float = Field(ge=0.0, le=1.0, description="Regime mismatch risk 0-1")
    entropy_risk: float = Field(ge=0.0, le=1.0, description="Entropy state risk 0-1")
    drift_risk: float = Field(ge=0.0, le=1.0, description="Drift state risk 0-1")
    suppression_risk: float = Field(ge=0.0, le=1.0, description="Suppression risk 0-1")
    fundamental_risk: float = Field(ge=0.0, le=1.0, description="Earnings/event proximity risk 0-1")
    
    composite_risk: float = Field(ge=0.0, le=1.0, description="Weighted composite risk score")
    risk_tier: str = Field(description="low/medium/high/extreme")
    
    permission: RiskPermission = RiskPermission.ALLOW
    override_code: Optional[str] = None
    denial_reasons: list[str] = Field(default_factory=list)
    
    compliance_note: str = "Risk assessment for research purposes only - not trading advice"


class RiskEngine:
    """
    Deterministic risk assessment engine.
    
    Evaluates multiple risk factors and provides structured risk assessment
    for research and simulation purposes only.
    """
    
    VOLATILITY_WEIGHT = 0.25
    SPREAD_WEIGHT = 0.10
    REGIME_WEIGHT = 0.20
    ENTROPY_WEIGHT = 0.15
    DRIFT_WEIGHT = 0.10
    SUPPRESSION_WEIGHT = 0.10
    FUNDAMENTAL_WEIGHT = 0.10
    
    RISK_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "extreme": 1.0
    }
    
    def __init__(self):
        self.denial_threshold = 0.8
        self.restriction_threshold = 0.6
    
    def assess(
        self,
        symbol: str,
        quantra_score: float,
        regime: str,
        entropy_state: str,
        drift_state: Optional[str] = None,
        suppression_state: Optional[str] = None,
        volatility_ratio: float = 1.0,
        spread_pct: float = 0.001,
        earnings_days_away: Optional[int] = None,
    ) -> RiskAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            symbol: Ticker symbol
            quantra_score: QuantraScore (0-100)
            regime: Market regime (trending_up, trending_down, volatile, range_bound, unknown)
            entropy_state: Entropy state (stable, elevated, chaotic)
            drift_state: Drift state if available
            suppression_state: Suppression state if available
            volatility_ratio: Current/historical volatility ratio
            spread_pct: Bid-ask spread percentage
            earnings_days_away: Days until earnings (None if unknown)
        
        Returns:
            RiskAssessment with all risk metrics and permission decision.
        """
        volatility_risk = self._assess_volatility(volatility_ratio)
        spread_risk = self._assess_spread(spread_pct)
        regime_risk = self._assess_regime(regime, quantra_score)
        entropy_risk = self._assess_entropy(entropy_state)
        drift_risk = self._assess_drift(drift_state)
        suppression_risk = self._assess_suppression(suppression_state)
        fundamental_risk = self._assess_fundamentals(earnings_days_away)
        
        composite_risk = (
            volatility_risk * self.VOLATILITY_WEIGHT +
            spread_risk * self.SPREAD_WEIGHT +
            regime_risk * self.REGIME_WEIGHT +
            entropy_risk * self.ENTROPY_WEIGHT +
            drift_risk * self.DRIFT_WEIGHT +
            suppression_risk * self.SUPPRESSION_WEIGHT +
            fundamental_risk * self.FUNDAMENTAL_WEIGHT
        )
        
        composite_risk = max(0.0, min(1.0, composite_risk))
        
        risk_tier = self._determine_tier(composite_risk)
        permission, override_code, reasons = self._determine_permission(
            composite_risk, entropy_state, drift_state, regime
        )
        
        return RiskAssessment(
            symbol=symbol,
            volatility_risk=volatility_risk,
            spread_risk=spread_risk,
            regime_risk=regime_risk,
            entropy_risk=entropy_risk,
            drift_risk=drift_risk,
            suppression_risk=suppression_risk,
            fundamental_risk=fundamental_risk,
            composite_risk=composite_risk,
            risk_tier=risk_tier,
            permission=permission,
            override_code=override_code,
            denial_reasons=reasons,
        )
    
    def _assess_volatility(self, volatility_ratio: float) -> float:
        """Assess risk from current volatility level."""
        if volatility_ratio <= 0.5:
            return 0.1
        elif volatility_ratio <= 1.0:
            return 0.2
        elif volatility_ratio <= 1.5:
            return 0.4
        elif volatility_ratio <= 2.0:
            return 0.6
        elif volatility_ratio <= 3.0:
            return 0.8
        else:
            return 1.0
    
    def _assess_spread(self, spread_pct: float) -> float:
        """Assess risk from bid-ask spread."""
        if spread_pct <= 0.001:
            return 0.1
        elif spread_pct <= 0.005:
            return 0.3
        elif spread_pct <= 0.01:
            return 0.5
        elif spread_pct <= 0.02:
            return 0.7
        else:
            return 1.0
    
    def _assess_regime(self, regime: str, quantra_score: float) -> float:
        """Assess risk from regime mismatch."""
        base_risk = {
            "trending_up": 0.2,
            "trending_down": 0.4,
            "range_bound": 0.3,
            "volatile": 0.7,
            "unknown": 0.5,
        }.get(regime.lower(), 0.5)
        
        if quantra_score < 40:
            base_risk += 0.2
        elif quantra_score > 70:
            base_risk -= 0.1
        
        return max(0.0, min(1.0, base_risk))
    
    def _assess_entropy(self, entropy_state: str) -> float:
        """Assess risk from entropy state."""
        return {
            "stable": 0.1,
            "EntropyState.STABLE": 0.1,
            "elevated": 0.5,
            "EntropyState.ELEVATED": 0.5,
            "chaotic": 0.9,
            "EntropyState.CHAOTIC": 0.9,
        }.get(str(entropy_state), 0.5)
    
    def _assess_drift(self, drift_state: Optional[str]) -> float:
        """Assess risk from drift state."""
        if drift_state is None:
            return 0.3
        
        return {
            "stable": 0.1,
            "DriftState.STABLE": 0.1,
            "moderate": 0.4,
            "DriftState.MODERATE": 0.4,
            "critical": 0.9,
            "DriftState.CRITICAL": 0.9,
        }.get(str(drift_state), 0.4)
    
    def _assess_suppression(self, suppression_state: Optional[str]) -> float:
        """Assess risk from suppression state."""
        if suppression_state is None:
            return 0.3
        
        return {
            "none": 0.1,
            "SuppressionState.NONE": 0.1,
            "mild": 0.3,
            "SuppressionState.MILD": 0.3,
            "moderate": 0.5,
            "SuppressionState.MODERATE": 0.5,
            "strong": 0.8,
            "SuppressionState.STRONG": 0.8,
        }.get(str(suppression_state), 0.4)
    
    def _assess_fundamentals(self, earnings_days_away: Optional[int]) -> float:
        """Assess risk from fundamental events (earnings proximity)."""
        if earnings_days_away is None:
            return 0.2
        
        if earnings_days_away <= 1:
            return 1.0
        elif earnings_days_away <= 3:
            return 0.8
        elif earnings_days_away <= 7:
            return 0.5
        elif earnings_days_away <= 14:
            return 0.3
        else:
            return 0.1
    
    def _determine_tier(self, composite_risk: float) -> str:
        """Determine risk tier from composite score."""
        if composite_risk < self.RISK_THRESHOLDS["low"]:
            return "low"
        elif composite_risk < self.RISK_THRESHOLDS["medium"]:
            return "medium"
        elif composite_risk < self.RISK_THRESHOLDS["high"]:
            return "high"
        else:
            return "extreme"
    
    def _determine_permission(
        self,
        composite_risk: float,
        entropy_state: str,
        drift_state: Optional[str],
        regime: str,
    ) -> tuple[RiskPermission, Optional[str], list[str]]:
        """Determine permission based on risk factors."""
        reasons = []
        
        if composite_risk >= self.denial_threshold:
            reasons.append(f"Composite risk {composite_risk:.2f} exceeds threshold")
            return RiskPermission.DENY, "RISK_THRESHOLD_EXCEEDED", reasons
        
        if "chaotic" in str(entropy_state).lower():
            reasons.append("Chaotic entropy state detected")
            return RiskPermission.DENY, "OMEGA_2_ENTROPY_OVERRIDE", reasons
        
        if drift_state and "critical" in str(drift_state).lower():
            reasons.append("Critical drift state detected")
            return RiskPermission.DENY, "OMEGA_3_DRIFT_OVERRIDE", reasons
        
        if regime.lower() == "volatile" and composite_risk >= 0.5:
            reasons.append("Volatile regime with elevated risk")
            return RiskPermission.RESTRICTED, "VOLATILE_REGIME_RESTRICTION", reasons
        
        if composite_risk >= self.restriction_threshold:
            reasons.append(f"Elevated composite risk {composite_risk:.2f}")
            return RiskPermission.RESTRICTED, None, reasons
        
        return RiskPermission.ALLOW, None, reasons
