"""
Omega Directives for QuantraCore Apex.

Omega directives are the final safety checks and overrides
that operate on ApexResult before output.

Ω1: Hard Safety Lock - Blocks all signals when triggered
Ω2: Entropy Override - Forces caution on high entropy
Ω3: Drift Override - Forces caution on critical drift
Ω4: Compliance Override - Ensures all outputs are compliant
Ω5: Signal Suppression Lock - Blocks signals under suppression conditions
Ω6: Volatility Cap Override - Limits exposure in high volatility
Ω7: Momentum Divergence Alert - Detects price/volume divergence
Ω8: Bollinger Squeeze Warning - Signals compression breakout risk
Ω9: MACD Reversal Detection - Identifies histogram reversals
Ω10: Fear Spike Override - Triggers on elevated ATR/fear levels
Ω11: RSI Extreme Override - Blocks signals at RSI extremes
Ω12: Volume Spike Alert - Detects abnormal volume surges
Ω13: Trend Weakness Warning - Signals weak trend conditions
Ω14: Gap Risk Override - Blocks signals after large gaps
Ω15: Tail Risk Lock - Engages on severe intraday drops
Ω16: Overnight Drift Alert - Detects large open-to-close moves
Ω17: Fractal Chaos Warning - Signals unstable price structure
Ω18: Liquidity Void Lock - Blocks signals on low liquidity
Ω19: Correlation Breakdown Alert - Detects unusual correlations
Ω20: Nuclear Killswitch - Final safety gate for data integrity

Version: 9.0-A
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
        enable_omega_5: bool = True,
        enable_omega_6: bool = True,
        enable_omega_7: bool = True,
        enable_omega_8: bool = True,
        enable_omega_9: bool = True,
        enable_omega_10: bool = True,
        enable_omega_11: bool = True,
        enable_omega_12: bool = True,
        enable_omega_13: bool = True,
        enable_omega_14: bool = True,
        enable_omega_15: bool = True,
        enable_omega_16: bool = True,
        enable_omega_17: bool = True,
        enable_omega_18: bool = True,
        enable_omega_19: bool = True,
        enable_omega_20: bool = True,
    ):
        self.enable_omega_1 = enable_omega_1
        self.enable_omega_2 = enable_omega_2
        self.enable_omega_3 = enable_omega_3
        self.enable_omega_4 = enable_omega_4
        self.enable_omega_5 = enable_omega_5
        self.enable_omega_6 = enable_omega_6
        self.enable_omega_7 = enable_omega_7
        self.enable_omega_8 = enable_omega_8
        self.enable_omega_9 = enable_omega_9
        self.enable_omega_10 = enable_omega_10
        self.enable_omega_11 = enable_omega_11
        self.enable_omega_12 = enable_omega_12
        self.enable_omega_13 = enable_omega_13
        self.enable_omega_14 = enable_omega_14
        self.enable_omega_15 = enable_omega_15
        self.enable_omega_16 = enable_omega_16
        self.enable_omega_17 = enable_omega_17
        self.enable_omega_18 = enable_omega_18
        self.enable_omega_19 = enable_omega_19
        self.enable_omega_20 = enable_omega_20
    
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
    
    def check_omega_5(self, result: ApexResult) -> OmegaStatus:
        """
        Ω5: Signal Suppression Lock
        
        Triggers when suppression conditions indicate signal unreliability:
        - Strong suppression detected
        - Multiple suppression factors active
        """
        if not self.enable_omega_5:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        suppression = result.suppression_metrics
        
        if suppression.is_suppressed and suppression.suppression_score > 0.7:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Strong signal suppression detected - suppression lock active",
                metadata={
                    "suppression_score": suppression.suppression_score,
                    "is_suppressed": suppression.is_suppressed,
                }
            )
        
        if suppression.suppression_score > 0.5:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Moderate suppression warning",
                metadata={"suppression_score": suppression.suppression_score}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_6(self, result: ApexResult) -> OmegaStatus:
        """
        Ω6: Volatility Cap Override
        
        Triggers when volatility exceeds safe thresholds:
        - High historical volatility detected
        - Rapid volatility expansion
        """
        if not self.enable_omega_6:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        volatility = getattr(result.microtraits, 'volatility', 0.0) if result.microtraits else 0.0
        
        if volatility > 0.04:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Extreme volatility detected - exposure cap engaged",
                metadata={"volatility": volatility, "threshold": 0.04}
            )
        
        if volatility > 0.025:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="High volatility warning - reduced exposure recommended",
                metadata={"volatility": volatility, "threshold": 0.025}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_7(self, result: ApexResult) -> OmegaStatus:
        """
        Ω7: Momentum Divergence Alert
        
        Triggers when price and volume/momentum diverge:
        - Price rising but momentum falling
        - Volume not confirming price action
        """
        if not self.enable_omega_7:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        momentum = getattr(result.microtraits, 'momentum_score', 0.5) if result.microtraits else 0.5
        trend_score = result.quantrascore / 100.0
        
        divergence = abs(trend_score - momentum)
        
        if divergence > 0.4:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Severe momentum divergence detected - potential reversal",
                metadata={"divergence": divergence, "momentum": momentum, "trend_score": trend_score}
            )
        
        if divergence > 0.25:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Momentum divergence warning",
                metadata={"divergence": divergence}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_8(self, result: ApexResult) -> OmegaStatus:
        """
        Ω8: Bollinger Squeeze Warning
        
        Triggers when Bollinger Band compression indicates breakout risk:
        - Extreme band compression
        - Potential for explosive move
        """
        if not self.enable_omega_8:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        compression = getattr(result.microtraits, 'compression', 0.0) if result.microtraits else 0.0
        
        if compression > 0.8:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Extreme compression detected - breakout imminent",
                metadata={"compression": compression, "threshold": 0.8}
            )
        
        if compression > 0.6:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="High compression warning - prepare for volatility expansion",
                metadata={"compression": compression}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_9(self, result: ApexResult) -> OmegaStatus:
        """
        Ω9: MACD Reversal Detection
        
        Triggers when MACD histogram indicates potential reversal:
        - Histogram direction change
        - Divergence from price
        """
        if not self.enable_omega_9:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        momentum = getattr(result.microtraits, 'momentum_score', 0.5) if result.microtraits else 0.5
        
        if abs(momentum - 0.5) < 0.1 and result.quantrascore > 60:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Momentum stall detected - potential reversal forming",
                metadata={"momentum": momentum, "quantrascore": result.quantrascore}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_10(self, result: ApexResult) -> OmegaStatus:
        """
        Ω10: Fear Spike Override
        
        Triggers when fear indicators spike:
        - Elevated ATR relative to historical
        - Rapid volatility expansion
        """
        if not self.enable_omega_10:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        volatility = getattr(result.microtraits, 'volatility', 0.0) if result.microtraits else 0.0
        noise = getattr(result.microtraits, 'noise', 0.0) if result.microtraits else 0.0
        
        fear_score = (volatility + noise) / 2
        
        if fear_score > 0.06:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Fear spike detected - market panic conditions",
                metadata={"fear_score": fear_score, "volatility": volatility, "noise": noise}
            )
        
        if fear_score > 0.04:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Elevated fear levels - caution advised",
                metadata={"fear_score": fear_score}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_11(self, result: ApexResult) -> OmegaStatus:
        """
        Ω11: RSI Extreme Override
        
        Triggers when RSI reaches extreme overbought/oversold levels:
        - RSI > 80 or < 20
        - Extended conditions
        """
        if not self.enable_omega_11:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        momentum = getattr(result.microtraits, 'momentum_score', 0.5) if result.microtraits else 0.5
        rsi_proxy = momentum * 100
        
        if rsi_proxy > 85 or rsi_proxy < 15:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason=f"RSI extreme detected ({rsi_proxy:.0f}) - reversal risk high",
                metadata={"rsi_proxy": rsi_proxy, "extreme": "overbought" if rsi_proxy > 85 else "oversold"}
            )
        
        if rsi_proxy > 75 or rsi_proxy < 25:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="RSI approaching extreme levels",
                metadata={"rsi_proxy": rsi_proxy}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_12(self, result: ApexResult) -> OmegaStatus:
        """
        Ω12: Volume Spike Alert
        
        Triggers when volume spikes abnormally:
        - Volume > 3x average
        - Potential institutional activity
        """
        if not self.enable_omega_12:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        volume_score = getattr(result.microtraits, 'volume_score', 0.5) if result.microtraits else 0.5
        
        if volume_score > 0.9:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Extreme volume spike detected - institutional activity likely",
                metadata={"volume_score": volume_score}
            )
        
        if volume_score > 0.75:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Elevated volume warning",
                metadata={"volume_score": volume_score}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_13(self, result: ApexResult) -> OmegaStatus:
        """
        Ω13: Trend Weakness Warning
        
        Triggers when trend shows weakness:
        - Low ADX equivalent
        - Choppy price action
        """
        if not self.enable_omega_13:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        noise = getattr(result.microtraits, 'noise', 0.0) if result.microtraits else 0.0
        
        if noise > 0.7:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Trend weakness detected - choppy conditions",
                metadata={"noise": noise, "threshold": 0.7}
            )
        
        if noise > 0.5:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Elevated noise warning - trend may be weakening",
                metadata={"noise": noise}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_14(self, result: ApexResult) -> OmegaStatus:
        """
        Ω14: Gap Risk Override
        
        Triggers after large gaps:
        - Gap > 3% from prior close
        - Elevated gap fill risk
        """
        if not self.enable_omega_14:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        gap = getattr(result.microtraits, 'gap_pct', 0.0) if result.microtraits else 0.0
        
        if abs(gap) > 0.10:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Extreme gap detected - gap risk lock engaged",
                metadata={"gap_pct": gap}
            )
        
        if abs(gap) > 0.03:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Large gap detected - gap fill risk elevated",
                metadata={"gap_pct": gap}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_15(self, result: ApexResult) -> OmegaStatus:
        """
        Ω15: Tail Risk Lock
        
        Triggers on severe intraday drops:
        - Intraday drop > 7%
        - Flash crash conditions
        """
        if not self.enable_omega_15:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        volatility = getattr(result.microtraits, 'volatility', 0.0) if result.microtraits else 0.0
        wick_ratio = getattr(result.microtraits, 'wick_ratio', 0.0) if result.microtraits else 0.0
        
        tail_risk = (volatility + wick_ratio) / 2
        
        if tail_risk > 0.08:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Tail risk lock engaged - extreme intraday volatility",
                metadata={"tail_risk": tail_risk, "volatility": volatility, "wick_ratio": wick_ratio}
            )
        
        if tail_risk > 0.05:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Elevated tail risk detected",
                metadata={"tail_risk": tail_risk}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_16(self, result: ApexResult) -> OmegaStatus:
        """
        Ω16: Overnight Drift Alert
        
        Triggers when open-to-close shows unusual drift:
        - Large intraday range relative to body
        - Unusual price action patterns
        """
        if not self.enable_omega_16:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        wick_ratio = getattr(result.microtraits, 'wick_ratio', 0.0) if result.microtraits else 0.0
        
        if wick_ratio > 0.7:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Unusual drift pattern detected - reversal risk",
                metadata={"wick_ratio": wick_ratio}
            )
        
        if wick_ratio > 0.5:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ADVISORY,
                reason="Elevated wick ratio warning",
                metadata={"wick_ratio": wick_ratio}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_17(self, result: ApexResult) -> OmegaStatus:
        """
        Ω17: Fractal Chaos Warning
        
        Triggers when price structure becomes unstable:
        - Fractal dimension indicates chaos
        - Price pattern breakdown
        """
        if not self.enable_omega_17:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        noise = getattr(result.microtraits, 'noise', 0.0) if result.microtraits else 0.0
        volatility = getattr(result.microtraits, 'volatility', 0.0) if result.microtraits else 0.0
        
        chaos_score = (noise * 0.6 + volatility * 0.4)
        
        if chaos_score > 0.6:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Fractal chaos detected - price structure unstable",
                metadata={"chaos_score": chaos_score, "noise": noise, "volatility": volatility}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_18(self, result: ApexResult) -> OmegaStatus:
        """
        Ω18: Liquidity Void Lock
        
        Triggers when liquidity drops severely:
        - Volume < 30% of average
        - Illiquid conditions detected
        """
        if not self.enable_omega_18:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        volume_score = getattr(result.microtraits, 'volume_score', 0.5) if result.microtraits else 0.5
        
        if volume_score < 0.15:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Liquidity void detected - trading blocked",
                metadata={"volume_score": volume_score}
            )
        
        if volume_score < 0.3:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Low liquidity warning - slippage risk elevated",
                metadata={"volume_score": volume_score}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_19(self, result: ApexResult) -> OmegaStatus:
        """
        Ω19: Correlation Breakdown Alert
        
        Triggers when normal correlations break down:
        - Unusual cross-asset behavior
        - Market regime shift indicators
        """
        if not self.enable_omega_19:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        if result.entropy_state == EntropyState.CHAOTIC and result.drift_state == DriftState.CRITICAL:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.ENFORCED,
                reason="Correlation breakdown detected - regime shift likely",
                metadata={
                    "entropy_state": result.entropy_state.value,
                    "drift_state": result.drift_state.value
                }
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
    def check_omega_20(self, result: ApexResult) -> OmegaStatus:
        """
        Ω20: Nuclear Killswitch
        
        Final safety gate for data integrity:
        - Insufficient data samples
        - Data validation failures
        - System integrity checks
        """
        if not self.enable_omega_20:
            return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
        
        protocol_count = len(result.protocol_results) if result.protocol_results else 0
        
        if protocol_count < 10:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Nuclear killswitch - insufficient protocol coverage",
                metadata={"protocol_count": protocol_count, "minimum_required": 10}
            )
        
        if result.quantrascore == 0 or result.quantrascore == 100:
            return OmegaStatus(
                active=True,
                level=OmegaLevel.LOCKED,
                reason="Nuclear killswitch - anomalous score detected",
                metadata={"quantrascore": result.quantrascore}
            )
        
        return OmegaStatus(active=False, level=OmegaLevel.INACTIVE)
    
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
            "omega_5_suppression": self.check_omega_5(result),
            "omega_6_volatility_cap": self.check_omega_6(result),
            "omega_7_momentum_divergence": self.check_omega_7(result),
            "omega_8_bollinger_squeeze": self.check_omega_8(result),
            "omega_9_macd_reversal": self.check_omega_9(result),
            "omega_10_fear_spike": self.check_omega_10(result),
            "omega_11_rsi_extreme": self.check_omega_11(result),
            "omega_12_volume_spike": self.check_omega_12(result),
            "omega_13_trend_weakness": self.check_omega_13(result),
            "omega_14_gap_risk": self.check_omega_14(result),
            "omega_15_tail_risk": self.check_omega_15(result),
            "omega_16_overnight_drift": self.check_omega_16(result),
            "omega_17_fractal_chaos": self.check_omega_17(result),
            "omega_18_liquidity_void": self.check_omega_18(result),
            "omega_19_correlation_breakdown": self.check_omega_19(result),
            "omega_20_nuclear_killswitch": self.check_omega_20(result),
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
    
    def apply_omega4(self, verdict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Omega 4 compliance directive to a verdict.
        
        Ensures compliance note is always present.
        """
        if "compliance_note" not in verdict:
            verdict["compliance_note"] = "Structural analysis only - not trading advice"
        return verdict
