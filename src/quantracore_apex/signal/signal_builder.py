"""
Signal Builder for QuantraCore Apex.

Generates unified trade signals from:
- QuantraScore
- Regime classification
- Risk assessment
- Protocol outputs
- Prediction stack

All signals are structural probabilities for research only.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SignalStrength(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeSignal(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    direction: SignalDirection
    strength: SignalStrength
    confidence: float = Field(ge=0.0, le=1.0, description="Signal confidence 0-1")
    
    quantra_score: float = Field(ge=0.0, le=100.0)
    regime: str
    risk_tier: str
    
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    
    expected_move_pct: Optional[float] = None
    expected_holding_days: Optional[int] = None
    
    fired_protocols: List[str] = Field(default_factory=list)
    supporting_factors: List[str] = Field(default_factory=list)
    warning_factors: List[str] = Field(default_factory=list)
    
    risk_approved: bool = False
    risk_notes: str = ""
    
    compliance_note: str = "Signal is structural probability only - not trading advice"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()


class SignalBuilder:
    """
    Builds unified trade signals from Apex engine outputs.
    
    Combines QuantraScore, regime, risk, and protocol data
    into actionable research signals.
    """
    
    SCORE_THRESHOLDS = {
        "strong_long": 70,
        "moderate_long": 60,
        "neutral_high": 55,
        "neutral_low": 45,
        "moderate_short": 40,
        "strong_short": 30,
    }
    
    FAVORABLE_REGIMES = {"trending_up", "range_bound"}
    UNFAVORABLE_REGIMES = {"volatile", "unknown"}
    
    def __init__(self):
        self.signals: List[TradeSignal] = []
    
    def build_signal(
        self,
        symbol: str,
        quantra_score: float,
        regime: str,
        risk_tier: str,
        entropy_state: str,
        current_price: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        fired_protocols: Optional[List[str]] = None,
        expected_move: Optional[float] = None,
        risk_approved: bool = False,
        risk_notes: str = "",
    ) -> TradeSignal:
        """
        Build a trade signal from Apex engine outputs.
        
        Args:
            symbol: Ticker symbol
            quantra_score: QuantraScore (0-100)
            regime: Market regime
            risk_tier: Risk tier (low/medium/high/extreme)
            entropy_state: Entropy state
            current_price: Current market price
            volatility_pct: Current volatility percentage
            fired_protocols: List of fired protocol IDs
            expected_move: Expected move prediction
            risk_approved: Whether risk engine approved
            risk_notes: Notes from risk assessment
        
        Returns:
            TradeSignal with direction, strength, and levels
        """
        direction = self._determine_direction(quantra_score, regime)
        strength = self._determine_strength(quantra_score, regime, risk_tier)
        confidence = self._calculate_confidence(quantra_score, regime, entropy_state, risk_tier)
        
        entry_low, entry_high, stop, t1, t2 = self._calculate_levels(
            current_price, direction, volatility_pct, expected_move
        )
        
        supporting, warnings = self._analyze_factors(
            quantra_score, regime, risk_tier, entropy_state, fired_protocols or []
        )
        
        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            quantra_score=quantra_score,
            regime=regime,
            risk_tier=risk_tier,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            stop_loss=stop,
            target_1=t1,
            target_2=t2,
            expected_move_pct=expected_move,
            fired_protocols=fired_protocols or [],
            supporting_factors=supporting,
            warning_factors=warnings,
            risk_approved=risk_approved,
            risk_notes=risk_notes,
        )
        
        self.signals.append(signal)
        return signal
    
    def _determine_direction(self, score: float, regime: str) -> SignalDirection:
        """Determine signal direction from score and regime."""
        if score >= self.SCORE_THRESHOLDS["moderate_long"]:
            return SignalDirection.LONG
        elif score <= self.SCORE_THRESHOLDS["moderate_short"]:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def _determine_strength(self, score: float, regime: str, risk_tier: str) -> SignalStrength:
        """Determine signal strength."""
        if risk_tier == "extreme":
            return SignalStrength.WEAK
        
        if score >= self.SCORE_THRESHOLDS["strong_long"]:
            if regime in self.FAVORABLE_REGIMES:
                return SignalStrength.STRONG
            return SignalStrength.MODERATE
        
        elif score <= self.SCORE_THRESHOLDS["strong_short"]:
            return SignalStrength.STRONG if risk_tier != "high" else SignalStrength.MODERATE
        
        elif score >= self.SCORE_THRESHOLDS["moderate_long"]:
            return SignalStrength.MODERATE if regime in self.FAVORABLE_REGIMES else SignalStrength.WEAK
        
        elif score <= self.SCORE_THRESHOLDS["moderate_short"]:
            return SignalStrength.MODERATE
        
        return SignalStrength.NEUTRAL
    
    def _calculate_confidence(
        self,
        score: float,
        regime: str,
        entropy_state: str,
        risk_tier: str,
    ) -> float:
        """Calculate signal confidence (0-1)."""
        base_confidence = abs(score - 50) / 50
        
        if regime in self.FAVORABLE_REGIMES:
            base_confidence *= 1.1
        elif regime in self.UNFAVORABLE_REGIMES:
            base_confidence *= 0.8
        
        if "chaotic" in str(entropy_state).lower():
            base_confidence *= 0.6
        elif "elevated" in str(entropy_state).lower():
            base_confidence *= 0.85
        
        risk_multipliers = {
            "low": 1.0,
            "medium": 0.9,
            "high": 0.7,
            "extreme": 0.4,
        }
        base_confidence *= risk_multipliers.get(risk_tier, 0.8)
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_levels(
        self,
        price: Optional[float],
        direction: SignalDirection,
        volatility_pct: Optional[float],
        expected_move: Optional[float],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate entry, stop, and target levels."""
        if price is None:
            return None, None, None, None, None
        
        vol = volatility_pct or 2.0
        move = expected_move or vol * 2
        
        if direction == SignalDirection.LONG:
            entry_low = price * (1 - vol / 200)
            entry_high = price * (1 + vol / 400)
            stop = price * (1 - vol / 50)
            t1 = price * (1 + move / 100)
            t2 = price * (1 + move * 1.5 / 100)
        elif direction == SignalDirection.SHORT:
            entry_low = price * (1 - vol / 400)
            entry_high = price * (1 + vol / 200)
            stop = price * (1 + vol / 50)
            t1 = price * (1 - move / 100)
            t2 = price * (1 - move * 1.5 / 100)
        else:
            return price * 0.99, price * 1.01, None, None, None
        
        return round(entry_low, 2), round(entry_high, 2), round(stop, 2), round(t1, 2), round(t2, 2)
    
    def _analyze_factors(
        self,
        score: float,
        regime: str,
        risk_tier: str,
        entropy_state: str,
        protocols: List[str],
    ) -> tuple[List[str], List[str]]:
        """Analyze supporting and warning factors."""
        supporting = []
        warnings = []
        
        if score >= 70:
            supporting.append("High conviction QuantraScore")
        elif score >= 60:
            supporting.append("Moderate conviction QuantraScore")
        
        if regime == "trending_up":
            supporting.append("Favorable trending regime")
        elif regime == "volatile":
            warnings.append("Volatile regime detected")
        elif regime == "unknown":
            warnings.append("Regime unclear")
        
        if risk_tier == "low":
            supporting.append("Low risk tier")
        elif risk_tier == "high":
            warnings.append("Elevated risk tier")
        elif risk_tier == "extreme":
            warnings.append("EXTREME risk tier - caution advised")
        
        if "chaotic" in str(entropy_state).lower():
            warnings.append("Chaotic entropy state")
        elif "stable" in str(entropy_state).lower():
            supporting.append("Stable entropy")
        
        tier_protocols = [p for p in protocols if p.startswith("T")]
        if len(tier_protocols) >= 5:
            supporting.append(f"{len(tier_protocols)} tier protocols firing")
        
        return supporting, warnings
    
    def get_signals(
        self,
        direction: Optional[SignalDirection] = None,
        min_strength: Optional[SignalStrength] = None,
        min_confidence: float = 0.0,
    ) -> List[TradeSignal]:
        """Get filtered signals."""
        signals = self.signals
        
        if direction:
            signals = [s for s in signals if s.direction == direction]
        
        if min_strength:
            strength_order = [SignalStrength.WEAK, SignalStrength.NEUTRAL, 
                           SignalStrength.MODERATE, SignalStrength.STRONG]
            min_idx = strength_order.index(min_strength)
            signals = [s for s in signals if strength_order.index(s.strength) >= min_idx]
        
        signals = [s for s in signals if s.confidence >= min_confidence]
        
        return signals
    
    def clear(self):
        """Clear all signals."""
        self.signals.clear()
