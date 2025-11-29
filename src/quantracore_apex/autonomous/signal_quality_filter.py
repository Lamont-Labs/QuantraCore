"""
Institutional-Grade Signal Quality Filter.

Enforces strict quality standards for autonomous trading:
- Minimum QuantraScore threshold (default 75+)
- Quality tier requirements (A+/A only)
- Risk tier limits (no extreme)
- Omega directive compliance
- Liquidity requirements
- Position limits

Only the highest quality signals pass through for execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, TYPE_CHECKING

from .models import (
    FilterResult,
    FilterRejectionReason,
    QualityThresholds,
    OrchestratorConfig,
)

if TYPE_CHECKING:
    from src.quantracore_apex.core.schemas import ApexResult
    from src.quantracore_apex.protocols.omega.omega import OmegaDirectives, OmegaLevel

try:
    from src.quantracore_apex.core.schemas import ApexResult as ApexResultClass
    from src.quantracore_apex.protocols.omega.omega import OmegaDirectives as OmegaDirectivesClass, OmegaLevel as OmegaLevelEnum
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    ApexResultClass = None
    OmegaDirectivesClass = None
    OmegaLevelEnum = None


logger = logging.getLogger(__name__)


class SignalQualityFilter:
    """
    Institutional-grade signal quality filter.
    
    Applies multiple layers of filtering to ensure only
    the highest quality signals are considered for trading.
    
    Filter Layers:
    1. QuantraScore threshold
    2. Quality tier (A+/A)
    3. Risk tier (not extreme)
    4. Omega directive status
    5. Liquidity requirements
    6. Position limits
    7. Symbol cooldowns
    8. Regime alignment
    """
    
    RISK_TIER_ORDER = ["low", "medium", "high", "extreme"]
    LIQUIDITY_ORDER = ["very_low", "low", "medium", "high", "very_high"]
    QUALITY_TIER_ORDER = ["D", "C", "B", "A", "A+"]
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        thresholds: Optional[QualityThresholds] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.thresholds = thresholds or self.config.quality_thresholds
        
        self._symbol_cooldowns: Dict[str, datetime] = {}
        self._open_positions: Set[str] = set()
        self._current_exposure: float = 0.0
        
        self._omega_directives = None
        if APEX_AVAILABLE and OmegaDirectivesClass is not None:
            try:
                self._omega_directives = OmegaDirectivesClass()
            except Exception as e:
                logger.warning(f"Could not initialize OmegaDirectives: {e}")
        
        self._filter_stats = {
            "total_evaluated": 0,
            "passed": 0,
            "rejected_quantrascore": 0,
            "rejected_quality_tier": 0,
            "rejected_risk_tier": 0,
            "rejected_omega": 0,
            "rejected_liquidity": 0,
            "rejected_position_limit": 0,
            "rejected_cooldown": 0,
            "rejected_avoid_flag": 0,
        }
    
    def update_positions(self, open_symbols: Set[str], exposure: float) -> None:
        """Update current position state from PositionMonitor."""
        self._open_positions = open_symbols
        self._current_exposure = exposure
    
    def add_cooldown(self, symbol: str, duration_seconds: Optional[float] = None) -> None:
        """Add cooldown for a symbol after trade."""
        duration = duration_seconds or self.config.symbol_cooldown_seconds
        self._symbol_cooldowns[symbol] = datetime.utcnow() + timedelta(seconds=duration)
    
    def clear_expired_cooldowns(self) -> None:
        """Remove expired cooldowns."""
        now = datetime.utcnow()
        expired = [s for s, t in self._symbol_cooldowns.items() if t <= now]
        for symbol in expired:
            del self._symbol_cooldowns[symbol]
    
    def filter(self, result: ApexResult) -> FilterResult:
        """
        Apply all quality filters to an ApexResult.
        
        Returns FilterResult indicating pass/fail with reasons.
        """
        self._filter_stats["total_evaluated"] += 1
        self.clear_expired_cooldowns()
        
        rejection_reasons: List[FilterRejectionReason] = []
        metadata: Dict[str, Any] = {}
        
        signal_id = getattr(result, "signal_id", "") or str(id(result))
        symbol = getattr(result, "symbol", "UNKNOWN")
        quantrascore = getattr(result, "quantrascore", 0.0)
        quality_tier = self._get_quality_tier(result)
        risk_tier = self._get_risk_tier(result)
        
        metadata["quantrascore"] = quantrascore
        metadata["quality_tier"] = quality_tier
        metadata["risk_tier"] = risk_tier
        
        if quantrascore < self.thresholds.min_quantrascore:
            rejection_reasons.append(FilterRejectionReason.QUANTRASCORE_TOO_LOW)
            metadata["quantrascore_threshold"] = self.thresholds.min_quantrascore
            self._filter_stats["rejected_quantrascore"] += 1
        
        if quality_tier not in self.thresholds.required_quality_tiers:
            rejection_reasons.append(FilterRejectionReason.QUALITY_TIER_INSUFFICIENT)
            metadata["required_tiers"] = self.thresholds.required_quality_tiers
            self._filter_stats["rejected_quality_tier"] += 1
        
        if not self._risk_tier_acceptable(risk_tier):
            rejection_reasons.append(FilterRejectionReason.RISK_TIER_TOO_HIGH)
            metadata["max_risk_tier"] = self.thresholds.max_risk_tier
            self._filter_stats["rejected_risk_tier"] += 1
        
        omega_block, omega_reason = self._check_omega_directives(result)
        if omega_block:
            rejection_reasons.append(FilterRejectionReason.OMEGA_DIRECTIVE_BLOCKED)
            metadata["omega_reason"] = omega_reason
            self._filter_stats["rejected_omega"] += 1
        
        liquidity_band = self._get_liquidity_band(result)
        if not self._liquidity_acceptable(liquidity_band):
            rejection_reasons.append(FilterRejectionReason.LIQUIDITY_INSUFFICIENT)
            metadata["liquidity_band"] = liquidity_band
            self._filter_stats["rejected_liquidity"] += 1
        
        if len(self._open_positions) >= self.config.max_concurrent_positions:
            rejection_reasons.append(FilterRejectionReason.MAX_POSITIONS_REACHED)
            metadata["current_positions"] = len(self._open_positions)
            metadata["max_positions"] = self.config.max_concurrent_positions
            self._filter_stats["rejected_position_limit"] += 1
        
        if self._current_exposure >= self.config.max_portfolio_exposure:
            rejection_reasons.append(FilterRejectionReason.MAX_EXPOSURE_REACHED)
            metadata["current_exposure"] = self._current_exposure
            metadata["max_exposure"] = self.config.max_portfolio_exposure
            self._filter_stats["rejected_position_limit"] += 1
        
        if symbol in self._symbol_cooldowns:
            rejection_reasons.append(FilterRejectionReason.COOLDOWN_ACTIVE)
            metadata["cooldown_until"] = self._symbol_cooldowns[symbol].isoformat()
            self._filter_stats["rejected_cooldown"] += 1
        
        if symbol in self._open_positions:
            rejection_reasons.append(FilterRejectionReason.MAX_POSITIONS_REACHED)
            metadata["already_has_position"] = True
        
        avoid_prob = self._get_avoid_probability(result)
        if avoid_prob > self.thresholds.max_avoid_probability:
            rejection_reasons.append(FilterRejectionReason.AVOID_FLAG_SET)
            metadata["avoid_probability"] = avoid_prob
            self._filter_stats["rejected_avoid_flag"] += 1
        
        runner_prob = self._get_runner_probability(result)
        if runner_prob < self.thresholds.min_runner_probability:
            rejection_reasons.append(FilterRejectionReason.RUNNER_PROB_LOW)
            metadata["runner_probability"] = runner_prob
        
        if self.thresholds.require_regime_alignment:
            regime = self._get_regime(result)
            if regime not in self.thresholds.allowed_regimes:
                metadata["regime"] = regime
                metadata["regime_warning"] = "not_in_allowed_list"
        
        passed = len(rejection_reasons) == 0
        
        if passed:
            self._filter_stats["passed"] += 1
            logger.info(
                f"[QualityFilter] PASSED: {symbol} | "
                f"Score={quantrascore:.1f} | Tier={quality_tier} | Risk={risk_tier}"
            )
        else:
            reasons_str = ", ".join(r.value for r in rejection_reasons[:3])
            logger.debug(
                f"[QualityFilter] REJECTED: {symbol} | "
                f"Score={quantrascore:.1f} | Reasons: {reasons_str}"
            )
        
        return FilterResult(
            passed=passed,
            signal_id=signal_id,
            symbol=symbol,
            quantrascore=quantrascore,
            quality_tier=quality_tier,
            risk_tier=risk_tier,
            rejection_reasons=rejection_reasons,
            metadata=metadata,
        )
    
    def _get_quality_tier(self, result: ApexResult) -> str:
        """Extract quality tier from result."""
        if hasattr(result, "quality_tier"):
            return str(result.quality_tier)
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return result.verdict.get("quality_tier", "C")
        return "C"
    
    def _get_risk_tier(self, result: ApexResult) -> str:
        """Extract risk tier from result."""
        if hasattr(result, "risk_tier"):
            tier = result.risk_tier
            if hasattr(tier, "value"):
                return tier.value.lower()
            return str(tier).lower()
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return result.verdict.get("risk_tier", "high").lower()
        return "high"
    
    def _get_liquidity_band(self, result: ApexResult) -> str:
        """Extract liquidity band from result."""
        if hasattr(result, "liquidity_band"):
            band = result.liquidity_band
            if hasattr(band, "value"):
                return band.value.lower()
            return str(band).lower()
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return result.verdict.get("liquidity_band", "medium").lower()
        return "medium"
    
    def _get_regime(self, result: ApexResult) -> str:
        """Extract regime from result."""
        if hasattr(result, "regime"):
            regime = result.regime
            if hasattr(regime, "value"):
                return regime.value.lower()
            return str(regime).lower()
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return result.verdict.get("regime", "unknown").lower()
        return "unknown"
    
    def _get_runner_probability(self, result: ApexResult) -> float:
        """Extract runner probability from result."""
        if hasattr(result, "runner_probability"):
            return float(result.runner_probability)
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return float(result.verdict.get("runner_probability", 0.0))
        return 0.0
    
    def _get_avoid_probability(self, result: ApexResult) -> float:
        """Extract avoid trade probability from result."""
        if hasattr(result, "avoid_probability"):
            return float(result.avoid_probability)
        if hasattr(result, "verdict") and isinstance(result.verdict, dict):
            return float(result.verdict.get("avoid_probability", 0.0))
        return 0.0
    
    def _risk_tier_acceptable(self, risk_tier: str) -> bool:
        """Check if risk tier is within acceptable range."""
        try:
            current_idx = self.RISK_TIER_ORDER.index(risk_tier.lower())
            max_idx = self.RISK_TIER_ORDER.index(self.thresholds.max_risk_tier.lower())
            return current_idx <= max_idx
        except ValueError:
            return False
    
    def _liquidity_acceptable(self, liquidity_band: str) -> bool:
        """Check if liquidity is sufficient."""
        try:
            current_idx = self.LIQUIDITY_ORDER.index(liquidity_band.lower())
            min_idx = self.LIQUIDITY_ORDER.index(self.thresholds.min_liquidity_band.lower())
            return current_idx >= min_idx
        except ValueError:
            return True
    
    def _check_omega_directives(self, result: Any) -> tuple:
        """
        Check Omega directives for blocking conditions.
        
        Returns (is_blocked, reason).
        """
        if not APEX_AVAILABLE or not self._omega_directives:
            return False, None
        
        try:
            if hasattr(self._omega_directives, 'check_all'):
                statuses = self._omega_directives.check_all(result)
            else:
                return False, None
            
            for directive_name, status in statuses.items():
                if hasattr(status, 'active') and status.active:
                    if hasattr(status, 'level'):
                        level_name = status.level.name if hasattr(status.level, 'name') else str(status.level)
                        if level_name in ["LOCKED", "ENFORCED"]:
                            reason = getattr(status, 'reason', 'Omega directive triggered')
                            return True, f"{directive_name}: {reason}"
            
            return False, None
            
        except Exception as e:
            logger.warning(f"Error checking Omega directives: {e}")
            return False, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        total = self._filter_stats["total_evaluated"]
        passed = self._filter_stats["passed"]
        return {
            **self._filter_stats,
            "pass_rate": passed / total if total > 0 else 0.0,
            "open_positions": len(self._open_positions),
            "current_exposure": self._current_exposure,
            "active_cooldowns": len(self._symbol_cooldowns),
        }
    
    def reset_stats(self) -> None:
        """Reset filter statistics."""
        for key in self._filter_stats:
            self._filter_stats[key] = 0
