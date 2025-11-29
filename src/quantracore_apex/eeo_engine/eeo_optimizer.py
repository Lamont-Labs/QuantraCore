"""
EEO Entry/Exit Optimizer

Main coordinator that builds complete entry/exit plans from signal contexts.
"""

import logging
from typing import Optional, Dict, Any

from .enums import SignalDirection, QualityTier
from .models import (
    EntryExitPlan,
    PlanMetadata,
    AbortCondition,
)
from .contexts import (
    SignalContext,
    PredictiveContext,
    MarketMicrostructure,
    RiskContext,
    EEOContext,
)
from .profiles import (
    EEOProfile,
    ProfileType,
    get_profile,
    BALANCED_PROFILE,
)
from .entry_optimizer import EntryOptimizer
from .exit_optimizer import ExitOptimizer

logger = logging.getLogger(__name__)


class EntryExitOptimizer:
    """
    Main coordinator for entry/exit optimization.
    
    This class sits between the Apex signal engine and the Execution Engine.
    For every structural signal, it calculates:
    - Best entry price zone
    - Best entry style (market vs limit vs stop)
    - Best exit plan (targets, stops, trailing, time-based exits)
    
    All within deterministic, risk-aware, and compliance-friendly constraints.
    """
    
    def __init__(
        self,
        profile: Optional[EEOProfile] = None,
        profile_type: Optional[ProfileType] = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            profile: Explicit profile to use
            profile_type: Profile type to load (ignored if profile is provided)
        """
        if profile:
            self.profile = profile
        elif profile_type:
            self.profile = get_profile(profile_type)
        else:
            self.profile = BALANCED_PROFILE
        
        self.entry_optimizer = EntryOptimizer(self.profile)
        self.exit_optimizer = ExitOptimizer(self.profile)
        
        logger.info(f"EEO Optimizer initialized with profile: {self.profile.name}")
    
    def build_plan(
        self,
        context: EEOContext,
        override_size_notional: Optional[float] = None,
    ) -> EntryExitPlan:
        """
        Build a complete entry/exit plan for the given context.
        
        Args:
            context: Complete EEO context with signal, predictive,
                    microstructure, and risk information
            override_size_notional: Override calculated position size
        
        Returns:
            EntryExitPlan with all entry and exit parameters
        """
        if not context.should_proceed():
            logger.warning(f"Context conditions unfavorable for {context.signal.symbol}")
            return self._create_abort_plan(context, "Unfavorable conditions")
        
        entry_opt = self.entry_optimizer.optimize(context)
        
        entry_price = entry_opt.base_entry.entry_price
        exit_opt = self.exit_optimizer.optimize(context, entry_price)
        
        size_notional = override_size_notional or self._calculate_size(
            context, entry_price, exit_opt.protective_stop
        )
        
        if context.signal.liquidity_band.value == "low":
            size_notional *= 0.7
        
        metadata = self._build_metadata(context)
        
        plan = EntryExitPlan(
            symbol=context.signal.symbol,
            direction=context.signal.direction,
            entry_mode=entry_opt.entry_mode,
            exit_mode=exit_opt.exit_mode,
            size_notional=round(size_notional, 2),
            base_entry=entry_opt.base_entry,
            scaled_entries=entry_opt.scaled_entries,
            protective_stop=exit_opt.protective_stop,
            profit_targets=exit_opt.profit_targets,
            trailing_stop=exit_opt.trailing_stop,
            time_based_exit=exit_opt.time_based_exit,
            abort_conditions=exit_opt.abort_conditions,
            metadata=metadata,
            source_signal_id=context.signal.signal_id,
        )
        
        rr = plan.risk_reward_ratio()
        rr_str = f"{rr:.2f}" if rr else "N/A"
        logger.info(
            f"Built plan {plan.plan_id} for {plan.symbol}: "
            f"entry={entry_price:.2f}, size=${size_notional:.0f}, "
            f"R:R={rr_str}"
        )
        
        return plan
    
    def build_plan_from_signal(
        self,
        signal_context: SignalContext,
        current_price: float,
        account_equity: float = 100000.0,
        predictive_context: Optional[PredictiveContext] = None,
        atr: float = 0.0,
    ) -> EntryExitPlan:
        """
        Convenience method to build a plan from minimal inputs.
        
        Args:
            signal_context: Signal context from Apex engine
            current_price: Current market price
            account_equity: Account equity for sizing
            predictive_context: Optional predictive context from ApexCore
            atr: Average True Range for volatility calculations
        
        Returns:
            EntryExitPlan
        """
        if predictive_context is None:
            predictive_context = PredictiveContext()
        
        microstructure = MarketMicrostructure.from_price(current_price, atr)
        
        risk_context = RiskContext(
            account_equity=account_equity,
            per_trade_risk_fraction=self.profile.per_trade_risk_fraction,
            max_notional_per_symbol=account_equity * 0.1,
        )
        
        context = EEOContext(
            signal=signal_context,
            predictive=predictive_context,
            microstructure=microstructure,
            risk=risk_context,
        )
        
        return self.build_plan(context)
    
    def _calculate_size(
        self,
        context: EEOContext,
        entry_price: float,
        protective_stop,
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Uses fixed-fraction risk sizing:
        size = (account_equity * risk_fraction) / stop_distance
        """
        risk = context.risk
        
        max_risk_dollars = risk.max_risk_dollars()
        
        if protective_stop and protective_stop.enabled:
            stop_distance = abs(entry_price - protective_stop.stop_price)
            if stop_distance > 0:
                shares = max_risk_dollars / stop_distance
                size_notional = shares * entry_price
            else:
                size_notional = max_risk_dollars * 2
        else:
            size_notional = max_risk_dollars * 2
        
        size_notional = min(size_notional, risk.max_notional_per_symbol)
        size_notional = min(size_notional, risk.available_capital() * 0.2)
        
        return max(0, size_notional)
    
    def _build_metadata(self, context: EEOContext) -> PlanMetadata:
        """Build metadata for the plan."""
        predictive = context.predictive
        
        return PlanMetadata(
            quality_label=predictive.future_quality_tier,
            runner_bucket=predictive.get_runner_bucket(),
            confidence=1.0 - predictive.ensemble_disagreement,
            profile_used=self.profile.name,
        )
    
    def _create_abort_plan(
        self,
        context: EEOContext,
        reason: str,
    ) -> EntryExitPlan:
        """Create an invalid plan when conditions prevent optimization."""
        return EntryExitPlan(
            symbol=context.signal.symbol,
            direction=context.signal.direction,
            size_notional=0.0,
            source_signal_id=context.signal.signal_id,
            abort_conditions=[
                AbortCondition(
                    condition_type="pre_optimization_abort",
                    threshold=0.0,
                    description=reason,
                )
            ],
            compliance_note=f"Plan aborted: {reason}",
        )
    
    def set_profile(self, profile: EEOProfile) -> None:
        """Change the active profile."""
        self.profile = profile
        self.entry_optimizer = EntryOptimizer(profile)
        self.exit_optimizer = ExitOptimizer(profile)
        logger.info(f"Profile changed to: {profile.name}")
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get current profile information."""
        return self.profile.to_dict()
