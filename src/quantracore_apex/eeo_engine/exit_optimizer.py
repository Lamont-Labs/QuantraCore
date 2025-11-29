"""
EEO Exit Optimizer

Calculates optimal exit parameters including stops, targets, and trailing stops.
Implements the "Best Exit" logic from the specification.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .enums import (
    SignalDirection,
    OrderTypeEEO,
    ExitMode,
    ExitStyle,
    TrailingStopMode,
    VolatilityBand,
)
from .models import (
    ProtectiveStop,
    ProfitTarget,
    TrailingStop,
    TimeBasedExit,
    AbortCondition,
)
from .contexts import EEOContext
from .profiles import EEOProfile

logger = logging.getLogger(__name__)


@dataclass
class ExitOptimization:
    """Result of exit optimization."""
    protective_stop: Optional[ProtectiveStop]
    profit_targets: List[ProfitTarget]
    trailing_stop: Optional[TrailingStop]
    time_based_exit: Optional[TimeBasedExit]
    abort_conditions: List[AbortCondition]
    exit_mode: ExitMode
    rationale: str


class ExitOptimizer:
    """
    Optimizes exit parameters for trading signals.
    
    Goals:
    - Systematically encode good exits; no random guessing
    - Use Estimated Move + volatility to place realistic targets and stops
    - Allow multi-leg exits for partial profit-taking
    - Always include a protective stop and/or time-based exit
    """
    
    def __init__(self, profile: EEOProfile):
        self.profile = profile
    
    def optimize(
        self,
        context: EEOContext,
        entry_price: float,
    ) -> ExitOptimization:
        """
        Calculate optimal exit parameters.
        
        Args:
            context: Complete EEO context
            entry_price: The planned entry price
        
        Returns:
            ExitOptimization with stops, targets, and exit rules
        """
        protective_stop = self._calculate_protective_stop(context, entry_price)
        
        profit_targets = self._calculate_profit_targets(context, entry_price)
        
        trailing_stop = self._calculate_trailing_stop(context)
        
        time_based_exit = self._calculate_time_exit(context)
        
        abort_conditions = self._calculate_abort_conditions(context)
        
        exit_mode = ExitMode.SCALED_OUT if len(profit_targets) > 1 else ExitMode.SINGLE
        
        rationale = self._build_rationale(
            protective_stop, profit_targets, trailing_stop, time_based_exit
        )
        
        return ExitOptimization(
            protective_stop=protective_stop,
            profit_targets=profit_targets,
            trailing_stop=trailing_stop,
            time_based_exit=time_based_exit,
            abort_conditions=abort_conditions,
            exit_mode=exit_mode,
            rationale=rationale,
        )
    
    def _calculate_protective_stop(
        self,
        context: EEOContext,
        entry_price: float,
    ) -> Optional[ProtectiveStop]:
        """
        Calculate protective stop-loss.
        
        Rules:
        - For LONG: stop below structural support (recent swing low)
        - Distance set as max(1-2 * ATR, small percentage of price)
        - ZDE research influence: tight stops if MAE rare
        - Volatility adjustment: widen stops but reduce size
        """
        if not self.profile.require_protective_stop:
            return None
        
        micro = context.microstructure
        signal = context.signal
        predictive = context.predictive
        
        atr = micro.atr_14 if micro.atr_14 > 0 else entry_price * 0.02
        
        base_stop_distance = atr * self.profile.stop_atr_multiple
        
        min_stop_distance = entry_price * 0.005
        base_stop_distance = max(base_stop_distance, min_stop_distance)
        
        if signal.volatility_band == VolatilityBand.HIGH:
            base_stop_distance *= 1.3
            rationale = "Stop widened for high volatility"
        elif signal.volatility_band == VolatilityBand.LOW:
            base_stop_distance *= 0.8
            rationale = "Tight stop in low volatility"
        else:
            rationale = "Standard ATR-based stop"
        
        if signal.zde_label:
            base_stop_distance *= 0.85
            rationale = f"{rationale}; tightened for ZDE research"
        
        if signal.direction == SignalDirection.LONG:
            stop_price = entry_price - base_stop_distance
        else:
            stop_price = entry_price + base_stop_distance
        
        stop_price = round(stop_price, 4)
        
        return ProtectiveStop(
            enabled=True,
            stop_price=stop_price,
            stop_type=OrderTypeEEO.STOP,
            rationale=rationale,
        )
    
    def _calculate_profit_targets(
        self,
        context: EEOContext,
        entry_price: float,
    ) -> List[ProfitTarget]:
        """
        Calculate profit targets.
        
        Uses Estimated Move median/upper band to place T1 and T2:
        - T1 = current_price + 0.5 * median_move
        - T2 = current_price + 0.8 * max_move
        
        Runner probability boost:
        - High runner_prob AND quality A+/A → extended targets
        """
        targets = []
        micro = context.microstructure
        signal = context.signal
        predictive = context.predictive
        
        atr = micro.atr_14 if micro.atr_14 > 0 else entry_price * 0.02
        
        if predictive.has_estimated_move() and predictive.estimated_move_median:
            median_move = predictive.estimated_move_median
            max_move = predictive.estimated_move_max or (median_move * 1.5)
            
            t1_move = median_move * self.profile.target_1_move_fraction
            t2_move = max_move * self.profile.target_2_move_fraction
            
            t1_rationale = "Median estimated move"
            t2_rationale = "Upper estimated move band"
        else:
            t1_move = atr * 1.5
            t2_move = atr * 2.5
            
            t1_rationale = "ATR-based target (1.5x)"
            t2_rationale = "ATR-based target (2.5x)"
        
        if predictive.is_runner_candidate() and predictive.is_high_quality():
            t2_move *= 1.3
            t2_rationale = f"{t2_rationale}; extended for runner candidate"
        
        if signal.direction == SignalDirection.LONG:
            t1_price = entry_price + t1_move
            t2_price = entry_price + t2_move
        else:
            t1_price = entry_price - t1_move
            t2_price = entry_price - t2_move
        
        targets.append(ProfitTarget(
            target_price=round(t1_price, 4),
            fraction_to_exit=self.profile.target_1_exit_fraction,
            style=ExitStyle.LIMIT,
            rationale=t1_rationale,
        ))
        
        targets.append(ProfitTarget(
            target_price=round(t2_price, 4),
            fraction_to_exit=self.profile.target_2_exit_fraction,
            style=ExitStyle.LIMIT,
            rationale=t2_rationale,
        ))
        
        return targets
    
    def _calculate_trailing_stop(
        self,
        context: EEOContext,
    ) -> Optional[TrailingStop]:
        """
        Calculate trailing stop parameters.
        
        Usage:
        - Optional; recommended more for high runner_prob cases
        
        Modes:
        - percent_trail: Trail by X% from maximum favorable excursion
        - atr_trail: Trail by K * ATR below highest close
        - structural_trail: Trail under swing structures
        """
        if not self.profile.use_trailing_stop:
            return None
        
        predictive = context.predictive
        micro = context.microstructure
        
        if predictive.is_runner_candidate():
            mode = TrailingStopMode.ATR
            atr_multiple = 1.5
            percent_trail = 0.015
            structural_ref = "highest close after entry"
        else:
            mode = TrailingStopMode.PERCENT
            atr_multiple = 2.0
            percent_trail = 0.02
            structural_ref = ""
        
        return TrailingStop(
            enabled=True,
            mode=mode,
            atr_multiple=atr_multiple,
            percent_trail=percent_trail,
            structural_reference=structural_ref,
        )
    
    def _calculate_time_exit(
        self,
        context: EEOContext,
    ) -> Optional[TimeBasedExit]:
        """
        Calculate time-based exit rules.
        
        Purpose:
        - Prevent capital from being stuck in dead trades
        
        Rules:
        - If trade spends > N bars without hitting target or stop → close
        - End-of-day exit option for intraday strategies
        """
        max_bars = self.profile.time_based_exit_bars
        
        signal = context.signal
        end_of_day = signal.timeframe in ("1m", "5m", "15m", "1h")
        
        rationale = f"Exit after {max_bars} bars to prevent capital lock-up"
        if end_of_day:
            rationale = f"{rationale}; EOD exit for intraday"
        
        return TimeBasedExit(
            enabled=True,
            max_bars_in_trade=max_bars,
            end_of_day_exit=end_of_day,
            rationale=rationale,
        )
    
    def _calculate_abort_conditions(
        self,
        context: EEOContext,
    ) -> List[AbortCondition]:
        """
        Calculate conditions that would abort the entry before fill.
        
        Conditions:
        - avoid_trade_prob spikes beyond threshold before fill
        - volatility regime flips to crash or similar
        """
        conditions = []
        
        predictive = context.predictive
        
        avoid_threshold = 0.7
        conditions.append(AbortCondition(
            condition_type="avoid_trade_spike",
            threshold=avoid_threshold,
            description=f"Abort if avoid_trade_prob exceeds {avoid_threshold} before fill",
        ))
        
        conditions.append(AbortCondition(
            condition_type="regime_flip_crash",
            threshold=0.0,
            description="Abort if volatility regime flips to crash before fill",
        ))
        
        conditions.append(AbortCondition(
            condition_type="spread_explosion",
            threshold=self.profile.max_spread_tolerance_pct * 2,
            description="Abort if spread exceeds 2x tolerance before fill",
        ))
        
        return conditions
    
    def _build_rationale(
        self,
        stop: Optional[ProtectiveStop],
        targets: List[ProfitTarget],
        trailing: Optional[TrailingStop],
        time_exit: Optional[TimeBasedExit],
    ) -> str:
        """Build a summary rationale for the exit optimization."""
        parts = []
        
        if stop and stop.enabled:
            parts.append(f"Stop: {stop.rationale}")
        
        if targets:
            target_summary = f"{len(targets)} profit targets"
            parts.append(target_summary)
        
        if trailing and trailing.enabled:
            parts.append(f"Trailing: {trailing.mode.value} mode")
        
        if time_exit and time_exit.enabled:
            parts.append(f"Time exit: {time_exit.max_bars_in_trade} bars")
        
        return "; ".join(parts) if parts else "No exit rules configured"
