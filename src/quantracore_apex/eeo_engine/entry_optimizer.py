"""
EEO Entry Optimizer

Calculates optimal entry zones, prices, and order types for trading signals.
Implements the "Best Entry" logic from the specification.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .enums import (
    SignalDirection,
    OrderTypeEEO,
    EntryMode,
    VolatilityBand,
    LiquidityBand,
    RegimeType,
)
from .models import BaseEntry, ScaledEntry, EntryZone
from .contexts import EEOContext
from .profiles import EEOProfile

logger = logging.getLogger(__name__)


@dataclass
class EntryOptimization:
    """Result of entry optimization."""
    base_entry: BaseEntry
    scaled_entries: List[ScaledEntry]
    entry_mode: EntryMode
    rationale: str
    strategy_used: str


class EntryOptimizer:
    """
    Optimizes entry parameters for trading signals.
    
    Goals:
    - Avoid chasing into bad fills
    - Use structure, volatility, and spreads to pick attractive entry zones
    - Respect ZDE research while allowing realistic fills
    - Remain deterministic and auditable
    """
    
    STRATEGY_BASELINE_LONG = "baseline_long"
    STRATEGY_BASELINE_SHORT = "baseline_short"
    STRATEGY_ZDE_AWARE = "zde_aware_adjustment"
    STRATEGY_HIGH_VOLATILITY = "high_volatility_mode"
    STRATEGY_LOW_LIQUIDITY = "low_liquidity_mode"
    STRATEGY_RUNNER_ANTICIPATION = "runner_anticipation_adjust"
    
    def __init__(self, profile: EEOProfile):
        self.profile = profile
    
    def optimize(self, context: EEOContext) -> EntryOptimization:
        """
        Calculate optimal entry parameters.
        
        Args:
            context: Complete EEO context with signal, predictive, 
                    microstructure, and risk information
        
        Returns:
            EntryOptimization with base entry, scaled entries, and rationale
        """
        strategy, rationale = self._select_strategy(context)
        
        if strategy == self.STRATEGY_HIGH_VOLATILITY:
            return self._apply_high_volatility_strategy(context, strategy, rationale)
        elif strategy == self.STRATEGY_LOW_LIQUIDITY:
            return self._apply_low_liquidity_strategy(context, strategy, rationale)
        elif strategy == self.STRATEGY_RUNNER_ANTICIPATION:
            return self._apply_runner_strategy(context, strategy, rationale)
        elif strategy == self.STRATEGY_ZDE_AWARE:
            return self._apply_zde_aware_strategy(context, strategy, rationale)
        else:
            return self._apply_baseline_strategy(context, strategy, rationale)
    
    def _select_strategy(self, context: EEOContext) -> Tuple[str, str]:
        """Select the most appropriate entry strategy based on context."""
        signal = context.signal
        predictive = context.predictive
        micro = context.microstructure
        
        if signal.volatility_band == VolatilityBand.HIGH or \
           signal.regime_type in (RegimeType.SQUEEZE, RegimeType.CRASH):
            return self.STRATEGY_HIGH_VOLATILITY, "High volatility detected"
        
        if signal.liquidity_band == LiquidityBand.LOW or micro.is_wide_spread():
            return self.STRATEGY_LOW_LIQUIDITY, "Low liquidity conditions"
        
        if predictive.is_runner_candidate() and predictive.is_high_quality():
            return self.STRATEGY_RUNNER_ANTICIPATION, "High runner probability with quality signal"
        
        if signal.zde_label:
            return self.STRATEGY_ZDE_AWARE, "ZDE research applied"
        
        if signal.direction == SignalDirection.LONG:
            return self.STRATEGY_BASELINE_LONG, "Standard long entry"
        else:
            return self.STRATEGY_BASELINE_SHORT, "Standard short entry"
    
    def _calculate_entry_zone(
        self,
        context: EEOContext,
        alpha: float,
        beta: float,
    ) -> Tuple[EntryZone, float]:
        """
        Calculate entry zone using mid price as anchor.
        
        For LONG: zone is [anchor - alpha*ATR, anchor + beta*ATR]
        For SHORT: zone is [anchor - beta*ATR, anchor + alpha*ATR]
        """
        micro = context.microstructure
        signal = context.signal
        
        anchor = micro.mid_price
        atr = micro.atr_14 if micro.atr_14 > 0 else anchor * 0.02
        
        if signal.direction == SignalDirection.LONG:
            lower = anchor - (alpha * atr)
            upper = anchor + (beta * atr)
            entry_price = anchor - (micro.spread * 0.25)
        else:
            lower = anchor - (beta * atr)
            upper = anchor + (alpha * atr)
            entry_price = anchor + (micro.spread * 0.25)
        
        entry_price = max(lower, min(upper, entry_price))
        
        return EntryZone(lower=lower, upper=upper), entry_price
    
    def _select_order_type(self, context: EEOContext) -> OrderTypeEEO:
        """Select appropriate order type based on conditions."""
        micro = context.microstructure
        
        if not self.profile.allow_market_orders:
            return OrderTypeEEO.LIMIT
        
        if micro.is_wide_spread(self.profile.max_spread_tolerance_pct):
            return OrderTypeEEO.LIMIT
        
        if self.profile.prefer_limit_orders:
            return OrderTypeEEO.LIMIT
        
        return OrderTypeEEO.MARKET
    
    def _apply_baseline_strategy(
        self,
        context: EEOContext,
        strategy: str,
        rationale: str,
    ) -> EntryOptimization:
        """Apply baseline entry strategy for normal conditions."""
        alpha = self.profile.entry_zone_atr_alpha
        beta = self.profile.entry_zone_atr_beta
        
        entry_zone, entry_price = self._calculate_entry_zone(context, alpha, beta)
        order_type = self._select_order_type(context)
        
        base_entry = BaseEntry(
            order_type=order_type,
            entry_price=round(entry_price, 4),
            entry_zone=entry_zone,
            time_window_bars=10,
        )
        
        scaled_entries = []
        entry_mode = EntryMode.SINGLE
        
        if self.profile.max_entries_scaled > 1 and self.profile.entry_aggressiveness != "LOW":
            scaled_entries, entry_mode = self._create_scaled_entries(
                context, entry_zone, order_type
            )
        
        return EntryOptimization(
            base_entry=base_entry,
            scaled_entries=scaled_entries,
            entry_mode=entry_mode,
            rationale=rationale,
            strategy_used=strategy,
        )
    
    def _apply_high_volatility_strategy(
        self,
        context: EEOContext,
        strategy: str,
        rationale: str,
    ) -> EntryOptimization:
        """
        High volatility mode:
        - Avoid MARKET orders where spreads are large
        - Prefer LIMIT with more conservative prices
        - Scale entries to avoid full size in one print
        """
        alpha = self.profile.entry_zone_atr_alpha * 1.5
        beta = self.profile.entry_zone_atr_beta * 0.8
        
        entry_zone, entry_price = self._calculate_entry_zone(context, alpha, beta)
        
        order_type = OrderTypeEEO.LIMIT
        
        base_entry = BaseEntry(
            order_type=order_type,
            entry_price=round(entry_price, 4),
            entry_zone=entry_zone,
            time_window_bars=15,
        )
        
        if self.profile.max_entries_scaled > 1:
            scaled_entries, entry_mode = self._create_scaled_entries(
                context, entry_zone, order_type, num_entries=min(3, self.profile.max_entries_scaled)
            )
        else:
            scaled_entries = []
            entry_mode = EntryMode.SINGLE
        
        return EntryOptimization(
            base_entry=base_entry,
            scaled_entries=scaled_entries,
            entry_mode=entry_mode,
            rationale=f"{rationale}; using conservative limits and scaled entries",
            strategy_used=strategy,
        )
    
    def _apply_low_liquidity_strategy(
        self,
        context: EEOContext,
        strategy: str,
        rationale: str,
    ) -> EntryOptimization:
        """
        Low liquidity mode:
        - Reduce size_notional (handled by caller)
        - Place more conservative LIMIT entries further inside the spread
        """
        alpha = self.profile.entry_zone_atr_alpha * 1.3
        beta = self.profile.entry_zone_atr_beta * 0.6
        
        entry_zone, entry_price = self._calculate_entry_zone(context, alpha, beta)
        
        micro = context.microstructure
        signal = context.signal
        
        if signal.direction == SignalDirection.LONG:
            entry_price = entry_price - (micro.spread * 0.3)
        else:
            entry_price = entry_price + (micro.spread * 0.3)
        
        entry_price = max(entry_zone.lower, min(entry_zone.upper, entry_price))
        
        base_entry = BaseEntry(
            order_type=OrderTypeEEO.LIMIT,
            entry_price=round(entry_price, 4),
            entry_zone=entry_zone,
            time_window_bars=20,
        )
        
        return EntryOptimization(
            base_entry=base_entry,
            scaled_entries=[],
            entry_mode=EntryMode.SINGLE,
            rationale=f"{rationale}; conservative limit inside spread",
            strategy_used=strategy,
        )
    
    def _apply_runner_strategy(
        self,
        context: EEOContext,
        strategy: str,
        rationale: str,
    ) -> EntryOptimization:
        """
        Runner anticipation mode:
        - Allow slightly more aggressive entry within entry_zone
        - But never override risk or ZDE constraints
        """
        alpha = self.profile.entry_zone_atr_alpha * 0.8
        beta = self.profile.entry_zone_atr_beta * 1.2
        
        entry_zone, entry_price = self._calculate_entry_zone(context, alpha, beta)
        
        micro = context.microstructure
        signal = context.signal
        
        if signal.direction == SignalDirection.LONG:
            entry_price = min(micro.mid_price + (micro.spread * 0.1), entry_zone.upper)
        else:
            entry_price = max(micro.mid_price - (micro.spread * 0.1), entry_zone.lower)
        
        order_type = self._select_order_type(context)
        if order_type == OrderTypeEEO.LIMIT and self.profile.allow_market_orders:
            order_type = OrderTypeEEO.MARKET
        
        base_entry = BaseEntry(
            order_type=order_type,
            entry_price=round(entry_price, 4),
            entry_zone=entry_zone,
            time_window_bars=8,
        )
        
        return EntryOptimization(
            base_entry=base_entry,
            scaled_entries=[],
            entry_mode=EntryMode.SINGLE,
            rationale=f"{rationale}; aggressive entry for runner candidate",
            strategy_used=strategy,
        )
    
    def _apply_zde_aware_strategy(
        self,
        context: EEOContext,
        strategy: str,
        rationale: str,
    ) -> EntryOptimization:
        """
        ZDE-aware adjustment:
        - If ZDE rate historically high: tighten entry_zone (narrow range)
        - If ZDE rate low: require better discount for entry
        """
        alpha = self.profile.entry_zone_atr_alpha * 0.7
        beta = self.profile.entry_zone_atr_beta * 0.7
        
        entry_zone, entry_price = self._calculate_entry_zone(context, alpha, beta)
        
        base_entry = BaseEntry(
            order_type=OrderTypeEEO.LIMIT,
            entry_price=round(entry_price, 4),
            entry_zone=entry_zone,
            time_window_bars=12,
        )
        
        return EntryOptimization(
            base_entry=base_entry,
            scaled_entries=[],
            entry_mode=EntryMode.SINGLE,
            rationale=f"{rationale}; tightened zone based on ZDE research",
            strategy_used=strategy,
        )
    
    def _create_scaled_entries(
        self,
        context: EEOContext,
        entry_zone: EntryZone,
        order_type: OrderTypeEEO,
        num_entries: int = 2,
    ) -> Tuple[List[ScaledEntry], EntryMode]:
        """Create scaled entry positions across the entry zone."""
        scaled_entries = []
        
        if num_entries < 2:
            return [], EntryMode.SINGLE
        
        num_entries = min(num_entries, self.profile.max_entries_scaled)
        fraction_per_entry = 1.0 / num_entries
        
        zone_width = entry_zone.width()
        
        for i in range(num_entries):
            progress = i / (num_entries - 1) if num_entries > 1 else 0.5
            
            if context.signal.direction == SignalDirection.LONG:
                price = entry_zone.lower + (zone_width * (1 - progress) * 0.8)
            else:
                price = entry_zone.upper - (zone_width * (1 - progress) * 0.8)
            
            scaled_entries.append(ScaledEntry(
                fraction_of_size=round(fraction_per_entry, 2),
                order_type=order_type,
                entry_price=round(price, 4),
            ))
        
        return scaled_entries, EntryMode.SCALED_IN
