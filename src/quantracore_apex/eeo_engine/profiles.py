"""
EEO Engine Policy Profiles

Defines configurable entry/exit "personalities" for different risk appetites.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class ProfileType(str, Enum):
    """Available profile types."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE_RESEARCH = "aggressive_research"


@dataclass
class EEOProfile:
    """
    Entry/Exit Optimization Profile.
    
    Defines the parameters that control how aggressively
    entries and exits are optimized.
    """
    name: str
    profile_type: ProfileType
    description: str
    
    per_trade_risk_fraction: float
    entry_aggressiveness: str
    max_entries_scaled: int
    use_trailing_stop: bool
    time_based_exit_bars: int
    
    entry_zone_atr_alpha: float = 0.5
    entry_zone_atr_beta: float = 0.3
    stop_atr_multiple: float = 2.0
    target_1_move_fraction: float = 0.5
    target_2_move_fraction: float = 0.8
    target_1_exit_fraction: float = 0.5
    target_2_exit_fraction: float = 0.5
    
    allow_market_orders: bool = True
    prefer_limit_orders: bool = True
    max_spread_tolerance_pct: float = 0.5
    
    require_protective_stop: bool = True
    allow_naked_entries: bool = False
    
    def is_aggressive(self) -> bool:
        """Check if profile is aggressive."""
        return self.entry_aggressiveness == "HIGH"
    
    def is_conservative(self) -> bool:
        """Check if profile is conservative."""
        return self.entry_aggressiveness == "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "profile_type": self.profile_type.value,
            "description": self.description,
            "per_trade_risk_fraction": self.per_trade_risk_fraction,
            "entry_aggressiveness": self.entry_aggressiveness,
            "max_entries_scaled": self.max_entries_scaled,
            "use_trailing_stop": self.use_trailing_stop,
            "time_based_exit_bars": self.time_based_exit_bars,
            "entry_zone_atr_alpha": self.entry_zone_atr_alpha,
            "entry_zone_atr_beta": self.entry_zone_atr_beta,
            "stop_atr_multiple": self.stop_atr_multiple,
            "target_1_move_fraction": self.target_1_move_fraction,
            "target_2_move_fraction": self.target_2_move_fraction,
            "allow_market_orders": self.allow_market_orders,
            "require_protective_stop": self.require_protective_stop,
        }


CONSERVATIVE_PROFILE = EEOProfile(
    name="Conservative",
    profile_type=ProfileType.CONSERVATIVE,
    description="Tight risk, modest targets, slow to enter.",
    per_trade_risk_fraction=0.005,
    entry_aggressiveness="LOW",
    max_entries_scaled=1,
    use_trailing_stop=False,
    time_based_exit_bars=30,
    entry_zone_atr_alpha=0.7,
    entry_zone_atr_beta=0.2,
    stop_atr_multiple=1.5,
    target_1_move_fraction=0.4,
    target_2_move_fraction=0.6,
    target_1_exit_fraction=0.6,
    target_2_exit_fraction=0.4,
    allow_market_orders=False,
    prefer_limit_orders=True,
    max_spread_tolerance_pct=0.3,
    require_protective_stop=True,
    allow_naked_entries=False,
)


BALANCED_PROFILE = EEOProfile(
    name="Balanced",
    profile_type=ProfileType.BALANCED,
    description="Default institutional-style profile.",
    per_trade_risk_fraction=0.01,
    entry_aggressiveness="MEDIUM",
    max_entries_scaled=2,
    use_trailing_stop=True,
    time_based_exit_bars=50,
    entry_zone_atr_alpha=0.5,
    entry_zone_atr_beta=0.3,
    stop_atr_multiple=2.0,
    target_1_move_fraction=0.5,
    target_2_move_fraction=0.8,
    target_1_exit_fraction=0.5,
    target_2_exit_fraction=0.5,
    allow_market_orders=True,
    prefer_limit_orders=True,
    max_spread_tolerance_pct=0.5,
    require_protective_stop=True,
    allow_naked_entries=False,
)


AGGRESSIVE_RESEARCH_PROFILE = EEOProfile(
    name="Aggressive Research",
    profile_type=ProfileType.AGGRESSIVE_RESEARCH,
    description="For research/backtest only; not for retail or live.",
    per_trade_risk_fraction=0.02,
    entry_aggressiveness="HIGH",
    max_entries_scaled=3,
    use_trailing_stop=True,
    time_based_exit_bars=80,
    entry_zone_atr_alpha=0.3,
    entry_zone_atr_beta=0.5,
    stop_atr_multiple=2.5,
    target_1_move_fraction=0.6,
    target_2_move_fraction=1.0,
    target_1_exit_fraction=0.4,
    target_2_exit_fraction=0.6,
    allow_market_orders=True,
    prefer_limit_orders=False,
    max_spread_tolerance_pct=0.8,
    require_protective_stop=True,
    allow_naked_entries=False,
)


PROFILES: Dict[ProfileType, EEOProfile] = {
    ProfileType.CONSERVATIVE: CONSERVATIVE_PROFILE,
    ProfileType.BALANCED: BALANCED_PROFILE,
    ProfileType.AGGRESSIVE_RESEARCH: AGGRESSIVE_RESEARCH_PROFILE,
}


def get_profile(profile_type: ProfileType) -> EEOProfile:
    """Get a profile by type."""
    return PROFILES.get(profile_type, BALANCED_PROFILE)


def get_profile_by_name(name: str) -> EEOProfile:
    """Get a profile by name string."""
    name_lower = name.lower()
    if "conservative" in name_lower:
        return CONSERVATIVE_PROFILE
    elif "aggressive" in name_lower:
        return AGGRESSIVE_RESEARCH_PROFILE
    return BALANCED_PROFILE
