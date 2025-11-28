"""
Sector context module for QuantraCore Apex.

Provides sector-aware analysis adjustments.
"""

from typing import Optional, Dict
from .schemas import RegimeType


SECTOR_VOLATILITY_PROFILES = {
    "technology": {"base_vol": 0.25, "sensitivity": 1.2},
    "healthcare": {"base_vol": 0.20, "sensitivity": 1.0},
    "financials": {"base_vol": 0.22, "sensitivity": 1.1},
    "energy": {"base_vol": 0.30, "sensitivity": 1.3},
    "utilities": {"base_vol": 0.12, "sensitivity": 0.7},
    "consumer_discretionary": {"base_vol": 0.23, "sensitivity": 1.1},
    "consumer_staples": {"base_vol": 0.14, "sensitivity": 0.8},
    "industrials": {"base_vol": 0.20, "sensitivity": 1.0},
    "materials": {"base_vol": 0.25, "sensitivity": 1.2},
    "real_estate": {"base_vol": 0.18, "sensitivity": 0.9},
    "communication_services": {"base_vol": 0.24, "sensitivity": 1.1},
    "default": {"base_vol": 0.20, "sensitivity": 1.0},
}


class SectorContext:
    """
    Provides sector-specific context for analysis.
    """
    
    def __init__(self, sector: Optional[str] = None):
        self.sector = sector or "default"
        self.profile = SECTOR_VOLATILITY_PROFILES.get(
            self.sector.lower().replace(" ", "_"),
            SECTOR_VOLATILITY_PROFILES["default"]
        )
    
    def get_volatility_baseline(self) -> float:
        """Get expected baseline volatility for sector."""
        return self.profile["base_vol"]
    
    def get_sensitivity_multiplier(self) -> float:
        """Get sensitivity multiplier for sector."""
        return self.profile["sensitivity"]
    
    def adjust_score_for_sector(self, score: float) -> float:
        """
        Adjust score based on sector characteristics.
        """
        sensitivity = self.get_sensitivity_multiplier()
        
        base_adjustment = (score - 50) * (sensitivity - 1.0) * 0.1
        
        return float(max(0, min(100, score + base_adjustment)))
    
    def get_sector_regime_bias(self, regime: RegimeType) -> float:
        """
        Get sector-specific regime bias.
        Some sectors naturally trend more than others.
        """
        if self.sector in ["utilities", "consumer_staples"]:
            if regime == RegimeType.RANGE_BOUND:
                return 0.1
            elif regime in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]:
                return -0.05
        elif self.sector in ["technology", "energy"]:
            if regime == RegimeType.VOLATILE:
                return 0.05
            elif regime == RegimeType.COMPRESSED:
                return 0.15
        
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sector": self.sector,
            "base_volatility": self.profile["base_vol"],
            "sensitivity": self.profile["sensitivity"],
        }


def apply_sector_context(score: float, sector: Optional[str] = None) -> float:
    """
    Convenience function to apply sector context to a score.
    
    Args:
        score: The QuantraScore to adjust
        sector: Optional sector name
        
    Returns:
        Adjusted score based on sector characteristics
    """
    ctx = SectorContext(sector)
    return ctx.adjust_score_for_sector(score)
