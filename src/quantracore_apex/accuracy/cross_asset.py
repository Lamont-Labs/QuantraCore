"""
Cross-Asset Features - Sector Correlations, VIX, Market Internals.

Provides:
- VIX regime classification
- Sector momentum and rotation
- Market breadth indicators
- Inter-market correlations
- Risk-on/Risk-off signals

These market-wide features improve accuracy by providing context
that symbol-specific analysis misses.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class VIXRegime(Enum):
    """VIX volatility regime classifications."""
    COMPLACENT = "complacent"
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


class MarketRegimeType(Enum):
    """Market-wide regime types."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    NEUTRAL = "neutral"
    TRANSITION = "transition"


class SectorName(Enum):
    """Standard sector classifications."""
    TECHNOLOGY = "XLK"
    FINANCIALS = "XLF"
    HEALTHCARE = "XLV"
    CONSUMER_DISCRETIONARY = "XLY"
    CONSUMER_STAPLES = "XLP"
    ENERGY = "XLE"
    UTILITIES = "XLU"
    INDUSTRIALS = "XLI"
    MATERIALS = "XLB"
    REAL_ESTATE = "XLRE"
    COMMUNICATION = "XLC"


@dataclass
class VIXAnalysis:
    """VIX analysis result."""
    current_level: float
    regime: VIXRegime
    percentile: float
    term_structure: str
    change_1d: float
    change_5d: float
    spike_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


@dataclass
class SectorAnalysis:
    """Sector momentum and rotation analysis."""
    sector_returns: Dict[str, float]
    leading_sectors: List[str]
    lagging_sectors: List[str]
    rotation_signal: str
    sector_dispersion: float
    defensive_momentum: float
    cyclical_momentum: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketBreadthAnalysis:
    """Market breadth indicators."""
    advance_decline_ratio: float
    new_highs_lows_ratio: float
    percent_above_50ma: float
    percent_above_200ma: float
    mcclellan_oscillator: float
    breadth_thrust: bool
    breadth_signal: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossAssetFeatures:
    """Combined cross-asset feature set."""
    timestamp: datetime
    vix: VIXAnalysis
    sectors: SectorAnalysis
    breadth: MarketBreadthAnalysis
    market_regime: MarketRegimeType
    risk_appetite_score: float
    correlation_regime: str
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp.isoformat(),
            "vix": self.vix.to_dict(),
            "sectors": self.sectors.to_dict(),
            "breadth": self.breadth.to_dict(),
            "market_regime": self.market_regime.value,
            "risk_appetite_score": self.risk_appetite_score,
            "correlation_regime": self.correlation_regime,
        }
        return d
    
    def to_feature_vector(self) -> Dict[str, float]:
        """Convert to flat feature dictionary for model input."""
        return {
            "vix_level": self.vix.current_level,
            "vix_percentile": self.vix.percentile,
            "vix_change_1d": self.vix.change_1d,
            "vix_regime_encoded": list(VIXRegime).index(self.vix.regime),
            "sector_dispersion": self.sectors.sector_dispersion,
            "defensive_momentum": self.sectors.defensive_momentum,
            "cyclical_momentum": self.sectors.cyclical_momentum,
            "advance_decline": self.breadth.advance_decline_ratio,
            "pct_above_50ma": self.breadth.percent_above_50ma,
            "pct_above_200ma": self.breadth.percent_above_200ma,
            "risk_appetite": self.risk_appetite_score,
            "market_regime_encoded": list(MarketRegimeType).index(self.market_regime),
        }


class VIXAnalyzer:
    """Analyzes VIX for volatility regime and signals."""
    
    VIX_THRESHOLDS = {
        "complacent": 12,
        "low": 15,
        "normal": 20,
        "elevated": 25,
        "high": 30,
        "extreme": 40,
    }
    
    def __init__(self, history_length: int = 252):
        self._history: deque = deque(maxlen=history_length)
    
    def update(self, vix_level: float) -> None:
        """Add new VIX observation."""
        self._history.append((datetime.utcnow(), vix_level))
    
    def analyze(
        self,
        current_vix: Optional[float] = None,
        vix_1d_ago: Optional[float] = None,
        vix_5d_ago: Optional[float] = None,
    ) -> VIXAnalysis:
        """
        Analyze current VIX conditions.
        
        Args:
            current_vix: Current VIX level
            vix_1d_ago: VIX level 1 day ago
            vix_5d_ago: VIX level 5 days ago
        """
        if current_vix is None:
            if self._history:
                current_vix = self._history[-1][1]
            else:
                current_vix = 20.0
        
        if current_vix < self.VIX_THRESHOLDS["complacent"]:
            regime = VIXRegime.COMPLACENT
        elif current_vix < self.VIX_THRESHOLDS["low"]:
            regime = VIXRegime.LOW
        elif current_vix < self.VIX_THRESHOLDS["normal"]:
            regime = VIXRegime.NORMAL
        elif current_vix < self.VIX_THRESHOLDS["elevated"]:
            regime = VIXRegime.ELEVATED
        elif current_vix < self.VIX_THRESHOLDS["high"]:
            regime = VIXRegime.HIGH
        else:
            regime = VIXRegime.EXTREME
        
        if len(self._history) >= 252:
            all_vix = [v for _, v in self._history]
            percentile = float(np.sum(np.array(all_vix) < current_vix) / len(all_vix) * 100)
        else:
            percentile = 50.0
        
        if vix_1d_ago:
            change_1d = (current_vix - vix_1d_ago) / vix_1d_ago * 100
        else:
            change_1d = 0.0
        
        if vix_5d_ago:
            change_5d = (current_vix - vix_5d_ago) / vix_5d_ago * 100
        else:
            change_5d = 0.0
        
        spike_detected = change_1d > 20 or (change_5d > 50 and current_vix > 25)
        
        term_structure = "contango" if current_vix < 20 else "backwardation"
        
        return VIXAnalysis(
            current_level=current_vix,
            regime=regime,
            percentile=percentile,
            term_structure=term_structure,
            change_1d=change_1d,
            change_5d=change_5d,
            spike_detected=spike_detected,
        )


class SectorAnalyzer:
    """Analyzes sector momentum and rotation."""
    
    DEFENSIVE_SECTORS = [SectorName.UTILITIES, SectorName.CONSUMER_STAPLES, SectorName.HEALTHCARE]
    CYCLICAL_SECTORS = [SectorName.TECHNOLOGY, SectorName.CONSUMER_DISCRETIONARY, SectorName.FINANCIALS]
    
    def __init__(self):
        self._sector_returns: Dict[str, deque] = {
            s.value: deque(maxlen=20) for s in SectorName
        }
    
    def update(self, sector: str, return_value: float) -> None:
        """Add new sector return observation."""
        if sector in self._sector_returns:
            self._sector_returns[sector].append(return_value)
    
    def analyze(
        self,
        sector_returns_5d: Optional[Dict[str, float]] = None,
    ) -> SectorAnalysis:
        """
        Analyze sector momentum and rotation.
        
        Args:
            sector_returns_5d: Dict of sector -> 5-day return
        """
        if sector_returns_5d is None:
            sector_returns_5d = {s.value: 0.0 for s in SectorName}
        
        sorted_sectors = sorted(
            sector_returns_5d.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        leading = [s for s, _ in sorted_sectors[:3]]
        lagging = [s for s, _ in sorted_sectors[-3:]]
        
        returns = list(sector_returns_5d.values())
        dispersion = float(np.std(returns)) if returns else 0.0
        
        defensive_rets = [
            sector_returns_5d.get(s.value, 0.0) for s in self.DEFENSIVE_SECTORS
        ]
        cyclical_rets = [
            sector_returns_5d.get(s.value, 0.0) for s in self.CYCLICAL_SECTORS
        ]
        
        defensive_momentum = float(np.mean(defensive_rets)) if defensive_rets else 0.0
        cyclical_momentum = float(np.mean(cyclical_rets)) if cyclical_rets else 0.0
        
        if cyclical_momentum > defensive_momentum + 1.0:
            rotation_signal = "risk_on_rotation"
        elif defensive_momentum > cyclical_momentum + 1.0:
            rotation_signal = "risk_off_rotation"
        else:
            rotation_signal = "neutral"
        
        return SectorAnalysis(
            sector_returns=sector_returns_5d,
            leading_sectors=leading,
            lagging_sectors=lagging,
            rotation_signal=rotation_signal,
            sector_dispersion=dispersion,
            defensive_momentum=defensive_momentum,
            cyclical_momentum=cyclical_momentum,
        )


class BreadthAnalyzer:
    """Analyzes market breadth indicators."""
    
    def __init__(self):
        self._ad_history: deque = deque(maxlen=20)
    
    def analyze(
        self,
        advances: int = 0,
        declines: int = 0,
        new_highs: int = 0,
        new_lows: int = 0,
        pct_above_50ma: float = 50.0,
        pct_above_200ma: float = 50.0,
    ) -> MarketBreadthAnalysis:
        """
        Analyze market breadth.
        
        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks
            new_highs: Number of new 52-week highs
            new_lows: Number of new 52-week lows
            pct_above_50ma: Percent of stocks above 50-day MA
            pct_above_200ma: Percent of stocks above 200-day MA
        """
        total = advances + declines
        if total > 0:
            ad_ratio = advances / total
        else:
            ad_ratio = 0.5
        
        self._ad_history.append(advances - declines)
        
        hl_total = new_highs + new_lows
        if hl_total > 0:
            hl_ratio = new_highs / hl_total
        else:
            hl_ratio = 0.5
        
        if len(self._ad_history) >= 19:
            ema_19 = float(np.mean(list(self._ad_history)[-19:]))
            ema_39 = float(np.mean(list(self._ad_history)))
            mcclellan = ema_19 - ema_39
        else:
            mcclellan = 0.0
        
        breadth_thrust = pct_above_50ma > 80 and ad_ratio > 0.9
        
        if pct_above_50ma > 70 and ad_ratio > 0.6:
            breadth_signal = "strong_bullish"
        elif pct_above_50ma > 50 and ad_ratio > 0.5:
            breadth_signal = "bullish"
        elif pct_above_50ma < 30 and ad_ratio < 0.4:
            breadth_signal = "bearish"
        elif pct_above_50ma < 20 and ad_ratio < 0.3:
            breadth_signal = "strong_bearish"
        else:
            breadth_signal = "neutral"
        
        return MarketBreadthAnalysis(
            advance_decline_ratio=ad_ratio,
            new_highs_lows_ratio=hl_ratio,
            percent_above_50ma=pct_above_50ma,
            percent_above_200ma=pct_above_200ma,
            mcclellan_oscillator=mcclellan,
            breadth_thrust=breadth_thrust,
            breadth_signal=breadth_signal,
        )


class CrossAssetAnalyzer:
    """
    Combined cross-asset feature generator.
    
    Aggregates VIX, sector, and breadth analysis into a unified
    feature set for model input.
    
    Usage:
        analyzer = CrossAssetAnalyzer()
        
        # Update with market data
        analyzer.update_vix(18.5)
        analyzer.update_sectors({"XLK": 2.5, "XLF": 1.0, ...})
        
        # Get combined features
        features = analyzer.get_features()
    """
    
    def __init__(
        self,
        save_dir: str = "data/cross_asset",
    ):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        
        self._vix_analyzer = VIXAnalyzer()
        self._sector_analyzer = SectorAnalyzer()
        self._breadth_analyzer = BreadthAnalyzer()
        
        self._current_vix: float = 20.0
        self._current_sectors: Dict[str, float] = {}
        self._current_breadth: Dict[str, Any] = {}
        
        self._last_update: Optional[datetime] = None
    
    def update_vix(
        self,
        current_level: float,
        level_1d_ago: Optional[float] = None,
        level_5d_ago: Optional[float] = None,
    ) -> VIXAnalysis:
        """Update VIX data."""
        self._current_vix = current_level
        self._vix_analyzer.update(current_level)
        self._last_update = datetime.utcnow()
        
        return self._vix_analyzer.analyze(current_level, level_1d_ago, level_5d_ago)
    
    def update_sectors(
        self,
        sector_returns_5d: Dict[str, float],
    ) -> SectorAnalysis:
        """Update sector returns."""
        self._current_sectors = sector_returns_5d
        self._last_update = datetime.utcnow()
        
        for sector, ret in sector_returns_5d.items():
            self._sector_analyzer.update(sector, ret)
        
        return self._sector_analyzer.analyze(sector_returns_5d)
    
    def update_breadth(
        self,
        advances: int,
        declines: int,
        new_highs: int = 0,
        new_lows: int = 0,
        pct_above_50ma: float = 50.0,
        pct_above_200ma: float = 50.0,
    ) -> MarketBreadthAnalysis:
        """Update breadth data."""
        self._current_breadth = {
            "advances": advances,
            "declines": declines,
            "new_highs": new_highs,
            "new_lows": new_lows,
            "pct_above_50ma": pct_above_50ma,
            "pct_above_200ma": pct_above_200ma,
        }
        self._last_update = datetime.utcnow()
        
        return self._breadth_analyzer.analyze(**self._current_breadth)
    
    def get_features(self) -> CrossAssetFeatures:
        """Get combined cross-asset features."""
        vix = self._vix_analyzer.analyze(self._current_vix)
        sectors = self._sector_analyzer.analyze(self._current_sectors)
        breadth = self._breadth_analyzer.analyze(**self._current_breadth) if self._current_breadth else MarketBreadthAnalysis(
            advance_decline_ratio=0.5,
            new_highs_lows_ratio=0.5,
            percent_above_50ma=50.0,
            percent_above_200ma=50.0,
            mcclellan_oscillator=0.0,
            breadth_thrust=False,
            breadth_signal="neutral",
        )
        
        risk_score = 0.0
        
        if vix.regime in [VIXRegime.COMPLACENT, VIXRegime.LOW]:
            risk_score += 0.3
        elif vix.regime in [VIXRegime.HIGH, VIXRegime.EXTREME]:
            risk_score -= 0.3
        
        if sectors.rotation_signal == "risk_on_rotation":
            risk_score += 0.2
        elif sectors.rotation_signal == "risk_off_rotation":
            risk_score -= 0.2
        
        if breadth.breadth_signal in ["bullish", "strong_bullish"]:
            risk_score += 0.2
        elif breadth.breadth_signal in ["bearish", "strong_bearish"]:
            risk_score -= 0.2
        
        risk_score = (risk_score + 0.5) / 1.0
        risk_score = max(0.0, min(1.0, risk_score))
        
        if risk_score > 0.6:
            market_regime = MarketRegimeType.RISK_ON
        elif risk_score < 0.4:
            market_regime = MarketRegimeType.RISK_OFF
        else:
            market_regime = MarketRegimeType.NEUTRAL
        
        if vix.regime == VIXRegime.ELEVATED:
            correlation_regime = "stress"
        elif vix.regime in [VIXRegime.COMPLACENT, VIXRegime.LOW]:
            correlation_regime = "complacent"
        else:
            correlation_regime = "normal"
        
        return CrossAssetFeatures(
            timestamp=datetime.utcnow(),
            vix=vix,
            sectors=sectors,
            breadth=breadth,
            market_regime=market_regime,
            risk_appetite_score=risk_score,
            correlation_regime=correlation_regime,
        )
    
    def get_feature_vector(self) -> Dict[str, float]:
        """Get features as flat dictionary for model input."""
        return self.get_features().to_feature_vector()
    
    def save_snapshot(self) -> None:
        """Save current state to disk."""
        features = self.get_features()
        
        snapshot_file = self._save_dir / f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(snapshot_file, "w") as f:
            json.dump(features.to_dict(), f, indent=2)
    
    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update


_cross_asset_analyzer: Optional[CrossAssetAnalyzer] = None


def get_cross_asset_analyzer() -> CrossAssetAnalyzer:
    """Get global cross-asset analyzer instance."""
    global _cross_asset_analyzer
    if _cross_asset_analyzer is None:
        _cross_asset_analyzer = CrossAssetAnalyzer()
    return _cross_asset_analyzer


__all__ = [
    "CrossAssetAnalyzer",
    "CrossAssetFeatures",
    "VIXAnalyzer",
    "VIXAnalysis",
    "VIXRegime",
    "SectorAnalyzer",
    "SectorAnalysis",
    "SectorName",
    "BreadthAnalyzer",
    "MarketBreadthAnalysis",
    "MarketRegimeType",
    "get_cross_asset_analyzer",
]
