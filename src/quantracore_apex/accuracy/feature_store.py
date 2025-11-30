"""
Feature Store - Centralized Feature Management with Data Quality Audits.

Provides:
- Unified feature registry for all models
- Data quality audits (completeness, freshness, drift)
- Cross-asset features (sector correlations, VIX, market internals)
- Feature versioning and lineage tracking
- Automatic stale data detection

Ensures consistent, high-quality features across all prediction heads.
"""

import os
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """Definition of a feature in the store."""
    name: str
    dtype: str
    category: str
    description: str
    source: str
    compute_fn: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureQualityReport:
    """Quality report for a feature."""
    feature_name: str
    completeness: float
    freshness_hours: float
    drift_score: float
    null_rate: float
    outlier_rate: float
    status: str
    issues: List[str] = field(default_factory=list)
    
    @property
    def is_healthy(self) -> bool:
        return (
            self.completeness >= 0.95 and
            self.null_rate < 0.05 and
            self.drift_score < 2.0 and
            self.freshness_hours < 24
        )


@dataclass
class FeatureSnapshot:
    """Point-in-time feature values for a symbol."""
    symbol: str
    timestamp: datetime
    features: Dict[str, float]
    regime: str = "unknown"
    sector: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureRegistry:
    """Registry of all available features."""
    
    CORE_FEATURES = {
        "quantra_score": FeatureDefinition(
            name="quantra_score",
            dtype="float",
            category="signal",
            description="Primary composite signal score (0-100)",
            source="apex_engine",
        ),
        "regime_encoded": FeatureDefinition(
            name="regime_encoded",
            dtype="int",
            category="context",
            description="Market regime (0=trend_up, 1=trend_down, 2=chop, 3=squeeze, 4=crash)",
            source="apex_engine",
        ),
        "entropy_band": FeatureDefinition(
            name="entropy_band",
            dtype="int",
            category="risk",
            description="Entropy level (0=low, 1=mid, 2=high)",
            source="apex_engine",
        ),
        "volatility_band": FeatureDefinition(
            name="volatility_band",
            dtype="int",
            category="risk",
            description="Volatility level (0=low, 1=mid, 2=high)",
            source="apex_engine",
        ),
        "liquidity_band": FeatureDefinition(
            name="liquidity_band",
            dtype="int",
            category="risk",
            description="Liquidity level (0=low, 1=mid, 2=high)",
            source="apex_engine",
        ),
        "protocol_count": FeatureDefinition(
            name="protocol_count",
            dtype="int",
            category="signal",
            description="Number of protocols that fired",
            source="apex_engine",
        ),
        "ret_1d": FeatureDefinition(
            name="ret_1d",
            dtype="float",
            category="return",
            description="1-day forward return",
            source="market_data",
        ),
        "ret_3d": FeatureDefinition(
            name="ret_3d",
            dtype="float",
            category="return",
            description="3-day forward return",
            source="market_data",
        ),
        "ret_5d": FeatureDefinition(
            name="ret_5d",
            dtype="float",
            category="return",
            description="5-day forward return",
            source="market_data",
        ),
        "atr_pct": FeatureDefinition(
            name="atr_pct",
            dtype="float",
            category="volatility",
            description="ATR as percentage of price",
            source="market_data",
        ),
        "volume_ratio": FeatureDefinition(
            name="volume_ratio",
            dtype="float",
            category="volume",
            description="Current volume / 20-day average volume",
            source="market_data",
        ),
        "rsi_14": FeatureDefinition(
            name="rsi_14",
            dtype="float",
            category="momentum",
            description="14-period RSI",
            source="market_data",
        ),
        "macd_hist": FeatureDefinition(
            name="macd_hist",
            dtype="float",
            category="momentum",
            description="MACD histogram value",
            source="market_data",
        ),
        "bb_position": FeatureDefinition(
            name="bb_position",
            dtype="float",
            category="volatility",
            description="Position within Bollinger Bands (0-1)",
            source="market_data",
        ),
    }
    
    CROSS_ASSET_FEATURES = {
        "vix_level": FeatureDefinition(
            name="vix_level",
            dtype="float",
            category="market",
            description="VIX volatility index level",
            source="market_data",
        ),
        "vix_percentile": FeatureDefinition(
            name="vix_percentile",
            dtype="float",
            category="market",
            description="VIX percentile rank (0-100)",
            source="market_data",
        ),
        "spy_trend": FeatureDefinition(
            name="spy_trend",
            dtype="int",
            category="market",
            description="SPY trend direction (-1, 0, 1)",
            source="market_data",
        ),
        "sector_momentum": FeatureDefinition(
            name="sector_momentum",
            dtype="float",
            category="sector",
            description="Sector relative strength",
            source="market_data",
        ),
        "market_breadth": FeatureDefinition(
            name="market_breadth",
            dtype="float",
            category="market",
            description="Advance-decline ratio",
            source="market_data",
        ),
        "correlation_spy": FeatureDefinition(
            name="correlation_spy",
            dtype="float",
            category="correlation",
            description="20-day correlation with SPY",
            source="market_data",
        ),
        "beta": FeatureDefinition(
            name="beta",
            dtype="float",
            category="risk",
            description="20-day beta to SPY",
            source="market_data",
        ),
    }
    
    @classmethod
    def get_all_features(cls) -> Dict[str, FeatureDefinition]:
        """Get all registered features."""
        return {**cls.CORE_FEATURES, **cls.CROSS_ASSET_FEATURES}
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of all feature names."""
        return list(cls.get_all_features().keys())


class FeatureStore:
    """
    Centralized feature management with quality audits.
    
    Usage:
        store = FeatureStore()
        
        # Store features for a symbol
        store.put_features("AAPL", {
            "quantra_score": 75.5,
            "regime_encoded": 0,
            ...
        })
        
        # Get features with quality check
        features = store.get_features("AAPL")
        
        # Run quality audit
        report = store.audit_feature("quantra_score")
    """
    
    def __init__(
        self,
        data_dir: str = "data/feature_store",
        max_cache_size: int = 10000,
        stale_threshold_hours: float = 24.0,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        self._max_cache_size = max_cache_size
        self._stale_threshold = stale_threshold_hours
        
        self._feature_cache: Dict[str, FeatureSnapshot] = {}
        self._feature_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = defaultdict(list)
        self._feature_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "count": 0, "sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf")
        })
        
        self._lock = threading.Lock()
        self._registry = FeatureRegistry()
    
    def put_features(
        self,
        symbol: str,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None,
        regime: str = "unknown",
        sector: str = "unknown",
    ) -> None:
        """
        Store feature snapshot for a symbol.
        
        Args:
            symbol: Symbol identifier
            features: Dictionary of feature name -> value
            timestamp: Optional timestamp (default: now)
            regime: Market regime
            sector: Sector classification
        """
        ts = timestamp or datetime.utcnow()
        
        snapshot = FeatureSnapshot(
            symbol=symbol,
            timestamp=ts,
            features=features,
            regime=regime,
            sector=sector,
        )
        
        with self._lock:
            self._feature_cache[symbol] = snapshot
            
            self._feature_history[symbol].append((ts, features))
            if len(self._feature_history[symbol]) > 1000:
                self._feature_history[symbol] = self._feature_history[symbol][-500:]
            
            for name, value in features.items():
                if value is not None and not np.isnan(value):
                    stats = self._feature_stats[name]
                    stats["count"] += 1
                    stats["sum"] += value
                    stats["sum_sq"] += value ** 2
                    stats["min"] = min(stats["min"], value)
                    stats["max"] = max(stats["max"], value)
            
            if len(self._feature_cache) > self._max_cache_size:
                self._evict_stale_entries()
    
    def get_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Get features for a symbol.
        
        Args:
            symbol: Symbol identifier
            feature_names: Optional list of specific features to return
            
        Returns:
            Feature dictionary or None if not found
        """
        with self._lock:
            snapshot = self._feature_cache.get(symbol)
            if snapshot is None:
                return None
            
            features = snapshot.features
            
            if feature_names:
                return {k: features.get(k) for k in feature_names}
            
            return features.copy()
    
    def get_snapshot(self, symbol: str) -> Optional[FeatureSnapshot]:
        """Get complete feature snapshot for a symbol."""
        with self._lock:
            return self._feature_cache.get(symbol)
    
    def get_feature_vector(
        self,
        symbol: str,
        feature_order: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get features as numpy array in specified order.
        
        Args:
            symbol: Symbol identifier
            feature_order: List of feature names in desired order
            
        Returns:
            Numpy array of feature values
        """
        features = self.get_features(symbol)
        if features is None:
            return None
        
        if feature_order is None:
            feature_order = list(self._registry.get_all_features().keys())
        
        vector = []
        for name in feature_order:
            value = features.get(name, 0.0)
            vector.append(value if value is not None else 0.0)
        
        return np.array(vector)
    
    def audit_feature(self, feature_name: str) -> FeatureQualityReport:
        """
        Run quality audit on a feature.
        
        Returns quality metrics including completeness, drift, and issues.
        """
        issues = []
        
        with self._lock:
            stats = self._feature_stats.get(feature_name, {})
            count = stats.get("count", 0)
            
            if count < 10:
                return FeatureQualityReport(
                    feature_name=feature_name,
                    completeness=0.0,
                    freshness_hours=float("inf"),
                    drift_score=0.0,
                    null_rate=1.0,
                    outlier_rate=0.0,
                    status="insufficient_data",
                    issues=["Less than 10 samples"]
                )
            
            total_symbols = len(self._feature_cache)
            symbols_with_feature = sum(
                1 for s in self._feature_cache.values()
                if feature_name in s.features and s.features[feature_name] is not None
            )
            completeness = symbols_with_feature / total_symbols if total_symbols > 0 else 0.0
            
            if completeness < 0.95:
                issues.append(f"Low completeness: {completeness:.1%}")
            
            recent_values = []
            now = datetime.utcnow()
            min_freshness = float("inf")
            
            for symbol, snapshot in self._feature_cache.items():
                if feature_name in snapshot.features:
                    age_hours = (now - snapshot.timestamp).total_seconds() / 3600
                    min_freshness = min(min_freshness, age_hours)
                    if age_hours < 24:
                        val = snapshot.features[feature_name]
                        if val is not None:
                            recent_values.append(val)
            
            if min_freshness > self._stale_threshold:
                issues.append(f"Stale data: {min_freshness:.1f} hours old")
            
            null_rate = 1.0 - (len(recent_values) / max(len(self._feature_cache), 1))
            
            if null_rate > 0.05:
                issues.append(f"High null rate: {null_rate:.1%}")
            
            if len(recent_values) > 10:
                mean = stats["sum"] / count
                variance = (stats["sum_sq"] / count) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                
                recent_mean = np.mean(recent_values)
                drift_score = abs(recent_mean - mean) / (std + 1e-10)
                
                if drift_score > 2.0:
                    issues.append(f"Feature drift detected: z={drift_score:.2f}")
                
                outliers = sum(1 for v in recent_values if abs(v - mean) > 3 * std)
                outlier_rate = outliers / len(recent_values)
            else:
                drift_score = 0.0
                outlier_rate = 0.0
            
            status = "healthy" if len(issues) == 0 else "degraded"
            
            return FeatureQualityReport(
                feature_name=feature_name,
                completeness=completeness,
                freshness_hours=min_freshness,
                drift_score=drift_score,
                null_rate=null_rate,
                outlier_rate=outlier_rate,
                status=status,
                issues=issues,
            )
    
    def audit_all_features(self) -> Dict[str, FeatureQualityReport]:
        """Run quality audit on all features."""
        all_feature_names: Set[str] = set()
        
        with self._lock:
            for snapshot in self._feature_cache.values():
                all_feature_names.update(snapshot.features.keys())
        
        return {name: self.audit_feature(name) for name in all_feature_names}
    
    def get_cross_asset_features(self) -> Dict[str, float]:
        """
        Get current cross-asset features (VIX, market breadth, etc).
        
        These are market-wide features, not symbol-specific.
        """
        cross_features = self.get_features("_MARKET_")
        if cross_features:
            return cross_features
        
        return {
            "vix_level": 20.0,
            "vix_percentile": 50.0,
            "spy_trend": 0,
            "market_breadth": 1.0,
        }
    
    def update_cross_asset_features(
        self,
        vix_level: float = 20.0,
        spy_trend: int = 0,
        market_breadth: float = 1.0,
        sector_data: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update market-wide cross-asset features."""
        features = {
            "vix_level": vix_level,
            "vix_percentile": min(100, max(0, (vix_level - 10) / 40 * 100)),
            "spy_trend": spy_trend,
            "market_breadth": market_breadth,
        }
        
        if sector_data:
            features.update(sector_data)
        
        self.put_features("_MARKET_", features)
    
    def _evict_stale_entries(self) -> None:
        """Remove stale entries from cache."""
        now = datetime.utcnow()
        stale_symbols = []
        
        for symbol, snapshot in self._feature_cache.items():
            age_hours = (now - snapshot.timestamp).total_seconds() / 3600
            if age_hours > self._stale_threshold * 2:
                stale_symbols.append(symbol)
        
        for symbol in stale_symbols[:100]:
            del self._feature_cache[symbol]
        
        logger.debug(f"[FeatureStore] Evicted {len(stale_symbols)} stale entries")
    
    def save_state(self) -> None:
        """Persist feature store state to disk."""
        state = {
            "stats": dict(self._feature_stats),
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        state_file = self._data_dir / "feature_store_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get feature store summary."""
        with self._lock:
            return {
                "cached_symbols": len(self._feature_cache),
                "tracked_features": len(self._feature_stats),
                "total_samples": sum(s.get("count", 0) for s in self._feature_stats.values()),
            }


_feature_store_instance: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    """Get global feature store instance."""
    global _feature_store_instance
    if _feature_store_instance is None:
        _feature_store_instance = FeatureStore()
    return _feature_store_instance


__all__ = [
    "FeatureStore",
    "FeatureRegistry",
    "FeatureDefinition",
    "FeatureSnapshot",
    "FeatureQualityReport",
    "get_feature_store",
]
