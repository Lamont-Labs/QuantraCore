"""
ApexLab V2 - Enhanced Label Schema and Dataset Builder.

Provides institutional-grade label generation with:
- Comprehensive future outcomes (returns, drawdowns, run-ups)
- Quality tier classification (A_PLUS, A, B, C, D)
- Runner and monster runner detection
- Regime and sector context labels
- Safety labels for trade avoidance
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import hashlib
import numpy as np
import pandas as pd

from src.quantracore_apex.core.schemas import OhlcvWindow, ApexContext
from src.quantracore_apex.core.engine import ApexEngine


class QualityTier(Enum):
    A_PLUS = "A_PLUS"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class MarketCapBand(Enum):
    MEGA = "mega"
    LARGE = "large"
    MID = "mid"
    SMALL = "small"
    MICRO = "micro"
    NANO = "nano"
    PENNY = "penny"


class VolatilityBand(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class LiquidityBand(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class EntropyBand(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class SuppressionState(Enum):
    NONE = "none"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"


class RegimeType(Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    SQUEEZE = "squeeze"
    CRASH = "crash"


class SectorRegime(Enum):
    SECTOR_UP = "sector_up"
    SECTOR_DOWN = "sector_down"
    SECTOR_CHOP = "sector_chop"
    SECTOR_ROTATION = "sector_rotation"


@dataclass
class ApexLabV2Row:
    """
    Complete training row for ApexCore V2.
    
    Contains meta fields, structural inputs from Apex engine,
    future outcomes, quality labels, runner flags, and safety labels.
    """
    symbol: str
    event_time: datetime
    timeframe: str
    engine_snapshot_id: str
    scanner_snapshot_id: str
    sector: str = "unknown"
    market_cap_band: str = MarketCapBand.MID.value
    
    quantra_score: float = 50.0
    risk_tier: str = "medium"
    entropy_band: str = EntropyBand.MID.value
    suppression_state: str = SuppressionState.NONE.value
    regime_type: str = RegimeType.CHOP.value
    volatility_band: str = VolatilityBand.MID.value
    liquidity_band: str = LiquidityBand.MID.value
    protocol_ids: List[str] = field(default_factory=list)
    protocol_vector: List[float] = field(default_factory=list)
    
    ret_1d: float = 0.0
    ret_3d: float = 0.0
    ret_5d: float = 0.0
    ret_10d: float = 0.0
    max_runup_5d: float = 0.0
    max_drawdown_5d: float = 0.0
    time_to_peak_5d: int = 0
    
    future_quality_tier: str = QualityTier.C.value
    hit_runner_threshold: int = 0
    hit_monster_runner_threshold: int = 0
    
    avoid_trade: int = 0
    
    regime_label: str = RegimeType.CHOP.value
    sector_regime: str = SectorRegime.SECTOR_CHOP.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApexLabV2Row":
        """Create from dictionary."""
        return cls(**data)


RUNNER_THRESHOLD = 0.15
MONSTER_RUNNER_THRESHOLD = 0.25
QUALITY_THRESHOLDS = {
    "A_PLUS": {"min_runup": 0.20, "max_drawdown": -0.05},
    "A": {"min_runup": 0.10, "max_drawdown": -0.07},
    "B": {"min_runup": 0.05, "max_drawdown": -0.10},
    "D": {"max_ret_5d": -0.05},
}


def assign_quality_tier(
    max_runup_5d: float,
    max_drawdown_5d: float,
    ret_5d: float
) -> str:
    """
    Assign quality tier based on future outcomes.
    
    A_PLUS: max_runup_5d >= +20% AND max_drawdown_5d >= -5%
    A: max_runup_5d >= +10% AND max_drawdown_5d >= -7%
    B: max_runup_5d >= +5% AND max_drawdown_5d >= -10%
    D: ret_5d <= -5%
    C: otherwise
    """
    if max_runup_5d >= 0.20 and max_drawdown_5d >= -0.05:
        return QualityTier.A_PLUS.value
    if max_runup_5d >= 0.10 and max_drawdown_5d >= -0.07:
        return QualityTier.A.value
    if max_runup_5d >= 0.05 and max_drawdown_5d >= -0.10:
        return QualityTier.B.value
    if ret_5d <= -0.05:
        return QualityTier.D.value
    return QualityTier.C.value


def compute_runner_flags(
    max_runup_5d: float,
    runner_threshold: float = RUNNER_THRESHOLD,
    monster_threshold: float = MONSTER_RUNNER_THRESHOLD
) -> Tuple[int, int]:
    """
    Compute runner and monster runner flags.
    
    Returns:
        (hit_runner_threshold, hit_monster_runner_threshold)
    """
    hit_runner = 1 if max_runup_5d >= runner_threshold else 0
    hit_monster = 1 if max_runup_5d >= monster_threshold else 0
    return hit_runner, hit_monster


def compute_regime_label(regime: str) -> str:
    """Map Apex regime to V2 regime label."""
    mapping = {
        "trending_up": RegimeType.TREND_UP.value,
        "trending_down": RegimeType.TREND_DOWN.value,
        "range_bound": RegimeType.CHOP.value,
        "volatile": RegimeType.CRASH.value,
        "compressed": RegimeType.SQUEEZE.value,
    }
    return mapping.get(regime, RegimeType.CHOP.value)


def compute_volatility_band(volatility: float) -> str:
    """Classify volatility into bands."""
    if volatility < 0.15:
        return VolatilityBand.LOW.value
    elif volatility < 0.30:
        return VolatilityBand.MID.value
    return VolatilityBand.HIGH.value


def compute_liquidity_band(avg_volume: float, avg_dollar_volume: float) -> str:
    """Classify liquidity into bands."""
    if avg_dollar_volume < 100000:
        return LiquidityBand.LOW.value
    elif avg_dollar_volume < 10000000:
        return LiquidityBand.MID.value
    return LiquidityBand.HIGH.value


def compute_entropy_band(entropy_state: str) -> str:
    """Map entropy state to band."""
    mapping = {
        "stable": EntropyBand.LOW.value,
        "elevated": EntropyBand.MID.value,
        "chaotic": EntropyBand.HIGH.value,
    }
    return mapping.get(entropy_state, EntropyBand.MID.value)


def compute_suppression_state(suppression: str) -> str:
    """Map suppression to V2 state."""
    if suppression in ("none", "light"):
        return SuppressionState.NONE.value
    elif suppression in ("moderate",):
        return SuppressionState.SUPPRESSED.value
    return SuppressionState.BLOCKED.value


def compute_avoid_trade(
    volatility_band: str,
    liquidity_band: str,
    suppression_state: str,
    risk_tier: str,
) -> int:
    """
    Determine if trade should be avoided.
    
    Returns 1 if any safety condition is triggered.
    """
    if volatility_band == VolatilityBand.HIGH.value:
        return 1
    if liquidity_band == LiquidityBand.LOW.value:
        return 1
    if suppression_state == SuppressionState.BLOCKED.value:
        return 1
    if risk_tier == "extreme":
        return 1
    return 0


def compute_future_returns(
    entry_price: float,
    future_prices: np.ndarray,
    bars_per_day: int = 1
) -> Dict[str, float]:
    """
    Compute future return metrics from price series.
    
    Args:
        entry_price: Entry price at event time
        future_prices: Array of future close prices
        bars_per_day: Number of bars per trading day
        
    Returns:
        Dictionary with ret_1d, ret_3d, ret_5d, ret_10d, max_runup_5d, 
        max_drawdown_5d, time_to_peak_5d
    """
    if entry_price <= 0 or len(future_prices) == 0:
        return {
            "ret_1d": 0.0, "ret_3d": 0.0, "ret_5d": 0.0, "ret_10d": 0.0,
            "max_runup_5d": 0.0, "max_drawdown_5d": 0.0, "time_to_peak_5d": 0
        }
    
    returns = future_prices / entry_price - 1.0
    
    def get_return_at_bar(bar_idx: int) -> float:
        if bar_idx < len(returns):
            return float(returns[bar_idx])
        elif len(returns) > 0:
            return float(returns[-1])
        return 0.0
    
    ret_1d = get_return_at_bar(bars_per_day - 1)
    ret_3d = get_return_at_bar(3 * bars_per_day - 1)
    ret_5d = get_return_at_bar(5 * bars_per_day - 1)
    ret_10d = get_return_at_bar(10 * bars_per_day - 1)
    
    five_day_end = min(5 * bars_per_day, len(returns))
    five_day_returns = returns[:five_day_end] if five_day_end > 0 else np.array([0.0])
    
    max_runup_5d = float(np.max(five_day_returns))
    max_drawdown_5d = float(np.min(five_day_returns))
    time_to_peak_5d = int(np.argmax(five_day_returns)) if len(five_day_returns) > 0 else 0
    
    return {
        "ret_1d": ret_1d,
        "ret_3d": ret_3d,
        "ret_5d": ret_5d,
        "ret_10d": ret_10d,
        "max_runup_5d": max_runup_5d,
        "max_drawdown_5d": max_drawdown_5d,
        "time_to_peak_5d": time_to_peak_5d,
    }


def generate_engine_snapshot_id(engine_config: Dict[str, Any]) -> str:
    """Generate deterministic hash of engine configuration."""
    config_str = str(sorted(engine_config.items()))
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def generate_scanner_snapshot_id(ohlcv_data: np.ndarray) -> str:
    """Generate deterministic hash of OHLCV data block."""
    data_bytes = ohlcv_data.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


NUM_PROTOCOLS = 115


def encode_protocol_vector(protocol_ids: List[str], total_protocols: int = NUM_PROTOCOLS) -> List[float]:
    """
    Encode protocol IDs into fixed-length activation vector.
    
    Args:
        protocol_ids: List of protocol IDs (e.g., ["T03", "T17", "LP05"])
        total_protocols: Total number of protocols in system
        
    Returns:
        Fixed-length vector with 1.0 for active protocols
    """
    vector = [0.0] * total_protocols
    
    for pid in protocol_ids:
        if pid.startswith("T"):
            idx = int(pid[1:]) - 1
            if 0 <= idx < 80:
                vector[idx] = 1.0
        elif pid.startswith("LP"):
            idx = 80 + int(pid[2:]) - 1
            if 80 <= idx < 105:
                vector[idx] = 1.0
        elif pid.startswith("MR"):
            idx = 105 + int(pid[2:]) - 1
            if 105 <= idx < 110:
                vector[idx] = 1.0
        elif pid.startswith("Ω") or pid.startswith("O"):
            idx = 110 + int(pid[1:]) - 1
            if 110 <= idx < 115:
                vector[idx] = 1.0
    
    return vector


class ApexLabV2Builder:
    """
    Dataset builder for ApexLab V2.
    
    Generates training rows by running the Apex engine on historical data
    and computing future outcomes.
    """
    
    def __init__(
        self,
        engine: Optional[ApexEngine] = None,
        enable_logging: bool = False,
        runner_threshold: float = RUNNER_THRESHOLD,
        monster_threshold: float = MONSTER_RUNNER_THRESHOLD,
    ):
        self.engine = engine or ApexEngine(enable_logging=enable_logging)
        self.runner_threshold = runner_threshold
        self.monster_threshold = monster_threshold
        self._engine_snapshot_id = generate_engine_snapshot_id({
            "version": "9.0-A",
            "protocols": 115,
            "compliance_mode": True,
        })
    
    def build_row(
        self,
        window: OhlcvWindow,
        future_prices: np.ndarray,
        timeframe: str = "1d",
        sector: str = "unknown",
        market_cap_band: str = "mid",
    ) -> ApexLabV2Row:
        """
        Build a single training row from a window and future prices.
        
        Args:
            window: 100-bar OHLCV window at event time
            future_prices: Array of future close prices after event
            timeframe: Timeframe string (e.g., "1d", "1h")
            sector: Sector classification
            market_cap_band: Market cap band
            
        Returns:
            Complete ApexLabV2Row
        """
        context = ApexContext(seed=42, compliance_mode=True)
        result = self.engine.run(window, context)
        
        closes = np.array([bar.close for bar in window.bars])
        volumes = np.array([bar.volume for bar in window.bars])
        
        entry_price = closes[-1]
        bars_per_day = self._timeframe_to_bars_per_day(timeframe)
        future_returns = compute_future_returns(entry_price, future_prices, bars_per_day)
        
        quality_tier = assign_quality_tier(
            future_returns["max_runup_5d"],
            future_returns["max_drawdown_5d"],
            future_returns["ret_5d"],
        )
        
        hit_runner, hit_monster = compute_runner_flags(
            future_returns["max_runup_5d"],
            self.runner_threshold,
            self.monster_threshold,
        )
        
        volatility = np.std(np.diff(np.log(closes + 1e-10)))
        volatility_band = compute_volatility_band(volatility)
        
        avg_volume = np.mean(volumes)
        avg_dollar_volume = avg_volume * np.mean(closes)
        liquidity_band = compute_liquidity_band(avg_volume, avg_dollar_volume)
        
        entropy_band = compute_entropy_band(result.entropy_state.value)
        suppression_state = compute_suppression_state(result.suppression_state.value)
        regime_type = compute_regime_label(result.regime.value)
        
        avoid_trade = compute_avoid_trade(
            volatility_band, liquidity_band, suppression_state, result.risk_tier.value
        )
        
        protocol_ids = [p.protocol_id for p in result.protocol_results if hasattr(p, 'protocol_id') and p.fired]
        protocol_vector = encode_protocol_vector(protocol_ids)
        
        scanner_snapshot_id = generate_scanner_snapshot_id(closes)
        
        return ApexLabV2Row(
            symbol=window.symbol,
            event_time=datetime.utcnow(),
            timeframe=timeframe,
            engine_snapshot_id=self._engine_snapshot_id,
            scanner_snapshot_id=scanner_snapshot_id,
            sector=sector,
            market_cap_band=market_cap_band,
            quantra_score=result.quantrascore,
            risk_tier=result.risk_tier.value,
            entropy_band=entropy_band,
            suppression_state=suppression_state,
            regime_type=regime_type,
            volatility_band=volatility_band,
            liquidity_band=liquidity_band,
            protocol_ids=protocol_ids,
            protocol_vector=protocol_vector,
            ret_1d=future_returns["ret_1d"],
            ret_3d=future_returns["ret_3d"],
            ret_5d=future_returns["ret_5d"],
            ret_10d=future_returns["ret_10d"],
            max_runup_5d=future_returns["max_runup_5d"],
            max_drawdown_5d=future_returns["max_drawdown_5d"],
            time_to_peak_5d=future_returns["time_to_peak_5d"],
            future_quality_tier=quality_tier,
            hit_runner_threshold=hit_runner,
            hit_monster_runner_threshold=hit_monster,
            avoid_trade=avoid_trade,
            regime_label=regime_type,
            sector_regime=SectorRegime.SECTOR_CHOP.value,
        )
    
    def build_dataset(
        self,
        windows: List[OhlcvWindow],
        future_prices_list: List[np.ndarray],
        timeframe: str = "1d",
        sectors: Optional[List[str]] = None,
        market_cap_bands: Optional[List[str]] = None,
    ) -> List[ApexLabV2Row]:
        """
        Build dataset from multiple windows.
        
        Args:
            windows: List of OHLCV windows
            future_prices_list: List of future price arrays
            timeframe: Timeframe string
            sectors: Optional list of sectors per window
            market_cap_bands: Optional list of market cap bands
            
        Returns:
            List of ApexLabV2Row objects
        """
        rows = []
        for i, (window, future_prices) in enumerate(zip(windows, future_prices_list)):
            sector = sectors[i] if sectors else "unknown"
            mcb = market_cap_bands[i] if market_cap_bands else "mid"
            row = self.build_row(window, future_prices, timeframe, sector, mcb)
            rows.append(row)
        return rows
    
    def to_dataframe(self, rows: List[ApexLabV2Row]) -> pd.DataFrame:
        """Convert rows to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in rows])
    
    def to_parquet(self, rows: List[ApexLabV2Row], path: str) -> None:
        """Save rows to Parquet file."""
        df = self.to_dataframe(rows)
        df.to_parquet(path, index=False)
    
    def _timeframe_to_bars_per_day(self, timeframe: str) -> int:
        """Convert timeframe to number of bars per trading day."""
        mapping = {
            "1m": 390,
            "5m": 78,
            "15m": 26,
            "30m": 13,
            "1h": 7,
            "4h": 2,
            "1d": 1,
        }
        return mapping.get(timeframe, 1)


class ApexLabV2DatasetBuilder:
    """
    High-level dataset builder for ApexLab V2.
    
    Provides universe-wide dataset construction with time-aware splitting.
    """
    
    def __init__(self, enable_logging: bool = False):
        self.builder = ApexLabV2Builder(enable_logging=enable_logging)
        self._dataset: List[ApexLabV2Row] = []
    
    def build_dataset(
        self,
        universe: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        data_provider: Optional[Any] = None,
    ) -> "ApexLabV2DatasetBuilder":
        """
        Build dataset for a universe of symbols using real market data.
        
        Args:
            universe: List of symbols to process
            start_date: Start date for data fetch
            end_date: End date for data fetch
            timeframe: Timeframe (1d, 1h, etc)
            data_provider: Data manager instance for fetching OHLCV
            
        Returns:
            Self with populated dataset
        """
        from src.quantracore_apex.data_layer.adapters.data_manager import UnifiedDataManager
        from src.quantracore_apex.core.window_builder import WindowBuilder
        
        self._dataset = []
        
        if data_provider is None:
            data_provider = UnifiedDataManager()
        
        window_builder = WindowBuilder()
        lookback_days = (end_date - start_date).days
        
        for symbol in universe:
            try:
                bars = data_provider.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback_days=lookback_days + 20
                )
                
                if bars is None or len(bars) < 120:
                    continue
                
                normalized_bars = []
                for bar in bars:
                    if isinstance(bar, dict):
                        normalized_bars.append({
                            "open": float(bar.get("open", bar.get("o", 0))),
                            "high": float(bar.get("high", bar.get("h", 0))),
                            "low": float(bar.get("low", bar.get("l", 0))),
                            "close": float(bar.get("close", bar.get("c", 0))),
                            "volume": float(bar.get("volume", bar.get("v", 0))),
                        })
                    else:
                        normalized_bars.append(bar)
                
                for i in range(100, len(normalized_bars) - 15, 10):
                    window_slice = normalized_bars[i-100:i]
                    future_slice = normalized_bars[i:i+15]
                    
                    window = window_builder.build_single(window_slice, symbol, timeframe)
                    if window is None:
                        continue
                    
                    future_prices = np.array([bar["close"] for bar in future_slice])
                    
                    row = self.builder.build_row(
                        window=window,
                        future_prices=future_prices,
                        timeframe=timeframe,
                        sector="unknown",
                        market_cap_band="mid"
                    )
                    self._dataset.append(row)
                    
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        return self
    
    def add_row(self, row: ApexLabV2Row) -> None:
        """Add a single row to the dataset."""
        self._dataset.append(row)
    
    def add_rows(self, rows: List[ApexLabV2Row]) -> None:
        """Add multiple rows to the dataset."""
        self._dataset.extend(rows)
    
    @property
    def rows(self) -> List[ApexLabV2Row]:
        """Get all rows."""
        return self._dataset
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return self.builder.to_dataframe(self._dataset)
    
    def to_parquet(self, path: str) -> None:
        """Save to Parquet."""
        self.builder.to_parquet(self._dataset, path)
    
    def to_arrow(self, path: str) -> None:
        """Save to Arrow format."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = self.to_dataframe()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path)
    
    def get_feature_target_split(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Get features and targets for training.
        
        Returns:
            (features_array, targets_dict)
        """
        df = self.to_dataframe()
        
        feature_cols = [
            "quantra_score", "protocol_vector",
        ]
        
        target_cols = {
            "quantra_score": df["quantra_score"].values,
            "hit_runner_threshold": df["hit_runner_threshold"].values,
            "future_quality_tier": df["future_quality_tier"].values,
            "avoid_trade": df["avoid_trade"].values,
            "regime_label": df["regime_label"].values,
        }
        
        numeric_features = df["quantra_score"].values.reshape(-1, 1)
        
        return numeric_features, target_cols


__all__ = [
    "ApexLabV2Row",
    "ApexLabV2Builder",
    "ApexLabV2DatasetBuilder",
    "QualityTier",
    "MarketCapBand",
    "VolatilityBand",
    "LiquidityBand",
    "EntropyBand",
    "SuppressionState",
    "RegimeType",
    "SectorRegime",
    "assign_quality_tier",
    "compute_runner_flags",
    "compute_regime_label",
    "compute_future_returns",
    "encode_protocol_vector",
]
