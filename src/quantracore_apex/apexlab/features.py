"""
Enhanced Feature Extractor for ApexLab - Swing Trade Optimized.

Extracts comprehensive feature vectors from EOD OHLCV windows for swing trade prediction.
Optimized for 2-10 day holding periods using daily bar data.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.core.entropy import compute_entropy
from src.quantracore_apex.core.suppression import compute_suppression
from src.quantracore_apex.core.drift import compute_drift
import logging

logger = logging.getLogger(__name__)


@dataclass
class SwingFeatures:
    """Container for swing-specific features."""
    momentum_features: Dict[str, float]
    volatility_features: Dict[str, float]
    volume_features: Dict[str, float]
    pattern_features: Dict[str, float]
    structure_features: Dict[str, float]
    

@dataclass 
class MultiHorizonLabels:
    """Multi-horizon forward return labels for swing trades."""
    return_3d: float
    return_5d: float
    return_8d: float
    return_10d: float
    max_adverse_excursion_5d: float
    max_favorable_excursion_5d: float
    hit_rate_threshold_3pct: bool
    hit_rate_threshold_5pct: bool
    gap_risk_5d: float


class SwingFeatureExtractor:
    """
    Enhanced Feature Extractor optimized for swing trade predictions.
    
    Extracts 80+ features from EOD data covering:
    - Multi-scale momentum (3/5/10/20/40 day)
    - Volatility regime (ATR compression, HV percentiles)
    - Volume texture (OBV, CMF, relative volume)
    - Candlestick patterns (NR4, Inside Day, Engulfing, etc.)
    - Price structure (swing highs/lows, Fibonacci levels)
    - Cross-sectional features (percentile ranks)
    """
    
    FEATURE_NAMES = [
        # Original core features (30)
        "wick_ratio", "body_ratio", "bullish_pct_last20", "compression_score",
        "noise_score", "strength_slope", "range_density", "volume_intensity",
        "trend_consistency", "volatility_ratio",
        "price_entropy", "volume_entropy", "combined_entropy", "entropy_floor",
        "suppression_level", "coil_factor", "breakout_probability",
        "drift_magnitude", "drift_direction", "mean_reversion_pressure",
        "sma_20_ratio", "sma_50_ratio", "rsi_14", "atr_ratio",
        "recent_return_5", "recent_return_10", "recent_return_20",
        "high_low_ratio", "close_position", "volume_sma_ratio",
        
        # Multi-scale momentum features (10)
        "momentum_3d", "momentum_5d", "momentum_10d", "momentum_20d", "momentum_40d",
        "momentum_slope_5_20", "momentum_acceleration", "log_return_5d", 
        "overnight_gap_avg", "gap_fill_rate",
        
        # Volatility regime features (10)
        "atr_14", "atr_compression_ratio", "hv_20_percentile", "hv_60_percentile",
        "volatility_contraction_score", "range_contraction_days", "bollinger_width",
        "keltner_squeeze", "atr_trend", "volatility_breakout_signal",
        
        # Volume texture features (10)
        "obv_slope", "obv_divergence", "cmf_20", "volume_spike_persistence",
        "relative_volume_5d", "relative_volume_20d", "volume_trend",
        "accumulation_distribution", "volume_price_trend", "smart_money_flow",
        
        # Candlestick pattern features (10)
        "nr4_signal", "nr7_signal", "inside_day", "outside_day",
        "engulfing_bullish", "engulfing_bearish", "hammer", "shooting_star",
        "doji", "marubozu",
        
        # Price structure features (10)
        "swing_high_distance", "swing_low_distance", "fib_382_distance", "fib_618_distance",
        "higher_highs_count", "higher_lows_count", "lower_highs_count", "lower_lows_count",
        "channel_position", "trend_strength_adx",
        
        # Technical indicators (10)
        "macd_histogram", "macd_signal_cross", "stochastic_k", "stochastic_d",
        "williams_r", "cci_20", "mfi_14", "roc_10", "trix_15", "ultimate_oscillator",
    ]
    
    def __init__(self, lookback_short: int = 20, lookback_medium: int = 60, lookback_long: int = 120):
        """
        Initialize the swing feature extractor.
        
        Args:
            lookback_short: Short-term lookback for oscillators (default: 20)
            lookback_medium: Medium-term lookback for momentum (default: 60)
            lookback_long: Long-term lookback for regime detection (default: 120)
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.feature_dim = len(self.FEATURE_NAMES)
        
    def extract(self, window: OhlcvWindow) -> np.ndarray:
        """
        Extract comprehensive swing trade feature vector from a window.
        
        Args:
            window: OhlcvWindow with at least 60 bars (ideally 120)
            
        Returns:
            numpy array of shape (feature_dim,) with 80 features
        """
        bars = window.bars
        n_bars = len(bars)
        
        if n_bars < 30:
            logger.warning(f"Window too short ({n_bars} bars), padding with zeros")
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        closes = np.array([b.close for b in bars])
        opens = np.array([b.open for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        volumes = np.array([b.volume for b in bars])
        
        # Core features from existing modules
        core_features = self._extract_core_features(window, closes, highs, lows, volumes)
        
        # Swing-specific features
        momentum_features = self._extract_momentum_features(closes, opens, highs, lows)
        volatility_features = self._extract_volatility_features(closes, highs, lows)
        volume_features = self._extract_volume_features(closes, highs, lows, volumes)
        pattern_features = self._extract_pattern_features(opens, highs, lows, closes)
        structure_features = self._extract_structure_features(closes, highs, lows)
        technical_features = self._extract_technical_features(closes, highs, lows, volumes)
        
        # Combine all features
        all_features = np.concatenate([
            core_features,
            momentum_features,
            volatility_features,
            volume_features,
            pattern_features,
            structure_features,
            technical_features,
        ])
        
        # Clean up NaN/Inf values
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return all_features.astype(np.float32)
    
    def _extract_core_features(self, window: OhlcvWindow, closes: np.ndarray, 
                                highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Extract original 30 core features."""
        microtraits = compute_microtraits(window)
        entropy_metrics = compute_entropy(window)
        suppression_metrics = compute_suppression(window)
        drift_metrics = compute_drift(window)
        
        current_price = closes[-1]
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
        
        sma_20_ratio = current_price / sma_20 if sma_20 > 0 else 1.0
        sma_50_ratio = current_price / sma_50 if sma_50 > 0 else 1.0
        
        rsi_14 = self._compute_rsi(closes, 14) / 100.0
        
        ranges = highs - lows
        atr = np.mean(ranges[-14:])
        atr_hist = np.mean(ranges[-28:-14]) if len(ranges) >= 28 else atr
        atr_ratio = atr / atr_hist if atr_hist > 0 else 1.0
        
        recent_return_5 = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 5 and closes[-6] != 0 else 0
        recent_return_10 = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] != 0 else 0
        recent_return_20 = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 and closes[-21] != 0 else 0
        
        period_high = np.max(highs[-20:])
        period_low = np.min(lows[-20:])
        high_low_ratio = (period_high - period_low) / period_low if period_low > 0 else 0
        close_position = (current_price - period_low) / (period_high - period_low) if period_high != period_low else 0.5
        
        volume_sma = np.mean(volumes[-20:])
        volume_sma_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
        
        return np.array([
            microtraits.wick_ratio,
            microtraits.body_ratio,
            microtraits.bullish_pct_last20,
            microtraits.compression_score,
            microtraits.noise_score,
            np.clip(microtraits.strength_slope / 10, -1, 1),
            microtraits.range_density,
            np.clip(microtraits.volume_intensity, 0, 3),
            microtraits.trend_consistency,
            np.clip(microtraits.volatility_ratio, 0, 3),
            entropy_metrics.price_entropy,
            entropy_metrics.volume_entropy,
            entropy_metrics.combined_entropy,
            entropy_metrics.entropy_floor,
            suppression_metrics.suppression_level,
            np.clip(suppression_metrics.coil_factor, 0, 2),
            suppression_metrics.breakout_probability,
            np.clip(drift_metrics.drift_magnitude, 0, 1),
            drift_metrics.drift_direction,
            drift_metrics.mean_reversion_pressure,
            np.clip(sma_20_ratio, 0.5, 1.5),
            np.clip(sma_50_ratio, 0.5, 1.5),
            rsi_14,
            np.clip(atr_ratio, 0, 3),
            np.clip(recent_return_5, -0.2, 0.2),
            np.clip(recent_return_10, -0.3, 0.3),
            np.clip(recent_return_20, -0.4, 0.4),
            np.clip(high_low_ratio, 0, 0.3),
            close_position,
            np.clip(volume_sma_ratio, 0, 5),
        ], dtype=np.float32)
    
    def _extract_momentum_features(self, closes: np.ndarray, opens: np.ndarray,
                                    highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """Extract multi-scale momentum features (10 features)."""
        n = len(closes)
        
        # Multi-scale returns
        momentum_3d = (closes[-1] / closes[-4] - 1) if n > 3 else 0
        momentum_5d = (closes[-1] / closes[-6] - 1) if n > 5 else 0
        momentum_10d = (closes[-1] / closes[-11] - 1) if n > 10 else 0
        momentum_20d = (closes[-1] / closes[-21] - 1) if n > 20 else 0
        momentum_40d = (closes[-1] / closes[-41] - 1) if n > 40 else 0
        
        # Momentum slope (short vs long term)
        momentum_slope_5_20 = momentum_5d - (momentum_20d / 4) if n > 20 else 0
        
        # Momentum acceleration (change in momentum)
        if n > 10:
            prev_momentum_5d = (closes[-6] / closes[-11] - 1)
            momentum_acceleration = momentum_5d - prev_momentum_5d
        else:
            momentum_acceleration = 0
        
        # Log returns for better normalization
        log_return_5d = np.log(closes[-1] / closes[-6]) if n > 5 and closes[-6] > 0 else 0
        
        # Overnight gap analysis
        if n > 5:
            gaps = opens[1:] - closes[:-1]
            overnight_gap_avg = np.mean(gaps[-5:]) / closes[-1] if closes[-1] > 0 else 0
            # Gap fill rate (how often gaps get filled same day)
            gap_fills = 0
            for i in range(-5, 0):
                if i < -1:
                    gap = opens[i+1] - closes[i]
                    if gap > 0 and lows[i+1] <= closes[i]:
                        gap_fills += 1
                    elif gap < 0 and highs[i+1] >= closes[i]:
                        gap_fills += 1
            gap_fill_rate = gap_fills / 5
        else:
            overnight_gap_avg = 0
            gap_fill_rate = 0
        
        return np.array([
            np.clip(momentum_3d, -0.15, 0.15),
            np.clip(momentum_5d, -0.20, 0.20),
            np.clip(momentum_10d, -0.30, 0.30),
            np.clip(momentum_20d, -0.40, 0.40),
            np.clip(momentum_40d, -0.50, 0.50),
            np.clip(momentum_slope_5_20, -0.10, 0.10),
            np.clip(momentum_acceleration, -0.10, 0.10),
            np.clip(log_return_5d, -0.20, 0.20),
            np.clip(overnight_gap_avg, -0.05, 0.05),
            gap_fill_rate,
        ], dtype=np.float32)
    
    def _extract_volatility_features(self, closes: np.ndarray, highs: np.ndarray, 
                                      lows: np.ndarray) -> np.ndarray:
        """Extract volatility regime features (10 features)."""
        n = len(closes)
        ranges = highs - lows
        
        # ATR and compression
        atr_14 = np.mean(ranges[-14:]) / closes[-1] if n >= 14 and closes[-1] > 0 else 0
        atr_28 = np.mean(ranges[-28:]) / closes[-1] if n >= 28 and closes[-1] > 0 else atr_14
        atr_compression_ratio = atr_14 / atr_28 if atr_28 > 0 else 1.0
        
        # Historical volatility percentiles
        if n >= 60:
            returns = np.diff(np.log(closes))
            hv_20 = np.std(returns[-20:]) * np.sqrt(252)
            hv_60 = np.std(returns[-60:]) * np.sqrt(252)
            
            # Compute percentile of current HV vs historical
            rolling_hvs = [np.std(returns[i:i+20]) * np.sqrt(252) for i in range(0, n-20, 5)]
            hv_20_percentile = np.sum(np.array(rolling_hvs) < hv_20) / len(rolling_hvs) if rolling_hvs else 0.5
            hv_60_percentile = np.sum(np.array(rolling_hvs) < hv_60) / len(rolling_hvs) if rolling_hvs else 0.5
        else:
            hv_20_percentile = 0.5
            hv_60_percentile = 0.5
        
        # Volatility contraction score (low vol precedes big moves)
        recent_range = np.mean(ranges[-5:])
        hist_range = np.mean(ranges[-20:]) if n >= 20 else recent_range
        volatility_contraction_score = 1 - (recent_range / hist_range) if hist_range > 0 else 0
        
        # Count consecutive narrow range days
        avg_range = np.mean(ranges[-20:]) if n >= 20 else np.mean(ranges)
        range_contraction_days = 0
        for i in range(-1, -min(10, n), -1):
            if ranges[i] < avg_range * 0.7:
                range_contraction_days += 1
            else:
                break
        
        # Bollinger Band width
        sma_20 = np.mean(closes[-20:]) if n >= 20 else np.mean(closes)
        std_20 = np.std(closes[-20:]) if n >= 20 else np.std(closes)
        bollinger_width = (2 * std_20 * 2) / sma_20 if sma_20 > 0 else 0
        
        # Keltner Channel squeeze (Bollinger inside Keltner)
        keltner_width = atr_14 * 1.5 * 2
        keltner_squeeze = 1.0 if bollinger_width < keltner_width else 0.0
        
        # ATR trend (expanding or contracting)
        if n >= 28:
            atr_recent = np.mean(ranges[-7:])
            atr_prev = np.mean(ranges[-14:-7])
            atr_trend = (atr_recent - atr_prev) / atr_prev if atr_prev > 0 else 0
        else:
            atr_trend = 0
        
        # Volatility breakout signal
        volatility_breakout_signal = 1.0 if (volatility_contraction_score > 0.3 and keltner_squeeze > 0) else 0.0
        
        return np.array([
            np.clip(atr_14 * 100, 0, 10),  # ATR as percentage
            np.clip(atr_compression_ratio, 0.5, 2.0),
            hv_20_percentile,
            hv_60_percentile,
            np.clip(volatility_contraction_score, 0, 1),
            range_contraction_days / 10,
            np.clip(bollinger_width, 0, 0.5),
            keltner_squeeze,
            np.clip(atr_trend, -0.5, 0.5),
            volatility_breakout_signal,
        ], dtype=np.float32)
    
    def _extract_volume_features(self, closes: np.ndarray, highs: np.ndarray,
                                  lows: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Extract volume texture features (10 features)."""
        n = len(closes)
        
        # On-Balance Volume (OBV)
        obv = np.zeros(n)
        obv[0] = volumes[0]
        for i in range(1, n):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        # OBV slope (trend)
        if n >= 10:
            obv_slope = (obv[-1] - obv[-10]) / (np.std(obv[-20:]) + 1e-10)
            # OBV divergence (price up but OBV down or vice versa)
            price_change = closes[-1] - closes[-10]
            obv_change = obv[-1] - obv[-10]
            obv_divergence = -1 if (price_change > 0 and obv_change < 0) else (1 if (price_change < 0 and obv_change > 0) else 0)
        else:
            obv_slope = 0
            obv_divergence = 0
        
        # Chaikin Money Flow (CMF)
        if n >= 20:
            mfv = np.zeros(n)
            for i in range(n):
                hl_range = highs[i] - lows[i]
                if hl_range > 0:
                    mfv[i] = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range * volumes[i]
            cmf_20 = np.sum(mfv[-20:]) / (np.sum(volumes[-20:]) + 1e-10)
        else:
            cmf_20 = 0
        
        # Volume spike persistence
        avg_vol = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        spike_threshold = avg_vol * 2
        volume_spike_persistence = sum(1 for v in volumes[-5:] if v > spike_threshold) / 5
        
        # Relative volume
        relative_volume_5d = np.mean(volumes[-5:]) / avg_vol if avg_vol > 0 else 1
        relative_volume_20d = np.mean(volumes[-20:]) / np.mean(volumes[-60:]) if n >= 60 and np.mean(volumes[-60:]) > 0 else 1
        
        # Volume trend
        if n >= 20:
            vol_sma_10 = np.mean(volumes[-10:])
            vol_sma_20 = np.mean(volumes[-20:])
            volume_trend = (vol_sma_10 - vol_sma_20) / vol_sma_20 if vol_sma_20 > 0 else 0
        else:
            volume_trend = 0
        
        # Accumulation/Distribution line
        ad = np.zeros(n)
        for i in range(n):
            hl_range = highs[i] - lows[i]
            if hl_range > 0:
                ad[i] = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl_range * volumes[i]
        accumulation_distribution = np.sum(ad[-20:]) / (np.sum(volumes[-20:]) + 1e-10) if n >= 20 else 0
        
        # Volume-price trend
        vpt = np.zeros(n)
        for i in range(1, n):
            if closes[i-1] > 0:
                vpt[i] = vpt[i-1] + volumes[i] * (closes[i] - closes[i-1]) / closes[i-1]
        volume_price_trend = (vpt[-1] - vpt[-10]) / (np.std(vpt[-20:]) + 1e-10) if n >= 20 else 0
        
        # Smart money flow (high volume on up days vs down days)
        if n >= 10:
            up_volume = sum(volumes[i] for i in range(-10, 0) if closes[i] > closes[i-1])
            down_volume = sum(volumes[i] for i in range(-10, 0) if closes[i] < closes[i-1])
            total_vol = up_volume + down_volume
            smart_money_flow = (up_volume - down_volume) / total_vol if total_vol > 0 else 0
        else:
            smart_money_flow = 0
        
        return np.array([
            np.clip(obv_slope, -3, 3),
            obv_divergence,
            np.clip(cmf_20, -1, 1),
            volume_spike_persistence,
            np.clip(relative_volume_5d, 0, 5),
            np.clip(relative_volume_20d, 0.5, 2),
            np.clip(volume_trend, -1, 1),
            np.clip(accumulation_distribution, -1, 1),
            np.clip(volume_price_trend, -3, 3),
            np.clip(smart_money_flow, -1, 1),
        ], dtype=np.float32)
    
    def _extract_pattern_features(self, opens: np.ndarray, highs: np.ndarray,
                                   lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Extract candlestick pattern features (10 features)."""
        n = len(closes)
        if n < 7:
            return np.zeros(10, dtype=np.float32)
        
        ranges = highs - lows
        bodies = np.abs(closes - opens)
        
        # NR4 (Narrow Range 4) - today's range is smallest of last 4
        nr4_signal = 1.0 if ranges[-1] == np.min(ranges[-4:]) else 0.0
        
        # NR7 (Narrow Range 7) - today's range is smallest of last 7
        nr7_signal = 1.0 if ranges[-1] == np.min(ranges[-7:]) else 0.0
        
        # Inside Day - today's range is within yesterday's range
        inside_day = 1.0 if (highs[-1] <= highs[-2] and lows[-1] >= lows[-2]) else 0.0
        
        # Outside Day - today's range engulfs yesterday's
        outside_day = 1.0 if (highs[-1] > highs[-2] and lows[-1] < lows[-2]) else 0.0
        
        # Bullish Engulfing
        engulfing_bullish = 1.0 if (
            closes[-2] < opens[-2] and  # Yesterday bearish
            closes[-1] > opens[-1] and  # Today bullish
            opens[-1] <= closes[-2] and  # Today opens at/below yesterday close
            closes[-1] >= opens[-2]  # Today closes at/above yesterday open
        ) else 0.0
        
        # Bearish Engulfing
        engulfing_bearish = 1.0 if (
            closes[-2] > opens[-2] and  # Yesterday bullish
            closes[-1] < opens[-1] and  # Today bearish
            opens[-1] >= closes[-2] and  # Today opens at/above yesterday close
            closes[-1] <= opens[-2]  # Today closes at/below yesterday open
        ) else 0.0
        
        # Hammer (bullish reversal) - small body at top, long lower wick
        body = bodies[-1]
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        total_range = ranges[-1]
        hammer = 1.0 if (
            total_range > 0 and
            lower_wick > 2 * body and
            upper_wick < body * 0.5 and
            closes[-1] > opens[-1]
        ) else 0.0
        
        # Shooting Star (bearish reversal) - small body at bottom, long upper wick
        shooting_star = 1.0 if (
            total_range > 0 and
            upper_wick > 2 * body and
            lower_wick < body * 0.5 and
            closes[-1] < opens[-1]
        ) else 0.0
        
        # Doji - very small body
        avg_body = np.mean(bodies[-10:])
        doji = 1.0 if body < avg_body * 0.1 else 0.0
        
        # Marubozu - no wicks, all body
        avg_range = np.mean(ranges[-10:])
        marubozu = 1.0 if (body > avg_range * 0.9 and total_range > 0) else 0.0
        
        return np.array([
            nr4_signal,
            nr7_signal,
            inside_day,
            outside_day,
            engulfing_bullish,
            engulfing_bearish,
            hammer,
            shooting_star,
            doji,
            marubozu,
        ], dtype=np.float32)
    
    def _extract_structure_features(self, closes: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray) -> np.ndarray:
        """Extract price structure features (10 features)."""
        n = len(closes)
        if n < 20:
            return np.zeros(10, dtype=np.float32)
        
        current_price = closes[-1]
        
        # Find swing highs and lows (local extremes)
        swing_highs = []
        swing_lows = []
        lookback = 5
        
        for i in range(lookback, n - lookback):
            if highs[i] == np.max(highs[i-lookback:i+lookback+1]):
                swing_highs.append((i, highs[i]))
            if lows[i] == np.min(lows[i-lookback:i+lookback+1]):
                swing_lows.append((i, lows[i]))
        
        # Distance to nearest swing high/low
        if swing_highs:
            nearest_swing_high = min(swing_highs, key=lambda x: abs(x[1] - current_price))[1]
            swing_high_distance = (current_price - nearest_swing_high) / current_price
        else:
            swing_high_distance = 0
            
        if swing_lows:
            nearest_swing_low = min(swing_lows, key=lambda x: abs(x[1] - current_price))[1]
            swing_low_distance = (current_price - nearest_swing_low) / current_price
        else:
            swing_low_distance = 0
        
        # Fibonacci levels
        period_high = np.max(highs[-60:]) if n >= 60 else np.max(highs)
        period_low = np.min(lows[-60:]) if n >= 60 else np.min(lows)
        fib_range = period_high - period_low
        
        if fib_range > 0:
            fib_382 = period_low + fib_range * 0.382
            fib_618 = period_low + fib_range * 0.618
            fib_382_distance = (current_price - fib_382) / current_price
            fib_618_distance = (current_price - fib_618) / current_price
        else:
            fib_382_distance = 0
            fib_618_distance = 0
        
        # Count higher highs/lows pattern
        higher_highs_count = 0
        higher_lows_count = 0
        lower_highs_count = 0
        lower_lows_count = 0
        
        for i in range(1, min(10, len(swing_highs))):
            if swing_highs[-(i)][1] > swing_highs[-(i+1)][1]:
                higher_highs_count += 1
            else:
                lower_highs_count += 1
                
        for i in range(1, min(10, len(swing_lows))):
            if swing_lows[-(i)][1] > swing_lows[-(i+1)][1]:
                higher_lows_count += 1
            else:
                lower_lows_count += 1
        
        # Channel position (where price is in recent range)
        range_20 = np.max(highs[-20:]) - np.min(lows[-20:])
        channel_position = (current_price - np.min(lows[-20:])) / range_20 if range_20 > 0 else 0.5
        
        # ADX-like trend strength
        if n >= 14:
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            atr = np.mean(tr[-14:])
            
            plus_dm = np.where(
                (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                np.maximum(highs[1:] - highs[:-1], 0),
                0
            )
            minus_dm = np.where(
                (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                np.maximum(lows[:-1] - lows[1:], 0),
                0
            )
            
            plus_di = 100 * np.mean(plus_dm[-14:]) / atr if atr > 0 else 0
            minus_di = 100 * np.mean(minus_dm[-14:]) / atr if atr > 0 else 0
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            trend_strength_adx = dx / 100
        else:
            trend_strength_adx = 0
        
        return np.array([
            np.clip(swing_high_distance, -0.2, 0.2),
            np.clip(swing_low_distance, -0.2, 0.2),
            np.clip(fib_382_distance, -0.2, 0.2),
            np.clip(fib_618_distance, -0.2, 0.2),
            higher_highs_count / 10,
            higher_lows_count / 10,
            lower_highs_count / 10,
            lower_lows_count / 10,
            channel_position,
            np.clip(trend_strength_adx, 0, 1),
        ], dtype=np.float32)
    
    def _extract_technical_features(self, closes: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Extract technical indicator features (10 features)."""
        n = len(closes)
        
        # MACD
        if n >= 26:
            ema_12 = self._ema(closes, 12)
            ema_26 = self._ema(closes, 26)
            macd_line = ema_12 - ema_26
            signal_line = self._ema(np.array([macd_line]), 9) if isinstance(macd_line, (int, float)) else self._ema(macd_line[-9:] if len(macd_line) > 9 else macd_line, 9)
            macd_histogram = (macd_line - signal_line) / closes[-1] if closes[-1] > 0 else 0
            
            # MACD signal cross (bullish cross = 1, bearish = -1)
            if n >= 27:
                prev_macd = self._ema(closes[:-1], 12) - self._ema(closes[:-1], 26)
                macd_signal_cross = 1 if (macd_line > signal_line and prev_macd <= signal_line) else (-1 if (macd_line < signal_line and prev_macd >= signal_line) else 0)
            else:
                macd_signal_cross = 0
        else:
            macd_histogram = 0
            macd_signal_cross = 0
        
        # Stochastic
        if n >= 14:
            lowest_low = np.min(lows[-14:])
            highest_high = np.max(highs[-14:])
            stochastic_k = (closes[-1] - lowest_low) / (highest_high - lowest_low) if highest_high != lowest_low else 0.5
            stochastic_d = np.mean([
                (closes[-i] - np.min(lows[-14-i:-i] if i > 0 else lows[-14:])) / 
                (np.max(highs[-14-i:-i] if i > 0 else highs[-14:]) - np.min(lows[-14-i:-i] if i > 0 else lows[-14:]) + 1e-10)
                for i in range(3)
            ]) if n >= 17 else stochastic_k
        else:
            stochastic_k = 0.5
            stochastic_d = 0.5
        
        # Williams %R
        williams_r = -stochastic_k if n >= 14 else -0.5
        
        # CCI
        if n >= 20:
            typical_price = (closes + highs + lows) / 3
            sma_tp = np.mean(typical_price[-20:])
            mean_dev = np.mean(np.abs(typical_price[-20:] - sma_tp))
            cci_20 = (typical_price[-1] - sma_tp) / (0.015 * mean_dev) if mean_dev > 0 else 0
        else:
            cci_20 = 0
        
        # Money Flow Index (MFI)
        if n >= 14:
            typical_prices = (closes + highs + lows) / 3
            raw_money_flow = typical_prices * volumes
            
            positive_flow = sum(raw_money_flow[i] for i in range(-14, 0) if typical_prices[i] > typical_prices[i-1])
            negative_flow = sum(raw_money_flow[i] for i in range(-14, 0) if typical_prices[i] < typical_prices[i-1])
            
            mfi_14 = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10))) if negative_flow > 0 else 100
            mfi_14 = mfi_14 / 100  # Normalize to 0-1
        else:
            mfi_14 = 0.5
        
        # Rate of Change
        roc_10 = (closes[-1] - closes[-11]) / closes[-11] if n > 10 and closes[-11] > 0 else 0
        
        # TRIX
        if n >= 45:
            ema1 = self._ema(closes, 15)
            ema2 = self._ema(np.array([ema1]), 15) if isinstance(ema1, (int, float)) else ema1
            ema3 = self._ema(np.array([ema2]), 15) if isinstance(ema2, (int, float)) else ema2
            trix_15 = (ema3 - self._ema(closes[:-1], 15)) / self._ema(closes[:-1], 15) if self._ema(closes[:-1], 15) > 0 else 0
        else:
            trix_15 = 0
        
        # Ultimate Oscillator
        if n >= 28:
            bp = closes - np.minimum(lows, np.roll(closes, 1))
            tr = np.maximum(highs - lows, np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1))
            ))
            
            avg7 = np.sum(bp[-7:]) / np.sum(tr[-7:]) if np.sum(tr[-7:]) > 0 else 0
            avg14 = np.sum(bp[-14:]) / np.sum(tr[-14:]) if np.sum(tr[-14:]) > 0 else 0
            avg28 = np.sum(bp[-28:]) / np.sum(tr[-28:]) if np.sum(tr[-28:]) > 0 else 0
            
            ultimate_oscillator = (4 * avg7 + 2 * avg14 + avg28) / 7
        else:
            ultimate_oscillator = 0.5
        
        return np.array([
            np.clip(macd_histogram * 100, -5, 5),
            macd_signal_cross,
            stochastic_k,
            stochastic_d,
            williams_r,
            np.clip(cci_20 / 200, -1, 1),
            mfi_14,
            np.clip(roc_10, -0.3, 0.3),
            np.clip(trix_15 * 1000, -1, 1),
            ultimate_oscillator,
        ], dtype=np.float32)
    
    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Compute RSI indicator."""
        if len(closes) < period + 1:
            return 50.0
        
        changes = np.diff(closes[-period-1:])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Compute Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def extract_batch(self, windows: List[OhlcvWindow]) -> np.ndarray:
        """
        Extract features for multiple windows.
        
        Returns:
            numpy array of shape (n_windows, feature_dim)
        """
        features = [self.extract(w) for w in windows]
        return np.stack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.FEATURE_NAMES.copy()
    
    def compute_forward_returns(self, current_bars: List[OhlcvBar], 
                                 future_bars: List[OhlcvBar]) -> MultiHorizonLabels:
        """
        Compute multi-horizon forward return labels.
        
        Args:
            current_bars: Historical bars up to prediction point
            future_bars: Future bars (at least 10 days)
            
        Returns:
            MultiHorizonLabels with forward returns and risk metrics
        """
        if len(future_bars) < 10:
            logger.warning(f"Insufficient future bars ({len(future_bars)}), returning zeros")
            return MultiHorizonLabels(
                return_3d=0, return_5d=0, return_8d=0, return_10d=0,
                max_adverse_excursion_5d=0, max_favorable_excursion_5d=0,
                hit_rate_threshold_3pct=False, hit_rate_threshold_5pct=False,
                gap_risk_5d=0
            )
        
        entry_price = current_bars[-1].close
        
        # Forward returns
        return_3d = (future_bars[2].close - entry_price) / entry_price
        return_5d = (future_bars[4].close - entry_price) / entry_price
        return_8d = (future_bars[7].close - entry_price) / entry_price if len(future_bars) >= 8 else return_5d
        return_10d = (future_bars[9].close - entry_price) / entry_price if len(future_bars) >= 10 else return_8d
        
        # Max adverse/favorable excursion in first 5 days
        future_lows = [b.low for b in future_bars[:5]]
        future_highs = [b.high for b in future_bars[:5]]
        
        max_adverse_excursion_5d = (min(future_lows) - entry_price) / entry_price
        max_favorable_excursion_5d = (max(future_highs) - entry_price) / entry_price
        
        # Hit rate thresholds
        hit_rate_threshold_3pct = max_favorable_excursion_5d >= 0.03
        hit_rate_threshold_5pct = max_favorable_excursion_5d >= 0.05
        
        # Gap risk (largest overnight gap in 5 days)
        gap_risk_5d = 0
        for i in range(1, min(5, len(future_bars))):
            gap = abs(future_bars[i].open - future_bars[i-1].close) / future_bars[i-1].close
            gap_risk_5d = max(gap_risk_5d, gap)
        
        return MultiHorizonLabels(
            return_3d=return_3d,
            return_5d=return_5d,
            return_8d=return_8d,
            return_10d=return_10d,
            max_adverse_excursion_5d=max_adverse_excursion_5d,
            max_favorable_excursion_5d=max_favorable_excursion_5d,
            hit_rate_threshold_3pct=hit_rate_threshold_3pct,
            hit_rate_threshold_5pct=hit_rate_threshold_5pct,
            gap_risk_5d=gap_risk_5d
        )


# Legacy FeatureExtractor for backward compatibility
class FeatureExtractor:
    """
    Original Feature Extractor (30 features).
    Use SwingFeatureExtractor for enhanced swing trade predictions.
    """
    
    FEATURE_NAMES = [
        "wick_ratio", "body_ratio", "bullish_pct_last20", "compression_score",
        "noise_score", "strength_slope", "range_density", "volume_intensity",
        "trend_consistency", "volatility_ratio",
        "price_entropy", "volume_entropy", "combined_entropy", "entropy_floor",
        "suppression_level", "coil_factor", "breakout_probability",
        "drift_magnitude", "drift_direction", "mean_reversion_pressure",
        "sma_20_ratio", "sma_50_ratio", "rsi_14", "atr_ratio",
        "recent_return_5", "recent_return_10", "recent_return_20",
        "high_low_ratio", "close_position", "volume_sma_ratio",
    ]
    
    def __init__(self):
        self.feature_dim = len(self.FEATURE_NAMES)
        self._swing_extractor = SwingFeatureExtractor()
    
    def extract(self, window: OhlcvWindow) -> np.ndarray:
        """Extract feature vector (returns first 30 features for compatibility)."""
        full_features = self._swing_extractor.extract(window)
        return full_features[:30]
    
    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Compute RSI indicator."""
        return self._swing_extractor._compute_rsi(closes, period)
    
    def extract_batch(self, windows: List[OhlcvWindow]) -> np.ndarray:
        """Extract features for multiple windows."""
        features = [self.extract(w) for w in windows]
        return np.stack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.FEATURE_NAMES.copy()


# Module-level instances
_default_extractor = None
_swing_extractor = None


def extract_features(windows: List[OhlcvWindow]) -> np.ndarray:
    """
    Module-level function to extract features from windows (legacy 30 features).
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = FeatureExtractor()
    return _default_extractor.extract_batch(windows)


def extract_swing_features(windows: List[OhlcvWindow]) -> np.ndarray:
    """
    Module-level function to extract enhanced swing trade features (80 features).
    """
    global _swing_extractor
    if _swing_extractor is None:
        _swing_extractor = SwingFeatureExtractor()
    return _swing_extractor.extract_batch(windows)


def get_swing_extractor() -> SwingFeatureExtractor:
    """Get the singleton swing feature extractor instance."""
    global _swing_extractor
    if _swing_extractor is None:
        _swing_extractor = SwingFeatureExtractor()
    return _swing_extractor


@dataclass
class RunnerSignals:
    """Container for breakout runner signals."""
    breakout_score: float  # 0-100 composite breakout probability
    timing_score: float    # 0-100 immediacy of expected move
    magnitude_score: float # 0-100 expected magnitude of move
    confidence: float      # 0-1 signal confidence
    signal_type: str       # 'immediate', 'imminent', 'developing', 'none'
    primary_catalyst: str  # Main driver of the signal
    secondary_catalysts: List[str]  # Supporting signals
    
    
class RunnerHunter:
    """
    10x ENHANCED RUNNER DETECTION SYSTEM
    
    Specialized detector for massive swing runners that break out immediately.
    Combines 150+ signals across 8 categories for maximum breakout probability.
    
    Categories:
    1. SQUEEZE DETECTION - Volatility compression before explosion
    2. MOMENTUM IGNITION - Early momentum signals before the crowd
    3. VOLUME SURGE - Institutional accumulation patterns
    4. CONSOLIDATION QUALITY - Base building and coiling patterns
    5. RELATIVE STRENGTH - Leadership vs market and sector
    6. BREAKOUT PROXIMITY - Distance to key resistance levels
    7. TIMING SIGNALS - Immediate move probability
    8. CATALYST ALIGNMENT - Multiple signal convergence
    """
    
    # Thresholds for runner classification
    RUNNER_THRESHOLDS = {
        'immediate': 85,   # Break out within 1-2 days
        'imminent': 70,    # Break out within 3-5 days
        'developing': 55,  # Break out within 5-10 days
    }
    
    # Feature names for all 150 runner-specific signals
    RUNNER_FEATURE_NAMES = [
        # Squeeze Detection (20 features)
        "bollinger_squeeze_intensity", "keltner_squeeze_intensity", "squeeze_duration_days",
        "squeeze_tightness_ratio", "volatility_percentile_20d", "volatility_percentile_60d",
        "atr_compression_5_20", "atr_compression_10_40", "range_compression_intensity",
        "historical_vol_floor", "implied_vol_ratio", "squeeze_fire_imminent",
        "tight_range_count", "narrowest_range_5d", "narrowest_range_10d",
        "vol_expansion_potential", "spring_compression", "energy_buildup",
        "pre_explosion_signal", "squeeze_quality_score",
        
        # Momentum Ignition (20 features)
        "rsi_breakout_signal", "rsi_thrust_intensity", "rsi_divergence_bullish",
        "macd_histogram_expansion", "macd_zero_cross_imminent", "macd_signal_strength",
        "price_acceleration_1d", "price_acceleration_3d", "price_acceleration_5d",
        "momentum_thrust_5d", "momentum_thrust_10d", "momentum_divergence",
        "rate_of_change_breakout", "velocity_increasing", "acceleration_positive",
        "force_index_surge", "elder_impulse_bullish", "william_ad_breakout",
        "tsi_breakout", "momentum_ignition_score",
        
        # Volume Surge (20 features)
        "volume_breakout_ratio", "volume_climax_signal", "pocket_pivot_signal",
        "accumulation_day_count", "distribution_day_count", "up_down_volume_ratio",
        "obv_breakout_signal", "obv_new_high", "volume_dry_up_signal",
        "smart_money_accumulation", "institutional_buying_signal", "volume_thrust",
        "relative_volume_1d", "relative_volume_3d", "relative_volume_5d",
        "volume_price_confirmation", "volume_precedes_price", "churn_signal",
        "volume_contraction_quality", "volume_surge_score",
        
        # Consolidation Quality (20 features)
        "base_length_days", "base_depth_percent", "tight_closes_count",
        "volatility_contraction_quality", "handle_formation", "cup_formation",
        "flat_base_score", "ascending_base_score", "double_bottom_score",
        "higher_lows_streak", "support_touch_count", "resistance_test_count",
        "consolidation_volume_pattern", "shakeout_recovery", "undercut_and_rally",
        "base_failure_count", "constructive_action", "tight_area_count",
        "orderly_pullback", "consolidation_quality_score",
        
        # Relative Strength (20 features)
        "rs_vs_spy_20d", "rs_vs_spy_60d", "rs_new_high_signal",
        "rs_breakout_signal", "rs_above_70_days", "sector_rs_rank",
        "industry_rs_rank", "rs_acceleration", "rs_momentum",
        "outperformance_streak", "relative_strength_line_slope", "rs_divergence",
        "sector_rotation_leader", "market_leader_signal", "institutional_sponsorship",
        "fund_ownership_increasing", "smart_money_flow_rs", "relative_volume_vs_sector",
        "alpha_generation", "relative_strength_score",
        
        # Breakout Proximity (20 features)
        "distance_to_52w_high", "distance_to_resistance", "distance_to_pivot",
        "breakout_pivot_distance", "prior_base_high_distance", "fibonacci_resistance_distance",
        "trendline_resistance_distance", "gap_resistance_distance", "round_number_distance",
        "overhead_supply_thin", "resistance_cluster_strength", "breakout_attempt_count",
        "failed_breakout_count", "successful_breakout_count", "breakout_success_rate",
        "price_ceiling_proximity", "clean_breakout_setup", "resistance_weakening",
        "buy_point_distance", "breakout_proximity_score",
        
        # Timing Signals (15 features)
        "earnings_catalyst_near", "sector_rotation_timing", "market_breadth_improving",
        "risk_on_environment", "seasonality_bullish", "day_of_week_optimal",
        "opening_range_breakout", "first_hour_strength", "gap_and_go_setup",
        "immediate_follow_through", "momentum_continuation", "breakout_day_volume",
        "institutional_participation", "confirmation_pending", "timing_score",
        
        # Catalyst Alignment (15 features)
        "signal_convergence_count", "multi_timeframe_alignment", "price_volume_confirmation",
        "indicator_confluence", "pattern_recognition_confidence", "setup_quality",
        "risk_reward_ratio", "expected_magnitude", "probability_weighted_return",
        "catalyst_strength", "bullish_signal_count", "bearish_signal_count",
        "net_signal_strength", "overall_conviction", "composite_runner_score",
    ]
    
    def __init__(self):
        """Initialize the Runner Hunter system."""
        self.feature_dim = len(self.RUNNER_FEATURE_NAMES)
        self._swing_extractor = SwingFeatureExtractor()
        logger.info(f"[RunnerHunter] Initialized with {self.feature_dim} runner-specific features")
    
    def hunt(self, window: OhlcvWindow) -> Tuple[RunnerSignals, np.ndarray]:
        """
        Hunt for massive swing runners in the given window.
        
        Args:
            window: OhlcvWindow with at least 60 bars
            
        Returns:
            Tuple of (RunnerSignals, feature_array)
        """
        features = self.extract_runner_features(window)
        signals = self._compute_runner_signals(features)
        return signals, features
    
    def extract_runner_features(self, window: OhlcvWindow) -> np.ndarray:
        """
        Extract all 150 runner-specific features from the window.
        
        Args:
            window: OhlcvWindow with at least 60 bars
            
        Returns:
            numpy array with 150 runner features
        """
        bars = window.bars
        if len(bars) < 60:
            logger.warning(f"[RunnerHunter] Insufficient bars: {len(bars)} < 60")
            return np.zeros(self.feature_dim)
        
        # Extract base price/volume arrays
        opens = np.array([b.open for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])
        
        # Extract each category of features
        squeeze_features = self._extract_squeeze_features(opens, highs, lows, closes, volumes)
        momentum_features = self._extract_momentum_ignition_features(opens, highs, lows, closes, volumes)
        volume_features = self._extract_volume_surge_features(opens, highs, lows, closes, volumes)
        consolidation_features = self._extract_consolidation_features(opens, highs, lows, closes, volumes)
        rs_features = self._extract_relative_strength_features(opens, highs, lows, closes, volumes)
        proximity_features = self._extract_breakout_proximity_features(opens, highs, lows, closes, volumes)
        timing_features = self._extract_timing_features(opens, highs, lows, closes, volumes)
        alignment_features = self._extract_catalyst_alignment_features(
            squeeze_features, momentum_features, volume_features, 
            consolidation_features, rs_features, proximity_features, timing_features
        )
        
        # Combine all features
        all_features = np.concatenate([
            squeeze_features,      # 20
            momentum_features,     # 20
            volume_features,       # 20
            consolidation_features,# 20
            rs_features,           # 20
            proximity_features,    # 20
            timing_features,       # 15
            alignment_features,    # 15
        ])
        
        return all_features
    
    def _extract_squeeze_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 squeeze detection features."""
        features = np.zeros(20)
        
        # Bollinger Bands
        sma_20 = np.mean(closes[-20:])
        std_20 = np.std(closes[-20:])
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        bb_width = (bb_upper - bb_lower) / sma_20 if sma_20 > 0 else 0
        
        # Keltner Channels (using simplified True Range)
        tr = np.maximum(highs[-20:] - lows[-20:], 
                       np.abs(highs[-20:] - closes[-21:-1]))
        atr_20 = np.mean(tr) if len(tr) > 0 else np.mean(highs[-20:] - lows[-20:])
        kc_upper = sma_20 + 1.5 * atr_20
        kc_lower = sma_20 - 1.5 * atr_20
        kc_width = (kc_upper - kc_lower) / sma_20 if sma_20 > 0 else 0
        
        # Squeeze detection
        squeeze_on = bb_width < kc_width
        features[0] = 1.0 if squeeze_on else 0.0  # bollinger_squeeze_intensity
        features[1] = min(kc_width / bb_width, 2.0) if bb_width > 0 else 0  # keltner_squeeze_intensity
        
        # Squeeze duration
        squeeze_days = 0
        for i in range(min(20, len(closes))):
            idx = -(i+1)
            sma = np.mean(closes[idx-20:idx]) if idx-20 >= -len(closes) else sma_20
            std = np.std(closes[idx-20:idx]) if idx-20 >= -len(closes) else std_20
            if std < atr_20 * 0.75:
                squeeze_days += 1
            else:
                break
        features[2] = min(squeeze_days / 10.0, 2.0)  # squeeze_duration_days
        
        # Tightness ratio
        range_20 = max(highs[-20:]) - min(lows[-20:])
        range_5 = max(highs[-5:]) - min(lows[-5:])
        features[3] = 1 - (range_5 / range_20) if range_20 > 0 else 0  # squeeze_tightness_ratio
        
        # Volatility percentiles
        returns = np.diff(np.log(closes[-60:]))
        vol_20 = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
        vol_60 = np.std(returns) * np.sqrt(252) if len(returns) >= 20 else 0
        features[4] = min(vol_20 / 0.5, 1.0)  # volatility_percentile_20d
        features[5] = min(vol_60 / 0.5, 1.0)  # volatility_percentile_60d
        
        # ATR compression ratios
        atr_5 = np.mean(np.maximum(highs[-5:] - lows[-5:], 0))
        atr_10 = np.mean(np.maximum(highs[-10:] - lows[-10:], 0))
        atr_40 = np.mean(np.maximum(highs[-40:] - lows[-40:], 0)) if len(highs) >= 40 else atr_20
        features[6] = 1 - (atr_5 / atr_20) if atr_20 > 0 else 0  # atr_compression_5_20
        features[7] = 1 - (atr_10 / atr_40) if atr_40 > 0 else 0  # atr_compression_10_40
        
        # Range compression
        daily_ranges = highs - lows
        avg_range_5 = np.mean(daily_ranges[-5:])
        avg_range_20 = np.mean(daily_ranges[-20:])
        features[8] = 1 - (avg_range_5 / avg_range_20) if avg_range_20 > 0 else 0  # range_compression_intensity
        
        # Historical volatility floor
        vol_min = np.min([np.std(returns[i:i+10]) for i in range(len(returns)-10)]) if len(returns) >= 20 else vol_20
        features[9] = 1 if np.std(returns[-10:]) <= vol_min * 1.1 else 0  # historical_vol_floor
        
        # Implied vol ratio (approximate)
        features[10] = 0.5  # implied_vol_ratio (placeholder - needs options data)
        
        # Squeeze fire imminent
        squeeze_firing = squeeze_on and (features[2] > 0.5) and (features[3] > 0.6)
        features[11] = 1.0 if squeeze_firing else 0.0  # squeeze_fire_imminent
        
        # Tight range counts
        nr_count = sum(1 for i in range(-5, 0) if (highs[i] - lows[i]) < np.mean(daily_ranges[-20:-5]) * 0.7)
        features[12] = min(nr_count / 3.0, 1.0)  # tight_range_count
        
        # Narrowest range signals
        range_5d_min = min(daily_ranges[-5:])
        range_10d_min = min(daily_ranges[-10:])
        features[13] = 1 if range_5d_min == min(daily_ranges[-20:]) else 0  # narrowest_range_5d
        features[14] = 1 if range_10d_min == min(daily_ranges[-30:]) else 0  # narrowest_range_10d
        
        # Volatility expansion potential
        vol_compression = 1 - (vol_20 / vol_60) if vol_60 > 0 else 0
        features[15] = max(0, vol_compression)  # vol_expansion_potential
        
        # Spring compression (price coiling)
        price_range_ratio = range_5 / (closes[-1] * 0.1) if closes[-1] > 0 else 0
        features[16] = max(0, 1 - price_range_ratio)  # spring_compression
        
        # Energy buildup
        volume_compression = 1 - (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 0
        energy = max(0, features[3] * 0.4 + features[16] * 0.3 + volume_compression * 0.3)
        features[17] = energy  # energy_buildup
        
        # Pre-explosion signal
        features[18] = 1.0 if (features[11] > 0 and features[17] > 0.6) else 0.0  # pre_explosion_signal
        
        # Composite squeeze quality score
        features[19] = np.mean([features[0], features[3], features[11], features[16], features[17]])  # squeeze_quality_score
        
        return features
    
    def _extract_momentum_ignition_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 momentum ignition features."""
        features = np.zeros(20)
        
        # RSI calculations
        rsi = self._compute_rsi(closes, 14)
        rsi_5d_ago = self._compute_rsi(closes[:-5], 14) if len(closes) > 19 else rsi
        
        # RSI breakout signals
        features[0] = 1 if (rsi > 60 and rsi_5d_ago < 50) else 0  # rsi_breakout_signal
        features[1] = max(0, (rsi - rsi_5d_ago) / 20)  # rsi_thrust_intensity
        
        # RSI divergence (price lower low, RSI higher low)
        price_ll = closes[-1] < min(closes[-10:-1])
        rsi_hl = rsi > self._compute_rsi(closes[:-5], 14)
        features[2] = 1 if (price_ll and rsi_hl) else 0  # rsi_divergence_bullish
        
        # MACD calculations
        ema_12 = self._compute_ema(closes, 12)
        ema_26 = self._compute_ema(closes, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._compute_ema(np.array([macd_line]), 9) if isinstance(macd_line, (int, float)) else macd_line
        histogram = macd_line - signal_line if isinstance(signal_line, (int, float)) else 0
        
        features[3] = max(0, histogram / (closes[-1] * 0.01)) if closes[-1] > 0 else 0  # macd_histogram_expansion
        features[4] = 1 if (macd_line < 0 and macd_line > -closes[-1] * 0.01) else 0  # macd_zero_cross_imminent
        features[5] = max(0, macd_line / (closes[-1] * 0.02)) if closes[-1] > 0 else 0  # macd_signal_strength
        
        # Price acceleration
        ret_1d = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
        ret_3d = (closes[-1] - closes[-4]) / closes[-4] if closes[-4] > 0 else 0
        ret_5d = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] > 0 else 0
        
        features[6] = max(-1, min(1, ret_1d * 20))  # price_acceleration_1d
        features[7] = max(-1, min(1, ret_3d * 10))  # price_acceleration_3d
        features[8] = max(-1, min(1, ret_5d * 5))   # price_acceleration_5d
        
        # Momentum thrust
        ret_10d = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] > 0 else 0
        features[9] = max(-1, min(1, ret_5d * 10))   # momentum_thrust_5d
        features[10] = max(-1, min(1, ret_10d * 5))  # momentum_thrust_10d
        
        # Momentum divergence
        price_trend = (closes[-1] - closes[-10]) / closes[-10] if len(closes) > 10 and closes[-10] > 0 else 0
        mom_trend = ret_5d - (closes[-6] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] > 0 else 0
        features[11] = max(0, mom_trend) if price_trend < 0 else 0  # momentum_divergence
        
        # Rate of change breakout
        roc_10 = ret_10d * 100
        roc_breakout = roc_10 > 5  # 5% breakout threshold
        features[12] = 1 if roc_breakout else 0  # rate_of_change_breakout
        
        # Velocity and acceleration
        velocity = ret_5d - (closes[-6] - closes[-11]) / closes[-11] if len(closes) > 10 and closes[-11] > 0 else 0
        features[13] = 1 if velocity > 0.02 else 0  # velocity_increasing
        features[14] = 1 if (velocity > 0 and ret_1d > ret_3d / 3) else 0  # acceleration_positive
        
        # Force index
        force = ret_1d * volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 0
        features[15] = max(0, min(1, force))  # force_index_surge
        
        # Elder impulse
        elder_bullish = (ema_12 > ema_26) and (histogram > 0 if isinstance(histogram, (int, float)) else True)
        features[16] = 1 if elder_bullish else 0  # elder_impulse_bullish
        
        # Williams AD breakout
        features[17] = 1 if (closes[-1] > closes[-2] and volumes[-1] > np.mean(volumes[-5:])) else 0  # william_ad_breakout
        
        # TSI breakout (approximate)
        features[18] = 1 if (ret_5d > 0.03 and ret_1d > 0.01) else 0  # tsi_breakout
        
        # Composite momentum ignition score
        features[19] = np.mean([features[0], features[3], features[6], features[9], features[12], features[16]])  # momentum_ignition_score
        
        return features
    
    def _extract_volume_surge_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 volume surge features."""
        features = np.zeros(20)
        
        avg_vol_5 = np.mean(volumes[-5:])
        avg_vol_20 = np.mean(volumes[-20:])
        avg_vol_50 = np.mean(volumes[-50:]) if len(volumes) >= 50 else avg_vol_20
        
        # Volume breakout ratio
        features[0] = volumes[-1] / avg_vol_50 if avg_vol_50 > 0 else 1  # volume_breakout_ratio
        
        # Climax volume (unusually high)
        features[1] = 1 if volumes[-1] > avg_vol_20 * 2 else 0  # volume_climax_signal
        
        # Pocket pivot (volume surge on up day after consolidation)
        up_day = closes[-1] > closes[-2]
        vol_surge = volumes[-1] > max(volumes[-10:-1])
        features[2] = 1 if (up_day and vol_surge) else 0  # pocket_pivot_signal
        
        # Accumulation/Distribution day counts
        acc_days = sum(1 for i in range(-20, 0) 
                      if closes[i] > closes[i-1] and volumes[i] > avg_vol_20)
        dist_days = sum(1 for i in range(-20, 0) 
                       if closes[i] < closes[i-1] and volumes[i] > avg_vol_20)
        features[3] = acc_days / 10.0  # accumulation_day_count
        features[4] = dist_days / 10.0  # distribution_day_count
        
        # Up/down volume ratio
        up_vol = sum(volumes[i] for i in range(-10, 0) if closes[i] > closes[i-1])
        down_vol = sum(volumes[i] for i in range(-10, 0) if closes[i] < closes[i-1])
        features[5] = up_vol / (down_vol + 1)  # up_down_volume_ratio
        
        # OBV signals
        obv = np.cumsum(np.where(np.diff(closes[-30:]) > 0, volumes[-29:], 
                                np.where(np.diff(closes[-30:]) < 0, -volumes[-29:], 0)))
        obv_slope = (obv[-1] - obv[-10]) / (np.std(obv) + 1) if len(obv) >= 10 else 0
        features[6] = max(-1, min(1, obv_slope))  # obv_breakout_signal
        features[7] = 1 if obv[-1] == max(obv) else 0  # obv_new_high
        
        # Volume dry-up (low volume before breakout)
        vol_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1
        features[8] = 1 if vol_ratio < 0.7 else 0  # volume_dry_up_signal
        
        # Smart money accumulation
        close_location = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1]) if (highs[-1] - lows[-1]) > 0 else 0.5
        smart_money = close_location * volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 0
        features[9] = min(smart_money, 2.0)  # smart_money_accumulation
        
        # Institutional buying signal
        features[10] = 1 if (volumes[-1] > avg_vol_20 * 1.5 and closes[-1] > opens[-1]) else 0  # institutional_buying_signal
        
        # Volume thrust
        vol_thrust = (volumes[-1] - avg_vol_5) / avg_vol_20 if avg_vol_20 > 0 else 0
        features[11] = max(0, vol_thrust)  # volume_thrust
        
        # Relative volumes
        features[12] = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1  # relative_volume_1d
        features[13] = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1   # relative_volume_3d (using 5d)
        features[14] = avg_vol_5 / avg_vol_50 if avg_vol_50 > 0 else 1   # relative_volume_5d
        
        # Volume-price confirmation
        price_up = closes[-1] > closes[-2]
        vol_up = volumes[-1] > volumes[-2]
        features[15] = 1 if (price_up and vol_up) else 0  # volume_price_confirmation
        
        # Volume precedes price
        vol_expanding = avg_vol_5 > avg_vol_20
        price_flat = abs(closes[-1] - closes[-5]) / closes[-5] < 0.02 if closes[-5] > 0 else True
        features[16] = 1 if (vol_expanding and price_flat) else 0  # volume_precedes_price
        
        # Churn (high volume, small price change)
        churn = volumes[-1] / avg_vol_20 / (abs(closes[-1] - closes[-2]) / closes[-2] + 0.001) if closes[-2] > 0 else 0
        features[17] = min(churn / 100, 1.0)  # churn_signal
        
        # Volume contraction quality
        vol_contraction = 1 - (avg_vol_5 / avg_vol_20) if avg_vol_20 > 0 else 0
        features[18] = max(0, vol_contraction)  # volume_contraction_quality
        
        # Composite volume surge score
        features[19] = np.mean([features[0]/2, features[2], features[5]/3, features[9]/2, features[10]])  # volume_surge_score
        
        return features
    
    def _extract_consolidation_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 consolidation quality features."""
        features = np.zeros(20)
        
        # Find consolidation base
        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        range_20 = high_20 - low_20
        
        # Base metrics
        features[0] = 20 / 60  # base_length_days (normalized)
        features[1] = range_20 / closes[-1] if closes[-1] > 0 else 0  # base_depth_percent
        
        # Tight closes count (closes within 1% of each other)
        tight_closes = 0
        for i in range(-10, -1):
            if abs(closes[i] - closes[i+1]) / closes[i] < 0.01:
                tight_closes += 1
        features[2] = tight_closes / 9.0  # tight_closes_count
        
        # Volatility contraction quality
        vol_early = np.std(closes[-20:-10])
        vol_late = np.std(closes[-10:])
        features[3] = 1 - (vol_late / vol_early) if vol_early > 0 else 0  # volatility_contraction_quality
        
        # Pattern detection (simplified)
        features[4] = 0.5  # handle_formation (placeholder)
        features[5] = 0.5  # cup_formation (placeholder)
        
        # Base types
        flat_score = 1 - (range_20 / closes[-1] / 0.15) if closes[-1] > 0 else 0
        features[6] = max(0, min(1, flat_score))  # flat_base_score
        
        # Ascending base (higher lows)
        low_5 = min(lows[-5:])
        low_10 = min(lows[-10:-5])
        low_15 = min(lows[-15:-10])
        ascending = (low_5 > low_10) and (low_10 > low_15)
        features[7] = 1 if ascending else 0  # ascending_base_score
        
        # Double bottom
        features[8] = 0.5  # double_bottom_score (placeholder)
        
        # Higher lows streak
        hl_streak = 0
        for i in range(-1, -10, -1):
            if lows[i] > lows[i-1]:
                hl_streak += 1
            else:
                break
        features[9] = hl_streak / 5.0  # higher_lows_streak
        
        # Support/resistance touches
        support_level = low_20
        resistance_level = high_20
        support_touches = sum(1 for l in lows[-20:] if abs(l - support_level) / support_level < 0.01)
        resistance_touches = sum(1 for h in highs[-20:] if abs(h - resistance_level) / resistance_level < 0.01)
        features[10] = min(support_touches / 3.0, 1.0)    # support_touch_count
        features[11] = min(resistance_touches / 3.0, 1.0)  # resistance_test_count
        
        # Consolidation volume pattern
        vol_trend = (np.mean(volumes[-5:]) - np.mean(volumes[-20:-5])) / np.mean(volumes[-20:-5]) if np.mean(volumes[-20:-5]) > 0 else 0
        features[12] = max(-1, min(1, -vol_trend))  # consolidation_volume_pattern (declining is good)
        
        # Shakeout recovery
        features[13] = 0.5  # shakeout_recovery (placeholder)
        
        # Undercut and rally
        features[14] = 0.5  # undercut_and_rally (placeholder)
        
        # Base failure count
        features[15] = 0.0  # base_failure_count
        
        # Constructive action
        constructive = features[3] > 0.3 and features[9] > 0.4
        features[16] = 1 if constructive else 0  # constructive_action
        
        # Tight areas
        features[17] = features[2]  # tight_area_count
        
        # Orderly pullback
        pullback_orderly = vol_late < vol_early * 0.8
        features[18] = 1 if pullback_orderly else 0  # orderly_pullback
        
        # Composite consolidation quality
        features[19] = np.mean([features[3], features[6], features[7], features[9], features[16]])  # consolidation_quality_score
        
        return features
    
    def _extract_relative_strength_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 relative strength features."""
        features = np.zeros(20)
        
        # Calculate returns for RS
        ret_20d = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 and closes[-21] > 0 else 0
        ret_60d = (closes[-1] - closes[-60]) / closes[-60] if len(closes) >= 60 and closes[-60] > 0 else ret_20d
        
        # Assume market return (SPY proxy - in production, use actual SPY data)
        market_ret_20d = 0.02  # Placeholder
        market_ret_60d = 0.05  # Placeholder
        
        # Relative strength vs market
        features[0] = ret_20d - market_ret_20d  # rs_vs_spy_20d
        features[1] = ret_60d - market_ret_60d  # rs_vs_spy_60d
        
        # RS new high
        features[2] = 1 if closes[-1] > max(closes[-60:-1]) else 0  # rs_new_high_signal
        
        # RS breakout
        rs_20d_prev = (closes[-6] - closes[-26]) / closes[-26] if len(closes) > 25 and closes[-26] > 0 else 0
        features[3] = 1 if (ret_20d > rs_20d_prev + 0.05) else 0  # rs_breakout_signal
        
        # RS above 70 days
        days_above = sum(1 for i in range(-20, 0) if closes[i] > closes[i-1])
        features[4] = days_above / 20.0  # rs_above_70_days (approximation)
        
        # Sector/Industry RS (placeholders - need sector data)
        features[5] = 0.7  # sector_rs_rank
        features[6] = 0.7  # industry_rs_rank
        
        # RS acceleration
        rs_accel = ret_20d - rs_20d_prev
        features[7] = max(-0.5, min(0.5, rs_accel))  # rs_acceleration
        
        # RS momentum
        features[8] = max(-1, min(1, ret_20d * 5))  # rs_momentum
        
        # Outperformance streak
        streak = sum(1 for i in range(-10, 0) if (closes[i] - closes[i-1]) / closes[i-1] > 0.001)
        features[9] = streak / 10.0  # outperformance_streak
        
        # RS line slope
        rs_slope = (ret_20d - (closes[-11] - closes[-31]) / closes[-31]) if len(closes) > 30 and closes[-31] > 0 else 0
        features[10] = max(-1, min(1, rs_slope * 10))  # relative_strength_line_slope
        
        # RS divergence
        features[11] = 0.5  # rs_divergence (placeholder)
        
        # Leadership signals
        features[12] = 1 if ret_20d > 0.08 else 0  # sector_rotation_leader
        features[13] = 1 if (ret_20d > 0.1 and ret_60d > 0.2) else 0  # market_leader_signal
        
        # Institutional sponsorship (approximation based on volume)
        avg_vol = np.mean(volumes[-20:])
        avg_vol_60 = np.mean(volumes[-60:]) if len(volumes) >= 60 else avg_vol
        features[14] = min(avg_vol / avg_vol_60, 2.0) if avg_vol_60 > 0 else 1  # institutional_sponsorship
        
        # Fund ownership (placeholder)
        features[15] = 0.5  # fund_ownership_increasing
        
        # Smart money flow RS
        up_vol = sum(volumes[i] for i in range(-10, 0) if closes[i] > closes[i-1])
        total_vol = sum(volumes[-10:])
        features[16] = up_vol / total_vol if total_vol > 0 else 0.5  # smart_money_flow_rs
        
        # Relative volume vs sector (placeholder)
        features[17] = 1.0  # relative_volume_vs_sector
        
        # Alpha generation
        features[18] = max(0, ret_20d - market_ret_20d)  # alpha_generation
        
        # Composite RS score
        features[19] = np.mean([features[0]+0.5, features[2], features[3], features[8]+0.5, features[13]])  # relative_strength_score
        
        return features
    
    def _extract_breakout_proximity_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 20 breakout proximity features."""
        features = np.zeros(20)
        
        current_price = closes[-1]
        high_52w = max(highs[-252:]) if len(highs) >= 252 else max(highs)
        high_20d = max(highs[-20:])
        high_10d = max(highs[-10:])
        
        # Distance to highs
        features[0] = 1 - (current_price / high_52w) if high_52w > 0 else 0  # distance_to_52w_high
        features[1] = 1 - (current_price / high_20d) if high_20d > 0 else 0  # distance_to_resistance
        features[2] = 1 - (current_price / high_10d) if high_10d > 0 else 0  # distance_to_pivot
        
        # Breakout pivot distance
        pivot = (max(highs[-5:]) + min(lows[-5:]) + closes[-1]) / 3
        features[3] = (pivot - current_price) / current_price if current_price > 0 else 0  # breakout_pivot_distance
        
        # Prior base high
        features[4] = features[1]  # prior_base_high_distance
        
        # Fibonacci resistance (approximate)
        range_20 = max(highs[-20:]) - min(lows[-20:])
        fib_382 = min(lows[-20:]) + range_20 * 0.382
        fib_618 = min(lows[-20:]) + range_20 * 0.618
        features[5] = abs(current_price - fib_382) / current_price if current_price > 0 else 0  # fibonacci_resistance_distance
        features[6] = abs(current_price - fib_618) / current_price if current_price > 0 else 0  # fib_618_distance
        
        # Trendline resistance (simplified)
        features[7] = features[1]  # trendline_resistance_distance
        
        # Gap resistance
        gaps = []
        for i in range(-20, -1):
            gap = opens[i+1] - closes[i]
            if abs(gap) / closes[i] > 0.02:
                gaps.append(opens[i+1])
        if gaps:
            nearest_gap = min(gaps, key=lambda x: abs(x - current_price))
            features[8] = abs(current_price - nearest_gap) / current_price if current_price > 0 else 0
        else:
            features[8] = 0.1  # gap_resistance_distance
        
        # Round number distance
        round_num = round(current_price, -1) if current_price >= 10 else round(current_price)
        features[9] = abs(current_price - round_num) / current_price if current_price > 0 else 0  # round_number_distance
        
        # Overhead supply thin
        above_price_volume = sum(volumes[i] for i in range(-20, 0) if closes[i] > current_price)
        total_vol = sum(volumes[-20:])
        features[10] = 1 - (above_price_volume / total_vol) if total_vol > 0 else 0.5  # overhead_supply_thin
        
        # Resistance cluster strength
        resistance_tests = sum(1 for h in highs[-20:] if abs(h - high_20d) / high_20d < 0.01)
        features[11] = min(resistance_tests / 5.0, 1.0)  # resistance_cluster_strength
        
        # Breakout attempts
        breakout_attempts = sum(1 for h in highs[-10:] if h > high_20d * 0.99)
        features[12] = min(breakout_attempts / 3.0, 1.0)  # breakout_attempt_count
        
        # Failed breakouts
        failed = sum(1 for i in range(-10, 0) if highs[i] > high_20d * 0.99 and closes[i] < high_20d * 0.98)
        features[13] = min(failed / 2.0, 1.0)  # failed_breakout_count
        
        # Successful breakouts
        features[14] = max(0, features[12] - features[13])  # successful_breakout_count
        
        # Breakout success rate
        features[15] = features[14] / (features[12] + 0.1)  # breakout_success_rate
        
        # Price ceiling proximity
        features[16] = 1 - features[0]  # price_ceiling_proximity
        
        # Clean breakout setup
        clean = features[10] > 0.7 and features[3] < 0.03
        features[17] = 1 if clean else 0  # clean_breakout_setup
        
        # Resistance weakening
        features[18] = 1 if features[12] > 0.3 and features[13] < 0.3 else 0  # resistance_weakening
        
        # Buy point distance
        features[19] = max(0, 1 - features[1] * 10)  # buy_point_distance (closer = higher)
        
        return features
    
    def _extract_timing_features(self, opens, highs, lows, closes, volumes) -> np.ndarray:
        """Extract 15 timing features."""
        features = np.zeros(15)
        
        # Earnings catalyst (placeholder)
        features[0] = 0.5  # earnings_catalyst_near
        
        # Sector rotation timing
        features[1] = 0.5  # sector_rotation_timing
        
        # Market breadth (approximation)
        up_days = sum(1 for i in range(-10, 0) if closes[i] > closes[i-1])
        features[2] = up_days / 10.0  # market_breadth_improving
        
        # Risk environment
        vol_ratio = np.std(closes[-10:]) / np.std(closes[-30:]) if np.std(closes[-30:]) > 0 else 1
        features[3] = 1 if vol_ratio < 1.2 else 0  # risk_on_environment
        
        # Seasonality (placeholder)
        features[4] = 0.5  # seasonality_bullish
        
        # Day of week (placeholder)
        features[5] = 0.5  # day_of_week_optimal
        
        # Opening range breakout
        orb = closes[-1] > highs[-2] and volumes[-1] > np.mean(volumes[-5:])
        features[6] = 1 if orb else 0  # opening_range_breakout
        
        # First hour strength (approximation)
        first_move = (closes[-1] - opens[-1]) / opens[-1] if opens[-1] > 0 else 0
        features[7] = max(0, min(1, first_move * 10))  # first_hour_strength
        
        # Gap and go
        gap = (opens[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
        gap_and_go = gap > 0.02 and closes[-1] > opens[-1]
        features[8] = 1 if gap_and_go else 0  # gap_and_go_setup
        
        # Immediate follow through
        strong_close = closes[-1] > (highs[-1] + lows[-1]) / 2
        vol_confirm = volumes[-1] > np.mean(volumes[-5:])
        features[9] = 1 if (strong_close and vol_confirm) else 0  # immediate_follow_through
        
        # Momentum continuation
        mom_cont = closes[-1] > closes[-2] > closes[-3]
        features[10] = 1 if mom_cont else 0  # momentum_continuation
        
        # Breakout day volume
        features[11] = volumes[-1] / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1  # breakout_day_volume
        
        # Institutional participation
        features[12] = 1 if features[11] > 1.5 else 0  # institutional_participation
        
        # Confirmation pending
        near_breakout = closes[-1] > max(highs[-20:-1]) * 0.98
        features[13] = 1 if near_breakout else 0  # confirmation_pending
        
        # Composite timing score
        features[14] = np.mean([features[6], features[8], features[9], features[10], features[13]])  # timing_score
        
        return features
    
    def _extract_catalyst_alignment_features(
        self, squeeze, momentum, volume, consolidation, rs, proximity, timing
    ) -> np.ndarray:
        """Extract 15 catalyst alignment features."""
        features = np.zeros(15)
        
        # Signal convergence count
        signals = [
            squeeze[19] > 0.6,   # squeeze quality
            momentum[19] > 0.5,  # momentum ignition
            volume[19] > 0.5,    # volume surge
            consolidation[19] > 0.5,  # consolidation quality
            rs[19] > 0.5,        # relative strength
            proximity[19] > 0.7, # breakout proximity
            timing[14] > 0.5,    # timing score
        ]
        features[0] = sum(signals) / 7.0  # signal_convergence_count
        
        # Multi-timeframe alignment
        features[1] = (momentum[9] + momentum[10]) / 2  # multi_timeframe_alignment
        
        # Price-volume confirmation
        features[2] = volume[15]  # volume_price_confirmation
        
        # Indicator confluence
        confluence = sum([
            1 if squeeze[11] > 0 else 0,  # squeeze firing
            1 if momentum[0] > 0 else 0,   # RSI breakout
            1 if volume[2] > 0 else 0,     # pocket pivot
            1 if proximity[17] > 0 else 0, # clean setup
        ])
        features[3] = confluence / 4.0  # indicator_confluence
        
        # Pattern recognition confidence
        features[4] = consolidation[19]  # pattern_recognition_confidence
        
        # Setup quality
        features[5] = np.mean([squeeze[19], consolidation[19], proximity[17]])  # setup_quality
        
        # Risk/reward ratio
        potential_gain = proximity[16]  # price ceiling proximity
        potential_loss = 1 - consolidation[6]  # base depth risk
        features[6] = potential_gain / (potential_loss + 0.1)  # risk_reward_ratio
        
        # Expected magnitude
        features[7] = np.mean([momentum[19], rs[19], proximity[16]])  # expected_magnitude
        
        # Probability-weighted return
        prob = features[0]  # signal convergence as probability
        features[8] = prob * features[7]  # probability_weighted_return
        
        # Catalyst strength
        features[9] = max(squeeze[19], momentum[19], volume[19])  # catalyst_strength
        
        # Bullish signal count
        bullish = sum([
            squeeze[11] > 0, squeeze[18] > 0,
            momentum[0] > 0, momentum[12] > 0, momentum[16] > 0,
            volume[2] > 0, volume[10] > 0,
            rs[13] > 0,
            proximity[17] > 0,
            timing[6] > 0, timing[8] > 0,
        ])
        features[10] = bullish / 11.0  # bullish_signal_count
        
        # Bearish signal count
        bearish = sum([
            volume[4] > 0.3,  # distribution days
            rs[0] < -0.05,    # underperformance
        ])
        features[11] = bearish / 2.0  # bearish_signal_count
        
        # Net signal strength
        features[12] = features[10] - features[11]  # net_signal_strength
        
        # Overall conviction
        features[13] = np.mean([features[0], features[5], features[10]])  # overall_conviction
        
        # COMPOSITE RUNNER SCORE (the main output)
        composite = (
            squeeze[19] * 0.15 +      # Squeeze quality
            momentum[19] * 0.20 +     # Momentum ignition
            volume[19] * 0.15 +       # Volume surge
            consolidation[19] * 0.10 + # Consolidation quality
            rs[19] * 0.15 +           # Relative strength
            proximity[19] * 0.15 +    # Breakout proximity
            timing[14] * 0.10         # Timing score
        )
        features[14] = composite  # composite_runner_score
        
        return features
    
    def _compute_runner_signals(self, features: np.ndarray) -> RunnerSignals:
        """Compute final runner signals from extracted features."""
        
        # Extract key scores
        squeeze_score = features[19]       # squeeze_quality_score
        momentum_score = features[39]      # momentum_ignition_score
        volume_score = features[59]        # volume_surge_score
        consolidation_score = features[79] # consolidation_quality_score
        rs_score = features[99]            # relative_strength_score
        proximity_score = features[119]    # breakout_proximity_score
        timing_score = features[134]       # timing_score
        composite_score = features[149]    # composite_runner_score
        
        # Convert to 0-100 scale
        breakout_score = composite_score * 100
        timing_score_100 = timing_score * 100
        magnitude_score = (momentum_score + rs_score + proximity_score) / 3 * 100
        
        # Determine signal type
        if breakout_score >= self.RUNNER_THRESHOLDS['immediate']:
            signal_type = 'immediate'
        elif breakout_score >= self.RUNNER_THRESHOLDS['imminent']:
            signal_type = 'imminent'
        elif breakout_score >= self.RUNNER_THRESHOLDS['developing']:
            signal_type = 'developing'
        else:
            signal_type = 'none'
        
        # Identify primary catalyst
        scores = {
            'squeeze': squeeze_score,
            'momentum': momentum_score,
            'volume': volume_score,
            'consolidation': consolidation_score,
            'relative_strength': rs_score,
            'proximity': proximity_score,
            'timing': timing_score,
        }
        primary_catalyst = max(scores, key=scores.get)
        
        # Get secondary catalysts (top 3 excluding primary)
        sorted_catalysts = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        secondary_catalysts = [c[0] for c in sorted_catalysts[1:4]]
        
        # Confidence based on signal convergence
        confidence = features[135] / 0.7  # Normalize convergence count
        confidence = min(1.0, max(0.0, confidence))
        
        return RunnerSignals(
            breakout_score=breakout_score,
            timing_score=timing_score_100,
            magnitude_score=magnitude_score,
            confidence=confidence,
            signal_type=signal_type,
            primary_catalyst=primary_catalyst,
            secondary_catalysts=secondary_catalysts,
        )
    
    def _compute_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Compute RSI indicator."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _compute_ema(self, data: np.ndarray, period: int) -> float:
        """Compute EMA of data."""
        if len(data) < period:
            return float(data[-1]) if len(data) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = data[-period]
        for price in data[-period+1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all runner feature names."""
        return self.RUNNER_FEATURE_NAMES.copy()
    
    def scan_for_runners(
        self, 
        windows: List[OhlcvWindow],
        min_score: float = 70.0
    ) -> List[Tuple[str, RunnerSignals, np.ndarray]]:
        """
        Scan multiple windows for potential runners.
        
        Args:
            windows: List of OhlcvWindow to scan
            min_score: Minimum breakout score to include (default: 70)
            
        Returns:
            List of (symbol, signals, features) tuples for qualifying stocks
        """
        runners = []
        
        for window in windows:
            signals, features = self.hunt(window)
            if signals.breakout_score >= min_score:
                runners.append((window.symbol, signals, features))
        
        # Sort by breakout score descending
        runners.sort(key=lambda x: x[1].breakout_score, reverse=True)
        
        return runners


# Module-level runner hunter instance
_runner_hunter = None


def get_runner_hunter() -> RunnerHunter:
    """Get the singleton RunnerHunter instance."""
    global _runner_hunter
    if _runner_hunter is None:
        _runner_hunter = RunnerHunter()
    return _runner_hunter
