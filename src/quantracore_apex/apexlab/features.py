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
