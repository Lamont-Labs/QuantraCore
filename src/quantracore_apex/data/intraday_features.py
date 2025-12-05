"""
Intraday Feature Extractor for 1-Minute Bars

Optimized feature extraction for high-frequency data that captures:
- Microstructure patterns (bid-ask spread proxies, tick direction)
- Intraday momentum (opening drive, lunch lull, closing push)
- Volume profile patterns (VWAP deviation, volume clustering)
- Time-of-day effects (market open volatility, power hour)
- High-frequency regime detection
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, time

from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar

logger = logging.getLogger(__name__)


@dataclass
class IntradayFeatures:
    """Container for intraday-specific features."""
    microstructure: Dict[str, float]
    intraday_momentum: Dict[str, float]
    volume_profile: Dict[str, float]
    time_patterns: Dict[str, float]
    regime_features: Dict[str, float]


class IntradayFeatureExtractor:
    """
    Feature extractor optimized for 1-minute bar data.
    
    Extracts 120 features across 5 categories:
    1. Microstructure (25 features)
    2. Intraday Momentum (25 features)
    3. Volume Profile (25 features)
    4. Time Patterns (25 features)
    5. Regime Detection (20 features)
    """
    
    FEATURE_NAMES = [
        # Microstructure (25)
        "avg_bar_range", "avg_body_ratio", "tick_direction_bias",
        "bar_range_volatility", "wick_ratio_mean", "wick_ratio_std",
        "upper_wick_bias", "lower_wick_bias", "doji_frequency",
        "large_bar_frequency", "small_bar_frequency", "gap_frequency",
        "gap_up_frequency", "gap_down_frequency", "avg_gap_size",
        "bar_overlap_ratio", "trend_bar_frequency", "reversal_bar_frequency",
        "inside_bar_frequency", "outside_bar_frequency", "close_location_value",
        "price_efficiency", "noise_ratio", "directional_accuracy", "tick_imbalance",
        
        # Intraday Momentum (25)
        "return_5bars", "return_10bars", "return_20bars", "return_50bars",
        "momentum_5", "momentum_10", "momentum_20", "momentum_acceleration",
        "rsi_fast", "rsi_slow", "rsi_divergence", "macd_histogram",
        "macd_signal_dist", "price_velocity", "price_acceleration",
        "high_momentum", "low_momentum", "trend_strength", "trend_consistency",
        "pullback_depth", "extension_ratio", "mean_reversion_score",
        "breakout_strength", "breakdown_strength", "momentum_quality",
        
        # Volume Profile (25)
        "volume_mean", "volume_std", "volume_skew", "relative_volume",
        "volume_trend", "volume_acceleration", "vwap_deviation",
        "volume_at_highs", "volume_at_lows", "volume_balance",
        "accumulation_score", "distribution_score", "smart_money_flow",
        "volume_climax", "volume_dry_up", "volume_surge_count",
        "obv_slope", "obv_divergence", "cmf_value", "mfi_value",
        "volume_price_trend", "demand_index", "ease_of_movement",
        "force_index", "volume_efficiency",
        
        # Time Patterns (25)
        "is_market_open", "is_first_30min", "is_last_30min", "is_lunch_hour",
        "is_power_hour", "time_of_day_score", "session_progress",
        "open_range_breakout", "open_range_size", "morning_trend",
        "afternoon_trend", "lunch_consolidation", "closing_drive",
        "overnight_gap", "premarket_momentum", "first_bar_direction",
        "first_5bar_trend", "reversal_time_score", "trend_time_score",
        "volatility_time_pattern", "volume_time_pattern", "momentum_time_pattern",
        "session_high_time", "session_low_time", "range_expansion_time",
        
        # Regime Detection (20)
        "volatility_regime", "trend_regime", "volume_regime",
        "momentum_regime", "mean_reversion_regime", "breakout_regime",
        "consolidation_regime", "distribution_regime", "accumulation_regime",
        "panic_regime", "euphoria_regime", "neutral_regime",
        "regime_stability", "regime_transition_prob", "regime_duration",
        "regime_strength", "regime_confidence", "regime_trend",
        "multi_regime_score", "regime_alignment",
    ]
    
    def __init__(self):
        self.feature_dim = len(self.FEATURE_NAMES)
    
    def extract(self, window: OhlcvWindow) -> np.ndarray:
        """Extract all features from a 1-minute window."""
        bars = window.bars
        if len(bars) < 20:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        opens = np.array([b.open for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])
        timestamps = [b.timestamp for b in bars]
        
        microstructure = self._extract_microstructure(opens, highs, lows, closes, volumes)
        momentum = self._extract_intraday_momentum(opens, highs, lows, closes, volumes)
        volume_profile = self._extract_volume_profile(opens, highs, lows, closes, volumes)
        time_patterns = self._extract_time_patterns(opens, highs, lows, closes, volumes, timestamps)
        regime = self._extract_regime_features(opens, highs, lows, closes, volumes)
        
        features = np.concatenate([
            microstructure,
            momentum,
            volume_profile,
            time_patterns,
            regime,
        ])
        
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -10.0, 10.0)
        
        return features.astype(np.float32)
    
    def extract_batch(self, windows: List[OhlcvWindow]) -> np.ndarray:
        """Extract features for multiple windows."""
        features_list = [self.extract(w) for w in windows]
        return np.array(features_list)
    
    def _extract_microstructure(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Extract microstructure features (25)."""
        ranges = highs - lows
        bodies = np.abs(closes - opens)
        
        avg_range = np.mean(ranges) / (np.mean(closes) + 1e-8)
        body_ratio = np.mean(bodies / (ranges + 1e-8))
        
        tick_direction = np.sign(closes - opens)
        tick_bias = np.mean(tick_direction)
        
        range_vol = np.std(ranges) / (np.mean(ranges) + 1e-8)
        
        upper_wicks = highs - np.maximum(opens, closes)
        lower_wicks = np.minimum(opens, closes) - lows
        wick_ratios = (upper_wicks + lower_wicks) / (ranges + 1e-8)
        
        wick_mean = np.mean(wick_ratios)
        wick_std = np.std(wick_ratios)
        upper_bias = np.mean(upper_wicks / (ranges + 1e-8))
        lower_bias = np.mean(lower_wicks / (ranges + 1e-8))
        
        doji_mask = bodies < (ranges * 0.1)
        doji_freq = np.mean(doji_mask)
        
        large_mask = ranges > np.percentile(ranges, 90)
        small_mask = ranges < np.percentile(ranges, 10)
        large_freq = np.mean(large_mask)
        small_freq = np.mean(small_mask)
        
        gaps = opens[1:] - closes[:-1]
        gap_mask = np.abs(gaps) > np.std(closes) * 0.1
        gap_freq = np.mean(gap_mask)
        gap_up_freq = np.mean(gaps > np.std(closes) * 0.1)
        gap_down_freq = np.mean(gaps < -np.std(closes) * 0.1)
        avg_gap = np.mean(np.abs(gaps)) / (np.mean(closes) + 1e-8)
        
        overlaps = np.minimum(highs[:-1], highs[1:]) - np.maximum(lows[:-1], lows[1:])
        bar_overlap = np.mean(np.maximum(overlaps, 0) / (ranges[:-1] + 1e-8))
        
        trend_bars = np.abs(closes - opens) > ranges * 0.6
        trend_freq = np.mean(trend_bars)
        
        reversal_bars = (tick_direction[:-1] != tick_direction[1:])
        reversal_freq = np.mean(reversal_bars)
        
        inside_bars = (highs[1:] < highs[:-1]) & (lows[1:] > lows[:-1])
        inside_freq = np.mean(inside_bars)
        
        outside_bars = (highs[1:] > highs[:-1]) & (lows[1:] < lows[:-1])
        outside_freq = np.mean(outside_bars)
        
        clv = (2 * closes - highs - lows) / (ranges + 1e-8)
        close_location = np.mean(clv)
        
        total_range = np.sum(ranges)
        net_move = np.abs(closes[-1] - opens[0])
        price_efficiency = net_move / (total_range + 1e-8)
        
        noise_ratio = 1.0 - price_efficiency
        
        correct_direction = np.sign(closes[1:] - closes[:-1]) == np.sign(closes[-1] - closes[0])
        directional_accuracy = np.mean(correct_direction)
        
        up_ticks = int(np.sum(tick_direction > 0))
        down_ticks = int(np.sum(tick_direction < 0))
        tick_imbalance = (up_ticks - down_ticks) / (len(tick_direction) + 1e-8)
        
        return np.array([
            avg_range, body_ratio, tick_bias, range_vol, wick_mean,
            wick_std, upper_bias, lower_bias, doji_freq, large_freq,
            small_freq, gap_freq, gap_up_freq, gap_down_freq, avg_gap,
            bar_overlap, trend_freq, reversal_freq, inside_freq, outside_freq,
            close_location, price_efficiency, noise_ratio, directional_accuracy, tick_imbalance,
        ])
    
    def _extract_intraday_momentum(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Extract intraday momentum features (25)."""
        returns = np.diff(np.log(closes + 1e-8))
        
        ret_5 = np.sum(returns[-5:]) if len(returns) >= 5 else 0
        ret_10 = np.sum(returns[-10:]) if len(returns) >= 10 else 0
        ret_20 = np.sum(returns[-20:]) if len(returns) >= 20 else 0
        ret_50 = np.sum(returns[-50:]) if len(returns) >= 50 else 0
        
        mom_5 = closes[-1] / closes[-6] - 1 if len(closes) > 5 else 0
        mom_10 = closes[-1] / closes[-11] - 1 if len(closes) > 10 else 0
        mom_20 = closes[-1] / closes[-21] - 1 if len(closes) > 20 else 0
        
        mom_short = mom_5
        mom_long = mom_20
        mom_accel = mom_short - mom_long
        
        def calc_rsi(prices, period):
            if len(prices) < period + 1:
                return 50.0
            deltas = np.diff(prices)
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        rsi_fast = calc_rsi(closes, 7) / 100
        rsi_slow = calc_rsi(closes, 14) / 100
        rsi_div = rsi_fast - rsi_slow
        
        def calc_ema(data, period):
            if len(data) < period:
                return data
            alpha = 2 / (period + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
            return ema
        
        ema_12 = calc_ema(closes, 12)
        ema_26 = calc_ema(closes, 26)
        macd_line = ema_12 - ema_26
        signal_line = calc_ema(macd_line, 9)
        macd_hist = (macd_line[-1] - signal_line[-1]) / (np.std(closes) + 1e-8)
        macd_sig_dist = macd_line[-1] - signal_line[-1]
        
        velocity = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        accel = velocity - (np.mean(returns[-10:-5]) if len(returns) >= 10 else 0)
        
        high_mom = (closes[-1] - np.min(lows[-20:])) / (np.max(highs[-20:]) - np.min(lows[-20:]) + 1e-8)
        low_mom = (np.max(highs[-20:]) - closes[-1]) / (np.max(highs[-20:]) - np.min(lows[-20:]) + 1e-8)
        
        trend_str = np.abs(ret_20) / (np.std(returns[-20:]) + 1e-8) if len(returns) >= 20 else 0
        
        directions = np.sign(returns)
        trend_cons = np.mean(directions[-20:] == np.sign(ret_20)) if len(returns) >= 20 else 0.5
        
        max_high = np.max(highs)
        pullback = (max_high - closes[-1]) / (max_high - np.min(lows) + 1e-8)
        
        ext_ratio = (closes[-1] - closes[0]) / (np.max(highs) - np.min(lows) + 1e-8)
        
        mean_price = np.mean(closes)
        mean_rev = (mean_price - closes[-1]) / (np.std(closes) + 1e-8)
        
        recent_high = np.max(highs[-10:])
        breakout_str = (closes[-1] - recent_high) / (np.std(closes) + 1e-8) if closes[-1] > recent_high else 0
        
        recent_low = np.min(lows[-10:])
        breakdown_str = (recent_low - closes[-1]) / (np.std(closes) + 1e-8) if closes[-1] < recent_low else 0
        
        mom_quality = trend_cons * np.abs(ret_20)
        
        return np.array([
            ret_5, ret_10, ret_20, ret_50, mom_5, mom_10, mom_20, mom_accel,
            rsi_fast, rsi_slow, rsi_div, macd_hist, macd_sig_dist,
            velocity, accel, high_mom, low_mom, trend_str, trend_cons,
            pullback, ext_ratio, mean_rev, breakout_str, breakdown_str, mom_quality,
        ])
    
    def _extract_volume_profile(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Extract volume profile features (25)."""
        vol_mean = np.mean(volumes) / (np.max(volumes) + 1e-8)
        vol_std = np.std(volumes) / (np.mean(volumes) + 1e-8)
        
        vol_skew = 0.0
        if np.std(volumes) > 0:
            vol_skew = np.mean(((volumes - np.mean(volumes)) / np.std(volumes)) ** 3)
        
        recent_vol = np.mean(volumes[-5:])
        avg_vol = np.mean(volumes)
        rel_vol = recent_vol / (avg_vol + 1e-8)
        
        vol_sma_short = np.mean(volumes[-5:])
        vol_sma_long = np.mean(volumes[-20:]) if len(volumes) >= 20 else vol_sma_short
        vol_trend = (vol_sma_short - vol_sma_long) / (vol_sma_long + 1e-8)
        
        vol_accel = vol_trend - ((np.mean(volumes[-10:-5]) - vol_sma_long) / (vol_sma_long + 1e-8) if len(volumes) >= 10 else 0)
        
        typical_price = (highs + lows + closes) / 3
        vwap = np.sum(typical_price * volumes) / (np.sum(volumes) + 1e-8)
        vwap_dev = (closes[-1] - vwap) / (np.std(closes) + 1e-8)
        
        high_idx = np.argmax(highs)
        low_idx = np.argmin(lows)
        vol_at_highs = volumes[high_idx] / (avg_vol + 1e-8)
        vol_at_lows = volumes[low_idx] / (avg_vol + 1e-8)
        
        up_vol = np.sum(volumes[closes > opens])
        down_vol = np.sum(volumes[closes < opens])
        vol_balance = (up_vol - down_vol) / (up_vol + down_vol + 1e-8)
        
        accum_score = vol_balance * vol_trend if vol_balance > 0 else 0
        dist_score = -vol_balance * vol_trend if vol_balance < 0 else 0
        
        price_returns = np.diff(closes) / closes[:-1]
        vol_returns = np.diff(volumes) / (volumes[:-1] + 1e-8)
        smart_money = np.corrcoef(price_returns[-19:], vol_returns[-19:])[0, 1] if len(price_returns) >= 19 else 0
        if np.isnan(smart_money):
            smart_money = 0
        
        vol_climax = 1.0 if volumes[-1] > np.percentile(volumes, 95) else 0.0
        vol_dryup = 1.0 if volumes[-1] < np.percentile(volumes, 5) else 0.0
        
        vol_spikes = np.sum(volumes > np.mean(volumes) * 2)
        vol_surge = vol_spikes / len(volumes)
        
        obv = np.cumsum(np.where(closes > np.roll(closes, 1), volumes, -volumes))
        obv_slope = (obv[-1] - obv[-10]) / (np.std(obv) + 1e-8) if len(obv) >= 10 else 0
        
        price_trend = closes[-1] - closes[-10] if len(closes) >= 10 else 0
        obv_trend = obv[-1] - obv[-10] if len(obv) >= 10 else 0
        obv_div = 1.0 if (price_trend > 0 and obv_trend < 0) or (price_trend < 0 and obv_trend > 0) else 0.0
        
        mf_multiplier = (2 * closes - highs - lows) / (highs - lows + 1e-8)
        mf_volume = mf_multiplier * volumes
        cmf = np.sum(mf_volume[-20:]) / (np.sum(volumes[-20:]) + 1e-8) if len(volumes) >= 20 else 0
        
        raw_mf = typical_price * volumes
        pos_mf = np.sum(raw_mf[typical_price > np.roll(typical_price, 1)])
        neg_mf = np.sum(raw_mf[typical_price < np.roll(typical_price, 1)])
        mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))
        mfi = mfi / 100
        
        vpt = np.cumsum(((closes - np.roll(closes, 1)) / (np.roll(closes, 1) + 1e-8)) * volumes)
        vpt_val = vpt[-1] / (np.std(vpt) + 1e-8)
        
        demand_idx = vol_balance * rel_vol
        
        emv = ((highs - lows) / 2 - (np.roll(highs, 1) - np.roll(lows, 1)) / 2) / (volumes / (highs - lows + 1e-8) + 1e-8)
        emv_val = np.mean(emv[-10:]) if len(emv) >= 10 else 0
        
        force = (closes - np.roll(closes, 1)) * volumes
        force_idx = np.mean(force[-10:]) / (np.std(force) + 1e-8) if len(force) >= 10 else 0
        
        vol_eff = np.abs(closes[-1] - opens[0]) / (np.sum(volumes) + 1e-8) * 1e6
        
        return np.array([
            vol_mean, vol_std, vol_skew, rel_vol, vol_trend, vol_accel, vwap_dev,
            vol_at_highs, vol_at_lows, vol_balance, accum_score, dist_score, smart_money,
            vol_climax, vol_dryup, vol_surge, obv_slope, obv_div, cmf, mfi,
            vpt_val, demand_idx, emv_val, force_idx, vol_eff,
        ])
    
    def _extract_time_patterns(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: List[datetime],
    ) -> np.ndarray:
        """Extract time-based pattern features (25)."""
        if not timestamps:
            return np.zeros(25)
        
        last_time = timestamps[-1]
        if isinstance(last_time, datetime):
            hour = last_time.hour
            minute = last_time.minute
        else:
            hour = 12
            minute = 0
        
        is_market_open = 1.0 if hour == 9 and minute < 35 else 0.0
        is_first_30 = 1.0 if hour == 9 and minute < 60 else 0.0
        is_last_30 = 1.0 if hour == 15 and minute >= 30 else 0.0
        is_lunch = 1.0 if 11 <= hour <= 13 else 0.0
        is_power = 1.0 if hour >= 15 else 0.0
        
        minutes_from_open = (hour - 9) * 60 + minute - 30
        session_progress = max(0, min(1, minutes_from_open / 390))
        
        tod_score = 0.0
        if hour < 10:
            tod_score = 0.9
        elif hour < 11:
            tod_score = 0.7
        elif hour < 14:
            tod_score = 0.4
        elif hour < 15:
            tod_score = 0.6
        else:
            tod_score = 0.8
        
        n = min(30, len(closes))
        open_range = np.max(highs[:n]) - np.min(lows[:n]) if n > 0 else 0
        open_high = np.max(highs[:n]) if n > 0 else closes[-1]
        open_low = np.min(lows[:n]) if n > 0 else closes[-1]
        orb = 0.0
        if closes[-1] > open_high:
            orb = 1.0
        elif closes[-1] < open_low:
            orb = -1.0
        
        open_range_size = open_range / (np.mean(closes) + 1e-8)
        
        mid = len(closes) // 2
        morning_ret = (closes[mid] - closes[0]) / (closes[0] + 1e-8) if mid > 0 else 0
        afternoon_ret = (closes[-1] - closes[mid]) / (closes[mid] + 1e-8) if mid > 0 else 0
        
        lunch_start = len(closes) // 3
        lunch_end = 2 * len(closes) // 3
        lunch_range = np.max(highs[lunch_start:lunch_end]) - np.min(lows[lunch_start:lunch_end])
        total_range = np.max(highs) - np.min(lows)
        lunch_consol = 1 - (lunch_range / (total_range + 1e-8))
        
        close_n = min(30, len(closes))
        closing_drive = (closes[-1] - closes[-close_n]) / (np.std(closes) + 1e-8) if close_n > 1 else 0
        
        overnight_gap = (opens[0] - closes[0]) / (closes[0] + 1e-8) if len(closes) > 1 else 0
        
        premarket_mom = morning_ret
        
        first_bar_dir = 1.0 if closes[0] > opens[0] else -1.0
        
        first_5 = closes[:5] if len(closes) >= 5 else closes
        first_5_trend = (first_5[-1] - first_5[0]) / (np.std(first_5) + 1e-8) if len(first_5) > 1 else 0
        
        reversal_time = 0.0
        trend_time = 0.0
        
        if len(closes) > 10:
            mid_close = closes[len(closes)//2]
            if (closes[-1] > mid_close > closes[0]) or (closes[-1] < mid_close < closes[0]):
                trend_time = 1.0
            else:
                reversal_time = 1.0
        
        vol_pattern = np.std(volumes) / (np.mean(volumes) + 1e-8)
        mom_pattern = np.std(np.diff(closes)) / (np.mean(np.abs(np.diff(closes))) + 1e-8)
        
        high_idx = np.argmax(highs)
        low_idx = np.argmin(lows)
        session_high_time = high_idx / (len(highs) + 1e-8)
        session_low_time = low_idx / (len(lows) + 1e-8)
        
        ranges = highs - lows
        max_range_idx = np.argmax(ranges)
        range_exp_time = max_range_idx / (len(ranges) + 1e-8)
        
        return np.array([
            is_market_open, is_first_30, is_last_30, is_lunch, is_power,
            tod_score, session_progress, orb, open_range_size, morning_ret,
            afternoon_ret, lunch_consol, closing_drive, overnight_gap, premarket_mom,
            first_bar_dir, first_5_trend, reversal_time, trend_time, vol_pattern,
            mom_pattern, vol_pattern, session_high_time, session_low_time, range_exp_time,
        ])
    
    def _extract_regime_features(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """Extract regime detection features (20)."""
        returns = np.diff(closes) / closes[:-1]
        vol = np.std(returns) if len(returns) > 0 else 0
        vol_regime = 0.0
        if vol < 0.005:
            vol_regime = 0.2
        elif vol < 0.01:
            vol_regime = 0.4
        elif vol < 0.02:
            vol_regime = 0.6
        elif vol < 0.03:
            vol_regime = 0.8
        else:
            vol_regime = 1.0
        
        trend = (closes[-1] - closes[0]) / (closes[0] + 1e-8)
        trend_regime = 0.0
        if abs(trend) < 0.005:
            trend_regime = 0.0
        elif abs(trend) < 0.01:
            trend_regime = 0.3 * np.sign(trend)
        elif abs(trend) < 0.02:
            trend_regime = 0.6 * np.sign(trend)
        else:
            trend_regime = 1.0 * np.sign(trend)
        
        vol_change = np.mean(volumes[-10:]) / (np.mean(volumes[:10]) + 1e-8) if len(volumes) >= 20 else 1.0
        vol_reg = np.clip((vol_change - 0.5) / 1.5, -1, 1)
        
        mom_sum = np.sum(returns[-20:]) if len(returns) >= 20 else 0
        mom_regime = np.clip(mom_sum * 10, -1, 1)
        
        mean_price = np.mean(closes)
        mean_dev = (closes[-1] - mean_price) / (np.std(closes) + 1e-8)
        mean_rev_regime = np.clip(-mean_dev / 3, -1, 1)
        
        recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
        breakout_regime = 1.0 if closes[-1] > recent_high * 0.99 else 0.0
        
        ranges = highs - lows
        avg_range = np.mean(ranges)
        recent_range = np.mean(ranges[-10:]) if len(ranges) >= 10 else avg_range
        consol_regime = 1.0 if recent_range < avg_range * 0.5 else 0.0
        
        up_vol = np.sum(volumes[closes > opens])
        down_vol = np.sum(volumes[closes < opens])
        dist_regime = 1.0 if down_vol > up_vol * 1.5 else 0.0
        accum_regime = 1.0 if up_vol > down_vol * 1.5 else 0.0
        
        panic_regime = 1.0 if vol_regime > 0.8 and trend_regime < -0.5 else 0.0
        euph_regime = 1.0 if vol_regime > 0.8 and trend_regime > 0.5 else 0.0
        neutral_regime = 1.0 if abs(trend_regime) < 0.2 and vol_regime < 0.4 else 0.0
        
        regime_signals = [vol_regime, abs(trend_regime), mom_regime]
        regime_stability = 1.0 - np.std(regime_signals)
        
        regime_trans = np.abs(np.diff([vol_regime, trend_regime, mom_regime]))
        trans_prob = np.mean(regime_trans)
        
        regime_duration = 0.5
        
        regime_str = np.max([abs(trend_regime), vol_regime, abs(mom_regime)])
        
        regime_conf = regime_stability * regime_str
        
        regime_trend_val = (vol_regime + trend_regime + mom_regime) / 3
        
        multi_regime = vol_regime * 0.3 + abs(trend_regime) * 0.4 + abs(mom_regime) * 0.3
        
        regime_align = 1.0 if (trend_regime > 0 and mom_regime > 0) or (trend_regime < 0 and mom_regime < 0) else 0.0
        
        return np.array([
            vol_regime, trend_regime, vol_reg, mom_regime, mean_rev_regime,
            breakout_regime, consol_regime, dist_regime, accum_regime, panic_regime,
            euph_regime, neutral_regime, regime_stability, trans_prob, regime_duration,
            regime_str, regime_conf, regime_trend_val, multi_regime, regime_align,
        ])


def extract_intraday_features(windows: List[OhlcvWindow]) -> np.ndarray:
    """Module-level function to extract intraday features."""
    extractor = IntradayFeatureExtractor()
    return extractor.extract_batch(windows)
