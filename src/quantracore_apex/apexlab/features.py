"""
Feature Extractor for ApexLab.

Converts OHLCV windows into feature vectors for model training.
"""

import numpy as np
from typing import List
from src.quantracore_apex.core.schemas import OhlcvWindow
from src.quantracore_apex.core.microtraits import compute_microtraits
from src.quantracore_apex.core.entropy import compute_entropy
from src.quantracore_apex.core.suppression import compute_suppression
from src.quantracore_apex.core.drift import compute_drift


class FeatureExtractor:
    """
    Extracts feature vectors from OHLCV windows.
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
    
    def extract(self, window: OhlcvWindow) -> np.ndarray:
        """
        Extract feature vector from a window.
        
        Returns:
            numpy array of shape (feature_dim,)
        """
        bars = window.bars
        closes = np.array([b.close for b in bars])
        volumes = np.array([b.volume for b in bars])
        highs = np.array([b.high for b in bars])
        lows = np.array([b.low for b in bars])
        
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
        
        recent_return_5 = (closes[-1] - closes[-6]) / closes[-6] if closes[-6] != 0 else 0
        recent_return_10 = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] != 0 else 0
        recent_return_20 = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 20 and closes[-21] != 0 else 0
        
        period_high = np.max(highs[-20:])
        period_low = np.min(lows[-20:])
        high_low_ratio = (period_high - period_low) / period_low if period_low > 0 else 0
        close_position = (current_price - period_low) / (period_high - period_low) if period_high != period_low else 0.5
        
        volume_sma = np.mean(volumes[-20:])
        volume_sma_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1.0
        
        features = np.array([
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
        
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
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


_default_extractor = None

def extract_features(windows: List[OhlcvWindow]) -> np.ndarray:
    """
    Module-level function to extract features from windows.
    
    Args:
        windows: List of OhlcvWindow objects
        
    Returns:
        numpy array of shape (n_windows, feature_dim)
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = FeatureExtractor()
    return _default_extractor.extract_batch(windows)
