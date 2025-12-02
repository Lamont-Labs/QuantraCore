#!/usr/bin/env python3
"""
BREAKOUT TIMER - Predicts WHEN a stock will start running.
Outputs days-to-breakout countdown with confidence levels.
"""

import os
import sys
import logging
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MOONSHOT_DIR = Path("data/moonshots")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def extract_timing_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Extract features that predict breakout timing."""
    if idx < 50 or idx >= len(df):
        return {}
    
    window = df.iloc[idx-50:idx+1]
    
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    open_prices = window["open"].values if "open" in window.columns else close
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    
    vol_10d = np.std(returns[-10:])
    vol_20d = np.std(returns[-20:])
    vol_compression = vol_10d / (vol_20d + 0.001)
    
    range_10d = (np.max(high[-10:]) - np.min(low[-10:])) / (close[-1] + 0.001)
    range_20d = (np.max(high[-20:]) - np.min(low[-20:])) / (close[-1] + 0.001)
    range_compression = range_10d / (range_20d + 0.001)
    
    vol_ratio_5d = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1)
    vol_ratio_3d = np.mean(volume[-3:]) / (np.mean(volume[-20:]) + 1)
    vol_acceleration = vol_ratio_3d / (vol_ratio_5d + 0.001)
    
    mom_3d = (close[-1] - close[-4]) / (close[-4] + 0.001)
    mom_5d = (close[-1] - close[-6]) / (close[-6] + 0.001)
    mom_10d = (close[-1] - close[-11]) / (close[-11] + 0.001)
    mom_inflection = mom_3d - mom_5d
    
    sma_10 = np.mean(close[-10:])
    sma_20 = np.mean(close[-20:])
    sma_50 = np.mean(close[-50:])
    sma_slope = (sma_10 - sma_20) / (sma_20 + 0.001)
    
    price_vs_sma20 = close[-1] / (sma_20 + 0.001)
    price_vs_sma50 = close[-1] / (sma_50 + 0.001)
    
    daily_range = high[-10:] - low[-10:]
    close_position = (close[-10:] - low[-10:]) / (daily_range + 0.001)
    close_near_high = np.mean(close_position)
    
    upper_wick = (high[-10:] - np.maximum(close[-10:], open_prices[-10:])) / (daily_range + 0.001)
    wick_shrinking = np.mean(upper_wick[-3:]) - np.mean(upper_wick[-10:-3])
    
    high_20d = np.max(high[-20:])
    breakout_proximity = close[-1] / (high_20d + 0.001)
    
    tight_days = sum(1 for i in range(-10, 0) if abs(returns[i]) < 0.02)
    consolidation_days = tight_days
    
    dollar_vol = close[-20:] * volume[-20:]
    dv_ratio = np.mean(dollar_vol[-3:]) / (np.mean(dollar_vol[-20:]) + 1)
    
    down_days = sum(1 for r in returns[-5:] if r < 0)
    up_days = 5 - down_days
    
    return {
        "vol_compression": float(vol_compression),
        "range_compression": float(range_compression),
        "vol_ratio_5d": float(vol_ratio_5d),
        "vol_acceleration": float(vol_acceleration),
        "mom_3d": float(mom_3d),
        "mom_5d": float(mom_5d),
        "mom_10d": float(mom_10d),
        "mom_inflection": float(mom_inflection),
        "sma_slope": float(sma_slope),
        "price_vs_sma20": float(price_vs_sma20),
        "price_vs_sma50": float(price_vs_sma50),
        "close_near_high": float(close_near_high),
        "wick_shrinking": float(wick_shrinking),
        "breakout_proximity": float(breakout_proximity),
        "consolidation_days": float(consolidation_days),
        "dv_ratio": float(dv_ratio),
        "up_days_5d": float(up_days),
        "volatility_20d": float(vol_20d),
    }

def load_moonshot_data():
    """Load moonshot data with days_to_peak labels."""
    path = MOONSHOT_DIR / "moonshots_latest.pkl.gz"
    if not path.exists():
        logger.error("Moonshots not found! Run moonshot_hunter_fast.py first")
        return []
    
    with gzip.open(path, "rb") as f:
        moonshots = pickle.load(f)
    
    logger.info(f"Loaded {len(moonshots)} moonshot events")
    return moonshots

def build_timing_dataset(moonshots: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build training dataset for timing prediction."""
    X = []
    y_days = []
    y_bins = []
    
    symbols_processed = set()
    symbol_data = {}
    
    for ms in moonshots:
        symbol = ms["symbol"]
        if symbol not in symbol_data:
            path = CACHE_DIR / f"{symbol}.parquet"
            if path.exists():
                symbol_data[symbol] = pd.read_parquet(path)
    
    feature_cols = None
    
    for ms in moonshots:
        symbol = ms["symbol"]
        if symbol not in symbol_data:
            continue
        
        df = symbol_data[symbol]
        idx = ms.get("window_start_idx", 0)
        days_to_peak = ms.get("days_to_peak", 15)
        
        if days_to_peak > 30:
            days_to_peak = 30
        
        features = extract_timing_features(df, idx)
        if not features:
            continue
        
        if feature_cols is None:
            feature_cols = list(features.keys())
        
        row = [features.get(k, 0) for k in feature_cols]
        
        if any(np.isnan(v) or np.isinf(v) for v in row):
            continue
        
        if days_to_peak <= 3:
            timing_bin = 0
        elif days_to_peak <= 5:
            timing_bin = 1
        elif days_to_peak <= 10:
            timing_bin = 2
        elif days_to_peak <= 20:
            timing_bin = 3
        else:
            timing_bin = 4
        
        X.append(row)
        y_days.append(days_to_peak)
        y_bins.append(timing_bin)
    
    logger.info(f"Built dataset with {len(X)} samples")
    logger.info(f"Features: {feature_cols}")
    
    return np.array(X), np.array(y_days), np.array(y_bins), feature_cols

def train_timing_models(X, y_days, y_bins, feature_cols):
    """Train models for breakout timing prediction."""
    
    X_train, X_test, y_days_train, y_days_test, y_bins_train, y_bins_test = train_test_split(
        X, y_days, y_bins, test_size=0.2, random_state=42
    )
    
    logger.info("\n[1/2] Training Days-to-Peak Regressor...")
    regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    regressor.fit(X_train, y_days_train)
    
    y_days_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_days_test, y_days_pred)
    r2 = r2_score(y_days_test, y_days_pred)
    
    logger.info(f"  MAE: {mae:.2f} days")
    logger.info(f"  R2 Score: {r2:.3f}")
    
    logger.info("\n[2/2] Training Timing Bin Classifier...")
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    classifier.fit(X_train, y_bins_train)
    
    y_bins_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_bins_test, y_bins_pred)
    
    logger.info(f"  Accuracy: {accuracy:.2%}")
    
    bin_labels = ["0-3 days", "4-5 days", "6-10 days", "11-20 days", "21-30 days"]
    for i, label in enumerate(bin_labels):
        count = sum(1 for y in y_bins if y == i)
        logger.info(f"    {label}: {count} samples")
    
    return {
        "regressor": regressor,
        "classifier": classifier,
        "feature_cols": feature_cols,
        "metrics": {
            "mae_days": mae,
            "r2": r2,
            "bin_accuracy": accuracy,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "bin_labels": bin_labels,
    }

def save_timing_model(model_data: Dict):
    """Save the timing model."""
    path = MODELS_DIR / "breakout_timer.pkl.gz"
    with gzip.open(path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"Saved breakout timer to {path}")
    return path

def predict_breakout_timing(model_data: Dict, features: Dict[str, float]) -> Dict[str, Any]:
    """Predict breakout timing for a stock."""
    feature_cols = model_data["feature_cols"]
    row = np.array([[features.get(k, 0) for k in feature_cols]])
    
    expected_days = model_data["regressor"].predict(row)[0]
    
    bin_probs = model_data["classifier"].predict_proba(row)[0]
    predicted_bin = np.argmax(bin_probs)
    
    p_within_3 = bin_probs[0] if len(bin_probs) > 0 else 0
    p_within_5 = sum(bin_probs[:2]) if len(bin_probs) > 1 else p_within_3
    p_within_10 = sum(bin_probs[:3]) if len(bin_probs) > 2 else p_within_5
    
    return {
        "expected_days": round(expected_days, 1),
        "predicted_timing": model_data["bin_labels"][predicted_bin],
        "confidence": round(max(bin_probs) * 100, 1),
        "p_within_3_days": round(p_within_3 * 100, 1),
        "p_within_5_days": round(p_within_5 * 100, 1),
        "p_within_10_days": round(p_within_10 * 100, 1),
        "bin_probabilities": {
            label: round(prob * 100, 1) 
            for label, prob in zip(model_data["bin_labels"], bin_probs)
        }
    }

def main():
    logger.info("=" * 60)
    logger.info("BREAKOUT TIMER - Predicting WHEN stocks will run")
    logger.info("=" * 60)
    
    moonshots = load_moonshot_data()
    if not moonshots:
        sys.exit(1)
    
    days_list = [m.get("days_to_peak", 15) for m in moonshots]
    logger.info(f"\nDays-to-Peak Distribution:")
    logger.info(f"  Min: {min(days_list)} days")
    logger.info(f"  Max: {min(max(days_list), 30)} days (capped)")
    logger.info(f"  Median: {np.median(days_list):.1f} days")
    logger.info(f"  Mean: {np.mean(days_list):.1f} days")
    
    X, y_days, y_bins, feature_cols = build_timing_dataset(moonshots)
    
    if len(X) < 100:
        logger.error("Not enough training data!")
        sys.exit(1)
    
    model_data = train_timing_models(X, y_days, y_bins, feature_cols)
    
    save_timing_model(model_data)
    
    print("\n" + "=" * 60)
    print("BREAKOUT TIMER TRAINED!")
    print("=" * 60)
    print(f"MAE: {model_data['metrics']['mae_days']:.2f} days")
    print(f"Bin Accuracy: {model_data['metrics']['bin_accuracy']:.1%}")
    print(f"Training Samples: {model_data['metrics']['training_samples']}")
    print("\nTiming Bins:")
    for label in model_data['bin_labels']:
        print(f"  - {label}")
    print("=" * 60)
    
    logger.info("\nExample prediction with sample features:")
    sample_features = {
        "vol_compression": 0.7,
        "range_compression": 0.6,
        "vol_ratio_5d": 1.5,
        "vol_acceleration": 1.2,
        "mom_3d": 0.05,
        "mom_5d": 0.03,
        "mom_10d": 0.02,
        "mom_inflection": 0.02,
        "sma_slope": 0.03,
        "price_vs_sma20": 1.05,
        "price_vs_sma50": 1.1,
        "close_near_high": 0.8,
        "wick_shrinking": -0.1,
        "breakout_proximity": 0.95,
        "consolidation_days": 5,
        "dv_ratio": 1.3,
        "up_days_5d": 3,
        "volatility_20d": 0.04,
    }
    
    prediction = predict_breakout_timing(model_data, sample_features)
    print("\nSample Prediction:")
    print(f"  Expected Days to Breakout: {prediction['expected_days']}")
    print(f"  Predicted Timing: {prediction['predicted_timing']}")
    print(f"  Confidence: {prediction['confidence']}%")
    print(f"  P(within 3 days): {prediction['p_within_3_days']}%")
    print(f"  P(within 5 days): {prediction['p_within_5_days']}%")
    print(f"  P(within 10 days): {prediction['p_within_10_days']}%")

if __name__ == "__main__":
    main()
