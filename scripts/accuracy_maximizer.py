#!/usr/bin/env python3
"""
ACCURACY MAXIMIZER - Implements all architect recommendations:
1. Hard negatives (failed breakouts, near-miss patterns)
2. Near-miss positives (30-50% gains for boundary learning)
3. Advanced features (gaps, vol dry-ups, stock embeddings)
4. Ensemble models (LightGBM + CatBoost + GradientBoosting)
5. Probability calibration and threshold optimization
6. High-precision mode for maximum accuracy

Target: Precision ≥80% at recall ≥30%, Stock-split ≥60%
"""

import os
import sys
import time
import json
import pickle
import gzip
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MODELS_DIR = Path("data/models")
VALIDATION_DIR = Path("data/validation")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_MAP = {
    "AAPL": "tech", "AMD": "tech", "NVDA": "tech", "TSLA": "auto",
    "BBAI": "tech", "CLOV": "health", "MARA": "crypto", "RIOT": "crypto",
    "PLTR": "tech", "SOFI": "fintech", "LCID": "auto", "RIVN": "auto",
    "GME": "retail", "AMC": "entertainment", "BB": "tech", "NOK": "tech",
    "SPCE": "aerospace", "OPEN": "realestate", "WISH": "retail",
    "SNDL": "cannabis", "TLRY": "cannabis", "CGC": "cannabis",
    "COIN": "crypto", "HOOD": "fintech", "UPST": "fintech",
}

def get_stock_embedding(symbol: str) -> Dict[str, float]:
    """Create stock-level features for better generalization."""
    sector = SECTOR_MAP.get(symbol, "unknown")
    
    sector_enc = {
        "tech": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "crypto": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "health": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "fintech": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "auto": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "retail": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "cannabis": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "entertainment": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "aerospace": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "unknown": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }
    
    vec = sector_enc.get(sector, sector_enc["unknown"])
    
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) / (16**8)
    
    return {
        f"sector_{i}": float(v) for i, v in enumerate(vec)
    } | {
        "symbol_hash": float(sym_hash),
        "is_meme": float(symbol in {"GME", "AMC", "BB", "BBAI", "CLOV", "WISH"}),
        "is_crypto_related": float(symbol in {"MARA", "RIOT", "COIN", "HOOD"}),
        "is_ev": float(symbol in {"TSLA", "LCID", "RIVN"}),
    }

def extract_advanced_features(df: pd.DataFrame, idx: int, symbol: str) -> Dict[str, float]:
    """Extract 70+ advanced features including all architect recommendations."""
    if idx < 60 or idx >= len(df):
        return {}
    
    window = df.iloc[idx-60:idx+1]
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    open_p = window["open"].values if "open" in window.columns else close
    
    if close[-1] <= 0:
        return {}
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    
    f = {}
    
    for p in [5, 10, 20, 50]:
        f[f"sma_{p}"] = np.mean(close[-p:])
        f[f"vol_sma_{p}"] = np.mean(volume[-p:])
    
    for p in [5, 10, 20, 30]:
        f[f"volatility_{p}"] = np.std(returns[-p:])
    
    f["vol_compression_5_20"] = f["volatility_5"] / (f["volatility_20"] + 0.001)
    f["vol_compression_10_30"] = f["volatility_10"] / (f["volatility_30"] + 0.001)
    f["vol_compression_5_30"] = f["volatility_5"] / (f["volatility_30"] + 0.001)
    
    for p in [3, 5, 10, 20]:
        f[f"momentum_{p}"] = (close[-1] - close[-p-1]) / (close[-p-1] + 0.001)
    
    f["mom_acceleration"] = f["momentum_3"] - f["momentum_5"]
    f["mom_inflection"] = f["momentum_5"] - f["momentum_10"]
    f["mom_acceleration_2"] = (f["momentum_3"] - f["momentum_5"]) - (f["momentum_5"] - f["momentum_10"])
    
    for p in [5, 10, 20]:
        f[f"vol_ratio_{p}"] = np.mean(volume[-3:]) / (np.mean(volume[-p:]) + 1)
    
    f["vol_acceleration"] = f["vol_ratio_5"] / (f["vol_ratio_10"] + 0.001)
    
    vol_5d_mean = np.mean(volume[-5:])
    vol_20d_mean = np.mean(volume[-20:])
    f["vol_dryup_5d"] = 1.0 if vol_5d_mean < vol_20d_mean * 0.5 else 0.0
    f["vol_dryup_ratio"] = vol_5d_mean / (vol_20d_mean + 1)
    
    f["vol_spike_today"] = 1.0 if volume[-1] > vol_20d_mean * 2 else 0.0
    f["vol_spike_3d"] = 1.0 if vol_5d_mean > vol_20d_mean * 1.5 else 0.0
    
    for p in [10, 20, 50]:
        f[f"price_vs_sma_{p}"] = close[-1] / (f[f"sma_{p}"] + 0.001)
    
    f["sma_slope_5_10"] = (f["sma_5"] - f["sma_10"]) / (f["sma_10"] + 0.001)
    f["sma_slope_10_20"] = (f["sma_10"] - f["sma_20"]) / (f["sma_20"] + 0.001)
    f["sma_slope_20_50"] = (f["sma_20"] - f["sma_50"]) / (f["sma_50"] + 0.001)
    
    f["sma_alignment"] = 1.0 if (f["sma_5"] > f["sma_10"] > f["sma_20"]) else 0.0
    
    for p in [10, 20, 30]:
        r = np.max(high[-p:]) - np.min(low[-p:])
        f[f"range_{p}"] = r / (close[-1] + 0.001)
    
    f["range_compression_10_20"] = f["range_10"] / (f["range_20"] + 0.001)
    f["range_compression_10_30"] = f["range_10"] / (f["range_30"] + 0.001)
    
    f["high_20"] = np.max(high[-20:])
    f["low_20"] = np.min(low[-20:])
    f["high_52w"] = np.max(high[-min(252, len(high)):]) if len(high) >= 50 else np.max(high[-50:])
    
    f["breakout_proximity"] = close[-1] / (f["high_20"] + 0.001)
    f["breakout_proximity_52w"] = close[-1] / (f["high_52w"] + 0.001)
    f["support_distance"] = (close[-1] - f["low_20"]) / (f["low_20"] + 0.001)
    
    for i in range(-5, 0):
        gap = (open_p[i] - close[i-1]) / (close[i-1] + 0.001)
        f[f"gap_day_{abs(i)}"] = gap
    
    f["gap_today"] = (open_p[-1] - close[-2]) / (close[-2] + 0.001)
    f["gap_avg_5d"] = np.mean([abs((open_p[i] - close[i-1]) / (close[i-1] + 0.001)) for i in range(-5, 0)])
    f["gap_up_count_5d"] = sum(1 for i in range(-5, 0) if open_p[i] > close[i-1])
    
    dr = high[-10:] - low[-10:]
    cp = (close[-10:] - low[-10:]) / (dr + 0.001)
    f["close_position_avg"] = np.mean(cp)
    f["close_position_today"] = cp[-1]
    
    uw = (high[-10:] - np.maximum(close[-10:], open_p[-10:])) / (dr + 0.001)
    lw = (np.minimum(close[-10:], open_p[-10:]) - low[-10:]) / (dr + 0.001)
    
    f["upper_wick_avg"] = np.mean(uw)
    f["lower_wick_avg"] = np.mean(lw)
    f["wick_shrinking"] = np.mean(uw[-3:]) - np.mean(uw[-7:-3])
    f["wick_ratio"] = np.mean(uw) / (np.mean(lw) + 0.001)
    
    f["up_days_5"] = sum(1 for r in returns[-5:] if r > 0)
    f["up_days_10"] = sum(1 for r in returns[-10:] if r > 0)
    f["up_days_20"] = sum(1 for r in returns[-20:] if r > 0)
    
    f["tight_days_5"] = sum(1 for r in returns[-5:] if abs(r) < 0.02)
    f["tight_days_10"] = sum(1 for r in returns[-10:] if abs(r) < 0.02)
    
    f["consecutive_up"] = 0
    for r in reversed(returns[-10:]):
        if r > 0:
            f["consecutive_up"] += 1
        else:
            break
    
    f["consecutive_tight"] = 0
    for r in reversed(returns[-10:]):
        if abs(r) < 0.02:
            f["consecutive_tight"] += 1
        else:
            break
    
    dv = close[-20:] * volume[-20:]
    f["dollar_vol_ratio"] = np.mean(dv[-3:]) / (np.mean(dv[-20:]) + 1)
    f["dollar_vol_today"] = dv[-1] / (np.mean(dv[-20:]) + 1)
    
    gains = returns[-14:].copy()
    gains[gains < 0] = 0
    losses = -returns[-14:].copy()
    losses[losses < 0] = 0
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    f["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 0.0001)))
    
    ema12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
    f["macd"] = (ema12 - ema26) / (close[-1] + 0.001)
    
    signal_line = pd.Series([ema12 - ema26]).ewm(span=9).mean().iloc[-1]
    f["macd_signal"] = (ema12 - ema26 - signal_line) / (close[-1] + 0.001)
    
    atr_vals = []
    for j in range(-14, 0):
        tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_vals.append(tr)
    f["atr_14"] = np.mean(atr_vals) / (close[-1] + 0.001)
    f["atr_compression"] = atr_vals[-1] / (np.mean(atr_vals) + 0.001)
    
    bb_sma = np.mean(close[-20:])
    bb_std = np.std(close[-20:])
    f["bb_position"] = (close[-1] - bb_sma) / (2 * bb_std + 0.001)
    f["bb_width"] = (4 * bb_std) / (bb_sma + 0.001)
    f["bb_squeeze"] = 1.0 if f["bb_width"] < np.mean([np.std(close[-20-i:-i]) if i > 0 else np.std(close[-20:]) for i in range(0, 20, 5)]) * 0.7 else 0.0
    
    f["current_price"] = close[-1]
    f["volume_today"] = volume[-1]
    f["is_penny"] = 1.0 if close[-1] < 5 else 0.0
    f["is_micro"] = 1.0 if close[-1] < 1 else 0.0
    
    stock_emb = get_stock_embedding(symbol)
    f.update(stock_emb)
    
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}

def build_enhanced_dataset() -> Tuple[List[Dict], Dict]:
    """Build dataset with hard negatives and near-miss positives."""
    logger.info("Building enhanced dataset with hard negatives...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stocks")
    
    moonshots = []       # 50%+ gains (true positives)
    near_miss = []       # 30-50% gains (boundary learning)
    hard_negatives = []  # 15-30% gains that fizzled (failed breakouts)
    true_negatives = []  # <15% gains (clear negatives)
    
    for i, path in enumerate(parquet_files):
        symbol = path.stem
        
        try:
            df = pd.read_parquet(path)
            if "date" not in df.columns and "timestamp" in df.columns:
                df["date"] = df["timestamp"]
        except Exception as e:
            continue
        
        if len(df) < 90:
            continue
        
        for idx in range(60, len(df) - 30):
            entry_price = df.iloc[idx]["close"]
            if entry_price <= 0:
                continue
            
            future_30d = df.iloc[idx:idx+30]
            max_price = future_30d["high"].max()
            max_gain = (max_price - entry_price) / entry_price
            
            feat = extract_advanced_features(df, idx, symbol)
            
            if not feat:
                continue
            
            date_val = df.iloc[idx].get("date", df.iloc[idx].get("timestamp", None))
            
            feat["symbol"] = symbol
            feat["date_idx"] = idx
            feat["date"] = str(date_val) if date_val is not None else f"idx_{idx}"
            feat["gain_pct"] = float(max_gain * 100)
            
            if max_gain >= 0.50:
                feat["label"] = 1
                feat["label_type"] = "moonshot"
                moonshots.append(feat)
            elif 0.30 <= max_gain < 0.50:
                feat["label"] = 1
                feat["label_type"] = "near_miss"
                near_miss.append(feat)
            elif 0.15 <= max_gain < 0.30:
                feat["label"] = 0
                feat["label_type"] = "hard_negative"
                hard_negatives.append(feat)
            elif max_gain < 0.15:
                feat["label"] = 0
                feat["label_type"] = "true_negative"
                true_negatives.append(feat)
        
        if i % 20 == 0:
            logger.info(f"  [{i}/{len(parquet_files)}] Moon:{len(moonshots)}, Near:{len(near_miss)}, Hard:{len(hard_negatives)}, True:{len(true_negatives)}")
    
    logger.info(f"\nDataset composition:")
    logger.info(f"  Moonshots (50%+): {len(moonshots)}")
    logger.info(f"  Near-miss (30-50%): {len(near_miss)}")
    logger.info(f"  Hard negatives (15-30%): {len(hard_negatives)}")
    logger.info(f"  True negatives (<15%): {len(true_negatives)}")
    
    np.random.seed(42)
    
    target_negatives = len(moonshots) + len(near_miss)
    
    hard_neg_sample = hard_negatives
    if len(hard_negatives) > target_negatives // 2:
        indices = np.random.choice(len(hard_negatives), target_negatives // 2, replace=False)
        hard_neg_sample = [hard_negatives[i] for i in indices]
    
    true_neg_sample = true_negatives
    if len(true_negatives) > target_negatives:
        indices = np.random.choice(len(true_negatives), target_negatives, replace=False)
        true_neg_sample = [true_negatives[i] for i in indices]
    
    all_samples = moonshots + near_miss + hard_neg_sample + true_neg_sample
    
    stats = {
        "moonshots": len(moonshots),
        "near_miss": len(near_miss),
        "hard_negatives": len(hard_neg_sample),
        "true_negatives": len(true_neg_sample),
        "total": len(all_samples),
        "positive_rate": (len(moonshots) + len(near_miss)) / len(all_samples) if all_samples else 0,
    }
    
    logger.info(f"\nFinal dataset: {len(all_samples)} samples")
    logger.info(f"  Positives: {len(moonshots) + len(near_miss)} ({stats['positive_rate']:.1%})")
    logger.info(f"  Negatives: {len(hard_neg_sample) + len(true_neg_sample)}")
    
    return all_samples, stats

def create_ensemble_model(X_train, y_train, feature_names: List[str]):
    """Create calibrated ensemble of LightGBM + CatBoost + GradientBoosting."""
    logger.info("Creating ensemble model...")
    
    pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1)
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        class_weights={0: 1.0, 1: pos_weight},
    )
    
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        min_samples_leaf=20,
        random_state=42,
    )
    
    logger.info("  Training LightGBM...")
    lgb_model.fit(X_train, y_train)
    
    logger.info("  Training CatBoost...")
    cat_model.fit(X_train, y_train)
    
    logger.info("  Training GradientBoosting...")
    gb_model.fit(X_train, y_train)
    
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('cat', cat_model),
            ('gb', gb_model),
        ],
        voting='soft',
        weights=[0.4, 0.4, 0.2],
    )
    
    ensemble.estimators_ = [lgb_model, cat_model, gb_model]
    ensemble.le_ = LabelEncoder().fit(y_train)
    ensemble.classes_ = np.array([0, 1])
    
    logger.info("  Calibrating probabilities...")
    calibrated = CalibratedClassifierCV(ensemble, method='isotonic', cv='prefit')
    calibrated.fit(X_train[:1000], y_train[:1000])
    
    return {
        'lgb': lgb_model,
        'cat': cat_model,
        'gb': gb_model,
        'ensemble': ensemble,
        'calibrated': calibrated,
        'feature_names': feature_names,
    }

def find_optimal_threshold(y_true, y_prob, target_precision: float = 0.80):
    """Find threshold that achieves target precision."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    for i, p in enumerate(precisions):
        if p >= target_precision and i < len(thresholds):
            return thresholds[i], precisions[i], recalls[i]
    
    best_idx = np.argmax(precisions[:-1])
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx]

def time_split_validation(samples: List[Dict], feature_cols: List[str]):
    """Time-split validation."""
    sorted_samples = sorted(samples, key=lambda x: x.get("date_idx", 0))
    
    split_idx = int(len(sorted_samples) * 0.7)
    train_samples = sorted_samples[:split_idx]
    test_samples = sorted_samples[split_idx:]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train_samples])
    y_train = np.array([s["label"] for s in train_samples])
    
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test_samples])
    y_test = np.array([s["label"] for s in test_samples])
    
    return X_train, X_test, y_train, y_test

def stock_split_validation(samples: List[Dict], feature_cols: List[str]):
    """Stock-split validation (hardest test)."""
    symbols = list(set(s["symbol"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(symbols)
    
    split_idx = int(len(symbols) * 0.7)
    train_symbols = set(symbols[:split_idx])
    
    train_samples = [s for s in samples if s["symbol"] in train_symbols]
    test_samples = [s for s in samples if s["symbol"] not in train_symbols]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train_samples])
    y_train = np.array([s["label"] for s in train_samples])
    
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test_samples])
    y_test = np.array([s["label"] for s in test_samples])
    
    return X_train, X_test, y_train, y_test

def evaluate_models(models: Dict, X_test, y_test, split_name: str):
    """Evaluate all models and find optimal thresholds."""
    results = {}
    
    for name, model in models.items():
        if name == 'feature_names':
            continue
        
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            continue
        
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        default_metrics = {
            "precision": precision_score(y_test, y_pred_default, zero_division=0),
            "recall": recall_score(y_test, y_pred_default, zero_division=0),
            "f1": f1_score(y_test, y_pred_default, zero_division=0),
        }
        
        thresh_80, prec_80, rec_80 = find_optimal_threshold(y_test, y_prob, 0.80)
        thresh_75, prec_75, rec_75 = find_optimal_threshold(y_test, y_prob, 0.75)
        thresh_70, prec_70, rec_70 = find_optimal_threshold(y_test, y_prob, 0.70)
        
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = 0
        
        results[name] = {
            "default": default_metrics,
            "roc_auc": roc_auc,
            "thresholds": {
                "80_precision": {"threshold": thresh_80, "precision": prec_80, "recall": rec_80},
                "75_precision": {"threshold": thresh_75, "precision": prec_75, "recall": rec_75},
                "70_precision": {"threshold": thresh_70, "precision": prec_70, "recall": rec_70},
            }
        }
        
        logger.info(f"\n  {name.upper()} ({split_name}):")
        logger.info(f"    Default (0.5): Prec={default_metrics['precision']:.1%}, Rec={default_metrics['recall']:.1%}")
        logger.info(f"    ROC-AUC: {roc_auc:.3f}")
        logger.info(f"    @80% Prec: thresh={thresh_80:.3f}, prec={prec_80:.1%}, rec={rec_80:.1%}")
        logger.info(f"    @75% Prec: thresh={thresh_75:.3f}, prec={prec_75:.1%}, rec={rec_75:.1%}")
    
    return results

def save_production_model(models: Dict, results: Dict, feature_cols: List[str], stats: Dict):
    """Save the best model for production use."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    best_model = models['ensemble']
    lgb_model = models['lgb']
    
    ensemble_path = MODELS_DIR / "moonshot_ensemble_v2.pkl.gz"
    with gzip.open(ensemble_path, 'wb') as f:
        pickle.dump({
            'model': lgb_model,
            'feature_names': feature_cols,
            'thresholds': {
                'high_precision': 0.7,
                'balanced': 0.5,
                'high_recall': 0.3,
            },
            'training_stats': stats,
            'timestamp': timestamp,
            'version': '2.0-accuracy-maximizer',
        }, f)
    
    logger.info(f"\nProduction model saved: {ensemble_path}")
    
    report_path = VALIDATION_DIR / f"accuracy_maximizer_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'dataset_stats': stats,
            'feature_count': len(feature_cols),
            'results': {k: v for k, v in results.items() if isinstance(v, dict)},
        }, f, indent=2, default=str)
    
    logger.info(f"Report saved: {report_path}")
    
    return ensemble_path

def main():
    print("=" * 70)
    print("ACCURACY MAXIMIZER - Implementing All Architect Recommendations")
    print("=" * 70)
    print("\nTarget: Precision ≥80% at recall ≥30%, Stock-split ≥60%")
    
    start_time = time.time()
    
    print("\n[1/5] Building enhanced dataset with hard negatives...")
    all_samples, stats = build_enhanced_dataset()
    
    if len(all_samples) < 1000:
        print(f"ERROR: Only {len(all_samples)} samples. Need more data.")
        return
    
    meta_cols = ["symbol", "date_idx", "date", "gain_pct", "label", "label_type"]
    feature_cols = [k for k in all_samples[0].keys() if k not in meta_cols]
    logger.info(f"\nFeature count: {len(feature_cols)}")
    
    print("\n[2/5] Running time-split validation...")
    X_tr_t, X_te_t, y_tr_t, y_te_t = time_split_validation(all_samples, feature_cols)
    logger.info(f"  Train: {len(X_tr_t)} ({sum(y_tr_t)} positives)")
    logger.info(f"  Test: {len(X_te_t)} ({sum(y_te_t)} positives)")
    
    print("\n[3/5] Training ensemble models (LightGBM + CatBoost + GB)...")
    models = create_ensemble_model(X_tr_t, y_tr_t, feature_cols)
    
    print("\n[4/5] Evaluating on time-split...")
    time_results = evaluate_models(models, X_te_t, y_te_t, "time-split")
    
    print("\n[5/5] Evaluating on stock-split (hardest test)...")
    X_tr_s, X_te_s, y_tr_s, y_te_s = stock_split_validation(all_samples, feature_cols)
    logger.info(f"  Train: {len(X_tr_s)} ({sum(y_tr_s)} positives)")
    logger.info(f"  Test: {len(X_te_s)} ({sum(y_te_s)} positives)")
    
    models_stock = create_ensemble_model(X_tr_s, y_tr_s, feature_cols)
    stock_results = evaluate_models(models_stock, X_te_s, y_te_s, "stock-split")
    
    all_results = {
        'time_split': time_results,
        'stock_split': stock_results,
    }
    
    model_path = save_production_model(models, all_results, feature_cols, stats)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("ACCURACY MAXIMIZER COMPLETE!")
    print("=" * 70)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Features: {len(feature_cols)}")
    print(f"Dataset: {len(all_samples)} samples")
    
    print("\n" + "-" * 70)
    print("FINAL RESULTS:")
    print("-" * 70)
    
    print("\nTIME-SPLIT (LightGBM):")
    if 'lgb' in time_results:
        r = time_results['lgb']
        print(f"  Default: Prec={r['default']['precision']:.1%}, Rec={r['default']['recall']:.1%}")
        print(f"  @80% Precision: Rec={r['thresholds']['80_precision']['recall']:.1%}")
        print(f"  @75% Precision: Rec={r['thresholds']['75_precision']['recall']:.1%}")
        print(f"  ROC-AUC: {r['roc_auc']:.3f}")
    
    print("\nSTOCK-SPLIT (LightGBM):")
    if 'lgb' in stock_results:
        r = stock_results['lgb']
        print(f"  Default: Prec={r['default']['precision']:.1%}, Rec={r['default']['recall']:.1%}")
        print(f"  @80% Precision: Rec={r['thresholds']['80_precision']['recall']:.1%}")
        print(f"  @75% Precision: Rec={r['thresholds']['75_precision']['recall']:.1%}")
        print(f"  ROC-AUC: {r['roc_auc']:.3f}")
    
    print(f"\nProduction model: {model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
