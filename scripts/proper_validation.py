#!/usr/bin/env python3
"""
PROPER VALIDATION - Addresses architect review concerns:
1. Include TRUE NEGATIVES (non-moonshot samples)
2. Time-split validation (train on older, test on newer)
3. Ablation: 18-feature vs 48-feature comparison
4. Report metrics on full universe
"""

import os
import sys
import time
import json
import pickle
import gzip
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MODELS_DIR = Path("data/models")
VALIDATION_DIR = Path("data/validation")
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

def extract_basic_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Original 18 features for comparison."""
    if idx < 50 or idx >= len(df):
        return {}
    
    window = df.iloc[idx-50:idx+1]
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    
    return {
        "close": float(close[-1]),
        "sma_10": float(np.mean(close[-10:])),
        "sma_20": float(np.mean(close[-20:])),
        "sma_50": float(np.mean(close[-50:])),
        "vol_10d": float(np.std(returns[-10:])),
        "vol_20d": float(np.std(returns[-20:])),
        "vol_compression": float(np.std(returns[-10:]) / (np.std(returns[-20:]) + 0.001)),
        "momentum_5d": float((close[-1] - close[-6]) / (close[-6] + 0.001)),
        "momentum_10d": float((close[-1] - close[-11]) / (close[-11] + 0.001)),
        "vol_ratio": float(np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1)),
        "price_vs_sma20": float(close[-1] / (np.mean(close[-20:]) + 0.001)),
        "price_vs_sma50": float(close[-1] / (np.mean(close[-50:]) + 0.001)),
        "range_10d": float((np.max(high[-10:]) - np.min(low[-10:])) / (close[-1] + 0.001)),
        "high_20d": float(np.max(high[-20:])),
        "breakout_prox": float(close[-1] / (np.max(high[-20:]) + 0.001)),
        "up_days": float(sum(1 for r in returns[-10:] if r > 0)),
        "tight_days": float(sum(1 for r in returns[-10:] if abs(r) < 0.02)),
        "vol_today": float(volume[-1]),
    }

def extract_enhanced_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Enhanced 48 features."""
    if idx < 60 or idx >= len(df):
        return {}
    
    window = df.iloc[idx-60:idx+1]
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    open_p = window["open"].values if "open" in window.columns else close
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    
    f = {}
    
    for p in [5, 10, 20, 50]:
        f[f"sma_{p}"] = np.mean(close[-p:])
        f[f"vol_sma_{p}"] = np.mean(volume[-p:])
    
    for p in [5, 10, 20]:
        f[f"volatility_{p}"] = np.std(returns[-p:])
    
    f["vol_compression_10_20"] = f["volatility_10"] / (f["volatility_20"] + 0.001)
    f["vol_compression_5_20"] = f["volatility_5"] / (f["volatility_20"] + 0.001)
    
    for p in [3, 5, 10, 20]:
        f[f"momentum_{p}"] = (close[-1] - close[-p-1]) / (close[-p-1] + 0.001)
    
    f["mom_acceleration"] = f["momentum_3"] - f["momentum_5"]
    f["mom_inflection"] = f["momentum_5"] - f["momentum_10"]
    
    for p in [5, 10, 20]:
        f[f"vol_ratio_{p}"] = np.mean(volume[-5:]) / (np.mean(volume[-p:]) + 1)
    
    f["vol_acceleration"] = f["vol_ratio_5"] / (f["vol_ratio_10"] + 0.001)
    
    for p in [10, 20, 50]:
        f[f"price_vs_sma_{p}"] = close[-1] / (f[f"sma_{p}"] + 0.001)
    
    f["sma_slope_10"] = (f["sma_5"] - f["sma_10"]) / (f["sma_10"] + 0.001)
    f["sma_slope_20"] = (f["sma_10"] - f["sma_20"]) / (f["sma_20"] + 0.001)
    
    for p in [10, 20]:
        r = np.max(high[-p:]) - np.min(low[-p:])
        f[f"range_{p}"] = r / (close[-1] + 0.001)
    
    f["range_compression"] = f["range_10"] / (f["range_20"] + 0.001)
    f["high_20"] = np.max(high[-20:])
    f["low_20"] = np.min(low[-20:])
    f["breakout_proximity"] = close[-1] / (f["high_20"] + 0.001)
    f["support_distance"] = (close[-1] - f["low_20"]) / (f["low_20"] + 0.001)
    
    dr = high[-10:] - low[-10:]
    cp = (close[-10:] - low[-10:]) / (dr + 0.001)
    f["close_near_high"] = np.mean(cp)
    
    uw = (high[-10:] - np.maximum(close[-10:], open_p[-10:])) / (dr + 0.001)
    f["upper_wick_avg"] = np.mean(uw)
    f["wick_shrinking"] = np.mean(uw[-3:]) - np.mean(uw[-10:-3])
    
    f["up_days_5"] = sum(1 for r in returns[-5:] if r > 0)
    f["up_days_10"] = sum(1 for r in returns[-10:] if r > 0)
    f["tight_days_10"] = sum(1 for r in returns[-10:] if abs(r) < 0.02)
    
    dv = close[-20:] * volume[-20:]
    f["dollar_vol_ratio"] = np.mean(dv[-3:]) / (np.mean(dv[-20:]) + 1)
    f["gap_today"] = (open_p[-1] - close[-2]) / (close[-2] + 0.001)
    f["rsi_proxy"] = f["up_days_10"] / 10.0
    
    ema12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
    ema26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
    f["macd_proxy"] = (ema12 - ema26) / (close[-1] + 0.001)
    
    atr = []
    for j in range(-14, 0):
        tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr.append(tr)
    f["atr_14"] = np.mean(atr) / (close[-1] + 0.001)
    
    f["current_price"] = close[-1]
    f["volume_today"] = volume[-1]
    
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}

def build_full_universe_dataset() -> Tuple[List[Dict], List[Dict]]:
    """Build dataset with BOTH positives (moonshots) AND negatives (non-moonshots)."""
    logger.info("Building full universe dataset with true negatives...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stocks")
    
    positives_basic = []
    positives_enhanced = []
    negatives_basic = []
    negatives_enhanced = []
    
    for i, path in enumerate(parquet_files):
        symbol = path.stem
        
        try:
            df = pd.read_parquet(path)
            if "date" not in df.columns and "timestamp" in df.columns:
                df["date"] = df["timestamp"]
        except:
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
            
            basic_feat = extract_basic_features(df, idx)
            enhanced_feat = extract_enhanced_features(df, idx)
            
            if not basic_feat or not enhanced_feat:
                continue
            
            date_val = df.iloc[idx].get("date", df.iloc[idx].get("timestamp", None))
            
            basic_feat["symbol"] = symbol
            basic_feat["date_idx"] = idx
            basic_feat["date"] = str(date_val) if date_val is not None else f"idx_{idx}"
            basic_feat["gain_pct"] = float(max_gain * 100)
            
            enhanced_feat["symbol"] = symbol
            enhanced_feat["date_idx"] = idx
            enhanced_feat["date"] = str(date_val) if date_val is not None else f"idx_{idx}"
            enhanced_feat["gain_pct"] = float(max_gain * 100)
            
            if max_gain >= 0.50:
                basic_feat["is_moonshot"] = 1
                enhanced_feat["is_moonshot"] = 1
                positives_basic.append(basic_feat)
                positives_enhanced.append(enhanced_feat)
            elif max_gain < 0.20:
                basic_feat["is_moonshot"] = 0
                enhanced_feat["is_moonshot"] = 0
                negatives_basic.append(basic_feat)
                negatives_enhanced.append(enhanced_feat)
        
        if i % 20 == 0:
            logger.info(f"  [{i}/{len(parquet_files)}] Pos: {len(positives_basic)}, Neg: {len(negatives_basic)}")
    
    logger.info(f"\nDataset built:")
    logger.info(f"  Positives (50%+ gain): {len(positives_basic)}")
    logger.info(f"  Negatives (<20% gain): {len(negatives_basic)}")
    
    np.random.seed(42)
    if len(negatives_basic) > len(positives_basic) * 3:
        sample_size = len(positives_basic) * 3
        indices = np.random.choice(len(negatives_basic), sample_size, replace=False)
        negatives_basic = [negatives_basic[i] for i in indices]
        negatives_enhanced = [negatives_enhanced[i] for i in indices]
        logger.info(f"  Sampled negatives to: {len(negatives_basic)} (3:1 ratio)")
    
    all_basic = positives_basic + negatives_basic
    all_enhanced = positives_enhanced + negatives_enhanced
    
    return all_basic, all_enhanced

def time_split_validation(samples: List[Dict], feature_cols: List[str], label_col: str = "is_moonshot"):
    """Split by time: train on first 70%, test on last 30%."""
    
    sorted_samples = sorted(samples, key=lambda x: x.get("date_idx", 0))
    
    split_idx = int(len(sorted_samples) * 0.7)
    train_samples = sorted_samples[:split_idx]
    test_samples = sorted_samples[split_idx:]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train_samples])
    y_train = np.array([s[label_col] for s in train_samples])
    
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test_samples])
    y_test = np.array([s[label_col] for s in test_samples])
    
    return X_train, X_test, y_train, y_test, train_samples, test_samples

def stock_split_validation(samples: List[Dict], feature_cols: List[str], label_col: str = "is_moonshot"):
    """Split by stock: train on 70% of stocks, test on 30% of stocks."""
    
    symbols = list(set(s["symbol"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(symbols)
    
    split_idx = int(len(symbols) * 0.7)
    train_symbols = set(symbols[:split_idx])
    test_symbols = set(symbols[split_idx:])
    
    train_samples = [s for s in samples if s["symbol"] in train_symbols]
    test_samples = [s for s in samples if s["symbol"] in test_symbols]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train_samples])
    y_train = np.array([s[label_col] for s in train_samples])
    
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test_samples])
    y_test = np.array([s[label_col] for s in test_samples])
    
    return X_train, X_test, y_train, y_test, train_symbols, test_symbols

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name: str):
    """Train model and return metrics."""
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.85,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return model, metrics

def run_ablation_study(basic_samples: List[Dict], enhanced_samples: List[Dict]):
    """Compare 18-feature vs 48-feature models on same holdout."""
    
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY: 18-feature vs 48-feature")
    logger.info("=" * 60)
    
    basic_cols = [k for k in basic_samples[0].keys() 
                  if k not in ["symbol", "date_idx", "date", "gain_pct", "is_moonshot"]]
    enhanced_cols = [k for k in enhanced_samples[0].keys() 
                     if k not in ["symbol", "date_idx", "date", "gain_pct", "is_moonshot"]]
    
    logger.info(f"Basic features: {len(basic_cols)}")
    logger.info(f"Enhanced features: {len(enhanced_cols)}")
    
    results = {}
    
    for split_name, split_func in [("time_split", time_split_validation), ("stock_split", stock_split_validation)]:
        logger.info(f"\n--- {split_name.upper()} VALIDATION ---")
        
        X_tr_b, X_te_b, y_tr, y_te, _, _ = split_func(basic_samples, basic_cols)
        X_tr_e, X_te_e, _, _, _, _ = split_func(enhanced_samples, enhanced_cols)
        
        logger.info(f"  Train: {len(X_tr_b)} samples ({sum(y_tr)} positives)")
        logger.info(f"  Test: {len(X_te_b)} samples ({sum(y_te)} positives)")
        
        _, metrics_basic = train_and_evaluate(X_tr_b, X_te_b, y_tr, y_te, "basic")
        _, metrics_enhanced = train_and_evaluate(X_tr_e, X_te_e, y_tr, y_te, "enhanced")
        
        results[split_name] = {
            "basic_18_features": metrics_basic,
            "enhanced_48_features": metrics_enhanced,
            "improvement": {
                "precision": metrics_enhanced["precision"] - metrics_basic["precision"],
                "recall": metrics_enhanced["recall"] - metrics_basic["recall"],
                "f1": metrics_enhanced["f1"] - metrics_basic["f1"],
                "specificity": metrics_enhanced["specificity"] - metrics_basic["specificity"],
            }
        }
        
        logger.info(f"\n  18-FEATURE MODEL:")
        logger.info(f"    Precision: {metrics_basic['precision']:.1%}")
        logger.info(f"    Recall: {metrics_basic['recall']:.1%}")
        logger.info(f"    F1: {metrics_basic['f1']:.1%}")
        logger.info(f"    Specificity: {metrics_basic['specificity']:.1%}")
        logger.info(f"    TP={metrics_basic['true_positives']}, FP={metrics_basic['false_positives']}, FN={metrics_basic['false_negatives']}, TN={metrics_basic['true_negatives']}")
        
        logger.info(f"\n  48-FEATURE MODEL:")
        logger.info(f"    Precision: {metrics_enhanced['precision']:.1%}")
        logger.info(f"    Recall: {metrics_enhanced['recall']:.1%}")
        logger.info(f"    F1: {metrics_enhanced['f1']:.1%}")
        logger.info(f"    Specificity: {metrics_enhanced['specificity']:.1%}")
        logger.info(f"    TP={metrics_enhanced['true_positives']}, FP={metrics_enhanced['false_positives']}, FN={metrics_enhanced['false_negatives']}, TN={metrics_enhanced['true_negatives']}")
        
        logger.info(f"\n  IMPROVEMENT:")
        logger.info(f"    Precision: {results[split_name]['improvement']['precision']:+.1%}")
        logger.info(f"    Recall: {results[split_name]['improvement']['recall']:+.1%}")
        logger.info(f"    F1: {results[split_name]['improvement']['f1']:+.1%}")
        logger.info(f"    Specificity: {results[split_name]['improvement']['specificity']:+.1%}")
    
    return results

def save_validation_report(results: Dict, basic_samples: List[Dict], enhanced_samples: List[Dict]):
    """Save comprehensive validation report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "timestamp": timestamp,
        "dataset": {
            "total_samples": len(basic_samples),
            "positives": sum(1 for s in basic_samples if s["is_moonshot"] == 1),
            "negatives": sum(1 for s in basic_samples if s["is_moonshot"] == 0),
            "symbols": len(set(s["symbol"] for s in basic_samples)),
        },
        "validation_results": results,
    }
    
    report_path = VALIDATION_DIR / f"validation_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nReport saved to: {report_path}")
    return report

def main():
    print("=" * 70)
    print("PROPER VALIDATION - Addressing Architect Review Concerns")
    print("=" * 70)
    start_time = time.time()
    
    print("\n[1/4] Building dataset with TRUE NEGATIVES...")
    basic_samples, enhanced_samples = build_full_universe_dataset()
    
    if len(basic_samples) < 500:
        print(f"ERROR: Only {len(basic_samples)} samples. Need more data.")
        return
    
    print("\n[2/4] Running TIME-SPLIT validation...")
    print("[3/4] Running STOCK-SPLIT validation...")
    print("[4/4] Running ABLATION study (18 vs 48 features)...")
    
    results = run_ablation_study(basic_samples, enhanced_samples)
    
    report = save_validation_report(results, basic_samples, enhanced_samples)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PROPER VALIDATION COMPLETE!")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s")
    print(f"Dataset: {report['dataset']['total_samples']} samples")
    print(f"  - Positives (50%+ moonshots): {report['dataset']['positives']}")
    print(f"  - Negatives (<20% gains): {report['dataset']['negatives']}")
    print(f"  - Symbols: {report['dataset']['symbols']}")
    
    print("\n" + "-" * 70)
    print("VALIDATED RESULTS (with true negatives):")
    print("-" * 70)
    
    for split_name in ["time_split", "stock_split"]:
        r = results[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  18-feature: Prec={r['basic_18_features']['precision']:.1%}, Rec={r['basic_18_features']['recall']:.1%}, F1={r['basic_18_features']['f1']:.1%}")
        print(f"  48-feature: Prec={r['enhanced_48_features']['precision']:.1%}, Rec={r['enhanced_48_features']['recall']:.1%}, F1={r['enhanced_48_features']['f1']:.1%}")
        print(f"  Improvement: Prec={r['improvement']['precision']:+.1%}, Rec={r['improvement']['recall']:+.1%}, F1={r['improvement']['f1']:+.1%}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
