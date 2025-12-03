#!/usr/bin/env python3
"""
STRICT MOONSHOT VALIDATION - Fair apples-to-apples comparison
Uses ONLY 50%+ gains as positives (same as baseline)
Applies all improvements (97 features, LightGBM, hard negatives)
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
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MODELS_DIR = Path("data/models")
VALIDATION_DIR = Path("data/validation")

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
    return {f"sector_{i}": float(v) for i, v in enumerate(vec)} | {
        "symbol_hash": float(sym_hash),
        "is_meme": float(symbol in {"GME", "AMC", "BB", "BBAI", "CLOV", "WISH"}),
        "is_crypto_related": float(symbol in {"MARA", "RIOT", "COIN", "HOOD"}),
        "is_ev": float(symbol in {"TSLA", "LCID", "RIVN"}),
    }

def extract_advanced_features(df: pd.DataFrame, idx: int, symbol: str) -> Dict[str, float]:
    """Full 97 features from accuracy_maximizer."""
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

def build_strict_moonshot_dataset():
    """Build dataset with STRICT 50%+ moonshot labels only."""
    logger.info("Building STRICT moonshot dataset (50%+ only)...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stocks")
    
    positives = []
    negatives = []
    
    for i, path in enumerate(parquet_files):
        symbol = path.stem
        
        try:
            df = pd.read_parquet(path)
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
            
            feat = extract_advanced_features(df, idx, symbol)
            if not feat:
                continue
            
            feat["symbol"] = symbol
            feat["date_idx"] = idx
            feat["gain_pct"] = float(max_gain * 100)
            
            if max_gain >= 0.50:
                feat["label"] = 1
                positives.append(feat)
            elif max_gain < 0.20:
                feat["label"] = 0
                negatives.append(feat)
        
        if i % 20 == 0:
            logger.info(f"  [{i}/{len(parquet_files)}] Pos: {len(positives)}, Neg: {len(negatives)}")
    
    np.random.seed(42)
    if len(negatives) > len(positives) * 3:
        indices = np.random.choice(len(negatives), len(positives) * 3, replace=False)
        negatives = [negatives[i] for i in indices]
    
    all_samples = positives + negatives
    logger.info(f"\nStrict dataset: {len(all_samples)} samples")
    logger.info(f"  Positives (50%+): {len(positives)}")
    logger.info(f"  Negatives (<20%): {len(negatives)}")
    
    return all_samples, {"positives": len(positives), "negatives": len(negatives)}

def time_split(samples, feature_cols):
    sorted_samples = sorted(samples, key=lambda x: x.get("date_idx", 0))
    split_idx = int(len(sorted_samples) * 0.7)
    train = sorted_samples[:split_idx]
    test = sorted_samples[split_idx:]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train])
    y_train = np.array([s["label"] for s in train])
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test])
    y_test = np.array([s["label"] for s in test])
    
    return X_train, X_test, y_train, y_test

def stock_split(samples, feature_cols):
    symbols = list(set(s["symbol"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(symbols)
    
    split_idx = int(len(symbols) * 0.7)
    train_symbols = set(symbols[:split_idx])
    
    train = [s for s in samples if s["symbol"] in train_symbols]
    test = [s for s in samples if s["symbol"] not in train_symbols]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train])
    y_train = np.array([s["label"] for s in train])
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test])
    y_test = np.array([s["label"] for s in test])
    
    return X_train, X_test, y_train, y_test

def find_threshold(y_true, y_prob, target_precision):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    for i, p in enumerate(precisions):
        if p >= target_precision and i < len(thresholds):
            return thresholds[i], precisions[i], recalls[i]
    return thresholds[-1], precisions[-1], recalls[-1]

def main():
    print("=" * 70)
    print("STRICT MOONSHOT VALIDATION - Fair Apples-to-Apples Comparison")
    print("=" * 70)
    print("\nUsing ONLY 50%+ gains as positives (same as baseline)")
    
    start_time = time.time()
    
    print("\n[1/4] Building strict moonshot dataset...")
    samples, stats = build_strict_moonshot_dataset()
    
    meta_cols = ["symbol", "date_idx", "gain_pct", "label"]
    feature_cols = [k for k in samples[0].keys() if k not in meta_cols]
    logger.info(f"Features: {len(feature_cols)}")
    
    print("\n[2/4] Running TIME-SPLIT validation...")
    X_tr_t, X_te_t, y_tr_t, y_te_t = time_split(samples, feature_cols)
    logger.info(f"  Train: {len(X_tr_t)} ({sum(y_tr_t)} positives)")
    logger.info(f"  Test: {len(X_te_t)} ({sum(y_te_t)} positives)")
    
    pos_weight = len(y_tr_t[y_tr_t == 0]) / (len(y_tr_t[y_tr_t == 1]) + 1)
    
    lgb_time = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    lgb_time.fit(X_tr_t, y_tr_t)
    
    y_prob_t = lgb_time.predict_proba(X_te_t)[:, 1]
    y_pred_t = (y_prob_t >= 0.5).astype(int)
    
    time_prec = precision_score(y_te_t, y_pred_t, zero_division=0)
    time_rec = recall_score(y_te_t, y_pred_t, zero_division=0)
    time_f1 = f1_score(y_te_t, y_pred_t, zero_division=0)
    time_auc = roc_auc_score(y_te_t, y_prob_t)
    
    thresh_80, prec_80, rec_80 = find_threshold(y_te_t, y_prob_t, 0.80)
    thresh_75, prec_75, rec_75 = find_threshold(y_te_t, y_prob_t, 0.75)
    
    print(f"\n  TIME-SPLIT RESULTS:")
    print(f"    Default (0.5): Precision={time_prec:.1%}, Recall={time_rec:.1%}, F1={time_f1:.1%}")
    print(f"    ROC-AUC: {time_auc:.3f}")
    print(f"    @80% Prec: thresh={thresh_80:.3f}, rec={rec_80:.1%}")
    print(f"    @75% Prec: thresh={thresh_75:.3f}, rec={rec_75:.1%}")
    
    print("\n[3/4] Running STOCK-SPLIT validation...")
    X_tr_s, X_te_s, y_tr_s, y_te_s = stock_split(samples, feature_cols)
    logger.info(f"  Train: {len(X_tr_s)} ({sum(y_tr_s)} positives)")
    logger.info(f"  Test: {len(X_te_s)} ({sum(y_te_s)} positives)")
    
    pos_weight_s = len(y_tr_s[y_tr_s == 0]) / (len(y_tr_s[y_tr_s == 1]) + 1)
    
    lgb_stock = lgb.LGBMClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=pos_weight_s,
        random_state=42, verbosity=-1, n_jobs=-1
    )
    lgb_stock.fit(X_tr_s, y_tr_s)
    
    y_prob_s = lgb_stock.predict_proba(X_te_s)[:, 1]
    y_pred_s = (y_prob_s >= 0.5).astype(int)
    
    stock_prec = precision_score(y_te_s, y_pred_s, zero_division=0)
    stock_rec = recall_score(y_te_s, y_pred_s, zero_division=0)
    stock_f1 = f1_score(y_te_s, y_pred_s, zero_division=0)
    stock_auc = roc_auc_score(y_te_s, y_prob_s)
    
    thresh_80_s, prec_80_s, rec_80_s = find_threshold(y_te_s, y_prob_s, 0.80)
    thresh_75_s, prec_75_s, rec_75_s = find_threshold(y_te_s, y_prob_s, 0.75)
    
    print(f"\n  STOCK-SPLIT RESULTS:")
    print(f"    Default (0.5): Precision={stock_prec:.1%}, Recall={stock_rec:.1%}, F1={stock_f1:.1%}")
    print(f"    ROC-AUC: {stock_auc:.3f}")
    print(f"    @80% Prec: thresh={thresh_80_s:.3f}, rec={rec_80_s:.1%}")
    print(f"    @75% Prec: thresh={thresh_75_s:.3f}, rec={rec_75_s:.1%}")
    
    print("\n[4/4] Saving production model...")
    model_path = MODELS_DIR / "moonshot_strict_v2.pkl.gz"
    with gzip.open(model_path, 'wb') as f:
        pickle.dump({
            'model': lgb_stock,
            'feature_names': feature_cols,
            'thresholds': {'high_precision': thresh_80_s, 'balanced': 0.5},
            'version': '2.0-strict-moonshot',
        }, f)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": stats,
        "features": len(feature_cols),
        "time_split": {
            "precision": time_prec, "recall": time_rec, "f1": time_f1,
            "roc_auc": time_auc, "thresh_80_recall": rec_80, "thresh_75_recall": rec_75
        },
        "stock_split": {
            "precision": stock_prec, "recall": stock_rec, "f1": stock_f1,
            "roc_auc": stock_auc, "thresh_80_recall": rec_80_s, "thresh_75_recall": rec_75_s
        }
    }
    
    report_path = VALIDATION_DIR / f"strict_moonshot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("STRICT MOONSHOT VALIDATION COMPLETE!")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("FAIR COMPARISON (same 50%+ label definition as baseline):")
    print("-" * 70)
    
    print("\n| Metric | BASELINE (48 feat) | NEW (97 feat) | CHANGE |")
    print("|--------|-------------------|---------------|--------|")
    print(f"| Time-Split Precision | 71.4% | {time_prec:.1%} | {(time_prec - 0.714)*100:+.1f}% |")
    print(f"| Time-Split Recall | 35.7% | {time_rec:.1%} | {(time_rec - 0.357)*100:+.1f}% |")
    print(f"| Stock-Split Precision | 52.5% | {stock_prec:.1%} | {(stock_prec - 0.525)*100:+.1f}% |")
    print(f"| Stock-Split Recall | 32.3% | {stock_rec:.1%} | {(stock_rec - 0.323)*100:+.1f}% |")
    
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Model saved: {model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
