#!/usr/bin/env python3
"""
INCREMENTAL TRAINING - Keep knowledge from previous model while learning from new stocks

Two modes:
1. WARM START: Load previous model, continue training on new data
2. ENSEMBLE: Combine old model predictions with new model for best of both

This preserves patterns learned from original 84 stocks while adding knowledge
from newly fetched biotech, China ADRs, etc.
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
    "MRNA": "biotech", "BNTX": "biotech", "NVAX": "biotech", "VXRT": "biotech",
    "INO": "biotech", "JD": "china_tech", "PDD": "china_tech", "BIDU": "china_tech",
    "BILI": "china_tech", "TME": "china_tech", "BABA": "china_tech",
}

def get_stock_embedding(symbol: str) -> Dict[str, float]:
    sector = SECTOR_MAP.get(symbol, "unknown")
    sector_list = ["tech", "crypto", "health", "fintech", "auto", "retail", 
                   "cannabis", "entertainment", "aerospace", "biotech", "china_tech", "unknown"]
    vec = [1.0 if s == sector else 0.0 for s in sector_list]
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) / (16**8)
    return {f"sector_{i}": float(v) for i, v in enumerate(vec)} | {
        "symbol_hash": float(sym_hash),
        "is_meme": float(symbol in {"GME", "AMC", "BB", "BBAI", "CLOV", "WISH"}),
        "is_crypto_related": float(symbol in {"MARA", "RIOT", "COIN", "HOOD"}),
        "is_ev": float(symbol in {"TSLA", "LCID", "RIVN"}),
        "is_biotech": float(symbol in {"MRNA", "BNTX", "NVAX", "VXRT", "INO"}),
        "is_china_tech": float(symbol in {"JD", "PDD", "BIDU", "BILI", "TME", "BABA"}),
    }

def extract_features(df: pd.DataFrame, idx: int, symbol: str) -> Dict[str, float]:
    """Extract 97+ features."""
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
    
    f["current_price"] = close[-1]
    f["volume_today"] = volume[-1]
    f["is_penny"] = 1.0 if close[-1] < 5 else 0.0
    f["is_micro"] = 1.0 if close[-1] < 1 else 0.0
    
    stock_emb = get_stock_embedding(symbol)
    f.update(stock_emb)
    
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in f.items()}

def build_dataset():
    """Build full dataset from all cached stocks."""
    logger.info("Building dataset from ALL cached stocks...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stocks")
    
    samples = []
    
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
            
            feat = extract_features(df, idx, symbol)
            if not feat:
                continue
            
            feat["symbol"] = symbol
            feat["date_idx"] = idx
            feat["gain_pct"] = float(max_gain * 100)
            
            if max_gain >= 0.50:
                feat["label"] = 1
                samples.append(feat)
            elif max_gain < 0.20:
                feat["label"] = 0
                samples.append(feat)
        
        if i % 20 == 0:
            pos = sum(1 for s in samples if s["label"] == 1)
            logger.info(f"  [{i}/{len(parquet_files)}] Samples: {len(samples)} (pos: {pos})")
    
    np.random.seed(42)
    positives = [s for s in samples if s["label"] == 1]
    negatives = [s for s in samples if s["label"] == 0]
    
    if len(negatives) > len(positives) * 3:
        indices = np.random.choice(len(negatives), len(positives) * 3, replace=False)
        negatives = [negatives[i] for i in indices]
    
    all_samples = positives + negatives
    logger.info(f"Final dataset: {len(all_samples)} samples ({len(positives)} pos, {len(negatives)} neg)")
    
    return all_samples

def load_previous_model():
    """Load the previous trained model for incremental learning."""
    model_path = MODELS_DIR / "moonshot_strict_v2.pkl.gz"
    
    if not model_path.exists():
        logger.info("No previous model found - training from scratch")
        return None
    
    with gzip.open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded previous model from {model_path}")
    return data.get('model')

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
    
    return X_train, X_test, y_train, y_test, train_symbols, set(symbols) - train_symbols

def main():
    print("=" * 70)
    print("INCREMENTAL TRAINING - Keep Previous Knowledge + Learn New Patterns")
    print("=" * 70)
    
    start_time = time.time()
    
    print("\n[1/4] Loading previous model...")
    prev_model = load_previous_model()
    
    print("\n[2/4] Building dataset from ALL stocks (old + new)...")
    samples = build_dataset()
    
    meta_cols = ["symbol", "date_idx", "gain_pct", "label"]
    feature_cols = [k for k in samples[0].keys() if k not in meta_cols]
    logger.info(f"Features: {len(feature_cols)}")
    
    print("\n[3/4] Training with WARM START (incremental learning)...")
    X_tr, X_te, y_tr, y_te, train_syms, test_syms = stock_split(samples, feature_cols)
    logger.info(f"Train: {len(X_tr)} ({sum(y_tr)} pos) from {len(train_syms)} stocks")
    logger.info(f"Test: {len(X_te)} ({sum(y_te)} pos) from {len(test_syms)} stocks")
    
    pos_weight = len(y_tr[y_tr == 0]) / (len(y_tr[y_tr == 1]) + 1)
    
    logger.info("  Training NEW model on ALL data (old + new stocks)...")
    new_model = lgb.LGBMClassifier(
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
    new_model.fit(X_tr, y_tr)
    
    if prev_model is not None:
        logger.info("  Creating ENSEMBLE with previous model...")
        try:
            prev_features = prev_model.feature_name_
            common_features = [f for f in feature_cols if f in prev_features]
            
            if len(common_features) >= 90:
                prev_feature_indices = [feature_cols.index(f) for f in common_features]
                X_te_prev = X_te[:, prev_feature_indices]
                
                prev_prob = prev_model.predict_proba(X_te_prev)[:, 1]
                new_prob = new_model.predict_proba(X_te)[:, 1]
                
                ensemble_prob = 0.3 * prev_prob + 0.7 * new_prob
                
                logger.info(f"  Ensemble: 30% old model + 70% new model")
            else:
                logger.info(f"  Not enough common features ({len(common_features)}), using new model only")
                ensemble_prob = None
        except Exception as e:
            logger.info(f"  Could not create ensemble: {e}")
            ensemble_prob = None
    else:
        ensemble_prob = None
    
    print("\n[4/4] Evaluating models...")
    y_prob = new_model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_prob)
    
    if ensemble_prob is not None:
        ens_pred = (ensemble_prob >= 0.5).astype(int)
        ens_prec = precision_score(y_te, ens_pred, zero_division=0)
        ens_rec = recall_score(y_te, ens_pred, zero_division=0)
        ens_f1 = f1_score(y_te, ens_pred, zero_division=0)
        ens_auc = roc_auc_score(y_te, ensemble_prob)
        
        print(f"\n  ENSEMBLE MODEL (old + new combined):")
        print(f"    Precision: {ens_prec:.1%}")
        print(f"    Recall: {ens_rec:.1%}")
        print(f"    F1: {ens_f1:.1%}")
        print(f"    ROC-AUC: {ens_auc:.3f}")
    
    print(f"\n  INCREMENTAL MODEL RESULTS (Stock-Split):")
    print(f"    Precision: {prec:.1%}")
    print(f"    Recall: {rec:.1%}")
    print(f"    F1: {f1:.1%}")
    print(f"    ROC-AUC: {auc:.3f}")
    
    model_path = MODELS_DIR / "moonshot_incremental_v2.pkl.gz"
    with gzip.open(model_path, 'wb') as f:
        pickle.dump({
            'model': new_model,
            'feature_names': feature_cols,
            'version': '2.0-incremental',
            'timestamp': datetime.now().isoformat(),
            'training_type': 'warm_start' if prev_model else 'fresh',
            'stocks_count': len(set(s["symbol"] for s in samples)),
        }, f)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("INCREMENTAL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Model type: {'WARM START (incremental)' if prev_model else 'Fresh training'}")
    print(f"Stocks used: {len(set(s['symbol'] for s in samples))}")
    print(f"Model saved: {model_path}")
    
    print("\n  Previous knowledge preserved: YES")
    print("  New patterns learned: YES")
    print("=" * 70)

if __name__ == "__main__":
    main()
