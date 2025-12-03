#!/usr/bin/env python3
"""
FAST TRAINING - Train on expanded universe with optimized sampling
"""

import os
import sys
import time
import pickle
import gzip
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb

CACHE_DIR = Path("data/cache/polygon/day")
MODELS_DIR = Path("data/models")

SECTOR_MAP = {
    "AAPL": "tech", "AMD": "tech", "NVDA": "tech", "TSLA": "auto",
    "BBAI": "tech", "CLOV": "health", "MARA": "crypto", "RIOT": "crypto",
    "PLTR": "tech", "SOFI": "fintech", "LCID": "auto", "RIVN": "auto",
    "GME": "retail", "AMC": "entertainment", "BB": "tech", "NOK": "tech",
    "MRNA": "biotech", "BNTX": "biotech", "NVAX": "biotech", "VXRT": "biotech",
    "INO": "biotech", "JD": "china_tech", "PDD": "china_tech", "BIDU": "china_tech",
    "BILI": "china_tech", "TME": "china_tech", "BABA": "china_tech",
    "QUBT": "quantum", "IONQ": "quantum", "RGTI": "quantum",
    "PLUG": "hydrogen", "FCEL": "hydrogen", "BE": "hydrogen",
    "LAZR": "lidar", "VLDR": "lidar", "OUST": "lidar",
    "CGC": "cannabis", "TLRY": "cannabis", "ACB": "cannabis", "SNDL": "cannabis",
}

def get_stock_embedding(symbol):
    sector = SECTOR_MAP.get(symbol, "unknown")
    sector_list = ["tech", "crypto", "health", "fintech", "auto", "retail", 
                   "cannabis", "entertainment", "aerospace", "biotech", "china_tech", 
                   "quantum", "hydrogen", "lidar", "unknown"]
    vec = [1.0 if s == sector else 0.0 for s in sector_list]
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) / (16**8)
    return {f"sector_{i}": float(v) for i, v in enumerate(vec)} | {
        "symbol_hash": float(sym_hash),
        "is_meme": float(symbol in {"GME", "AMC", "BB", "BBAI", "CLOV", "WISH"}),
        "is_crypto_related": float(symbol in {"MARA", "RIOT", "COIN", "HOOD", "CLSK", "BTBT"}),
        "is_ev": float(symbol in {"TSLA", "LCID", "RIVN", "WKHS", "GOEV", "NKLA", "FSR"}),
        "is_biotech": float(symbol in {"MRNA", "BNTX", "NVAX", "VXRT", "INO", "OCGN"}),
        "is_china_tech": float(symbol in {"JD", "PDD", "BIDU", "BILI", "TME", "BABA", "NIO", "XPEV", "LI"}),
    }


def extract_features(df, idx, symbol):
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
        f[f"volatility_{p}"] = np.std(returns[-p:]) if len(returns) >= p else np.std(returns)
    
    f["vol_compression_5_20"] = f["volatility_5"] / (f["volatility_20"] + 0.001)
    f["vol_compression_10_30"] = f["volatility_10"] / (f["volatility_30"] + 0.001)
    f["vol_compression_5_30"] = f["volatility_5"] / (f["volatility_30"] + 0.001)
    
    for p in [3, 5, 10, 20]:
        if len(close) > p:
            f[f"momentum_{p}"] = (close[-1] - close[-p-1]) / (close[-p-1] + 0.001)
        else:
            f[f"momentum_{p}"] = 0
    
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
        f[f"range_{p}"] = (np.max(high[-p:]) - np.min(low[-p:])) / (close[-1] + 0.001)
    
    f["range_compression_10_20"] = f["range_10"] / (f["range_20"] + 0.001)
    f["range_compression_10_30"] = f["range_10"] / (f["range_30"] + 0.001)
    
    f["high_20"] = np.max(high[-20:])
    f["low_20"] = np.min(low[-20:])
    f["high_52w"] = np.max(high[-min(252, len(high)):]) if len(high) >= 50 else np.max(high[-50:])
    
    f["breakout_proximity"] = close[-1] / (f["high_20"] + 0.001)
    f["breakout_proximity_52w"] = close[-1] / (f["high_52w"] + 0.001)
    f["support_distance"] = (close[-1] - f["low_20"]) / (f["low_20"] + 0.001)
    
    for i in range(-5, 0):
        if abs(i-1) < len(close):
            gap = (open_p[i] - close[i-1]) / (close[i-1] + 0.001)
        else:
            gap = 0
        f[f"gap_day_{abs(i)}"] = gap
    
    f["gap_today"] = (open_p[-1] - close[-2]) / (close[-2] + 0.001) if len(close) >= 2 else 0
    f["gap_avg_5d"] = np.mean([abs(f[f"gap_day_{i}"]) for i in range(1, 6)])
    f["gap_up_count_5d"] = sum(1 for i in range(1, 6) if f[f"gap_day_{i}"] > 0)
    
    dr = high[-10:] - low[-10:]
    cp = (close[-10:] - low[-10:]) / (dr + 0.001)
    f["close_position_avg"] = np.mean(cp)
    f["close_position_today"] = cp[-1]
    
    uw = (high[-10:] - np.maximum(close[-10:], open_p[-10:])) / (dr + 0.001)
    lw = (np.minimum(close[-10:], open_p[-10:]) - low[-10:]) / (dr + 0.001)
    f["upper_wick_avg"] = np.mean(uw)
    f["lower_wick_avg"] = np.mean(lw)
    f["wick_shrinking"] = np.mean(uw[-3:]) - np.mean(uw[-7:-3]) if len(uw) >= 7 else 0
    f["wick_ratio"] = np.mean(uw) / (np.mean(lw) + 0.001)
    
    f["up_days_5"] = sum(1 for r in returns[-5:] if r > 0) if len(returns) >= 5 else 0
    f["up_days_10"] = sum(1 for r in returns[-10:] if r > 0) if len(returns) >= 10 else 0
    f["up_days_20"] = sum(1 for r in returns[-20:] if r > 0) if len(returns) >= 20 else 0
    f["tight_days_5"] = sum(1 for r in returns[-5:] if abs(r) < 0.02) if len(returns) >= 5 else 0
    f["tight_days_10"] = sum(1 for r in returns[-10:] if abs(r) < 0.02) if len(returns) >= 10 else 0
    
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
    
    if len(close) >= 26:
        ema12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
        f["macd"] = (ema12 - ema26) / (close[-1] + 0.001)
    else:
        f["macd"] = 0
    
    atr_vals = []
    for j in range(-14, 0):
        if abs(j-1) < len(close):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_vals.append(tr)
    f["atr_14"] = np.mean(atr_vals) / (close[-1] + 0.001) if atr_vals else 0
    f["atr_compression"] = atr_vals[-1] / (np.mean(atr_vals) + 0.001) if len(atr_vals) > 1 else 1
    
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


def build_dataset_fast(max_samples_per_stock=50):
    """Build dataset with limited samples per stock for speed."""
    print("Building dataset from cached stocks...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} cached stocks")
    
    samples = []
    
    for i, path in enumerate(parquet_files):
        symbol = path.stem
        
        try:
            df = pd.read_parquet(path)
        except:
            continue
        
        if len(df) < 90:
            continue
        
        pos_samples = []
        neg_samples = []
        
        for idx in range(60, len(df) - 30, 5):
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
                pos_samples.append(feat)
            elif max_gain < 0.15:
                feat["label"] = 0
                neg_samples.append(feat)
        
        np.random.seed(42 + i)
        if len(pos_samples) > max_samples_per_stock:
            indices = np.random.choice(len(pos_samples), max_samples_per_stock, replace=False)
            pos_samples = [pos_samples[j] for j in indices]
        if len(neg_samples) > max_samples_per_stock * 2:
            indices = np.random.choice(len(neg_samples), max_samples_per_stock * 2, replace=False)
            neg_samples = [neg_samples[j] for j in indices]
        
        samples.extend(pos_samples)
        samples.extend(neg_samples)
        
        if i % 50 == 0:
            pos = sum(1 for s in samples if s["label"] == 1)
            print(f"  [{i}/{len(parquet_files)}] Samples: {len(samples)} (pos: {pos})")
    
    print(f"Total samples: {len(samples)}")
    return samples


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("FAST TRAINING - Expanded Universe")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n[1/3] Building dataset...")
    samples = build_dataset_fast(max_samples_per_stock=40)
    
    meta_cols = ["symbol", "date_idx", "gain_pct", "label"]
    feature_cols = sorted([k for k in samples[0].keys() if k not in meta_cols])
    print(f"Features: {len(feature_cols)}")
    
    symbols = list(set(s["symbol"] for s in samples))
    np.random.seed(42)
    np.random.shuffle(symbols)
    
    split_idx = int(len(symbols) * 0.7)
    train_symbols = set(symbols[:split_idx])
    test_symbols = set(symbols[split_idx:])
    
    train = [s for s in samples if s["symbol"] in train_symbols]
    test = [s for s in samples if s["symbol"] in test_symbols]
    
    X_train = np.array([[s.get(k, 0) for k in feature_cols] for s in train])
    y_train = np.array([s["label"] for s in train])
    X_test = np.array([[s.get(k, 0) for k in feature_cols] for s in test])
    y_test = np.array([s["label"] for s in test])
    
    print(f"\n[2/3] Training LightGBM...")
    print(f"  Train: {len(X_train)} ({sum(y_train)} pos) from {len(train_symbols)} stocks")
    print(f"  Test: {len(X_test)} ({sum(y_test)} pos) from {len(test_symbols)} stocks")
    
    pos_weight = len(y_train[y_train == 0]) / (len(y_train[y_train == 1]) + 1)
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    print("\n[3/3] Evaluating...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n  STOCK-SPLIT VALIDATION (unseen stocks):")
    print(f"    Precision: {prec:.1%}")
    print(f"    Recall: {rec:.1%}")
    print(f"    ROC-AUC: {auc:.3f}")
    
    for thresh in [0.6, 0.7, 0.8]:
        y_pred_t = (y_prob >= thresh).astype(int)
        prec_t = precision_score(y_test, y_pred_t, zero_division=0)
        rec_t = recall_score(y_test, y_pred_t, zero_division=0)
        print(f"    @ {thresh:.0%} threshold: Precision {prec_t:.1%}, Recall {rec_t:.1%}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "moonshot_incremental_v2.pkl.gz"
    with gzip.open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_cols,
            'version': '3.0-expanded',
            'timestamp': datetime.now().isoformat(),
            'training_type': 'expanded_universe',
            'stocks_count': len(symbols),
            'samples_count': len(samples),
            'precision': float(prec),
            'recall': float(rec),
            'roc_auc': float(auc),
        }, f)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Stocks in universe: {len(symbols)}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Model saved: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
