#!/usr/bin/env python3
"""
SYSTEM UPGRADE - Comprehensive improvement of all models.
1. Refresh cache with latest data
2. Expand stock universe
3. Add more features
4. Tune hyperparameters
5. Retrain everything
"""

import os
import sys
import time
import json
import pickle
import gzip
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
CACHE_DIR = Path("data/cache/polygon/day")
MOONSHOT_DIR = Path("data/moonshots")
MODELS_DIR = Path("data/models")
UPGRADE_DIR = Path("data/upgrades")

for d in [CACHE_DIR, MOONSHOT_DIR, MODELS_DIR, UPGRADE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EXPANDED_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "AVGO",
    "GME", "AMC", "BBBY", "BB", "NOK", "PLTR", "WISH", "CLOV", "SOFI", "HOOD",
    "MARA", "RIOT", "HUT", "BITF", "HIVE", "COIN", "BITO",
    "LCID", "RIVN", "NIO", "XPEV", "LI", "FSR", "NKLA", "WKHS", "GOEV",
    "SPCE", "RKLB", "ASTR", "RDW", "ASTS",
    "DWAC", "PHUN", "BKKT", "MARK", "BBIG", "ATER", "PROG", "SDC", "EXPR", "KOSS",
    "SNDL", "TLRY", "CGC", "ACB", "HEXO", "OGI", "VFF", "GRWG",
    "UPST", "AFRM", "SQ", "PYPL", "SHOP", "SE", "MELI", "DKNG", "PENN",
    "CRWD", "NET", "ZS", "OKTA", "DDOG", "SNOW", "MDB", "CFLT",
    "U", "RBLX", "MTTR", "STEM", "CHPT", "BLNK", "EVGO",
    "OPEN", "Z", "RDFN", "CVNA", "CARG", "VRM",
    "ABNB", "UBER", "LYFT", "DASH", "GRUB",
    "SNAP", "PINS", "TWTR", "SPOT", "TTD",
    "ROKU", "FUBO", "NFLX", "DIS", "WBD", "PARA",
    "PLUG", "FCEL", "BE", "BLDP", "ENPH", "SEDG", "RUN", "NOVA",
    "SPWR", "CSIQ", "JKS", "DQ",
    "ICCT", "INDO", "HUSA", "IMPP", "BTCS", "OPTT", "RELI", "MGAM", "CEI",
    "MULN", "FFIE", "GOEV", "RIDE", "WKHS", "SOLO", "AYRO",
    "GSAT", "IRDM", "VORB",
    "DNA", "BEAM", "EDIT", "NTLA", "CRSP", "PACB",
    "MRNA", "BNTX", "NVAX", "OCGN", "VXRT",
    "BYND", "TTCF", "OTLY",
    "AI", "PATH", "DOCN", "ESTC", "GTLB", "SUMO",
    "IONQ", "RGTI", "QUBT",
]

def fetch_stock_data(symbol: str, days_back: int = 730) -> pd.DataFrame:
    """Fetch historical data for a stock from Polygon."""
    if not POLYGON_API_KEY:
        return pd.DataFrame()
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"apiKey": POLYGON_API_KEY, "limit": 5000}
    
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        
        data = resp.json()
        results = data.get("results", [])
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.rename(columns={
            "t": "timestamp", "o": "open", "h": "high", 
            "l": "low", "c": "close", "v": "volume"
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"]
        
        return df[["date", "open", "high", "low", "close", "volume"]]
    
    except Exception as e:
        logger.warning(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def refresh_cache(symbols: List[str], delay: float = 0.25):
    """Refresh cache with latest data for all symbols."""
    logger.info(f"Refreshing cache for {len(symbols)} symbols...")
    
    refreshed = 0
    failed = 0
    
    for i, symbol in enumerate(symbols):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(symbols)} ({refreshed} refreshed, {failed} failed)")
        
        df = fetch_stock_data(symbol)
        
        if len(df) >= 60:
            path = CACHE_DIR / f"{symbol}.parquet"
            df.to_parquet(path, index=False)
            refreshed += 1
        else:
            failed += 1
        
        time.sleep(delay)
    
    logger.info(f"Cache refresh complete: {refreshed} stocks updated, {failed} failed")
    return refreshed

def extract_enhanced_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Extract 50+ enhanced features for better predictions."""
    if idx < 60 or idx >= len(df):
        return {}
    
    window = df.iloc[idx-60:idx+1]
    
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    open_prices = window["open"].values if "open" in window.columns else close
    
    returns = np.diff(close) / (close[:-1] + 0.001)
    
    features = {}
    
    for period in [5, 10, 20, 50]:
        features[f"sma_{period}"] = np.mean(close[-period:])
        features[f"vol_sma_{period}"] = np.mean(volume[-period:])
    
    for period in [5, 10, 20]:
        features[f"volatility_{period}"] = np.std(returns[-period:])
    
    features["vol_compression_10_20"] = features["volatility_10"] / (features["volatility_20"] + 0.001)
    features["vol_compression_5_20"] = features["volatility_5"] / (features["volatility_20"] + 0.001)
    
    for period in [3, 5, 10, 20]:
        features[f"momentum_{period}"] = (close[-1] - close[-period-1]) / (close[-period-1] + 0.001)
    
    features["mom_acceleration"] = features["momentum_3"] - features["momentum_5"]
    features["mom_inflection"] = features["momentum_5"] - features["momentum_10"]
    
    for period in [5, 10, 20]:
        features[f"vol_ratio_{period}"] = np.mean(volume[-5:]) / (np.mean(volume[-period:]) + 1)
    
    features["vol_acceleration"] = features["vol_ratio_5"] / (features["vol_ratio_10"] + 0.001)
    
    for period in [10, 20, 50]:
        features[f"price_vs_sma_{period}"] = close[-1] / (features[f"sma_{period}"] + 0.001)
    
    features["sma_slope_10"] = (features["sma_5"] - features["sma_10"]) / (features["sma_10"] + 0.001)
    features["sma_slope_20"] = (features["sma_10"] - features["sma_20"]) / (features["sma_20"] + 0.001)
    
    for period in [10, 20]:
        range_val = np.max(high[-period:]) - np.min(low[-period:])
        features[f"range_{period}"] = range_val / (close[-1] + 0.001)
    
    features["range_compression"] = features["range_10"] / (features["range_20"] + 0.001)
    
    for period in [10, 20]:
        features[f"high_{period}"] = np.max(high[-period:])
        features[f"low_{period}"] = np.min(low[-period:])
    
    features["breakout_proximity"] = close[-1] / (features["high_20"] + 0.001)
    features["support_distance"] = (close[-1] - features["low_20"]) / (features["low_20"] + 0.001)
    
    daily_range = high[-10:] - low[-10:]
    close_position = (close[-10:] - low[-10:]) / (daily_range + 0.001)
    features["close_near_high"] = np.mean(close_position)
    
    upper_wick = (high[-10:] - np.maximum(close[-10:], open_prices[-10:])) / (daily_range + 0.001)
    lower_wick = (np.minimum(close[-10:], open_prices[-10:]) - low[-10:]) / (daily_range + 0.001)
    features["upper_wick_avg"] = np.mean(upper_wick)
    features["lower_wick_avg"] = np.mean(lower_wick)
    features["wick_shrinking"] = np.mean(upper_wick[-3:]) - np.mean(upper_wick[-10:-3])
    
    features["up_days_5"] = sum(1 for r in returns[-5:] if r > 0)
    features["up_days_10"] = sum(1 for r in returns[-10:] if r > 0)
    features["tight_days_10"] = sum(1 for r in returns[-10:] if abs(r) < 0.02)
    
    dollar_vol = close[-20:] * volume[-20:]
    features["dollar_vol_ratio"] = np.mean(dollar_vol[-3:]) / (np.mean(dollar_vol[-20:]) + 1)
    
    features["gap_today"] = (open_prices[-1] - close[-2]) / (close[-2] + 0.001)
    
    features["rsi_proxy"] = features["up_days_10"] / 10.0
    
    if len(returns) >= 20:
        ema_12 = pd.Series(close[-26:]).ewm(span=12).mean().iloc[-1]
        ema_26 = pd.Series(close[-26:]).ewm(span=26).mean().iloc[-1]
        features["macd_proxy"] = (ema_12 - ema_26) / (close[-1] + 0.001)
    else:
        features["macd_proxy"] = 0
    
    atr_values = []
    for j in range(-14, 0):
        tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_values.append(tr)
    features["atr_14"] = np.mean(atr_values) / (close[-1] + 0.001)
    
    features["current_price"] = close[-1]
    features["volume_today"] = volume[-1]
    
    cleaned = {}
    for k, v in features.items():
        if np.isfinite(v):
            cleaned[k] = float(v)
        else:
            cleaned[k] = 0.0
    
    return cleaned

def find_moonshots_enhanced(df: pd.DataFrame, symbol: str, threshold: float = 0.50) -> List[Dict]:
    """Find moonshots with enhanced feature extraction."""
    moonshots = []
    
    if len(df) < 60:
        return moonshots
    
    for i in range(60, len(df) - 30):
        entry_price = df.iloc[i]["close"]
        if entry_price <= 0:
            continue
        
        future_30d = df.iloc[i:i+30]
        max_price = future_30d["high"].max()
        max_gain = (max_price - entry_price) / entry_price
        
        if max_gain >= threshold:
            max_idx_local = future_30d["high"].values.argmax()
            
            features = extract_enhanced_features(df, i)
            if features:
                features["symbol"] = symbol
                features["gain_pct"] = float(max_gain * 100)
                features["days_to_peak"] = int(max_idx_local)
                features["entry_date"] = str(df.iloc[i].get("date", f"idx_{i}"))
                
                moonshots.append(features)
    
    return moonshots

def scan_all_stocks():
    """Scan all cached stocks for moonshots with enhanced features."""
    logger.info("Scanning all stocks for moonshots with enhanced features...")
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stocks")
    
    all_moonshots = []
    
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
        
        moonshots = find_moonshots_enhanced(df, symbol)
        
        if moonshots:
            if i % 20 == 0:
                logger.info(f"[{i+1}/{len(parquet_files)}] {symbol}: {len(moonshots)} moonshots")
            all_moonshots.extend(moonshots)
    
    logger.info(f"Total moonshots found: {len(all_moonshots)}")
    return all_moonshots

def tune_hyperparameters(X, y, model_type: str = "classifier"):
    """Perform grid search to find optimal hyperparameters."""
    logger.info(f"Tuning hyperparameters for {model_type}...")
    
    if model_type == "classifier":
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.8, 0.9, 1.0],
        }
    else:
        base_model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.8, 0.9, 1.0],
        }
    
    grid_search = GridSearchCV(
        base_model, param_grid, 
        cv=3, scoring="accuracy" if model_type == "classifier" else "neg_mean_absolute_error",
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X, y)
    
    logger.info(f"Best params: {grid_search.best_params_}")
    logger.info(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def train_enhanced_models(moonshots: List[Dict]):
    """Train all models with enhanced features and tuned hyperparameters."""
    logger.info("Training enhanced models...")
    
    feature_cols = [k for k in moonshots[0].keys() if k not in ["symbol", "gain_pct", "days_to_peak", "entry_date"]]
    
    X = np.array([[m.get(k, 0) for k in feature_cols] for m in moonshots])
    gains = np.array([m["gain_pct"] for m in moonshots])
    days = np.array([min(m["days_to_peak"], 30) for m in moonshots])
    
    y_100 = (gains >= 100).astype(int)
    y_200 = (gains >= 200).astype(int)
    y_500 = (gains >= 500).astype(int)
    
    if days.max() <= 3:
        y_timing = np.zeros(len(days), dtype=int)
    else:
        y_timing = np.digitize(days, bins=[0, 3, 5, 10, 20, 31]) - 1
    
    X_train, X_test, y_100_train, y_100_test = train_test_split(X, y_100, test_size=0.2, random_state=42)
    _, _, y_200_train, y_200_test = train_test_split(X, y_200, test_size=0.2, random_state=42)
    _, _, y_500_train, y_500_test = train_test_split(X, y_500, test_size=0.2, random_state=42)
    _, _, y_timing_train, y_timing_test = train_test_split(X, y_timing, test_size=0.2, random_state=42)
    _, _, days_train, days_test = train_test_split(X, days, test_size=0.2, random_state=42)
    
    results = {"feature_cols": feature_cols}
    
    logger.info("\n[1/5] Tuning 100%+ detector...")
    best_params_100 = tune_hyperparameters(X_train, y_100_train, "classifier")
    model_100 = GradientBoostingClassifier(**best_params_100, random_state=42)
    model_100.fit(X_train, y_100_train)
    pred_100 = model_100.predict(X_test)
    acc_100 = accuracy_score(y_100_test, pred_100)
    prec_100 = precision_score(y_100_test, pred_100, zero_division=0)
    rec_100 = recall_score(y_100_test, pred_100, zero_division=0)
    logger.info(f"  100%+ Model: Acc={acc_100:.1%}, Prec={prec_100:.1%}, Rec={rec_100:.1%}")
    results["100_plus"] = {"model": model_100, "accuracy": acc_100, "precision": prec_100, "recall": rec_100, "params": best_params_100}
    
    logger.info("\n[2/5] Tuning 200%+ detector...")
    best_params_200 = tune_hyperparameters(X_train, y_200_train, "classifier")
    model_200 = GradientBoostingClassifier(**best_params_200, random_state=42)
    model_200.fit(X_train, y_200_train)
    pred_200 = model_200.predict(X_test)
    acc_200 = accuracy_score(y_200_test, pred_200)
    prec_200 = precision_score(y_200_test, pred_200, zero_division=0)
    rec_200 = recall_score(y_200_test, pred_200, zero_division=0)
    logger.info(f"  200%+ Model: Acc={acc_200:.1%}, Prec={prec_200:.1%}, Rec={rec_200:.1%}")
    results["200_plus"] = {"model": model_200, "accuracy": acc_200, "precision": prec_200, "recall": rec_200, "params": best_params_200}
    
    logger.info("\n[3/5] Tuning 500%+ detector...")
    best_params_500 = tune_hyperparameters(X_train, y_500_train, "classifier")
    model_500 = GradientBoostingClassifier(**best_params_500, random_state=42)
    model_500.fit(X_train, y_500_train)
    pred_500 = model_500.predict(X_test)
    acc_500 = accuracy_score(y_500_test, pred_500)
    prec_500 = precision_score(y_500_test, pred_500, zero_division=0)
    rec_500 = recall_score(y_500_test, pred_500, zero_division=0)
    logger.info(f"  500%+ Model: Acc={acc_500:.1%}, Prec={prec_500:.1%}, Rec={rec_500:.1%}")
    results["500_plus"] = {"model": model_500, "accuracy": acc_500, "precision": prec_500, "recall": rec_500, "params": best_params_500}
    
    logger.info("\n[4/5] Tuning timing classifier...")
    best_params_timing = tune_hyperparameters(X_train, y_timing_train, "classifier")
    model_timing_cls = GradientBoostingClassifier(**best_params_timing, random_state=42)
    model_timing_cls.fit(X_train, y_timing_train)
    pred_timing = model_timing_cls.predict(X_test)
    acc_timing = accuracy_score(y_timing_test, pred_timing)
    logger.info(f"  Timing Classifier: Acc={acc_timing:.1%}")
    results["timing_classifier"] = {"model": model_timing_cls, "accuracy": acc_timing, "params": best_params_timing}
    
    logger.info("\n[5/5] Tuning days regressor...")
    best_params_reg = tune_hyperparameters(X_train, days_train, "regressor")
    model_timing_reg = GradientBoostingRegressor(**best_params_reg, random_state=42)
    model_timing_reg.fit(X_train, days_train)
    pred_days = model_timing_reg.predict(X_test)
    mae = mean_absolute_error(days_test, pred_days)
    logger.info(f"  Days Regressor: MAE={mae:.2f} days")
    results["timing_regressor"] = {"model": model_timing_reg, "mae": mae, "params": best_params_reg}
    
    results["bin_labels"] = ["0-3 days", "4-5 days", "6-10 days", "11-20 days", "21-30 days"]
    
    return results

def save_upgraded_models(results: Dict, moonshots: List[Dict]):
    """Save all upgraded models and data."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    moonshots_path = MOONSHOT_DIR / f"moonshots_enhanced_{timestamp}.pkl.gz"
    with gzip.open(moonshots_path, "wb") as f:
        pickle.dump(moonshots, f)
    logger.info(f"Saved {len(moonshots)} enhanced moonshots to {moonshots_path}")
    
    with gzip.open(MOONSHOT_DIR / "moonshots_enhanced_latest.pkl.gz", "wb") as f:
        pickle.dump(moonshots, f)
    
    detector = {
        "100_plus": {"model": results["100_plus"]["model"], "precision": results["100_plus"]["precision"], "recall": results["100_plus"]["recall"]},
        "200_plus": {"model": results["200_plus"]["model"], "precision": results["200_plus"]["precision"], "recall": results["200_plus"]["recall"]},
        "500_plus": {"model": results["500_plus"]["model"], "precision": results["500_plus"]["precision"], "recall": results["500_plus"]["recall"]},
        "feature_cols": results["feature_cols"],
    }
    with gzip.open(MODELS_DIR / "moonshot_detector_enhanced.pkl.gz", "wb") as f:
        pickle.dump(detector, f)
    logger.info("Saved enhanced moonshot detector")
    
    timer = {
        "regressor": results["timing_regressor"]["model"],
        "classifier": results["timing_classifier"]["model"],
        "feature_cols": results["feature_cols"],
        "bin_labels": results["bin_labels"],
        "metrics": {
            "mae_days": results["timing_regressor"]["mae"],
            "bin_accuracy": results["timing_classifier"]["accuracy"],
        }
    }
    with gzip.open(MODELS_DIR / "breakout_timer_enhanced.pkl.gz", "wb") as f:
        pickle.dump(timer, f)
    logger.info("Saved enhanced breakout timer")
    
    upgrade_report = {
        "timestamp": timestamp,
        "moonshots_found": len(moonshots),
        "features_used": len(results["feature_cols"]),
        "models": {
            "100_plus": {"precision": results["100_plus"]["precision"], "recall": results["100_plus"]["recall"], "params": results["100_plus"]["params"]},
            "200_plus": {"precision": results["200_plus"]["precision"], "recall": results["200_plus"]["recall"], "params": results["200_plus"]["params"]},
            "500_plus": {"precision": results["500_plus"]["precision"], "recall": results["500_plus"]["recall"], "params": results["500_plus"]["params"]},
            "timing": {"accuracy": results["timing_classifier"]["accuracy"], "mae": results["timing_regressor"]["mae"]},
        }
    }
    with open(UPGRADE_DIR / f"upgrade_report_{timestamp}.json", "w") as f:
        json.dump(upgrade_report, f, indent=2)
    
    return upgrade_report

def main():
    print("=" * 70)
    print("QUANTRACORE APEX - COMPREHENSIVE SYSTEM UPGRADE")
    print("=" * 70)
    start_time = time.time()
    
    print("\n[PHASE 1] Refreshing data cache...")
    stocks_refreshed = refresh_cache(EXPANDED_UNIVERSE[:100], delay=0.2)
    
    print("\n[PHASE 2] Scanning for moonshots with enhanced features...")
    moonshots = scan_all_stocks()
    
    if len(moonshots) < 100:
        print(f"ERROR: Only found {len(moonshots)} moonshots. Need at least 100.")
        return
    
    print("\n[PHASE 3] Training with hyperparameter optimization...")
    results = train_enhanced_models(moonshots)
    
    print("\n[PHASE 4] Saving upgraded models...")
    report = save_upgraded_models(results, moonshots)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("UPGRADE COMPLETE!")
    print("=" * 70)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Stocks refreshed: {stocks_refreshed}")
    print(f"Moonshots found: {report['moonshots_found']}")
    print(f"Features: {report['features_used']}")
    print("\nModel Performance:")
    print(f"  100%+ Detector: {report['models']['100_plus']['precision']:.1%} precision")
    print(f"  200%+ Detector: {report['models']['200_plus']['precision']:.1%} precision")
    print(f"  500%+ Detector: {report['models']['500_plus']['precision']:.1%} precision")
    print(f"  Timing Classifier: {report['models']['timing']['accuracy']:.1%} accuracy")
    print(f"  Days Predictor: {report['models']['timing']['mae']:.2f} days MAE")
    print("=" * 70)

if __name__ == "__main__":
    main()
