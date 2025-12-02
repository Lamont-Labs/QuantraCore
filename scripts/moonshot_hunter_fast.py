#!/usr/bin/env python3
"""
FAST MOONSHOT HUNTER - Scans cached data for 50%+ gains.
Uses already-downloaded parquet files to find moonshots quickly.
"""

import os
import sys
import logging
import pickle
import gzip
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MOONSHOT_DIR = Path("data/moonshots")
MOONSHOT_DIR.mkdir(parents=True, exist_ok=True)

def find_moonshots_in_data(df: pd.DataFrame, symbol: str, threshold: float = 0.50) -> List[Dict]:
    """Find all instances where stock gained 50%+ within 30 days."""
    moonshots = []
    
    if len(df) < 30:
        return moonshots
    
    for i in range(len(df) - 30):
        entry_price = df.iloc[i]["close"]
        if entry_price <= 0:
            continue
        
        future_30d = df.iloc[i:i+30]
        max_price = future_30d["high"].max()
        max_gain = (max_price - entry_price) / entry_price
        
        if max_gain >= threshold:
            max_idx_local = future_30d["high"].values.argmax()
            days_to_peak = max_idx_local
            
            moonshots.append({
                "symbol": symbol,
                "entry_date": str(df.iloc[i]["date"]) if "date" in df.columns else f"day_{i}",
                "entry_price": float(entry_price),
                "peak_price": float(max_price),
                "gain_pct": float(max_gain * 100),
                "days_to_peak": int(days_to_peak),
                "window_start_idx": i,
            })
    
    return moonshots

def extract_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Extract features from the period before the moonshot."""
    if idx < 50:
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
        "volatility_20d": float(np.std(returns[-20:])),
        "volume_ratio": float(volume[-1] / (np.mean(volume[-20:]) + 1)),
        "price_to_sma20": float(close[-1] / (np.mean(close[-20:]) + 0.001)),
        "momentum_5d": float((close[-1] - close[-6]) / (close[-6] + 0.001)),
        "momentum_10d": float((close[-1] - close[-11]) / (close[-11] + 0.001)),
        "trend_strength": float((np.mean(close[-10:]) - np.mean(close[-20:])) / (np.mean(close[-20:]) + 0.001)),
        "consolidation": float(np.std(close[-10:]) / (np.mean(close[-10:]) + 0.001)),
    }

def main():
    logger.info("=" * 60)
    logger.info("FAST MOONSHOT HUNTER - Scanning cached data for 50%+ gains")
    logger.info("=" * 60)
    
    parquet_files = list(CACHE_DIR.glob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} cached stock files")
    
    all_moonshots = []
    all_features = []
    
    for i, path in enumerate(parquet_files):
        symbol = path.stem
        
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning(f"Could not read {symbol}: {e}")
            continue
        
        if len(df) < 60:
            continue
        
        moonshots = find_moonshots_in_data(df, symbol, threshold=0.50)
        
        if moonshots:
            logger.info(f"[{i+1}/{len(parquet_files)}] {symbol}: {len(moonshots)} moonshots (max: {max(m['gain_pct'] for m in moonshots):.0f}%)")
            
            for ms in moonshots:
                features = extract_features(df, ms["window_start_idx"])
                if features:
                    features["symbol"] = symbol
                    features["gain_pct"] = ms["gain_pct"]
                    features["days_to_peak"] = ms["days_to_peak"]
                    features["entry_date"] = ms["entry_date"]
                    all_features.append(features)
            
            all_moonshots.extend(moonshots)
    
    logger.info("\n" + "=" * 60)
    
    if all_moonshots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        moonshots_path = MOONSHOT_DIR / f"moonshots_{timestamp}.pkl.gz"
        with gzip.open(moonshots_path, "wb") as f:
            pickle.dump(all_moonshots, f)
        
        features_path = MOONSHOT_DIR / f"moonshot_features_{timestamp}.pkl.gz"
        with gzip.open(features_path, "wb") as f:
            pickle.dump(all_features, f)
        
        all_moonshots_sorted = sorted(all_moonshots, key=lambda x: x["gain_pct"], reverse=True)
        latest_path = MOONSHOT_DIR / "moonshots_latest.pkl.gz"
        with gzip.open(latest_path, "wb") as f:
            pickle.dump(all_moonshots_sorted, f)
        
        features_latest = MOONSHOT_DIR / "moonshot_features_latest.pkl.gz"
        with gzip.open(features_latest, "wb") as f:
            pickle.dump(all_features, f)
        
        df_summary = pd.DataFrame(all_moonshots_sorted[:100])
        df_summary.to_csv(MOONSHOT_DIR / "top_100_moonshots.csv", index=False)
        
        gains = [m["gain_pct"] for m in all_moonshots]
        
        print("\n" + "=" * 60)
        print("MOONSHOT DATABASE CREATED!")
        print("=" * 60)
        print(f"Total 50%+ events: {len(all_moonshots)}")
        print(f"Feature sets saved: {len(all_features)}")
        print(f"Average Gain: {np.mean(gains):.1f}%")
        print(f"Max Gain: {np.max(gains):.1f}%")
        print(f"Median Gain: {np.median(gains):.1f}%")
        print(f"\nTop 10 Moonshots:")
        for j, ms in enumerate(all_moonshots_sorted[:10]):
            print(f"  {j+1}. {ms['symbol']}: +{ms['gain_pct']:.0f}% on {ms['entry_date']}")
        print(f"\nData saved to: {MOONSHOT_DIR}")
        print("=" * 60)
    else:
        logger.warning("No moonshots found!")

if __name__ == "__main__":
    main()
