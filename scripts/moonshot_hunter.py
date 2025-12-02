#!/usr/bin/env python3
"""
MOONSHOT HUNTER - Finds and downloads all 50%+ gain samples.
Scans historical data for massive runners and saves them for training.
"""

import os
import sys
import logging
import pickle
import gzip
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CACHE_DIR = Path("data/cache/polygon/day")
MOONSHOT_DIR = Path("data/moonshots")
MOONSHOT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_MOONSHOT_STOCKS = [
    "GME", "AMC", "BBIG", "SPRT", "IRNT", "BKKT", "DWAC", "PHUN", "MARK",
    "CEI", "PROG", "ATER", "SDC", "WISH", "CLOV", "BB", "EXPR", "KOSS",
    "NAKD", "SNDL", "TLRY", "MARA", "RIOT", "HUT", "BITF", "HIVE", "COIN",
    "HOOD", "RBLX", "PLTR", "SOFI", "LCID", "RIVN", "NIO", "XPEV", "LI",
    "TSLA", "NVDA", "AMD", "UPST", "AFRM", "DASH", "ABNB", "U", "SNOW",
    "CRWD", "DDOG", "NET", "ZS", "OKTA", "TWLO", "SQ", "PYPL", "SE",
    "MELI", "SHOP", "ETSY", "PINS", "SNAP", "ROKU", "TTD", "MGAM",
    "IMPP", "INDO", "HUSA", "BTCS", "OPTT", "RELI", "ICCT", "MULN",
    "BBBY", "APE", "HYMC", "FAZE", "GETY", "HKD", "MEGL", "TOP",
    "APRN", "NKLA", "WKHS", "GOEV", "RIDE", "FSR", "FFIE", "PSNY"
]

SCAN_YEARS = 3

def fetch_polygon_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily bars from Polygon.io."""
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set!")
        return pd.DataFrame()
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {
        "apiKey": POLYGON_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                df = pd.DataFrame(data["results"])
                df = df.rename(columns={
                    "o": "open", "h": "high", "l": "low", 
                    "c": "close", "v": "volume", "t": "timestamp"
                })
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df[["date", "open", "high", "low", "close", "volume"]]
        elif response.status_code == 429:
            logger.warning(f"Rate limited on {symbol}, waiting...")
            time.sleep(12)
            return fetch_polygon_bars(symbol, start_date, end_date)
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
    
    return pd.DataFrame()

def find_moonshots_in_data(df: pd.DataFrame, symbol: str, threshold: float = 0.50) -> List[Dict]:
    """Find all instances where stock gained 50%+ within 30 days."""
    moonshots = []
    
    if len(df) < 30:
        return moonshots
    
    for i in range(len(df) - 30):
        entry_price = df.iloc[i]["close"]
        
        future_30d = df.iloc[i:i+30]
        max_price = future_30d["high"].max()
        max_gain = (max_price - entry_price) / entry_price
        
        if max_gain >= threshold:
            max_idx = future_30d["high"].idxmax()
            days_to_peak = max_idx - i
            
            moonshots.append({
                "symbol": symbol,
                "entry_date": df.iloc[i]["date"],
                "entry_price": entry_price,
                "peak_price": max_price,
                "gain_pct": max_gain * 100,
                "days_to_peak": days_to_peak,
                "window_start_idx": i,
                "window_end_idx": min(i + 90, len(df)),
            })
    
    return moonshots

def extract_moonshot_features(df: pd.DataFrame, moonshot: Dict) -> Dict[str, Any]:
    """Extract features from the period before the moonshot."""
    idx = moonshot["window_start_idx"]
    if idx < 50:
        return {}
    
    window = df.iloc[idx-50:idx+1]
    
    close = window["close"].values
    high = window["high"].values
    low = window["low"].values
    volume = window["volume"].values
    
    returns = np.diff(close) / close[:-1]
    
    features = {
        "symbol": moonshot["symbol"],
        "entry_date": moonshot["entry_date"],
        "gain_pct": moonshot["gain_pct"],
        "days_to_peak": moonshot["days_to_peak"],
        "close": close[-1],
        "sma_10": np.mean(close[-10:]),
        "sma_20": np.mean(close[-20:]),
        "sma_50": np.mean(close[-50:]),
        "volatility_20d": np.std(returns[-20:]),
        "volume_ratio": volume[-1] / (np.mean(volume[-20:]) + 1),
        "price_to_sma20": close[-1] / np.mean(close[-20:]),
        "momentum_5d": (close[-1] - close[-6]) / close[-6],
        "momentum_10d": (close[-1] - close[-11]) / close[-11],
        "trend_strength": (np.mean(close[-10:]) - np.mean(close[-20:])) / np.mean(close[-20:]),
        "high_low_range": (np.max(high[-20:]) - np.min(low[-20:])) / close[-1],
        "volume_surge": volume[-1] / (np.mean(volume[-50:]) + 1),
        "consolidation": np.std(close[-10:]) / np.mean(close[-10:]),
        "breakout_potential": (close[-1] - np.min(low[-20:])) / (np.max(high[-20:]) - np.min(low[-20:]) + 0.01),
    }
    
    return features

def save_moonshot_data(moonshots: List[Dict], features: List[Dict]):
    """Save moonshot data for training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    moonshots_path = MOONSHOT_DIR / f"moonshots_{timestamp}.pkl.gz"
    with gzip.open(moonshots_path, "wb") as f:
        pickle.dump(moonshots, f)
    logger.info(f"Saved {len(moonshots)} moonshots to {moonshots_path}")
    
    features_path = MOONSHOT_DIR / f"moonshot_features_{timestamp}.pkl.gz"
    with gzip.open(features_path, "wb") as f:
        pickle.dump(features, f)
    logger.info(f"Saved {len(features)} feature sets to {features_path}")
    
    df = pd.DataFrame(moonshots)
    csv_path = MOONSHOT_DIR / "moonshots_summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")
    
    return moonshots_path, features_path

def main():
    logger.info("=" * 60)
    logger.info("MOONSHOT HUNTER - Finding 50%+ Gain Samples")
    logger.info("=" * 60)
    
    if not POLYGON_API_KEY:
        logger.error("POLYGON_API_KEY not set! Cannot fetch data.")
        sys.exit(1)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=SCAN_YEARS * 365)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Scanning {len(KNOWN_MOONSHOT_STOCKS)} stocks from {start_str} to {end_str}")
    logger.info(f"Looking for 50%+ gains within 30-day windows")
    
    all_moonshots = []
    all_features = []
    
    for i, symbol in enumerate(KNOWN_MOONSHOT_STOCKS):
        logger.info(f"\n[{i+1}/{len(KNOWN_MOONSHOT_STOCKS)}] Scanning {symbol}...")
        
        df = fetch_polygon_bars(symbol, start_str, end_str)
        
        if len(df) < 60:
            logger.warning(f"  Not enough data for {symbol} ({len(df)} bars)")
            continue
        
        cache_path = CACHE_DIR / f"{symbol}.parquet"
        df.to_parquet(cache_path)
        logger.info(f"  Cached {len(df)} bars")
        
        moonshots = find_moonshots_in_data(df, symbol, threshold=0.50)
        
        if moonshots:
            logger.info(f"  Found {len(moonshots)} moonshot events!")
            
            for ms in moonshots:
                logger.info(f"    {ms['entry_date'].strftime('%Y-%m-%d')}: +{ms['gain_pct']:.1f}% in {ms['days_to_peak']} days")
                
                features = extract_moonshot_features(df, ms)
                if features:
                    all_features.append(features)
            
            all_moonshots.extend(moonshots)
        else:
            logger.info(f"  No 50%+ moonshots found")
        
        time.sleep(0.25)
    
    logger.info("\n" + "=" * 60)
    logger.info("MOONSHOT HUNT COMPLETE!")
    logger.info("=" * 60)
    
    if all_moonshots:
        moonshots_path, features_path = save_moonshot_data(all_moonshots, all_features)
        
        gains = [m["gain_pct"] for m in all_moonshots]
        logger.info(f"\nTotal Moonshots Found: {len(all_moonshots)}")
        logger.info(f"Average Gain: {np.mean(gains):.1f}%")
        logger.info(f"Max Gain: {np.max(gains):.1f}%")
        logger.info(f"Median Gain: {np.median(gains):.1f}%")
        
        print("\n" + "=" * 60)
        print(f"MOONSHOT DATABASE CREATED!")
        print(f"Total 50%+ events: {len(all_moonshots)}")
        print(f"Feature sets saved: {len(all_features)}")
        print(f"Data saved to: {MOONSHOT_DIR}")
        print("=" * 60)
    else:
        logger.warning("No moonshots found in the scanned data!")

if __name__ == "__main__":
    main()
