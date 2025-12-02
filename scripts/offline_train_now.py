#!/usr/bin/env python3
"""
OFFLINE TRAINING - Runs immediately using cached Polygon data.
Bypasses all external APIs (FRED, Finnhub) that are rate limited.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/polygon/day")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RUNNER_STOCKS = [
    "AAPL", "ABBV", "AMZN", "AVGO", "BBIG", "BRK.B", "BTCS", "CLOV",
    "C", "CRM", "ELV", "ETSY", "GOOGL", "GOOG", "HD", "HON", "HUSA",
    "ICCT", "IMPP", "INDO", "JNJ", "JPM", "LLY", "LMT", "MARA", "META",
    "MGAM", "MSFT", "NFLX", "NVDA", "OPTT", "PLTR", "RELI", "RIOT",
    "ROKU", "SNAP", "SOFI", "SYK", "TSLA", "V", "WMT", "Z"
]

def load_cached_bars(symbol: str) -> pd.DataFrame:
    """Load cached parquet data for a symbol."""
    path = CACHE_DIR / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"  Loaded {len(df)} bars for {symbol}")
        return df
    return pd.DataFrame()

def compute_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute trading features from OHLCV data."""
    if len(df) < 20:
        return {}
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    returns = np.diff(close) / close[:-1]
    
    sma_10 = np.mean(close[-10:])
    sma_20 = np.mean(close[-20:])
    sma_50 = np.mean(close[-50:]) if len(close) >= 50 else sma_20
    
    vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    
    rsi_delta = np.diff(close[-15:])
    gains = np.where(rsi_delta > 0, rsi_delta, 0)
    losses = np.where(rsi_delta < 0, -rsi_delta, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0.001
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / max(avg_loss, 0.001)
    rsi = 100 - (100 / (1 + rs))
    
    atr_range = high[-14:] - low[-14:]
    atr = np.mean(atr_range)
    
    vol_ratio = volume[-1] / (np.mean(volume[-20:]) + 1)
    
    price_to_sma20 = close[-1] / sma_20 if sma_20 > 0 else 1.0
    momentum_5d = (close[-1] - close[-6]) / close[-6] if len(close) >= 6 else 0
    momentum_10d = (close[-1] - close[-11]) / close[-11] if len(close) >= 11 else 0
    
    return {
        'close': close[-1],
        'sma_10': sma_10,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'volatility_20d': vol_20,
        'rsi_14': rsi,
        'atr_14': atr,
        'volume_ratio': vol_ratio,
        'price_to_sma20': price_to_sma20,
        'momentum_5d': momentum_5d,
        'momentum_10d': momentum_10d,
        'trend_strength': (sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0,
    }

def compute_labels(df: pd.DataFrame, idx: int, horizon: int = 5) -> Dict[str, float]:
    """Compute future return labels."""
    if idx + horizon >= len(df):
        return {}
    
    current_close = df.iloc[idx]['close']
    future_close = df.iloc[idx + horizon]['close']
    future_high = df.iloc[idx:idx+horizon]['high'].max()
    future_low = df.iloc[idx:idx+horizon]['low'].min()
    
    future_return = (future_close - current_close) / current_close
    max_gain = (future_high - current_close) / current_close
    max_drawdown = (current_close - future_low) / current_close
    
    is_runner = 1.0 if max_gain >= 0.05 else 0.0
    is_mega_runner = 1.0 if max_gain >= 0.10 else 0.0
    is_moonshot = 1.0 if max_gain >= 0.50 else 0.0
    
    return {
        'future_return': future_return,
        'max_gain': max_gain,
        'max_drawdown': max_drawdown,
        'is_runner': is_runner,
        'is_mega_runner': is_mega_runner,
        'is_moonshot': is_moonshot,
    }

def generate_training_samples() -> List[Tuple[Dict, Dict]]:
    """Generate training samples from cached data."""
    samples = []
    
    available_symbols = [s for s in RUNNER_STOCKS if (CACHE_DIR / f"{s}.parquet").exists()]
    logger.info(f"Found {len(available_symbols)} symbols with cached data")
    
    for symbol in available_symbols:
        df = load_cached_bars(symbol)
        if len(df) < 60:
            continue
        
        for i in range(50, len(df) - 10, 3):
            window_df = df.iloc[i-50:i+1]
            features = compute_features(window_df)
            labels = compute_labels(df, i, horizon=5)
            
            if features and labels:
                features['symbol'] = symbol
                samples.append((features, labels))
    
    return samples

def train_runner_model(samples: List[Tuple[Dict, Dict]], target: str) -> Dict[str, Any]:
    """Train a gradient boosting model for runner detection."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    feature_cols = ['volatility_20d', 'rsi_14', 'volume_ratio', 'price_to_sma20', 
                    'momentum_5d', 'momentum_10d', 'trend_strength']
    
    X = []
    y = []
    
    for features, labels in samples:
        if all(k in features for k in feature_cols) and target in labels:
            row = [features[k] for k in feature_cols]
            X.append(row)
            y.append(int(labels[target]))
    
    X = np.array(X)
    y = np.array(y)
    
    runner_count = np.sum(y)
    non_runner_count = len(y) - runner_count
    logger.info(f"  {target}: {runner_count} positive / {non_runner_count} negative")
    
    if runner_count < 10:
        logger.warning(f"  Not enough positive samples for {target}")
        return {"error": "insufficient_positive_samples"}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'samples_trained': len(X_train),
        'samples_tested': len(X_test),
        'positive_samples': int(runner_count),
    }
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'target': target,
    }

def save_model(model_data: Dict[str, Any], name: str):
    """Save trained model to disk."""
    path = MODELS_DIR / f"{name}.pkl.gz"
    with gzip.open(path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"  Saved model to {path}")
    return path

def main():
    logger.info("=" * 60)
    logger.info("OFFLINE TRAINING - Using cached Polygon data")
    logger.info("Bypassing FRED/Finnhub rate limits")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    logger.info("\n[1/4] Loading cached data...")
    samples = generate_training_samples()
    logger.info(f"Generated {len(samples)} training samples")
    
    if len(samples) < 100:
        logger.error("Not enough training samples!")
        sys.exit(1)
    
    logger.info("\n[2/4] Training 5%+ Runner Model (apex_production)...")
    runner_result = train_runner_model(samples, 'is_runner')
    if 'error' not in runner_result:
        save_model(runner_result, 'apex_production_offline')
        logger.info(f"  Accuracy: {runner_result['metrics']['accuracy']:.2%}")
        logger.info(f"  Precision: {runner_result['metrics']['precision']:.2%}")
        logger.info(f"  Recall: {runner_result['metrics']['recall']:.2%}")
    
    logger.info("\n[3/4] Training 10%+ Mega Runner Model (mega_runners)...")
    mega_result = train_runner_model(samples, 'is_mega_runner')
    if 'error' not in mega_result:
        save_model(mega_result, 'mega_runners_offline')
        logger.info(f"  Accuracy: {mega_result['metrics']['accuracy']:.2%}")
        logger.info(f"  Precision: {mega_result['metrics']['precision']:.2%}")
        logger.info(f"  Recall: {mega_result['metrics']['recall']:.2%}")
    
    logger.info("\n[4/4] Training 50%+ Moonshot Model (moonshots)...")
    moonshot_result = train_runner_model(samples, 'is_moonshot')
    if 'error' not in moonshot_result:
        save_model(moonshot_result, 'moonshots_offline')
        logger.info(f"  Accuracy: {moonshot_result['metrics']['accuracy']:.2%}")
    else:
        logger.info("  Skipped - not enough 50%+ moves in cached data")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Time: {elapsed:.1f} seconds")
    logger.info(f"Samples: {len(samples)}")
    logger.info("=" * 60)
    
    print("\n=== TRAINING RESULTS ===")
    if 'error' not in runner_result:
        print(f"5%+ Runner Model: {runner_result['metrics']['accuracy']:.1%} accuracy, {runner_result['metrics']['positive_samples']} runners detected")
    if 'error' not in mega_result:
        print(f"10%+ Mega Runner: {mega_result['metrics']['accuracy']:.1%} accuracy, {mega_result['metrics']['positive_samples']} mega runners")
    if 'error' not in moonshot_result:
        print(f"50%+ Moonshot: {moonshot_result['metrics']['accuracy']:.1%} accuracy")
    print(f"\nTotal training time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
