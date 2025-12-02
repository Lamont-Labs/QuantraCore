#!/usr/bin/env python3
"""
MOONSHOT-ONLY TRAINING - Trains exclusively on 50%+ gain patterns.
Uses the moonshot database to build a specialized detector.
"""

import os
import sys
import logging
import pickle
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

MOONSHOT_DIR = Path("data/moonshots")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_moonshot_features():
    """Load saved moonshot features."""
    path = MOONSHOT_DIR / "moonshot_features_latest.pkl.gz"
    if not path.exists():
        logger.error(f"Moonshot features not found at {path}")
        return []
    
    with gzip.open(path, "rb") as f:
        features = pickle.load(f)
    
    logger.info(f"Loaded {len(features)} moonshot feature sets")
    return features

def categorize_gains(features):
    """Categorize moonshots by gain size."""
    categories = {
        "50_100": [],   # 50-100%
        "100_200": [],  # 100-200%
        "200_500": [],  # 200-500%
        "500_plus": [], # 500%+
    }
    
    for f in features:
        gain = f.get("gain_pct", 0)
        if gain >= 500:
            categories["500_plus"].append(f)
        elif gain >= 200:
            categories["200_500"].append(f)
        elif gain >= 100:
            categories["100_200"].append(f)
        else:
            categories["50_100"].append(f)
    
    return categories

def train_moonshot_classifier(features):
    """Train a classifier to recognize moonshot patterns."""
    feature_cols = [
        "volatility_20d", "volume_ratio", "price_to_sma20",
        "momentum_5d", "momentum_10d", "trend_strength", "consolidation"
    ]
    
    X = []
    y_100 = []
    y_200 = []
    y_500 = []
    
    for f in features:
        if not all(k in f for k in feature_cols):
            continue
        
        row = [f[k] for k in feature_cols]
        if any(np.isnan(v) or np.isinf(v) for v in row):
            continue
        
        gain = f.get("gain_pct", 0)
        
        X.append(row)
        y_100.append(1 if gain >= 100 else 0)
        y_200.append(1 if gain >= 200 else 0)
        y_500.append(1 if gain >= 500 else 0)
    
    X = np.array(X)
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"  100%+ samples: {sum(y_100)}")
    logger.info(f"  200%+ samples: {sum(y_200)}")
    logger.info(f"  500%+ samples: {sum(y_500)}")
    
    models = {}
    
    for name, y in [("100_plus", y_100), ("200_plus", y_200), ("500_plus", y_500)]:
        y = np.array(y)
        if sum(y) < 20:
            logger.warning(f"Not enough positive samples for {name}")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "positive_samples": int(sum(y)),
            "total_samples": len(y),
        }
        
        models[name] = {
            "model": model,
            "feature_cols": feature_cols,
            "metrics": metrics,
        }
        
        logger.info(f"\n{name} Model:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Precision: {metrics['precision']:.2%}")
        logger.info(f"  Recall: {metrics['recall']:.2%}")
        logger.info(f"  F1 Score: {metrics['f1']:.2%}")
    
    return models

def save_models(models):
    """Save trained models."""
    for name, data in models.items():
        path = MODELS_DIR / f"moonshot_{name}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {name} model to {path}")
    
    combined_path = MODELS_DIR / "moonshot_detector_latest.pkl.gz"
    with gzip.open(combined_path, "wb") as f:
        pickle.dump(models, f)
    logger.info(f"Saved combined detector to {combined_path}")

def main():
    logger.info("=" * 60)
    logger.info("MOONSHOT-ONLY TRAINING")
    logger.info("Training on 50%+ gain patterns exclusively")
    logger.info("=" * 60)
    
    features = load_moonshot_features()
    if not features:
        sys.exit(1)
    
    categories = categorize_gains(features)
    logger.info("\nGain Distribution:")
    for cat, items in categories.items():
        logger.info(f"  {cat}: {len(items)} samples")
    
    models = train_moonshot_classifier(features)
    
    if models:
        save_models(models)
        
        print("\n" + "=" * 60)
        print("MOONSHOT DETECTOR TRAINED!")
        print("=" * 60)
        for name, data in models.items():
            m = data["metrics"]
            print(f"{name}: {m['precision']:.1%} precision, {m['recall']:.1%} recall ({m['positive_samples']} samples)")
        print("=" * 60)
    else:
        logger.error("No models were trained!")

if __name__ == "__main__":
    main()
