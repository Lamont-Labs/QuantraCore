#!/usr/bin/env python3
"""
Train and Save ApexCore V3 Model.

Creates initial model artifacts for the ApexCore V3 prediction system.
Uses synthetic training data to bootstrap the models.
"""

import os
import sys
import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantracore_apex.prediction.apexcore_v3 import ApexCoreV3Model


def generate_training_data(n_samples: int = 500) -> list:
    """Generate synthetic training data for model bootstrap."""
    
    regimes = ["trend_up", "trend_down", "chop", "squeeze", "volatile"]
    entropy_bands = ["low", "mid", "high"]
    suppression_states = ["none", "suppressed", "blocked"]
    volatility_bands = ["low", "mid", "high"]
    liquidity_bands = ["low", "mid", "high"]
    risk_tiers = ["low", "medium", "high", "extreme"]
    quality_tiers = ["A", "B", "C", "D", "F"]
    
    tier_protocols = [f"T{i:02d}" for i in range(1, 81)]
    monster_protocols = [f"MR{i:02d}" for i in range(1, 21)]
    learning_protocols = [f"LP{i:02d}" for i in range(1, 26)]
    
    all_protocols = tier_protocols + monster_protocols + learning_protocols
    
    rows = []
    random.seed(42)
    np.random.seed(42)
    
    for i in range(n_samples):
        regime = random.choice(regimes)
        quantra_score = np.clip(np.random.normal(55, 20), 0, 100)
        
        if regime == "trend_up":
            ret_1d = np.random.normal(0.5, 1.0)
            ret_3d = np.random.normal(1.5, 2.0)
            ret_5d = np.random.normal(2.5, 3.0)
        elif regime == "trend_down":
            ret_1d = np.random.normal(-0.5, 1.0)
            ret_3d = np.random.normal(-1.5, 2.0)
            ret_5d = np.random.normal(-2.5, 3.0)
        else:
            ret_1d = np.random.normal(0, 0.8)
            ret_3d = np.random.normal(0, 1.5)
            ret_5d = np.random.normal(0, 2.5)
        
        num_protocols = random.randint(3, 25)
        protocol_ids = random.sample(all_protocols, num_protocols)
        
        hit_runner = 1 if (quantra_score > 70 and ret_5d > 5) else 0
        
        if quantra_score > 80 and ret_5d > 3:
            quality = "A"
        elif quantra_score > 65 and ret_5d > 1:
            quality = "B"
        elif quantra_score > 50:
            quality = "C"
        elif quantra_score > 35:
            quality = "D"
        else:
            quality = "F"
        
        avoid = 1 if (quantra_score < 30 or (regime == "volatile" and quantra_score < 50)) else 0
        
        max_runup = max(ret_1d, ret_3d, ret_5d, 0) + np.random.uniform(0, 2)
        max_drawdown = min(ret_1d, ret_3d, ret_5d, 0) - np.random.uniform(0, 2)
        
        row = {
            "symbol": f"SYM{i:04d}",
            "timestamp": (datetime.utcnow() - timedelta(days=random.randint(1, 365))).isoformat(),
            "quantra_score": quantra_score,
            "entropy_band": random.choice(entropy_bands),
            "suppression_state": random.choice(suppression_states),
            "regime_type": regime,
            "volatility_band": random.choice(volatility_bands),
            "liquidity_band": random.choice(liquidity_bands),
            "risk_tier": random.choice(risk_tiers),
            "protocol_ids": protocol_ids,
            "ret_1d": ret_1d,
            "ret_3d": ret_3d,
            "ret_5d": ret_5d,
            "max_runup_5d": max_runup,
            "max_drawdown_5d": max_drawdown,
            "hit_runner_threshold": hit_runner,
            "future_quality_tier": quality,
            "avoid_trade": avoid,
            "regime_label": regime,
        }
        rows.append(row)
    
    return rows


def train_and_save(model_size: str = "big", n_samples: int = 500):
    """Train ApexCore V3 model and save artifacts."""
    
    print(f"\n[ApexCoreV3] Training {model_size} model with {n_samples} samples...")
    
    enable_cal = n_samples >= 300
    
    model = ApexCoreV3Model(
        model_size=model_size,
        enable_calibration=enable_cal,
        enable_uncertainty=enable_cal,
        enable_multi_horizon=True,
    )
    
    print("[ApexCoreV3] Generating training data...")
    training_data = generate_training_data(n_samples)
    
    print("[ApexCoreV3] Training model...")
    metrics = model.fit(training_data)
    
    print(f"[ApexCoreV3] Training complete. Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    
    print("[ApexCoreV3] Saving model...")
    model.save()
    
    print(f"[ApexCoreV3] Model saved to models/apexcore_v3/{model_size}/")
    
    return model, metrics


def main():
    """Main entry point."""
    
    print("=" * 60)
    print("ApexCore V3 Model Training")
    print("=" * 60)
    
    big_model, big_metrics = train_and_save("big", n_samples=500)
    
    mini_model, mini_metrics = train_and_save("mini", n_samples=200)
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"\nBig Model:")
    print(f"  - QuantraScore RMSE: {big_metrics['quantrascore_rmse']:.4f}")
    print(f"  - Runner Accuracy: {big_metrics['runner_accuracy']:.4f}")
    print(f"  - Quality Accuracy: {big_metrics['quality_accuracy']:.4f}")
    
    print(f"\nMini Model:")
    print(f"  - QuantraScore RMSE: {mini_metrics['quantrascore_rmse']:.4f}")
    print(f"  - Runner Accuracy: {mini_metrics['runner_accuracy']:.4f}")
    print(f"  - Quality Accuracy: {mini_metrics['quality_accuracy']:.4f}")
    
    print("\n[ApexCoreV3] Bootstrap training complete!")
    print("Models ready at: models/apexcore_v3/")


if __name__ == "__main__":
    main()
