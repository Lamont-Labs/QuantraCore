#!/usr/bin/env python3
"""
ApexLab Demo Training Script.

Demonstrates the complete ApexLab training pipeline:
1. Generate synthetic data
2. Build training dataset
3. Train ApexCore demo model
4. Validate model alignment
"""

import sys
sys.path.insert(0, ".")

import logging

from src.quantracore_apex.apexlab.train_apexcore_demo import ApexCoreDemoTrainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_demo_training():
    """Run the demo training pipeline."""
    
    logger.info("=" * 60)
    logger.info("ApexLab — Demo Training Pipeline")
    logger.info("=" * 60)
    logger.info("")
    
    trainer = ApexCoreDemoTrainer()
    
    logger.info("Step 1: Generating demo data...")
    dataset = trainer.generate_demo_data(
        symbols=["DEMO1", "DEMO2", "DEMO3"],
        n_bars=200
    )
    logger.info(f"  Generated {dataset['metadata']['n_samples']} training samples")
    logger.info("")
    
    logger.info("Step 2: Training models...")
    metrics = trainer.train(dataset)
    logger.info(f"  Regime accuracy: {metrics['metrics']['regime_accuracy']:.4f}")
    logger.info(f"  Risk accuracy: {metrics['metrics']['risk_accuracy']:.4f}")
    logger.info(f"  QuantraScore MAE: {metrics['metrics']['score_mae']:.4f}")
    logger.info("")
    
    logger.info("Step 3: Saving models...")
    model_path = trainer.save()
    logger.info(f"  Model saved to: {model_path}")
    logger.info("")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    
    return model_path


if __name__ == "__main__":
    run_demo_training()
