#!/usr/bin/env python3
"""
ApexCore Validation Script.

Validates ApexCore model alignment with Apex engine teacher.
"""

import sys
sys.path.insert(0, ".")

import logging
from datetime import datetime, timedelta

from src.quantracore_apex.apexcore.interface import ApexCoreFull
from src.quantracore_apex.apexlab.validation import AlignmentValidator
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_validation(model_path: str = "data/training/models/apexcore_demo.pkl"):
    """Run ApexCore validation."""
    
    logger.info("=" * 60)
    logger.info("ApexCore — Alignment Validation")
    logger.info("=" * 60)
    logger.info("")
    
    logger.info("Loading model...")
    model = ApexCoreFull()
    try:
        model.load(model_path)
    except FileNotFoundError:
        logger.error(f"Model not found at {model_path}")
        logger.info("Run scripts/run_apexlab_demo.py first to train a model.")
        return None
    
    logger.info("Generating validation windows...")
    adapter = SyntheticAdapter(seed=99)
    window_builder = WindowBuilder(window_size=100, step=20)
    
    end_date = datetime(2024, 1, 1)
    start_date = end_date - timedelta(days=200)
    
    all_windows = []
    for symbol in ["VAL1", "VAL2", "VAL3"]:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        windows = window_builder.build(normalized_bars, symbol)
        all_windows.extend(windows[:5])
    
    logger.info(f"  Validation set: {len(all_windows)} windows")
    logger.info("")
    
    logger.info("Running validation...")
    validator = AlignmentValidator(model)
    report = validator.validate(all_windows)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Validation Results")
    logger.info("=" * 60)
    logger.info(f"  Samples validated: {report['n_samples']}")
    logger.info(f"  Regime accuracy: {report['metrics']['regime_accuracy']:.4f}")
    logger.info(f"  Risk accuracy: {report['metrics']['risk_accuracy']:.4f}")
    logger.info(f"  QuantraScore MAE: {report['metrics']['score_mae']:.4f}")
    logger.info(f"  QuantraScore RMSE: {report['metrics']['score_rmse']:.4f}")
    logger.info("")
    logger.info(f"  PASSED: {report['passed']}")
    
    report_path = validator.save_report(report)
    logger.info(f"  Report saved to: {report_path}")
    logger.info("=" * 60)
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/training/models/apexcore_demo.pkl")
    args = parser.parse_args()
    
    run_validation(args.model)
