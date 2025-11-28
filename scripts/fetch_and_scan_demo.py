#!/usr/bin/env python3
"""
Fetch and Scan Demo Script for QuantraCore Apex.

Demonstrates the full analysis pipeline:
1. Fetches OHLCV data (synthetic for demo)
2. Normalizes and caches data
3. Runs Apex engine on windows
4. Prints summary of results
"""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta
import logging

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.prediction.monster_runner import MonsterRunnerEngine


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_demo():
    """Run the demo scan."""
    
    symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
    
    adapter = SyntheticAdapter(seed=42)
    window_builder = WindowBuilder(window_size=100)
    engine = ApexEngine(enable_logging=False)
    monster_runner = MonsterRunnerEngine()
    
    end_date = datetime(2024, 1, 1)
    start_date = end_date - timedelta(days=150)
    
    logger.info("=" * 60)
    logger.info("QuantraCore Apex — Demo Scan")
    logger.info("=" * 60)
    logger.info("")
    
    results = []
    
    for symbol in symbols:
        logger.info(f"Scanning {symbol}...")
        
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        window = window_builder.build_single(normalized_bars, symbol)
        
        if window is None:
            logger.warning(f"  Insufficient data for {symbol}")
            continue
        
        context = ApexContext(seed=42, compliance_mode=True)
        result = engine.run(window, context)
        
        mr_output = monster_runner.analyze(window)
        
        results.append({
            "symbol": symbol,
            "quantrascore": result.quantrascore,
            "regime": result.regime.value,
            "risk_tier": result.risk_tier.value,
            "entropy_state": result.entropy_state.value,
            "suppression_state": result.suppression_state.value,
            "runner_state": mr_output.runner_state.value,
            "runner_prob": mr_output.runner_probability,
        })
        
        logger.info(f"  QuantraScore: {result.quantrascore:.1f}")
        logger.info(f"  Regime: {result.regime.value}")
        logger.info(f"  Risk: {result.risk_tier.value}")
        logger.info(f"  MonsterRunner: {mr_output.runner_state.value} ({mr_output.runner_probability:.2f})")
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    results.sort(key=lambda x: x["quantrascore"], reverse=True)
    
    print("\n{:<8} {:>10} {:>15} {:>10} {:>15}".format(
        "Symbol", "Score", "Regime", "Risk", "Runner State"
    ))
    print("-" * 60)
    
    for r in results:
        print("{:<8} {:>10.1f} {:>15} {:>10} {:>15}".format(
            r["symbol"],
            r["quantrascore"],
            r["regime"],
            r["risk_tier"],
            r["runner_state"]
        ))
    
    print("\n" + "=" * 60)
    print("Compliance Note: All outputs are structural probabilities.")
    print("This is NOT trading advice.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
