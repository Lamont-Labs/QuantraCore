#!/usr/bin/env python3
"""Nuclear determinism validation - single cycle."""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv

def run_nuclear_determinism_check():
    """Run a single cycle of nuclear determinism validation."""
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=42)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=60)
    
    results_run1 = []
    results_run2 = []
    
    for symbol in symbols:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        result1 = engine.run_scan(normalized_bars, symbol)
        result2 = engine.run_scan(normalized_bars, symbol)
        
        results_run1.append(result1)
        results_run2.append(result2)
    
    for i, symbol in enumerate(symbols):
        r1, r2 = results_run1[i], results_run2[i]
        
        if r1.quantrascore != r2.quantrascore:
            print(f"FAIL: {symbol} quantrascore mismatch: {r1.quantrascore} vs {r2.quantrascore}")
            return False
        
        if r1.verdict.action != r2.verdict.action:
            print(f"FAIL: {symbol} verdict action mismatch")
            return False
        
        if r1.window_hash != r2.window_hash:
            print(f"FAIL: {symbol} window hash mismatch")
            return False
        
        fired1 = sorted([p.protocol_id for p in r1.protocol_results if p.fired])
        fired2 = sorted([p.protocol_id for p in r2.protocol_results if p.fired])
        if fired1 != fired2:
            print(f"FAIL: {symbol} protocol fire sequence mismatch")
            return False
    
    return True

if __name__ == "__main__":
    success = run_nuclear_determinism_check()
    sys.exit(0 if success else 1)
