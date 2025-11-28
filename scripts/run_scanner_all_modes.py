#!/usr/bin/env python3
"""Universal scanner validation across all 8 modes."""

import sys
sys.path.insert(0, ".")

import yaml
from datetime import datetime, timedelta
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv

def load_scan_modes():
    """Load scan modes from config."""
    with open("config/scan_modes.yaml", "r") as f:
        return yaml.safe_load(f)

def load_symbol_universe():
    """Load symbol universe from config."""
    with open("config/symbol_universe.yaml", "r") as f:
        return yaml.safe_load(f)

def run_scanner_all_modes():
    """Validate scanner across all modes with synthetic data."""
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=42)
    
    scan_modes = load_scan_modes()
    symbol_universe = load_symbol_universe()
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=60)
    
    mode_names = list(scan_modes.get("modes", {}).keys())
    if not mode_names:
        mode_names = ["mega_cap", "large_cap", "mid_cap", "small_cap", 
                      "micro_cap", "nano_cap", "penny_stock", "full_universe"]
    
    bucket_keys = list(symbol_universe.get("buckets", {}).keys())
    if not bucket_keys:
        bucket_keys = ["mega_cap", "large_cap", "mid_cap", "small_cap", 
                       "micro_cap", "nano_cap", "penny_stock"]
    
    print(f"Testing {len(mode_names)} scan modes across {len(bucket_keys)} market cap buckets")
    
    test_symbols = {}
    for bucket in bucket_keys:
        bucket_data = symbol_universe.get("buckets", {}).get(bucket, {})
        symbols = bucket_data.get("symbols", [])[:3]
        if not symbols:
            symbols = [f"{bucket.upper()[:4]}{i}" for i in range(1, 4)]
        test_symbols[bucket] = symbols
    
    total_scans = 0
    successful_scans = 0
    
    for mode in mode_names:
        print(f"  Testing mode: {mode}")
        
        if mode == "full_universe":
            symbols_to_test = []
            for bucket_symbols in test_symbols.values():
                symbols_to_test.extend(bucket_symbols[:1])
        else:
            bucket_key = mode.replace("_cap", "_cap") if "_cap" in mode else mode
            symbols_to_test = test_symbols.get(bucket_key, test_symbols.get(list(test_symbols.keys())[0], ["TEST1"]))
        
        for symbol in symbols_to_test[:2]:
            total_scans += 1
            try:
                bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
                normalized_bars, _ = normalize_ohlcv(bars)
                result = engine.run_scan(normalized_bars, symbol)
                
                if result is None:
                    print(f"    FAIL: {symbol} returned None")
                    continue
                
                if not (0 <= result.quantrascore <= 100):
                    print(f"    FAIL: {symbol} quantrascore out of range: {result.quantrascore}")
                    continue
                
                successful_scans += 1
                
            except Exception as e:
                print(f"    FAIL: {symbol} raised exception: {e}")
                continue
    
    print(f"\nScanner validation: {successful_scans}/{total_scans} successful")
    
    if successful_scans < total_scans * 0.9:
        print("FAIL: Too many scan failures")
        return False
    
    return True

if __name__ == "__main__":
    success = run_scanner_all_modes()
    sys.exit(0 if success else 1)
