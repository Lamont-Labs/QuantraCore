#!/usr/bin/env python3
"""ApexLab + ApexCore training/inference pipeline validation."""

import sys
sys.path.insert(0, ".")

from datetime import datetime, timedelta
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder
from src.quantracore_apex.apexlab.features import FeatureExtractor
from src.quantracore_apex.apexlab.labels import LabelGenerator

def validate_apexcore_pipeline():
    """Validate the full ApexLab -> ApexCore pipeline."""
    print("Validating ApexLab components...")
    
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=42)
    window_builder = WindowBuilder(window_size=100, step=1, min_bars=100)
    feature_extractor = FeatureExtractor()
    label_generator = LabelGenerator(enable_logging=False)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=200)
    
    print("  1. Testing WindowBuilder...")
    windows = []
    for symbol in symbols:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        window = window_builder.build_single(normalized_bars, symbol)
        
        if window is None:
            print(f"    FAIL: WindowBuilder returned None for {symbol}")
            return False
        
        windows.append((symbol, window))
    print(f"    OK: Built {len(windows)} windows")
    
    print("  2. Testing FeatureExtractor...")
    features_list = []
    for symbol, window in windows:
        features = feature_extractor.extract(window)
        
        if features is None:
            print(f"    FAIL: FeatureExtractor returned None for {symbol}")
            return False
        
        if len(features) == 0:
            print(f"    FAIL: FeatureExtractor returned empty features for {symbol}")
            return False
        
        features_list.append((symbol, features))
    print(f"    OK: Extracted features for {len(features_list)} symbols")
    
    print("  3. Testing LabelGenerator...")
    labels_list = []
    for symbol, window in windows:
        label = label_generator.generate(window)
        
        if label is None:
            print(f"    FAIL: LabelGenerator returned None for {symbol}")
            return False
        
        if "quantrascore_numeric" not in label:
            print(f"    FAIL: LabelGenerator missing quantrascore_numeric for {symbol}")
            return False
        
        labels_list.append((symbol, label))
    print(f"    OK: Generated labels for {len(labels_list)} symbols")
    
    print("  4. Testing pipeline determinism...")
    for symbol, window in windows[:2]:
        features1 = feature_extractor.extract(window)
        features2 = feature_extractor.extract(window)
        
        label1 = label_generator.generate(window)
        label2 = label_generator.generate(window)
        
        import numpy as np
        if not np.allclose(features1, features2):
            print(f"    FAIL: Features not deterministic for {symbol}")
            return False
        
        if label1["quantrascore_numeric"] != label2["quantrascore_numeric"]:
            print(f"    FAIL: Labels not deterministic for {symbol}")
            return False
    print("    OK: Pipeline is deterministic")
    
    print("  5. Testing dataset construction...")
    dataset = []
    for i, (symbol, window) in enumerate(windows):
        features = features_list[i][1]
        label = labels_list[i][1]
        
        dataset.append({
            "symbol": symbol,
            "features": features,
            "quantrascore": label["quantrascore_numeric"],
            "regime_class": label["regime_class"]
        })
    
    if len(dataset) != len(symbols):
        print(f"    FAIL: Dataset size mismatch: {len(dataset)} vs {len(symbols)}")
        return False
    print(f"    OK: Constructed dataset with {len(dataset)} samples")
    
    print("\nApexLab/ApexCore pipeline validation: PASSED")
    return True

if __name__ == "__main__":
    success = validate_apexcore_pipeline()
    sys.exit(0 if success else 1)
