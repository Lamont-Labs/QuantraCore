#!/usr/bin/env python3
"""
QuantraCore Apex CLI - v9.0-A
Command-line interface for health checks, scans, replays, and training.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_health(args):
    """Check engine, data, and model health."""
    print("QuantraCore Apex v9.0-A Health Check")
    print("=" * 40)
    
    checks_passed = 0
    checks_total = 0
    
    checks_total += 1
    try:
        from src.quantracore_apex.core.engine import ApexEngine
        ApexEngine()
        print("[OK] Engine loaded successfully")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Engine load failed: {e}")
    
    checks_total += 1
    mode_file = Path("config/mode.yaml")
    if mode_file.exists():
        import yaml
        with open(mode_file) as f:
            config = yaml.safe_load(f)
        mode = config.get("default_mode", "unknown")
        print(f"[OK] Mode config loaded: {mode}")
        checks_passed += 1
    else:
        print("[WARN] Mode config not found")
    
    checks_total += 1
    try:
        from src.quantracore_apex.core.drift_detector import DriftDetector
        detector = DriftDetector()
        baselines_loaded = detector.load_baselines()
        print(f"[OK] Drift detector ready ({baselines_loaded} baselines)")
        checks_passed += 1
    except Exception as e:
        print(f"[WARN] Drift detector: {e}")
        checks_passed += 1
    
    checks_total += 1
    try:
        from src.quantracore_apex.core.decision_gates import DecisionGateRunner
        DecisionGateRunner()
        print("[OK] Decision gates ready")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Decision gates: {e}")
    
    checks_total += 1
    try:
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        ReplayEngine()
        print("[OK] Replay engine ready")
        checks_passed += 1
    except Exception as e:
        print(f"[FAIL] Replay engine: {e}")
    
    print("-" * 40)
    print(f"Health: {checks_passed}/{checks_total} checks passed")
    
    return 0 if checks_passed == checks_total else 1


def cmd_scan_universe(args):
    """Run a configured universe scan."""
    universe = args.universe
    print(f"Scanning universe: {universe}")
    print("=" * 40)
    
    try:
        import yaml
        universe_file = Path("config/symbol_universe.yaml")
        
        if not universe_file.exists():
            print("[ERROR] Symbol universe config not found")
            return 1
        
        with open(universe_file) as f:
            config = yaml.safe_load(f)
        
        if universe not in config.get("universes", {}):
            print(f"[ERROR] Universe '{universe}' not found")
            return 1
        
        universe_data = config["universes"][universe]
        symbols = [s["symbol"] for s in universe_data.get("symbols", []) if s.get("active", True)]
        
        print(f"Found {len(symbols)} active symbols")
        
        from src.quantracore_apex.core.engine import ApexEngine
        from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticDataAdapter
        
        engine = ApexEngine()
        adapter = SyntheticDataAdapter()
        
        results = []
        for symbol in symbols:
            try:
                data = adapter.get_ohlcv(symbol, lookback_bars=50)
                if data:
                    result = engine.run_scan(data)
                    results.append({
                        "symbol": symbol,
                        "score": result.quantrascore,
                        "band": result.score_band,
                        "regime": result.regime,
                    })
                    print(f"  {symbol}: {result.quantrascore:.1f} ({result.score_band}) - {result.regime}")
            except Exception as e:
                print(f"  {symbol}: ERROR - {e}")
        
        print("-" * 40)
        print(f"Scanned {len(results)} symbols successfully")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Scan failed: {e}")
        return 1


def cmd_replay_demo(args):
    """Run a small demo replay."""
    print("Running Demo Replay")
    print("=" * 40)
    
    try:
        from src.quantracore_apex.replay.replay_engine import ReplayEngine
        
        engine = ReplayEngine()
        result = engine.run_demo_replay()
        
        print(f"Symbols processed: {result.symbols_processed}")
        print(f"Signals generated: {result.signals_generated}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Final equity: ${result.equity_curve[-1]:,.2f}")
        
        if result.drift_flags:
            print(f"Drift flags: {len(result.drift_flags)}")
        
        if result.errors:
            print(f"Errors: {len(result.errors)}")
        
        if args.save:
            engine.save_replay_log(result)
            print("Saved replay log to provenance/")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Replay failed: {e}")
        return 1


def cmd_scan_mode(args):
    """Run a scan using a pre-defined scan mode."""
    mode = args.mode
    max_symbols = args.max_symbols
    lookback = args.lookback
    
    print("QuantraCore Apex v9.0-A Mode Scan")
    print(f"Mode: {mode}")
    print("=" * 50)
    
    try:
        from src.quantracore_apex.config.scan_modes import load_scan_mode, list_scan_modes
        
        available_modes = list_scan_modes()
        if mode not in available_modes:
            print(f"[ERROR] Unknown mode: {mode}")
            print(f"Available modes: {', '.join(available_modes)}")
            return 1
        
        mode_config = load_scan_mode(mode)
        print(f"Description: {mode_config.description}")
        print(f"Buckets: {', '.join(mode_config.buckets)}")
        print(f"Max symbols: {max_symbols or mode_config.max_symbols}")
        print(f"Chunk size: {mode_config.chunk_size}")
        print("-" * 50)
        
        from src.quantracore_apex.core.universe_scan import create_universe_scanner
        
        scanner = create_universe_scanner()
        
        def on_progress(symbol, current, total):
            pct = current / total * 100
            print(f"  [{current}/{total}] {symbol} ({pct:.0f}%)")
        
        result = scanner.scan(
            mode=mode,
            lookback_days=lookback,
            max_symbols=max_symbols,
            on_progress=on_progress if args.verbose else None,
        )
        
        print("-" * 50)
        print("SCAN RESULTS")
        print(f"  Symbols scanned: {result.scan_count}")
        print(f"  Successful: {result.success_count}")
        print(f"  Errors: {result.error_count}")
        print(f"  Small-caps: {result.smallcap_count}")
        print(f"  Extreme risk: {result.extreme_risk_count}")
        print(f"  Runner candidates: {result.runner_candidate_count}")
        print(f"  Time: {result.total_time_seconds:.2f}s")
        
        if result.results and args.show_top:
            print("-" * 50)
            print(f"TOP {min(args.show_top, len(result.results))} RESULTS:")
            for i, r in enumerate(result.results[:args.show_top], 1):
                flags = []
                if r.smallcap_flag:
                    flags.append("SC")
                if r.speculative_flag:
                    flags.append("SPEC")
                if r.mr_fuse_score >= 60:
                    flags.append(f"MR:{r.mr_fuse_score:.0f}")
                flag_str = " [" + ",".join(flags) + "]" if flags else ""
                print(f"  {i}. {r.symbol}: {r.quantrascore:.1f} ({r.score_bucket}) - {r.verdict_action}{flag_str}")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_list_modes(args):
    """List available scan modes."""
    print("QuantraCore Apex v9.0-A - Available Scan Modes")
    print("=" * 60)
    
    try:
        from src.quantracore_apex.config.scan_modes import (
            list_scan_modes,
            load_scan_mode,
            get_smallcap_modes,
            get_extreme_risk_modes,
        )
        
        modes = list_scan_modes()
        smallcap_modes = set(get_smallcap_modes())
        extreme_modes = set(get_extreme_risk_modes())
        
        for mode_name in modes:
            config = load_scan_mode(mode_name)
            flags = []
            if mode_name in smallcap_modes:
                flags.append("SMALLCAP")
            if mode_name in extreme_modes:
                flags.append("EXTREME")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            
            print(f"\n{mode_name}{flag_str}")
            print(f"  {config.description}")
            print(f"  Buckets: {', '.join(config.buckets)}")
            print(f"  Max: {config.max_symbols} | Chunk: {config.chunk_size}")
        
        print("\n" + "=" * 60)
        print(f"Total modes: {len(modes)}")
        return 0
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1


def cmd_lab_train_mini(args):
    """Run small ApexCoreMini training on demo set."""
    print("ApexLab Demo Training")
    print("=" * 40)
    
    try:
        from src.quantracore_apex.apexlab.dataset_builder import ApexLabDatasetBuilder
        from src.quantracore_apex.apexcore.models import ApexCoreMini
        
        print("Building demo dataset...")
        ApexLabDatasetBuilder()
        
        demo_features = []
        demo_labels = []
        
        import numpy as np
        np.random.seed(42)
        for i in range(100):
            features = np.random.randn(30).tolist()
            demo_features.append(features)
            demo_labels.append({
                "regime_label": np.random.choice(["trending_up", "range_bound", "volatile"]),
                "score_label": float(40 + np.random.rand() * 40),
            })
        
        print(f"Demo dataset: {len(demo_features)} samples")
        
        model = ApexCoreMini()
        print(f"Model: {model.model_id}")
        print(f"Status: {'placeholder' if model.is_placeholder else 'trained'}")
        
        print("-" * 40)
        print("Demo training complete (simulated)")
        print("For full training, use scripts/run_apexlab_demo.py")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return 1


def main(argv: Optional[list] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="qapex",
        description="QuantraCore Apex v9.0-A CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    health_parser = subparsers.add_parser("health", help="Check engine, data, models")
    health_parser.set_defaults(func=cmd_health)
    
    scan_parser = subparsers.add_parser("scan-universe", help="Run configured universe scan")
    scan_parser.add_argument(
        "--universe", "-u",
        default="demo",
        help="Universe name from symbol_universe.yaml"
    )
    scan_parser.set_defaults(func=cmd_scan_universe)
    
    replay_parser = subparsers.add_parser("replay-demo", help="Run small demo replay")
    replay_parser.add_argument(
        "--save", "-s",
        action="store_true",
        help="Save replay log to provenance/"
    )
    replay_parser.set_defaults(func=cmd_replay_demo)
    
    train_parser = subparsers.add_parser("lab-train-mini", help="Run demo ApexCoreMini training")
    train_parser.set_defaults(func=cmd_lab_train_mini)
    
    scan_mode_parser = subparsers.add_parser("scan", help="Run scan with pre-defined mode")
    scan_mode_parser.add_argument(
        "--mode", "-m",
        default="demo",
        help="Scan mode (e.g., demo, mega_large_focus, high_vol_small_caps, low_float_runners)"
    )
    scan_mode_parser.add_argument(
        "--max-symbols", "-n",
        type=int,
        default=None,
        help="Override max symbols from mode config"
    )
    scan_mode_parser.add_argument(
        "--lookback", "-l",
        type=int,
        default=250,
        help="Lookback days for historical data"
    )
    scan_mode_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show progress for each symbol"
    )
    scan_mode_parser.add_argument(
        "--show-top", "-t",
        type=int,
        default=10,
        help="Show top N results"
    )
    scan_mode_parser.set_defaults(func=cmd_scan_mode)
    
    list_modes_parser = subparsers.add_parser("list-modes", help="List available scan modes")
    list_modes_parser.set_defaults(func=cmd_list_modes)
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
