#!/usr/bin/env python3
"""
scripts/run_random_universe_scan.py

Random Universe Stress Scan for QuantraCore Apex v9.0-A

Runs a stress test scan over a random subset of the full US equity universe.
Uses the existing universal scanner with synthetic data for testing, or
live data providers when API keys are available.

Usage:
    python scripts/run_random_universe_scan.py --mode demo --sample-size 100
    python scripts/run_random_universe_scan.py --mode full_us_equities --sample-size 800
    python scripts/run_random_universe_scan.py --sample-size 500 --seed 42
"""

import sys
sys.path.insert(0, ".")

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.quantracore_apex.config.symbol_universe import (
    get_all_symbols,
    get_symbol_info,
    get_symbols_for_mode,
    load_symbol_universe,
    SymbolInfo,
)
from src.quantracore_apex.config.scan_modes import (
    load_scan_mode,
    list_scan_modes,
    ScanModeConfig,
)
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv


@dataclass
class ScanResultEntry:
    """Single symbol scan result."""
    symbol: str
    quantrascore: float
    score_bucket: str
    regime: str
    risk_tier: str
    entropy_state: str
    suppression_state: str
    drift_state: str
    protocol_fired_count: int
    omega_alerts: List[str]
    market_cap_bucket: str
    mr_fuse_score: float
    window_hash: str
    scan_time_ms: float
    error: Optional[str] = None


def sample_symbols(all_symbols: List[str], sample_size: int) -> List[str]:
    """Randomly sample symbols from the universe."""
    if sample_size >= len(all_symbols):
        return all_symbols[:]
    return random.sample(all_symbols, sample_size)


def scan_single_symbol(
    engine: ApexEngine,
    adapter: SyntheticAdapter,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> ScanResultEntry:
    """Scan a single symbol and return structured result."""
    start_time = time.time()
    
    try:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        if len(normalized_bars) < 50:
            return ScanResultEntry(
                symbol=symbol,
                quantrascore=0.0,
                score_bucket="unknown",
                regime="unknown",
                risk_tier="unknown",
                entropy_state="unknown",
                suppression_state="unknown",
                drift_state="unknown",
                protocol_fired_count=0,
                omega_alerts=[],
                market_cap_bucket="unknown",
                mr_fuse_score=0.0,
                window_hash="",
                scan_time_ms=(time.time() - start_time) * 1000,
                error=f"Insufficient data: {len(normalized_bars)} bars",
            )
        
        result = engine.run_scan(normalized_bars, symbol)
        
        symbol_info = get_symbol_info(symbol)
        market_cap_bucket = symbol_info.market_cap_bucket if symbol_info else "unknown"
        
        mr_fuse_score = 0.0
        if hasattr(result, 'microtraits') and result.microtraits:
            mr_fuse_score = result.quantrascore * (1 + result.microtraits.volatility_ratio / 10)
        
        fired_count = sum(1 for p in result.protocol_results if p.fired) if result.protocol_results else 0
        omega_alerts = list(result.omega_overrides.keys()) if result.omega_overrides else []
        
        return ScanResultEntry(
            symbol=symbol,
            quantrascore=result.quantrascore,
            score_bucket=result.score_bucket.value if hasattr(result.score_bucket, 'value') else str(result.score_bucket),
            regime=result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
            risk_tier=result.risk_tier.value if hasattr(result.risk_tier, 'value') else str(result.risk_tier),
            entropy_state=result.entropy_state.value if hasattr(result.entropy_state, 'value') else str(result.entropy_state),
            suppression_state=result.suppression_state.value if hasattr(result.suppression_state, 'value') else str(result.suppression_state),
            drift_state=result.drift_state.value if hasattr(result.drift_state, 'value') else str(result.drift_state),
            protocol_fired_count=fired_count,
            omega_alerts=omega_alerts,
            market_cap_bucket=market_cap_bucket,
            mr_fuse_score=round(mr_fuse_score, 2),
            window_hash=result.window_hash,
            scan_time_ms=(time.time() - start_time) * 1000,
            error=None,
        )
        
    except Exception as e:
        return ScanResultEntry(
            symbol=symbol,
            quantrascore=0.0,
            score_bucket="unknown",
            regime="unknown",
            risk_tier="unknown",
            entropy_state="unknown",
            suppression_state="unknown",
            drift_state="unknown",
            protocol_fired_count=0,
            omega_alerts=[],
            market_cap_bucket="unknown",
            mr_fuse_score=0.0,
            window_hash="",
            scan_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


def summarize_results(results: List[ScanResultEntry]) -> Dict[str, Any]:
    """Generate summary statistics from scan results."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    if not successful:
        return {
            "total_scanned": len(results),
            "successful": 0,
            "failed": len(failed),
            "avg_quantrascore": None,
            "max_quantrascore": None,
            "min_quantrascore": None,
            "regime_distribution": {},
            "bucket_distribution": {},
            "risk_tier_distribution": {},
            "runner_candidates": [],
            "avg_scan_time_ms": 0,
        }
    
    scores = [r.quantrascore for r in successful]
    
    regime_dist = {}
    bucket_dist = {}
    risk_dist = {}
    
    for r in successful:
        regime_dist[r.regime] = regime_dist.get(r.regime, 0) + 1
        bucket_dist[r.market_cap_bucket] = bucket_dist.get(r.market_cap_bucket, 0) + 1
        risk_dist[r.risk_tier] = risk_dist.get(r.risk_tier, 0) + 1
    
    runner_candidates = sorted(
        successful,
        key=lambda r: r.mr_fuse_score if r.mr_fuse_score > 0 else r.quantrascore,
        reverse=True,
    )[:50]
    
    avg_scan_time = sum(r.scan_time_ms for r in results) / len(results)
    
    return {
        "total_scanned": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "avg_quantrascore": round(sum(scores) / len(scores), 2),
        "max_quantrascore": round(max(scores), 2),
        "min_quantrascore": round(min(scores), 2),
        "regime_distribution": regime_dist,
        "bucket_distribution": bucket_dist,
        "risk_tier_distribution": risk_dist,
        "runner_candidates": [
            {
                "symbol": r.symbol,
                "quantrascore": r.quantrascore,
                "mr_fuse_score": r.mr_fuse_score,
                "regime": r.regime,
                "market_cap_bucket": r.market_cap_bucket,
                "risk_tier": r.risk_tier,
            }
            for r in runner_candidates
        ],
        "avg_scan_time_ms": round(avg_scan_time, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a random-universe QuantraCore Apex stress scan."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full_us_equities",
        help="Scan mode name from config/scan_modes.yaml (default: full_us_equities).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=800,
        help="Number of random symbols to scan (default: 800).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Symbols per batch (default: 50).",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=0.1,
        help="Delay between batches in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available scan modes and exit.",
    )
    args = parser.parse_args()
    
    if args.list_modes:
        print("Available scan modes:")
        for mode in list_scan_modes():
            print(f"  - {mode}")
        return
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Using seed: {args.seed}")
    
    print("=" * 70)
    print("QUANTRACORE APEX v9.0-A — RANDOM UNIVERSE STRESS SCAN")
    print("=" * 70)
    print()
    
    try:
        mode_config = load_scan_mode(args.mode)
        print(f"[INFO] Scan mode: {args.mode}")
        print(f"[INFO] Mode buckets: {mode_config.buckets}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        print(f"[INFO] Available modes: {list_scan_modes()}")
        return
    
    all_symbols = get_symbols_for_mode(args.mode)
    print(f"[INFO] Total symbols in mode universe: {len(all_symbols)}")
    
    if len(all_symbols) == 0:
        all_symbols = get_all_symbols()
        print(f"[INFO] Falling back to full universe: {len(all_symbols)} symbols")
    
    sampled = sample_symbols(all_symbols, args.sample_size)
    print(f"[INFO] Sampled {len(sampled)} symbols for this run")
    print(f"[INFO] Batch size: {args.batch_size}")
    print()
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("logs/random_scan") / ts
    base_dir.mkdir(parents=True, exist_ok=True)
    
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=args.seed or 42)
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=200)
    
    results: List[ScanResultEntry] = []
    total_batches = (len(sampled) + args.batch_size - 1) // args.batch_size
    overall_start = time.time()
    
    for batch_idx in range(0, len(sampled), args.batch_size):
        batch = sampled[batch_idx:batch_idx + args.batch_size]
        batch_num = batch_idx // args.batch_size + 1
        
        print(f"[BATCH {batch_num}/{total_batches}] Scanning {len(batch)} symbols...", end=" ")
        batch_start = time.time()
        
        for symbol in batch:
            result = scan_single_symbol(engine, adapter, symbol, start_date, end_date)
            results.append(result)
        
        batch_time = time.time() - batch_start
        successful_in_batch = sum(1 for r in results[-len(batch):] if r.error is None)
        print(f"Done in {batch_time:.2f}s ({successful_in_batch}/{len(batch)} OK)")
        
        if batch_idx + args.batch_size < len(sampled):
            time.sleep(args.batch_delay)
    
    total_time = time.time() - overall_start
    print()
    print(f"[INFO] Total scan time: {total_time:.2f}s")
    print(f"[INFO] Average per symbol: {(total_time / len(sampled) * 1000):.1f}ms")
    print()
    
    summary = summarize_results(results)
    
    results_dicts = [asdict(r) for r in results]
    failures = [asdict(r) for r in results if r.error is not None]
    
    results_path = base_dir / "results.json"
    failures_path = base_dir / "failures.json"
    summary_path = base_dir / "summary.json"
    
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results_dicts, f, indent=2, sort_keys=True)
    
    with failures_path.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    print("=" * 70)
    print("SCAN SUMMARY")
    print("=" * 70)
    print(f"  Total scanned:     {summary['total_scanned']}")
    print(f"  Successful:        {summary['successful']}")
    print(f"  Failed:            {summary['failed']}")
    print(f"  Avg QuantraScore:  {summary['avg_quantrascore']}")
    print(f"  Max QuantraScore:  {summary['max_quantrascore']}")
    print(f"  Min QuantraScore:  {summary['min_quantrascore']}")
    print()
    
    print("Regime Distribution:")
    for regime, count in sorted(summary['regime_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {regime}: {count}")
    print()
    
    print("Market Cap Distribution:")
    for bucket, count in sorted(summary['bucket_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {bucket}: {count}")
    print()
    
    print("Top 10 Runner Candidates:")
    for i, runner in enumerate(summary['runner_candidates'][:10], 1):
        print(f"  {i:2}. {runner['symbol']:6} | QS: {runner['quantrascore']:5.1f} | "
              f"MR: {runner['mr_fuse_score']:5.1f} | {runner['regime']:15} | {runner['market_cap_bucket']}")
    print()
    
    print("=" * 70)
    print(f"[OUTPUT] Results:  {results_path}")
    print(f"[OUTPUT] Failures: {failures_path}")
    print(f"[OUTPUT] Summary:  {summary_path}")
    print("=" * 70)
    print()
    print("[INFO] Random universe stress scan complete.")


if __name__ == "__main__":
    main()
