#!/usr/bin/env python3
"""
scripts/run_high_vol_smallcaps_scan.py

High-Volatility Small-Cap / Micro-Cap / Penny Stock Runner Sweep
for QuantraCore Apex v9.0-A

Focuses on small, micro, nano, and penny stocks to identify
potential MonsterRunner candidates with elevated volatility signals.

Usage:
    python scripts/run_high_vol_smallcaps_scan.py --sample-size 500
    python scripts/run_high_vol_smallcaps_scan.py --sample-size 1000 --seed 99
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
    get_symbols_by_bucket,
    get_symbol_info,
    get_smallcap_symbols,
    get_low_float_symbols,
)
from src.quantracore_apex.config.scan_modes import (
    load_scan_mode,
    list_scan_modes,
)
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv


@dataclass
class SmallCapScanResult:
    """Single small-cap symbol scan result."""
    symbol: str
    quantrascore: float
    score_bucket: str
    regime: str
    risk_tier: str
    entropy_state: str
    suppression_state: str
    drift_state: str
    protocol_fired_count: int
    omega_overrides: List[str]
    market_cap_bucket: str
    mr_fuse_score: float
    volatility_ratio: float
    compression_score: float
    is_runner_candidate: bool
    window_hash: str
    scan_time_ms: float
    error: Optional[str] = None


def get_smallcap_universe() -> List[str]:
    """Get all small-cap focused symbols."""
    buckets = ["small", "micro", "nano", "penny"]
    symbols = get_symbols_by_bucket(buckets)
    
    if len(symbols) == 0:
        symbols = get_smallcap_symbols()
    
    if len(symbols) == 0:
        symbols = get_low_float_symbols(max_float_millions=50.0)
    
    return symbols


def sample_symbols(all_symbols: List[str], sample_size: int) -> List[str]:
    """Randomly sample symbols."""
    if sample_size >= len(all_symbols):
        return all_symbols[:]
    return random.sample(all_symbols, sample_size)


def scan_single_symbol(
    engine: ApexEngine,
    adapter: SyntheticAdapter,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> SmallCapScanResult:
    """Scan a single small-cap symbol."""
    start_time = time.time()
    
    try:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        if len(normalized_bars) < 50:
            return SmallCapScanResult(
                symbol=symbol,
                quantrascore=0.0,
                score_bucket="unknown",
                regime="unknown",
                risk_tier="unknown",
                entropy_state="unknown",
                suppression_state="unknown",
                drift_state="unknown",
                protocol_fired_count=0,
                omega_overrides=[],
                market_cap_bucket="unknown",
                mr_fuse_score=0.0,
                volatility_ratio=0.0,
                compression_score=0.0,
                is_runner_candidate=False,
                window_hash="",
                scan_time_ms=(time.time() - start_time) * 1000,
                error=f"Insufficient data: {len(normalized_bars)} bars",
            )
        
        result = engine.run_scan(normalized_bars, symbol)
        
        symbol_info = get_symbol_info(symbol)
        market_cap_bucket = symbol_info.market_cap_bucket if symbol_info else "unknown"
        
        volatility_ratio = result.microtraits.volatility_ratio if result.microtraits else 0.0
        compression_score = result.microtraits.compression_score if result.microtraits else 0.0
        
        mr_fuse_score = result.quantrascore * (1 + volatility_ratio / 10)
        
        is_runner = (
            result.quantrascore >= 60 or
            mr_fuse_score >= 65 or
            (volatility_ratio > 1.5 and result.quantrascore >= 50)
        )
        
        fired_count = sum(1 for p in result.protocol_results if p.fired) if result.protocol_results else 0
        omega_overrides = list(result.omega_overrides.keys()) if result.omega_overrides else []
        
        return SmallCapScanResult(
            symbol=symbol,
            quantrascore=result.quantrascore,
            score_bucket=result.score_bucket.value if hasattr(result.score_bucket, 'value') else str(result.score_bucket),
            regime=result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
            risk_tier=result.risk_tier.value if hasattr(result.risk_tier, 'value') else str(result.risk_tier),
            entropy_state=result.entropy_state.value if hasattr(result.entropy_state, 'value') else str(result.entropy_state),
            suppression_state=result.suppression_state.value if hasattr(result.suppression_state, 'value') else str(result.suppression_state),
            drift_state=result.drift_state.value if hasattr(result.drift_state, 'value') else str(result.drift_state),
            protocol_fired_count=fired_count,
            omega_overrides=omega_overrides,
            market_cap_bucket=market_cap_bucket,
            mr_fuse_score=round(mr_fuse_score, 2),
            volatility_ratio=round(volatility_ratio, 3),
            compression_score=round(compression_score, 3),
            is_runner_candidate=is_runner,
            window_hash=result.window_hash,
            scan_time_ms=(time.time() - start_time) * 1000,
            error=None,
        )
        
    except Exception as e:
        return SmallCapScanResult(
            symbol=symbol,
            quantrascore=0.0,
            score_bucket="unknown",
            regime="unknown",
            risk_tier="unknown",
            entropy_state="unknown",
            suppression_state="unknown",
            drift_state="unknown",
            protocol_fired_count=0,
            omega_overrides=[],
            market_cap_bucket="unknown",
            mr_fuse_score=0.0,
            volatility_ratio=0.0,
            compression_score=0.0,
            is_runner_candidate=False,
            window_hash="",
            scan_time_ms=(time.time() - start_time) * 1000,
            error=str(e),
        )


def summarize_results(results: List[SmallCapScanResult]) -> Dict[str, Any]:
    """Generate summary statistics."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    runners = [r for r in successful if r.is_runner_candidate]
    
    if not successful:
        return {
            "total_scanned": len(results),
            "successful": 0,
            "failed": len(failed),
            "runner_candidates_found": 0,
            "avg_quantrascore": None,
            "max_quantrascore": None,
            "avg_volatility_ratio": None,
            "regime_distribution": {},
            "bucket_distribution": {},
            "top_runners": [],
        }
    
    scores = [r.quantrascore for r in successful]
    vol_ratios = [r.volatility_ratio for r in successful]
    
    regime_dist = {}
    bucket_dist = {}
    
    for r in successful:
        regime_dist[r.regime] = regime_dist.get(r.regime, 0) + 1
        bucket_dist[r.market_cap_bucket] = bucket_dist.get(r.market_cap_bucket, 0) + 1
    
    top_runners = sorted(
        successful,
        key=lambda r: r.mr_fuse_score,
        reverse=True,
    )[:50]
    
    return {
        "total_scanned": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "runner_candidates_found": len(runners),
        "avg_quantrascore": round(sum(scores) / len(scores), 2),
        "max_quantrascore": round(max(scores), 2),
        "min_quantrascore": round(min(scores), 2),
        "avg_volatility_ratio": round(sum(vol_ratios) / len(vol_ratios), 3),
        "max_volatility_ratio": round(max(vol_ratios), 3),
        "regime_distribution": regime_dist,
        "bucket_distribution": bucket_dist,
        "runner_candidates": [
            {
                "symbol": r.symbol,
                "quantrascore": r.quantrascore,
                "mr_fuse_score": r.mr_fuse_score,
                "volatility_ratio": r.volatility_ratio,
                "regime": r.regime,
                "market_cap_bucket": r.market_cap_bucket,
                "risk_tier": r.risk_tier,
            }
            for r in top_runners
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run high-vol small-cap runner sweep for QuantraCore Apex."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of symbols to scan (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible sampling.",
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
        help="Delay between batches (default: 0.1s).",
    )
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"[INFO] Using seed: {args.seed}")
    
    print("=" * 70)
    print("QUANTRACORE APEX v9.0-A — HIGH-VOL SMALLCAP RUNNER SWEEP")
    print("=" * 70)
    print()
    
    all_symbols = get_smallcap_universe()
    print(f"[INFO] Small-cap universe size: {len(all_symbols)}")
    
    if len(all_symbols) == 0:
        from src.quantracore_apex.config.symbol_universe import get_all_symbols
        all_symbols = get_all_symbols()
        all_symbols = [s for i, s in enumerate(all_symbols) if i % 3 == 0]
        print(f"[INFO] Fallback to general universe sample: {len(all_symbols)}")
    
    sampled = sample_symbols(all_symbols, args.sample_size)
    print(f"[INFO] Sampled {len(sampled)} symbols for this sweep")
    print(f"[INFO] Batch size: {args.batch_size}")
    print()
    
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("logs/high_vol_smallcaps") / ts
    base_dir.mkdir(parents=True, exist_ok=True)
    
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=args.seed or 99)
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=200)
    
    results: List[SmallCapScanResult] = []
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
        runners_in_batch = sum(1 for r in results[-len(batch):] if r.is_runner_candidate)
        print(f"Done in {batch_time:.2f}s ({runners_in_batch} runners)")
        
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
    runners_only = [asdict(r) for r in results if r.is_runner_candidate]
    
    with (base_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_dicts, f, indent=2, sort_keys=True)
    
    with (base_dir / "failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    
    with (base_dir / "runners.json").open("w", encoding="utf-8") as f:
        json.dump(runners_only, f, indent=2, sort_keys=True)
    
    with (base_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    print("=" * 70)
    print("SMALL-CAP RUNNER SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Total scanned:        {summary['total_scanned']}")
    print(f"  Successful:           {summary['successful']}")
    print(f"  Failed:               {summary['failed']}")
    print(f"  Runner candidates:    {summary['runner_candidates_found']}")
    print(f"  Avg QuantraScore:     {summary['avg_quantrascore']}")
    print(f"  Max QuantraScore:     {summary['max_quantrascore']}")
    print(f"  Avg Volatility Ratio: {summary['avg_volatility_ratio']}")
    print()
    
    print("Market Cap Distribution:")
    for bucket, count in sorted(summary['bucket_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {bucket}: {count}")
    print()
    
    print("Top 10 Runner Candidates:")
    for i, runner in enumerate(summary['runner_candidates'][:10], 1):
        print(f"  {i:2}. {runner['symbol']:6} | QS: {runner['quantrascore']:5.1f} | "
              f"MR: {runner['mr_fuse_score']:5.1f} | Vol: {runner['volatility_ratio']:.3f} | "
              f"{runner['market_cap_bucket']}")
    print()
    
    print("=" * 70)
    print(f"[OUTPUT] Results:  {base_dir / 'results.json'}")
    print(f"[OUTPUT] Runners:  {base_dir / 'runners.json'}")
    print(f"[OUTPUT] Summary:  {base_dir / 'summary.json'}")
    print("=" * 70)
    print()
    print("[INFO] High-vol small-cap runner sweep complete.")


if __name__ == "__main__":
    main()
