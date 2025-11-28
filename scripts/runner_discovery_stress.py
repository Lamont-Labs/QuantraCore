#!/usr/bin/env python3
"""
scripts/runner_discovery_stress.py

Runner Discovery Stress Test for QuantraCore Apex v9.x

Stress tests runner detection across all market cap buckets:
    - microcaps
    - smallcaps  
    - midcaps
    - largecaps
    - megacaps

Verifies:
    - Runner detection logic
    - MR-Fuse pattern recognition
    - Volatility-gated scanning
    - ZDE interaction with high-vol assets
"""

import sys
sys.path.insert(0, ".")

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.config.symbol_universe import (
    get_symbols_by_bucket,
    get_symbol_info,
    get_all_symbols,
)


@dataclass
class RunnerResult:
    """Single runner discovery result."""
    symbol: str
    bucket: str
    quantrascore: float
    mr_fuse_score: float
    volatility_ratio: float
    compression_score: float
    is_runner_candidate: bool
    regime: str
    risk_tier: str
    entropy_state: str
    omega_overrides: List[str]
    window_hash: str
    error: Optional[str] = None


BUCKET_SYMBOLS = {
    "microcaps": ["CEI", "SENS", "COSM", "KOSS", "SDIG", "ATER", "CLOV", "WISH"],
    "smallcaps": ["AFRM", "CVNA", "RIOT", "MARA", "IONQ", "WULF", "SOUN", "SOFI"],
    "midcaps": ["RBLX", "DKNG", "PLTR", "HOOD", "ROKU", "U", "UAA", "SNAP"],
    "largecaps": ["AAPL", "TSLA", "NFLX", "AMD", "NVDA", "MSFT", "GOOGL", "AMZN"],
    "megacaps": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK.B", "JPM"],
}


def get_bucket_symbols(bucket: str) -> List[str]:
    """Get symbols for a bucket, falling back to universe if needed."""
    if bucket in BUCKET_SYMBOLS:
        return BUCKET_SYMBOLS[bucket]
    
    all_symbols = get_all_symbols()
    return random.sample(all_symbols, min(8, len(all_symbols)))


def calculate_mr_fuse_score(result, volatility_ratio: float) -> float:
    """Calculate MR-Fuse score combining QuantraScore with volatility."""
    base_score = result.quantrascore
    vol_multiplier = 1 + (volatility_ratio / 10)
    
    compression = result.microtraits.compression_score if result.microtraits else 0.5
    compression_boost = 1 + (compression * 0.1)
    
    return base_score * vol_multiplier * compression_boost


def scan_symbol_for_runner(
    engine: ApexEngine,
    adapter: SyntheticAdapter,
    symbol: str,
    bucket: str,
    start_date: datetime,
    end_date: datetime,
) -> RunnerResult:
    """Scan a single symbol for runner potential."""
    try:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        if len(normalized_bars) < 50:
            return RunnerResult(
                symbol=symbol,
                bucket=bucket,
                quantrascore=0.0,
                mr_fuse_score=0.0,
                volatility_ratio=0.0,
                compression_score=0.0,
                is_runner_candidate=False,
                regime="unknown",
                risk_tier="unknown",
                entropy_state="unknown",
                omega_overrides=[],
                window_hash="",
                error=f"Insufficient data: {len(normalized_bars)} bars",
            )
        
        result = engine.run_scan(normalized_bars, symbol)
        
        volatility_ratio = result.microtraits.volatility_ratio if result.microtraits else 0.0
        compression_score = result.microtraits.compression_score if result.microtraits else 0.0
        
        mr_fuse_score = calculate_mr_fuse_score(result, volatility_ratio)
        
        is_runner = (
            result.quantrascore >= 60 or
            mr_fuse_score >= 65 or
            (volatility_ratio > 1.5 and result.quantrascore >= 50)
        )
        
        omega_overrides = list(result.omega_overrides.keys()) if result.omega_overrides else []
        
        return RunnerResult(
            symbol=symbol,
            bucket=bucket,
            quantrascore=round(result.quantrascore, 2),
            mr_fuse_score=round(mr_fuse_score, 2),
            volatility_ratio=round(volatility_ratio, 3),
            compression_score=round(compression_score, 3),
            is_runner_candidate=is_runner,
            regime=result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
            risk_tier=result.risk_tier.value if hasattr(result.risk_tier, 'value') else str(result.risk_tier),
            entropy_state=result.entropy_state.value if hasattr(result.entropy_state, 'value') else str(result.entropy_state),
            omega_overrides=omega_overrides,
            window_hash=result.window_hash,
            error=None,
        )
        
    except Exception as e:
        return RunnerResult(
            symbol=symbol,
            bucket=bucket,
            quantrascore=0.0,
            mr_fuse_score=0.0,
            volatility_ratio=0.0,
            compression_score=0.0,
            is_runner_candidate=False,
            regime="unknown",
            risk_tier="unknown",
            entropy_state="unknown",
            omega_overrides=[],
            window_hash="",
            error=str(e),
        )


def main() -> None:
    print("=" * 70)
    print("QUANTRACORE APEX v9.x — RUNNER DISCOVERY STRESS TEST")
    print("=" * 70)
    print()
    
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=99)
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=200)
    
    all_results: List[RunnerResult] = []
    bucket_stats: Dict[str, Dict[str, Any]] = {}
    
    for bucket in ["microcaps", "smallcaps", "midcaps", "largecaps", "megacaps"]:
        symbols = get_bucket_symbols(bucket)
        print(f"\n[RUNNER] Scanning {bucket} ({len(symbols)} symbols)...")
        
        bucket_results: List[RunnerResult] = []
        
        for symbol in symbols:
            result = scan_symbol_for_runner(
                engine, adapter, symbol, bucket, start_date, end_date
            )
            bucket_results.append(result)
            all_results.append(result)
            
            if result.error:
                print(f"[RUNNER][{bucket:10}] {symbol:6} → ERROR: {result.error}")
            else:
                runner_flag = "🔥" if result.is_runner_candidate else "  "
                print(f"[RUNNER][{bucket:10}] {symbol:6} → "
                      f"Q={result.quantrascore:5.2f}  "
                      f"MR={result.mr_fuse_score:5.2f}  "
                      f"Vol={result.volatility_ratio:.3f}  "
                      f"runner={result.is_runner_candidate} {runner_flag}")
        
        valid = [r for r in bucket_results if r.error is None]
        runners = [r for r in valid if r.is_runner_candidate]
        
        bucket_stats[bucket] = {
            "total": len(bucket_results),
            "valid": len(valid),
            "runners": len(runners),
            "avg_quantrascore": round(sum(r.quantrascore for r in valid) / len(valid), 2) if valid else 0,
            "avg_mr_fuse": round(sum(r.mr_fuse_score for r in valid) / len(valid), 2) if valid else 0,
            "avg_volatility": round(sum(r.volatility_ratio for r in valid) / len(valid), 3) if valid else 0,
        }
    
    print()
    print("=" * 70)
    print("RUNNER DISCOVERY SUMMARY BY BUCKET")
    print("=" * 70)
    
    for bucket, stats in bucket_stats.items():
        print(f"\n{bucket.upper()}:")
        print(f"  Scanned:         {stats['total']}")
        print(f"  Valid:           {stats['valid']}")
        print(f"  Runners found:   {stats['runners']}")
        print(f"  Avg QuantraScore: {stats['avg_quantrascore']}")
        print(f"  Avg MR-Fuse:      {stats['avg_mr_fuse']}")
        print(f"  Avg Volatility:   {stats['avg_volatility']}")
    
    print()
    print("=" * 70)
    print("TOP 15 RUNNER CANDIDATES")
    print("=" * 70)
    
    valid_results = [r for r in all_results if r.error is None]
    top_runners = sorted(valid_results, key=lambda r: r.mr_fuse_score, reverse=True)[:15]
    
    for i, r in enumerate(top_runners, 1):
        runner_flag = "🔥" if r.is_runner_candidate else ""
        print(f"{i:2}. {r.symbol:6} [{r.bucket:10}] "
              f"QS={r.quantrascore:5.1f}  MR={r.mr_fuse_score:5.1f}  "
              f"Vol={r.volatility_ratio:.3f}  {r.regime:12} {runner_flag}")
    
    output_dir = Path("logs/runner_stress")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "last_runners.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    stats_path = output_dir / "last_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(bucket_stats, f, indent=2)
    
    print(f"\n[RUNNER] Results saved to {output_path}")
    print(f"[RUNNER] Stats saved to {stats_path}")
    
    total_runners = sum(1 for r in valid_results if r.is_runner_candidate)
    print(f"\n[RUNNER] Total runner candidates: {total_runners}/{len(valid_results)}")
    print("[RUNNER] ✅ Runner discovery stress test COMPLETE")


if __name__ == "__main__":
    main()
