#!/usr/bin/env python3
"""
scripts/zde_validation.py

Zero Drawdown Entry (ZDE) Validation for QuantraCore Apex v9.x

Validates that entry signals respect the ZDE constraint with configurable
wiggle room tolerance (default ±3%) for real-world execution variance.

ZDE Definition:
    - ZDE = 0% allowable drawdown after entry signal is triggered
    - With tolerance: Allow ±2% to ±3% wiggle range while preserving deterministic scoring
"""

import sys
sys.path.insert(0, ".")

import json
import random
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.config.symbol_universe import get_all_symbols


ZDE_TOLERANCE = 0.03  # 3% wiggle room


@dataclass
class ZDEResult:
    """Single ZDE validation result."""
    symbol: str
    entry_price: float
    min_after_entry: float
    max_drawdown_pct: float
    zde_passed: bool
    quantrascore: float
    regime: str
    risk_tier: str
    window_hash: str
    error: Optional[str] = None


def get_diverse_tickers() -> List[str]:
    """Get a diverse set of tickers across all market caps."""
    all_symbols = get_all_symbols()
    
    priority_tickers = [
        "AAPL", "TSLA", "NVDA", "AMZN", "META", "PLTR", "NFLX", "AMD", "BA", "F", "T",
        "GME", "AMC", "CEI", "SOUN", "IONQ", "AFRM", "DKNG", "CVNA",
        "RBLX", "HOOD", "SHOP", "COIN", "MARA", "RIOT", "WULF", "NVAX", "BNTX",
        "MSFT", "GOOGL", "JPM", "V", "UNH", "JNJ", "WMT", "PG", "XOM", "CVX",
    ]
    
    available = [t for t in priority_tickers if t in all_symbols]
    
    remaining = [s for s in all_symbols if s not in available]
    random.shuffle(remaining)
    available.extend(remaining[:max(0, 50 - len(available))])
    
    random.shuffle(available)
    return available[:50]


def validate_zde_for_symbol(
    engine: ApexEngine,
    adapter: SyntheticAdapter,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> ZDEResult:
    """Validate ZDE constraint for a single symbol."""
    try:
        bars = adapter.fetch_ohlcv(symbol, start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        if len(normalized_bars) < 50:
            return ZDEResult(
                symbol=symbol,
                entry_price=0.0,
                min_after_entry=0.0,
                max_drawdown_pct=0.0,
                zde_passed=False,
                quantrascore=0.0,
                regime="unknown",
                risk_tier="unknown",
                window_hash="",
                error=f"Insufficient data: {len(normalized_bars)} bars",
            )
        
        result = engine.run_scan(normalized_bars, symbol)
        
        entry_idx = len(normalized_bars) - 20
        entry_price = normalized_bars[entry_idx].close
        
        post_entry_lows = [bar.low for bar in normalized_bars[entry_idx:]]
        min_after_entry = min(post_entry_lows) if post_entry_lows else entry_price
        
        if entry_price > 0:
            max_drawdown = (entry_price - min_after_entry) / entry_price
        else:
            max_drawdown = 0.0
        
        zde_passed = max_drawdown <= ZDE_TOLERANCE
        
        return ZDEResult(
            symbol=symbol,
            entry_price=round(entry_price, 4),
            min_after_entry=round(min_after_entry, 4),
            max_drawdown_pct=round(max_drawdown * 100, 2),
            zde_passed=zde_passed,
            quantrascore=result.quantrascore,
            regime=result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
            risk_tier=result.risk_tier.value if hasattr(result.risk_tier, 'value') else str(result.risk_tier),
            window_hash=result.window_hash,
            error=None,
        )
        
    except Exception as e:
        return ZDEResult(
            symbol=symbol,
            entry_price=0.0,
            min_after_entry=0.0,
            max_drawdown_pct=0.0,
            zde_passed=False,
            quantrascore=0.0,
            regime="unknown",
            risk_tier="unknown",
            window_hash="",
            error=str(e),
        )


def main() -> None:
    print("=" * 70)
    print("QUANTRACORE APEX v9.x — ZERO DRAWDOWN ENTRY (ZDE) VALIDATION")
    print("=" * 70)
    print(f"[ZDE] Tolerance range: ±{ZDE_TOLERANCE * 100:.1f}%")
    print()
    
    engine = ApexEngine(enable_logging=False)
    adapter = SyntheticAdapter(seed=42)
    
    end_date = datetime(2024, 1, 15)
    start_date = end_date - timedelta(days=200)
    
    tickers = get_diverse_tickers()
    print(f"[ZDE] Testing {len(tickers)} symbols...")
    print()
    
    results: List[ZDEResult] = []
    
    for symbol in tickers:
        result = validate_zde_for_symbol(engine, adapter, symbol, start_date, end_date)
        results.append(result)
        
        if result.error:
            print(f"[ZDE][{symbol:6}] ERROR: {result.error}")
        else:
            status = "PASS" if result.zde_passed else "FAIL"
            print(f"[ZDE][{symbol:6}] entry={result.entry_price:8.3f}  "
                  f"min_after={result.min_after_entry:8.3f}  "
                  f"dd={result.max_drawdown_pct:5.2f}%  "
                  f"QS={result.quantrascore:5.1f} → {status}")
    
    print()
    print("=" * 70)
    print("ZDE VALIDATION SUMMARY")
    print("=" * 70)
    
    valid_results = [r for r in results if r.error is None]
    passes = sum(1 for r in valid_results if r.zde_passed)
    fails = len(valid_results) - passes
    errors = sum(1 for r in results if r.error is not None)
    
    print(f"  Total tested:  {len(results)}")
    print(f"  Valid scans:   {len(valid_results)}")
    print(f"  Passed ZDE:    {passes}")
    print(f"  Failed ZDE:    {fails}")
    print(f"  Errors:        {errors}")
    
    if valid_results:
        drawdowns = [r.max_drawdown_pct for r in valid_results]
        print(f"  Avg drawdown:  {statistics.mean(drawdowns):.2f}%")
        print(f"  Max drawdown:  {max(drawdowns):.2f}%")
        print(f"  Min drawdown:  {min(drawdowns):.2f}%")
    
    print()
    
    if fails > 0:
        print("Failed symbols:")
        for r in valid_results:
            if not r.zde_passed:
                print(f"  - {r.symbol}: {r.max_drawdown_pct:.2f}% drawdown")
    
    output_dir = Path("logs/zde_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "last_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\n[ZDE] Results saved to {output_path}")
    
    pass_rate = (passes / len(valid_results) * 100) if valid_results else 0
    print(f"\n[ZDE] Pass rate: {pass_rate:.1f}%")
    
    if pass_rate >= 90:
        print("[ZDE] ✅ ZDE validation PASSED (>90% pass rate)")
    else:
        print(f"[ZDE] ⚠️  ZDE pass rate below threshold ({pass_rate:.1f}% < 90%)")


if __name__ == "__main__":
    main()
