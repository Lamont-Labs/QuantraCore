#!/usr/bin/env python3
"""
QUANTRACORE APEX — 500+ SYMBOL BACKTEST WITH CACHING
=====================================================
Runs sequential backtest with disk caching to handle rate limits.
Can resume from where it left off if interrupted.

Usage:
  python scripts/backtest_500.py              # Run all 500 symbols
  python scripts/backtest_500.py --resume     # Resume from last checkpoint
  python scripts/backtest_500.py --limit 100  # Only run 100 symbols
"""

import os
import sys
import json
import hashlib
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

try:
    from polygon import RESTClient
except ImportError:
    os.system("pip install polygon-api-client --quiet")
    from polygon import RESTClient

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    print("ERROR: POLYGON_API_KEY not found in environment")
    sys.exit(1)

CACHE_DIR = Path("data/polygon_cache")
PROOF_LOG = Path("proof_logs/backtest_500.jsonl")
CHECKPOINT_FILE = Path("proof_logs/.checkpoint_500")
SUMMARY_LOG = Path("proof_logs/backtest_500_summary.json")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROOF_LOG.parent.mkdir(parents=True, exist_ok=True)

START_DATE = "2021-01-01"
END_DATE = "2024-01-01"
MIN_CANDLES = 200
RATE_LIMIT_DELAY = 13

SP500_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "PYPL", "BAC", "ADBE",
    "CMCSA", "NFLX", "XOM", "VZ", "INTC", "T", "PFE", "KO", "PEP", "MRK",
    "ABT", "CVX", "ABBV", "CRM", "CSCO", "TMO", "ACN", "AVGO", "MCD", "COST",
    "DHR", "NKE", "TXN", "LLY", "NEE", "MDT", "PM", "UNP", "BMY", "ORCL",
    "AMGN", "IBM", "HON", "LIN", "LOW", "QCOM", "SBUX", "RTX", "GS", "BLK",
    "AMD", "CAT", "DE", "INTU", "ISRG", "NOW", "SPGI", "ADP", "GILD", "BKNG",
    "MDLZ", "ADI", "TJX", "MMC", "CVS", "CI", "SYK", "CB", "LRCX", "ZTS",
    "REGN", "MO", "TMUS", "PLD", "DUK", "SO", "BDX", "CL", "BSX", "ITW",
    "MMM", "USB", "PNC", "TFC", "SCHW", "ICE", "EOG", "HUM", "VRTX", "SLB",
    "FDX", "MU", "EW", "ATVI", "APD", "SHW", "NSC", "ETN", "AEP", "CME",
    "CCI", "D", "GD", "NOC", "PSA", "BIIB", "EMR", "ILMN", "KMB", "PXD",
    "ROP", "SPG", "WM", "AON", "AIG", "COF", "GM", "F", "FCX", "CTSH",
    "TRV", "SRE", "MCK", "HCA", "ORLY", "AFL", "AZO", "CMG", "MCHP", "KLAC",
    "PSX", "APH", "DXCM", "SNPS", "CDNS", "MSI", "TT", "IDXX", "FTNT", "A",
    "EXC", "WEC", "DD", "DOW", "ANET", "GIS", "XEL", "PAYX", "DLR", "CTAS",
    "STZ", "VLO", "EA", "ADM", "WELL", "DHI", "LEN", "NEM", "PCAR", "YUM",
    "FAST", "KR", "AME", "HSY", "PPG", "ROST", "OXY", "SYY", "EQR", "ED",
    "KHC", "VRSK", "AWK", "CBRE", "DTE", "WBA", "KEYS", "TROW", "MTD", "ALB",
    "HAL", "BKR", "WST", "WBD", "ZBRA", "ANSS", "CPRT", "ODFL", "CARR", "IQV",
    "WY", "VICI", "EFX", "OTIS", "FTV", "LYB", "WAT", "DLTR", "RMD", "GPN",
    "IR", "CDW", "EXR", "TSCO", "BR", "FE", "ROL", "AES", "PEG", "NTRS",
    "K", "FITB", "CFG", "RF", "HBAN", "KEY", "CNC", "HRL", "IFF", "J",
    "PKI", "LH", "HOLX", "BAX", "TER", "TDY", "FMC", "STE", "LKQ", "DGX",
    "VMC", "MLM", "MAA", "CINF", "L", "AEE", "ES", "NI", "CMS", "EVRG",
    "SBAC", "CNP", "WRB", "ATO", "LNT", "PNW", "NRG", "AMCR", "TAP", "CPB",
    "MKC", "CLX", "JBHT", "CHRW", "EXPD", "XYL", "ALLE", "NVR", "POOL", "LVS",
    "WYNN", "MGM", "BXP", "VTR", "UDR", "PEAK", "REG", "HST", "KIM", "SJM",
    "CAG", "HII", "IPG", "GL", "SEE", "BEN", "IVZ", "WRK", "NWSA", "NWS",
    "LW", "CE", "EMN", "ALK", "AAL", "UAL", "DAL", "LUV", "CCL", "NCLH",
    "RCL", "MAR", "HLT", "H", "EXPE", "BKNG", "ABNB", "MTCH", "LYFT", "UBER",
    "DOCU", "ZM", "CZR", "PENN", "DKNG", "RBLX", "COIN", "HOOD", "SOFI", "AFRM",
    "UPST", "OPEN", "WISH", "PLTR", "PATH", "SNOW", "DDOG", "NET", "CRWD", "ZS",
    "OKTA", "MDB", "TWLO", "SQ", "SHOP", "MELI", "SE", "GRAB", "NU", "RIVN",
    "LCID", "FSR", "NKLA", "GOEV", "ARVL", "BLNK", "CHPT", "EVGO", "QS", "MVST",
    "PLUG", "FCEL", "BE", "BLDP", "HYLN", "RIDE", "WKHS", "XL", "LAZR", "VLDR",
    "AEVA", "OUST", "MVIS", "LIDR", "INVZ", "CPNG", "BABA", "JD", "PDD", "BILI",
    "TME", "BIDU", "NIO", "XPEV", "LI", "BYD", "VWAGY", "STLA", "HMC", "TM",
    "GM", "F", "RACE", "TSLA", "RIVN", "LCID", "FSR", "SOLO", "AYRO", "ELMS",
    "ARVL", "NKLA", "HYLN", "RIDE", "WKHS", "GOEV", "FFIE", "MULN", "PSNY", "VFS",
    "AMD", "INTC", "NVDA", "QCOM", "AVGO", "TXN", "ADI", "MCHP", "NXPI", "ON",
    "MU", "LRCX", "KLAC", "AMAT", "ASML", "TSM", "UMC", "GFS", "WOLF", "CREE",
    "STM", "SWKS", "QRVO", "SLAB", "MXIM", "MPWR", "SYNA", "RMBS", "AMKR", "COHR"
]

SYMBOLS = list(dict.fromkeys([s for s in SP500_SYMBOLS if '.' not in s and len(s) <= 5]))[:500]


def hash_record(rec: dict) -> str:
    canonical = json.dumps(rec, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def load_cached_data(symbol: str) -> pd.DataFrame | None:
    cache_path = get_cache_path(symbol)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if len(df) >= MIN_CANDLES:
                return df
        except:
            pass
    return None


def save_to_cache(symbol: str, df: pd.DataFrame):
    cache_path = get_cache_path(symbol)
    df.to_parquet(cache_path, index=False)


def fetch_from_polygon(client: RESTClient, symbol: str) -> pd.DataFrame | None:
    try:
        aggs = list(client.get_aggs(
            symbol, 1, "day",
            START_DATE, END_DATE,
            adjusted=True,
            limit=50000
        ))
        
        if not aggs or len(aggs) < MIN_CANDLES:
            return None
        
        df = pd.DataFrame([{
            'timestamp': a.timestamp,
            'open': a.open,
            'high': a.high,
            'low': a.low,
            'close': a.close,
            'volume': a.volume
        } for a in aggs])
        
        return df
    except Exception as e:
        print(f"    API error for {symbol}: {str(e)[:50]}")
        return None


def df_to_bars(df: pd.DataFrame) -> list[OhlcvBar]:
    bars = []
    for _, row in df.iterrows():
        bars.append(OhlcvBar(
            timestamp=datetime.utcfromtimestamp(row['timestamp'] / 1000),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume'])
        ))
    return bars


def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(completed), f)


def main():
    parser = argparse.ArgumentParser(description='500+ Symbol Backtest with Caching')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--limit', type=int, default=500, help='Max symbols to process')
    parser.add_argument('--delay', type=int, default=RATE_LIMIT_DELAY, help='API delay in seconds')
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUANTRACORE APEX — 500+ SYMBOL BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Target: {min(args.limit, len(SYMBOLS))} symbols")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 70)
    
    client = RESTClient(api_key=POLYGON_API_KEY)
    engine = ApexEngine(auto_load_protocols=True)
    
    completed = load_checkpoint() if args.resume else set()
    symbols_to_process = [s for s in SYMBOLS[:args.limit] if s not in completed]
    
    if args.resume and completed:
        print(f"\nResuming from checkpoint: {len(completed)} already done")
    
    print(f"Symbols to process: {len(symbols_to_process)}")
    
    cached_count = sum(1 for s in symbols_to_process if get_cache_path(s).exists())
    print(f"Already cached: {cached_count} (will skip API calls)")
    print(f"Need to fetch: {len(symbols_to_process) - cached_count}")
    
    eta_minutes = (len(symbols_to_process) - cached_count) * args.delay / 60
    print(f"Estimated time: {eta_minutes:.1f} minutes\n")
    
    results = []
    failed = []
    api_calls = 0
    cache_hits = 0
    start_time = time.time()
    
    mode = 'a' if args.resume else 'w'
    with open(PROOF_LOG, mode) as f:
        for i, symbol in enumerate(symbols_to_process):
            progress = f"[{i+1}/{len(symbols_to_process)}]"
            
            df = load_cached_data(symbol)
            if df is not None:
                cache_hits += 1
                source = "cache"
            else:
                if api_calls > 0:
                    time.sleep(args.delay)
                
                df = fetch_from_polygon(client, symbol)
                api_calls += 1
                source = "API"
                
                if df is not None:
                    save_to_cache(symbol, df)
            
            if df is None:
                print(f"{progress} SKIP {symbol}: insufficient data")
                failed.append(symbol)
                completed.add(symbol)
                save_checkpoint(completed)
                continue
            
            try:
                bars = df_to_bars(df)
                scan_result = engine.run_scan(bars, symbol, seed=42)
                
                fired_protocols = [p.protocol_id for p in scan_result.protocol_results if p.fired]
                
                record = {
                    "symbol": symbol,
                    "period": f"{START_DATE} to {END_DATE}",
                    "quantra_score": round(scan_result.quantrascore, 2),
                    "regime": scan_result.regime.value if hasattr(scan_result.regime, 'value') else str(scan_result.regime),
                    "risk_tier": scan_result.risk_tier.value if hasattr(scan_result.risk_tier, 'value') else str(scan_result.risk_tier),
                    "entropy_state": str(scan_result.entropy_metrics.entropy_state),
                    "fired_protocols": fired_protocols,
                    "protocol_count": len(fired_protocols),
                    "num_candles": len(bars),
                    "window_hash": scan_result.window_hash,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "compliance_note": scan_result.verdict.compliance_note
                }
                record["proof_hash"] = hash_record(record)
                
                f.write(json.dumps(record) + "\n")
                f.flush()
                results.append(record)
                
                score = record["quantra_score"]
                regime = record["regime"]
                marker = "★" if score >= 60 else " "
                print(f"{progress} {marker} {symbol:5} → {score:5.2f} [{regime}] ({source})")
                
            except Exception as e:
                print(f"{progress} ERROR {symbol}: {str(e)[:40]}")
                failed.append(symbol)
            
            completed.add(symbol)
            save_checkpoint(completed)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"  Symbols processed: {len(results)}")
    print(f"  Symbols failed: {len(failed)}")
    print(f"  API calls: {api_calls}")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Proof log: {PROOF_LOG}")
    
    if results:
        scores = [r["quantra_score"] for r in results]
        regimes = [r["regime"] for r in results]
        
        print(f"\n--- QUANTRASCORE ANALYSIS ---")
        print(f"  Mean: {np.mean(scores):.2f}")
        print(f"  Std Dev: {np.std(scores):.2f}")
        print(f"  Min/Max: {np.min(scores):.2f} / {np.max(scores):.2f}")
        
        print(f"\n--- SIGNAL THRESHOLDS ---")
        high = [r for r in results if r["quantra_score"] >= 70]
        mid = [r for r in results if 60 <= r["quantra_score"] < 70]
        print(f"  >= 70 (high conviction): {len(high)} ({100*len(high)/len(results):.1f}%)")
        print(f"  60-69 (moderate): {len(mid)} ({100*len(mid)/len(results):.1f}%)")
        
        print(f"\n--- REGIME DISTRIBUTION ---")
        for regime in sorted(set(regimes)):
            count = regimes.count(regime)
            print(f"  {regime}: {count} ({100*count/len(regimes):.1f}%)")
        
        print(f"\n--- TOP 15 HIGHEST SCORES ---")
        top15 = sorted(results, key=lambda x: x["quantra_score"], reverse=True)[:15]
        for r in top15:
            print(f"  {r['symbol']:5} {r['quantra_score']:5.2f} [{r['regime']}/{r['risk_tier']}]")
        
        summary = {
            "backtest_period": f"{START_DATE} to {END_DATE}",
            "symbols_processed": len(results),
            "symbols_failed": len(failed),
            "api_calls": api_calls,
            "cache_hits": cache_hits,
            "runtime_minutes": round(elapsed / 60, 2),
            "mean_score": round(np.mean(scores), 2),
            "std_score": round(np.std(scores), 2),
            "high_conviction_count": len(high),
            "moderate_count": len(mid),
            "regime_distribution": {r: regimes.count(r) for r in set(regimes)},
            "top_15": [{"symbol": r["symbol"], "score": r["quantra_score"], "regime": r["regime"]} for r in top15],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        with open(SUMMARY_LOG, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary: {SUMMARY_LOG}")
    
    print("\n" + "=" * 70)
    print("COMPLIANCE: Structural probabilities only, NOT trading advice.")
    print("=" * 70)


if __name__ == "__main__":
    main()
