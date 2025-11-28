#!/usr/bin/env python3
"""
QuantraCore Apex — Backtest & Proof Logger

Runs 1-year backtest on 500 symbols with cryptographic proof logging.
All outputs are STRUCTURAL PROBABILITIES, not trading advice.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.polygon_adapter import PolygonAdapter

SYMBOLS_COUNT = 500
START_DATE = "2023-11-01"
END_DATE = "2024-11-01"
TIMEFRAME = "day"
PROOF_LOG_DIR = "proof_logs"
PROOF_LOG_FILE = os.path.join(PROOF_LOG_DIR, "backtest_proof.jsonl")

os.makedirs(PROOF_LOG_DIR, exist_ok=True)


def hash_decision(decision_dict: dict) -> str:
    """Generate SHA-256 hash of decision for proof logging."""
    canonical = json.dumps(decision_dict, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def run_backtest():
    """
    Run 1-year backtest on top 500 symbols.
    
    Logs all decisions with cryptographic proof hashes.
    """
    print("=" * 70)
    print("QuantraCore Apex — Backtest & Proof Logger")
    print("=" * 70)
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Target Symbols: {SYMBOLS_COUNT}")
    print(f"Proof Log: {PROOF_LOG_FILE}")
    print("=" * 70)
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("\nERROR: POLYGON_API_KEY not found in environment.")
        print("Please add your Polygon API key to the Secrets tab.")
        sys.exit(1)
    
    print("\nInitializing QuantraCore Apex engine...")
    engine = ApexEngine(auto_load_protocols=True)
    adapter = PolygonAdapter(api_key=api_key)
    
    print(f"Engine loaded with {len(engine.protocol_runner.protocols)} protocols")
    
    print("\nFetching top symbols by volume...")
    try:
        symbols = adapter.get_top_symbols(count=SYMBOLS_COUNT, min_volume=100000)
        print(f"Found {len(symbols)} qualifying symbols")
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        print("Using fallback symbol list...")
        symbols = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK.B",
            "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "CVX", "HD",
            "LLY", "MRK", "ABBV", "KO", "PEP", "COST", "AVGO", "TMO", "MCD",
            "CSCO", "ACN", "ABT", "DHR", "NFLX", "NKE", "ADBE", "CRM", "AMD",
            "TXN", "PM", "NEE", "UPS", "RTX", "HON", "LOW", "QCOM", "UNP",
            "IBM", "INTC", "SPGI", "GE", "BA"
        ]
    
    results = []
    errors = []
    
    print(f"\nStarting backtest on {len(symbols)} symbols...")
    
    with open(PROOF_LOG_FILE, "w") as logf:
        for symbol in tqdm(symbols, desc="Backtesting"):
            try:
                bars = adapter.fetch_range(
                    symbol=symbol,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    timeframe=TIMEFRAME
                )
                
                if len(bars) < 100:
                    errors.append(f"{symbol}: Insufficient data ({len(bars)} bars)")
                    continue
                
                scan_result = engine.run_scan(bars, symbol)
                
                protocol_trace = []
                for p in scan_result.protocol_results:
                    if hasattr(p, 'fired') and p.fired:
                        protocol_trace.append({
                            "protocol_id": p.protocol_id,
                            "confidence": p.confidence,
                            "signal_type": p.signal_type
                        })
                
                decision_record = {
                    "symbol": symbol,
                    "date_range": f"{START_DATE}_to_{END_DATE}",
                    "bars_analyzed": len(bars),
                    "quantra_score": scan_result.quantrascore,
                    "score_bucket": scan_result.score_bucket,
                    "regime": scan_result.regime.value,
                    "risk_tier": scan_result.risk_tier.value,
                    "entropy_state": scan_result.entropy_state,
                    "drift_state": scan_result.drift_state,
                    "protocols_fired": len(protocol_trace),
                    "protocol_trace": protocol_trace,
                    "verdict": {
                        "action": scan_result.verdict.action,
                        "confidence": scan_result.verdict.confidence,
                        "compliance_note": scan_result.verdict.compliance_note
                    },
                    "window_hash": scan_result.window_hash,
                    "timestamp_utc": datetime.utcnow().isoformat(),
                }
                
                decision_record["proof_hash"] = hash_decision(decision_record)
                
                logf.write(json.dumps(decision_record, default=str) + "\n")
                logf.flush()
                
                results.append(decision_record)
                
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Symbols processed: {len(results)}")
    print(f"Errors/Skipped: {len(errors)}")
    print(f"Proof log: {PROOF_LOG_FILE}")
    
    if results:
        scores = [r["quantra_score"] for r in results]
        regimes = {}
        for r in results:
            reg = r["regime"]
            regimes[reg] = regimes.get(reg, 0) + 1
        
        print(f"\nScore Distribution:")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")
        print(f"  Avg: {sum(scores)/len(scores):.2f}")
        
        print(f"\nRegime Distribution:")
        for reg, count in sorted(regimes.items(), key=lambda x: -x[1]):
            print(f"  {reg}: {count} ({100*count/len(results):.1f}%)")
        
        print(f"\nSample proof hash (last record):")
        print(f"  {results[-1]['proof_hash']}")
    
    if errors:
        print(f"\nErrors (first 10):")
        for err in errors[:10]:
            print(f"  {err}")
    
    print("\n" + "=" * 70)
    print("COMPLIANCE NOTE: All outputs are structural probabilities.")
    print("This is NOT trading advice. Use at your own risk.")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_backtest()
