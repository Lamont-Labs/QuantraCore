#!/usr/bin/env python3
"""
QuantraCore Apex — Live Stress Test

Runs 24-hour continuous stress test with real-time market data.
Uses Polygon WebSocket for live tick data.

COMPLIANCE: All outputs are structural probabilities, NOT trading advice.
"""

import os
import sys
import json
import signal
import asyncio
from datetime import datetime, timedelta
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.data_layer.adapters.polygon_adapter import PolygonAdapter
from src.quantracore_apex.core.schemas import OhlcvBar

TEST_DURATION_HOURS = 24
ANALYSIS_INTERVAL_SECONDS = 60
STRESS_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ"]
PROOF_LOG_FILE = "proof_logs/live_stress_test.jsonl"

os.makedirs("proof_logs", exist_ok=True)


class LiveStressTest:
    """
    Live stress test runner.
    
    Continuously analyzes market data and logs proof records.
    """
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        
        self.engine = ApexEngine(auto_load_protocols=True)
        self.adapter = PolygonAdapter(api_key=self.api_key)
        self.running = True
        self.stats = {
            "scans_completed": 0,
            "errors": 0,
            "start_time": datetime.utcnow(),
            "scores": deque(maxlen=1000)
        }
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nReceived shutdown signal. Finishing up...")
        self.running = False
    
    def run(self):
        """Run the stress test."""
        print("=" * 70)
        print("QuantraCore Apex — Live Stress Test")
        print("=" * 70)
        print(f"Duration: {TEST_DURATION_HOURS} hours")
        print(f"Analysis interval: {ANALYSIS_INTERVAL_SECONDS} seconds")
        print(f"Symbols: {', '.join(STRESS_SYMBOLS)}")
        print(f"Engine: {len(self.engine.protocol_runner.protocols)} protocols loaded")
        print("=" * 70)
        print("\nPress Ctrl+C to stop early\n")
        
        end_time = datetime.utcnow() + timedelta(hours=TEST_DURATION_HOURS)
        
        with open(PROOF_LOG_FILE, "a") as logf:
            while self.running and datetime.utcnow() < end_time:
                cycle_start = datetime.utcnow()
                
                for symbol in STRESS_SYMBOLS:
                    if not self.running:
                        break
                    
                    try:
                        bars = self.adapter.fetch(symbol, days=100)
                        
                        if len(bars) < 50:
                            continue
                        
                        result = self.engine.run_scan(bars, symbol)
                        
                        record = {
                            "symbol": symbol,
                            "timestamp_utc": datetime.utcnow().isoformat(),
                            "quantra_score": result.quantrascore,
                            "regime": result.regime.value,
                            "risk_tier": result.risk_tier.value,
                            "protocols_fired": sum(1 for p in result.protocol_results 
                                                   if hasattr(p, 'fired') and p.fired),
                            "window_hash": result.window_hash,
                            "compliance_note": "Structural analysis only - not trading advice"
                        }
                        
                        logf.write(json.dumps(record, default=str) + "\n")
                        logf.flush()
                        
                        self.stats["scans_completed"] += 1
                        self.stats["scores"].append(result.quantrascore)
                        
                        print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] "
                              f"{symbol}: Score={result.quantrascore:.1f} "
                              f"Regime={result.regime.value} "
                              f"Risk={result.risk_tier.value}")
                        
                    except Exception as e:
                        self.stats["errors"] += 1
                        print(f"[ERROR] {symbol}: {e}")
                
                elapsed = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(0, ANALYSIS_INTERVAL_SECONDS - elapsed)
                
                if self.running and sleep_time > 0:
                    print(f"\n--- Cycle complete. Next in {sleep_time:.0f}s ---\n")
                    
                    import time
                    time.sleep(sleep_time)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test summary."""
        duration = (datetime.utcnow() - self.stats["start_time"]).total_seconds() / 3600
        
        print("\n" + "=" * 70)
        print("STRESS TEST COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration:.2f} hours")
        print(f"Scans completed: {self.stats['scans_completed']}")
        print(f"Errors: {self.stats['errors']}")
        
        if self.stats["scores"]:
            scores = list(self.stats["scores"])
            print(f"\nScore Statistics (last {len(scores)} scans):")
            print(f"  Min: {min(scores):.2f}")
            print(f"  Max: {max(scores):.2f}")
            print(f"  Avg: {sum(scores)/len(scores):.2f}")
        
        print(f"\nProof log: {PROOF_LOG_FILE}")
        print("\n" + "=" * 70)
        print("COMPLIANCE: All outputs are structural probabilities.")
        print("This is NOT trading advice.")
        print("=" * 70)


def main():
    """Main entry point."""
    try:
        test = LiveStressTest()
        test.run()
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("Please add your Polygon API key to the Secrets tab.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
