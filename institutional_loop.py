#!/usr/bin/env python3
"""
QuantraCore Alpha Factory - 24/7 Live Research Loop.

Streams real-time data from Polygon (equities) and Binance (crypto),
runs ApexEngine scans, and tracks a simulated portfolio.

Usage:
    python institutional_loop.py

Environment Variables:
    POLYGON_API_KEY: Required for US equity data (free tier = 5 symbols)

Note: This is RESEARCH MODE ONLY - all trading is simulated.
"""

import os
import sys
import logging
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("AlphaFactory")


def main():
    """Main entry point for the alpha factory."""
    
    print("=" * 70)
    print("  QUANTRACORE ALPHA FACTORY v9.0")
    print("  Mini Institutional Research Engine")
    print("=" * 70)
    print()
    print("  Mode: RESEARCH ONLY (Paper Trading)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if polygon_key:
        print("  [OK] POLYGON_API_KEY found - Equity feeds enabled")
    else:
        print("  [--] POLYGON_API_KEY not set - Equity feeds disabled")
        print("       Add POLYGON_API_KEY in Secrets for live US equity data")
    
    print("  [OK] Binance public API - Crypto feeds enabled (no key needed)")
    print()
    print("=" * 70)
    print()
    
    from src.quantracore_apex.alpha_factory.loop import AlphaFactoryLoop
    from src.quantracore_apex.alpha_factory.dashboard import EquityCurvePlotter
    
    equity_symbols = ["AAPL", "NVDA", "TSLA", "SPY"] if polygon_key else []
    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    factory = AlphaFactoryLoop(
        initial_cash=1_000_000.0,
        equity_symbols=equity_symbols,
        crypto_symbols=crypto_symbols,
        min_score=60
    )
    
    plotter = EquityCurvePlotter(output_dir="static")
    
    def refresh_dashboard():
        """Periodically refresh the dashboard."""
        while True:
            time.sleep(900)
            try:
                plotter.refresh()
                logger.info("Dashboard refreshed")
            except Exception as e:
                logger.error(f"Dashboard refresh error: {e}")
    
    dashboard_thread = threading.Thread(target=refresh_dashboard, daemon=True)
    dashboard_thread.start()
    
    plotter.generate_dashboard_html()
    
    logger.info("Starting alpha factory...")
    factory.run_forever()


if __name__ == "__main__":
    main()
