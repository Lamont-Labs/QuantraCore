#!/usr/bin/env python3
"""
QuantraCore Apex - Autonomous Trading Runner.

Runs the institutional-grade autonomous trading orchestrator.

Usage:
    python scripts/run_autonomous.py --mode paper --duration 3600
    python scripts/run_autonomous.py --mode research --symbols AAPL,NVDA,TSLA
    python scripts/run_autonomous.py --demo  # Quick demo with simulated data

Options:
    --mode: paper (default) or research
    --duration: Run duration in seconds (None = run forever)
    --symbols: Comma-separated symbol list
    --simulated: Use simulated stream (default True)
    --demo: Run a quick 60-second demo
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantracore_apex.autonomous import (
    TradingOrchestrator,
    OrchestratorConfig,
    QualityThresholds,
)
from src.quantracore_apex.autonomous.trading_orchestrator import run_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="QuantraCore Apex Autonomous Trading Runner"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "research"],
        help="Trading mode: paper (execute on Alpaca paper) or research (log only)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Run duration in seconds (None = run forever)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbol list"
    )
    
    parser.add_argument(
        "--no-simulated",
        action="store_true",
        default=False,
        help="Disable simulated data stream (requires Polygon API)"
    )
    
    parser.add_argument(
        "--live-stream",
        action="store_true",
        help="Use live Polygon WebSocket stream (requires POLYGON_API_KEY)"
    )
    
    parser.add_argument(
        "--min-score",
        type=float,
        default=75.0,
        help="Minimum QuantraScore threshold (default: 75)"
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions (default: 5)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick 60-second demo with simulated data"
    )
    
    parser.add_argument(
        "--status-interval",
        type=int,
        default=30,
        help="Status report interval in seconds"
    )
    
    return parser.parse_args()


async def run_with_status(
    orchestrator: TradingOrchestrator,
    duration: int,
    status_interval: int,
) -> None:
    """Run orchestrator with periodic status reports."""
    
    async def status_reporter():
        while True:
            await asyncio.sleep(status_interval)
            status = orchestrator.get_status()
            metrics = status.get("metrics", {})
            
            logger.info("=" * 50)
            logger.info("ORCHESTRATOR STATUS REPORT")
            logger.info("=" * 50)
            logger.info(f"State: {status.get('state', 'unknown')}")
            logger.info(f"Uptime: {metrics.get('uptime_seconds', 0):.0f}s")
            logger.info(f"Signals scanned: {metrics.get('total_signals_scanned', 0)}")
            logger.info(f"Signals passed: {metrics.get('signals_passed_filter', 0)}")
            logger.info(f"Trades opened: {metrics.get('trades_opened', 0)}")
            logger.info(f"Trades closed: {metrics.get('trades_closed', 0)}")
            logger.info(f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
            logger.info(f"Total P&L: ${metrics.get('total_realized_pnl', 0):.2f}")
            logger.info(f"Current positions: {metrics.get('current_positions', 0)}")
            logger.info("=" * 50)
    
    status_task = asyncio.create_task(status_reporter())
    
    if duration:
        async def stop_after_duration():
            await asyncio.sleep(duration)
            logger.info(f"Duration limit ({duration}s) reached, stopping...")
            await orchestrator.stop()
        
        stop_task = asyncio.create_task(stop_after_duration())
    
    try:
        await orchestrator.start()
    finally:
        status_task.cancel()
        if duration:
            stop_task.cancel()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("QuantraCore Apex - Autonomous Trading System")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    use_simulated = not args.no_simulated and not args.live_stream
    
    if args.demo:
        args.duration = 60
        use_simulated = True
        args.min_score = 50.0
        print("DEMO MODE: Running 60-second demo with simulated data")
        print()
    
    if args.symbols:
        watchlist = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "GOOG", "META", "MSFT", "AMD"]
    
    thresholds = QualityThresholds(
        min_quantrascore=args.min_score,
        required_quality_tiers=["A+", "A"],
        max_risk_tier="high",
        min_liquidity_band="medium",
    )
    
    config = OrchestratorConfig(
        watchlist=watchlist,
        max_concurrent_positions=args.max_positions,
        quality_thresholds=thresholds,
        paper_mode=(args.mode == "paper"),
        scan_interval_seconds=10.0 if args.demo else 60.0,
        respect_market_hours=False,
    )
    
    print(f"Mode: {args.mode.upper()}")
    print(f"Stream: {'Simulated' if use_simulated else 'Polygon Live'}")
    print(f"Watchlist: {len(watchlist)} symbols")
    print(f"Min QuantraScore: {args.min_score}")
    print(f"Max positions: {args.max_positions}")
    print(f"Duration: {args.duration}s" if args.duration else "Duration: Forever")
    print("=" * 70)
    print()
    
    orchestrator = TradingOrchestrator(
        config=config,
        use_simulated_stream=use_simulated,
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        loop.create_task(orchestrator.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    ready, errors = orchestrator.is_ready()
    if not ready:
        print("\nERROR: Orchestrator not ready to start!")
        print("Missing components:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease ensure all required Apex components are installed.")
        sys.exit(1)
    
    try:
        loop.run_until_complete(
            run_with_status(
                orchestrator,
                args.duration,
                args.status_interval,
            )
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("Autonomous trading cannot start without required components.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received...")
    finally:
        loop.run_until_complete(orchestrator.stop())
        loop.close()
    
    print()
    print("=" * 70)
    print("FINAL STATUS")
    print("=" * 70)
    
    status = orchestrator.get_status()
    metrics = status.get("metrics", {})
    outcomes = status.get("outcomes", {})
    
    print(f"Total signals scanned: {metrics.get('total_signals_scanned', 0)}")
    print(f"Signals passed filter: {metrics.get('signals_passed_filter', 0)}")
    print(f"Trades executed: {metrics.get('trades_opened', 0)}")
    print(f"Trades closed: {metrics.get('trades_closed', 0)}")
    print(f"Profitable: {metrics.get('profitable_trades', 0)}")
    print(f"Losing: {metrics.get('losing_trades', 0)}")
    print(f"Win rate: {outcomes.get('win_rate', 0)*100:.1f}%")
    print(f"Total P&L: ${metrics.get('total_realized_pnl', 0):.2f}")
    print()
    print("Autonomous trading session complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
