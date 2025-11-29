#!/usr/bin/env python3
"""
Apex Paper Trader CLI

Run QuantraCore Apex in PAPER mode with Alpaca paper account.

Usage:
    # Set environment variables first:
    export ALPACA_PAPER_API_KEY="your-key"
    export ALPACA_PAPER_API_SECRET="your-secret"

    # Run paper trader:
    python scripts/apex_paper_trader.py --config config/broker.yaml

SAFETY: This script only supports PAPER trading. LIVE is disabled.
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantracore_apex.broker import (
    ExecutionEngine,
    ExecutionMode,
    ApexSignal,
    SignalDirection,
    load_broker_config,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_signal(symbol: str, direction: SignalDirection) -> ApexSignal:
    """Create a sample signal for testing."""
    import uuid
    return ApexSignal(
        signal_id=str(uuid.uuid4()),
        symbol=symbol,
        direction=direction,
        quantra_score=65.0,
        runner_prob=0.25,
        regime="trending",
        volatility_band="normal",
        size_hint=0.01,  # 1% of capital
        metadata={"source": "paper_trader_cli"},
    )


def main():
    parser = argparse.ArgumentParser(description="Apex Paper Trader")
    parser.add_argument(
        "--config",
        default="config/broker.yaml",
        help="Path to broker configuration file"
    )
    parser.add_argument(
        "--mode",
        choices=["RESEARCH", "PAPER"],
        default="PAPER",
        help="Execution mode (LIVE is disabled)"
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Symbol to test with (e.g., AAPL)"
    )
    parser.add_argument(
        "--direction",
        choices=["LONG", "EXIT"],
        default=None,
        help="Signal direction for test"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Just show status, don't execute"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_broker_config(args.config)
    
    # Override mode if specified
    if args.mode:
        config.execution_mode = ExecutionMode[args.mode]
    
    # Ensure we're not in LIVE mode
    if config.execution_mode == ExecutionMode.LIVE:
        logger.error("LIVE mode is DISABLED. Switching to PAPER mode.")
        config.execution_mode = ExecutionMode.PAPER
    
    logger.info(f"Execution mode: {config.execution_mode.value}")
    
    # Initialize execution engine
    try:
        engine = ExecutionEngine(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize execution engine: {e}")
        return 1
    
    # Show status
    status = engine.get_status()
    logger.info("=== Execution Engine Status ===")
    logger.info(f"Mode: {status['mode']}")
    logger.info(f"Adapter: {status['adapter']}")
    logger.info(f"Is Paper: {status['is_paper']}")
    logger.info(f"Equity: ${status['equity']:,.2f}")
    logger.info(f"Positions: {status['position_count']}")
    logger.info(f"Open Orders: {status['open_order_count']}")
    
    if args.status_only:
        return 0
    
    # Execute test signal if provided
    if args.symbol and args.direction:
        direction = SignalDirection[args.direction]
        signal = create_sample_signal(args.symbol.upper(), direction)
        
        logger.info(f"\n=== Executing Test Signal ===")
        logger.info(f"Symbol: {signal.symbol}")
        logger.info(f"Direction: {signal.direction.value}")
        logger.info(f"QuantraScore: {signal.quantra_score}")
        
        result = engine.execute_signal(signal)
        
        if result:
            logger.info(f"\n=== Execution Result ===")
            logger.info(f"Order ID: {result.order_id}")
            logger.info(f"Status: {result.status.value}")
            logger.info(f"Filled Qty: {result.filled_qty}")
            logger.info(f"Avg Fill Price: ${result.avg_fill_price:.2f}")
            if result.error_message:
                logger.warning(f"Error: {result.error_message}")
        else:
            logger.info("No order was placed (signal may have been filtered)")
    
    # Show updated status
    logger.info("\n=== Final Status ===")
    final_status = engine.get_status()
    logger.info(f"Equity: ${final_status['equity']:,.2f}")
    logger.info(f"Positions: {final_status['position_count']}")
    logger.info(f"Daily Turnover: ${final_status['daily_turnover']:,.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
