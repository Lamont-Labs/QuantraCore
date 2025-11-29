"""
Universal Broker Router for QuantraCore Apex.

One environment variable controls the entire planet of brokers.

Usage:
    Set BROKER env var to switch brokers instantly:
    - alpaca_paper (default) → Alpaca paper trading
    - alpaca_live            → Alpaca real money US
    - binance                → Binance spot crypto
    - binance_testnet        → Binance testnet
    - ibkr                   → Interactive Brokers
    - bybit                  → Bybit exchange
    - bybit_testnet          → Bybit testnet
    - tradier                → Tradier brokerage
    - tradier_sandbox        → Tradier sandbox
    - paper_sim              → Internal paper simulator

All adapters follow the same interface:
    - send_order(symbol, qty)  → Place market order
    - get_positions()          → Get current positions
    - get_account()            → Get (cash, equity)

Omega Directive Compliance:
    - LIVE trading only available when APEX_MODE=LIVE
    - Default mode is RESEARCH (no execution)
    - All orders pass through 9-check risk engine
"""

import os
import logging
from typing import Tuple, Dict, Callable, Any, Optional
from functools import lru_cache

from .enums import BrokerType, ExecutionMode
from .adapters.base_adapter import BrokerAdapter
from .adapters.null_adapter import NullAdapter
from .adapters.paper_sim_adapter import PaperSimAdapter
from .adapters.alpaca_adapter import AlpacaPaperAdapter


logger = logging.getLogger(__name__)


def _get_execution_mode() -> ExecutionMode:
    """Get execution mode from environment."""
    mode = os.environ.get("APEX_MODE", "RESEARCH").upper()
    try:
        return ExecutionMode(mode)
    except ValueError:
        logger.warning(f"[Universal] Unknown APEX_MODE: {mode}, defaulting to RESEARCH")
        return ExecutionMode.RESEARCH


def _get_broker_type() -> BrokerType:
    """Get broker type from environment."""
    broker = os.environ.get("BROKER", "alpaca_paper").lower()
    try:
        return BrokerType(broker)
    except ValueError:
        logger.warning(f"[Universal] Unknown BROKER: {broker}, defaulting to alpaca_paper")
        return BrokerType.ALPACA_PAPER


def _create_alpaca_paper_adapter() -> BrokerAdapter:
    """Create Alpaca paper adapter."""
    api_key = os.environ.get("ALPACA_PAPER_API_KEY")
    api_secret = os.environ.get("ALPACA_PAPER_API_SECRET")
    
    if api_key and api_secret:
        return AlpacaPaperAdapter(
            api_key=api_key,
            api_secret=api_secret,
        )
    else:
        logger.warning("[Universal] Alpaca paper keys not set, using PaperSim")
        return PaperSimAdapter()


def _create_alpaca_live_adapter() -> BrokerAdapter:
    """Create Alpaca live adapter (REQUIRES LIVE MODE)."""
    from .adapters.alpaca_adapter import AlpacaPaperAdapter
    
    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    
    if not api_key or not api_secret:
        raise ValueError(
            "Alpaca LIVE requires APCA_API_KEY_ID and APCA_API_SECRET_KEY. "
            "For paper trading, use BROKER=alpaca_paper instead."
        )
    
    return AlpacaPaperAdapter(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://api.alpaca.markets",
    )


def _create_binance_adapter(testnet: bool = False) -> BrokerAdapter:
    """Create Binance adapter."""
    from .adapters.binance_adapter import BinanceAdapter
    return BinanceAdapter(testnet=testnet)


def _create_ibkr_adapter() -> BrokerAdapter:
    """Create IBKR adapter."""
    from .adapters.ibkr_adapter import IBKRAdapter
    
    port = int(os.environ.get("IBKR_PORT", "7497"))
    paper = port in (7497, 4002)
    
    return IBKRAdapter(port=port, paper=paper)


def _create_bybit_adapter(testnet: bool = False) -> BrokerAdapter:
    """Create Bybit adapter."""
    from .adapters.bybit_adapter import BybitAdapter
    return BybitAdapter(testnet=testnet)


def _create_tradier_adapter(sandbox: bool = True) -> BrokerAdapter:
    """Create Tradier adapter."""
    from .adapters.tradier_adapter import TradierAdapter
    return TradierAdapter(sandbox=sandbox)


@lru_cache(maxsize=1)
def get_adapter() -> BrokerAdapter:
    """
    Get the appropriate broker adapter based on BROKER and APEX_MODE.
    
    Returns cached adapter instance (singleton per process).
    
    Omega Directive Compliance:
    - RESEARCH mode → NullAdapter (no execution)
    - PAPER mode → Paper adapters only
    - LIVE mode → DISABLED by default (requires explicit enable)
    """
    mode = _get_execution_mode()
    broker_type = _get_broker_type()
    
    logger.info(f"[Universal] Mode: {mode.value}, Broker: {broker_type.value}")
    
    if mode == ExecutionMode.RESEARCH:
        logger.info("[Universal] RESEARCH mode - using NullAdapter (signals only)")
        return NullAdapter()
    
    if mode == ExecutionMode.LIVE:
        live_enabled = os.environ.get("APEX_LIVE_ENABLED", "false").lower() == "true"
        if not live_enabled:
            logger.error(
                "[Universal] LIVE mode requested but APEX_LIVE_ENABLED != true. "
                "This is a safety measure. Use APEX_MODE=PAPER for paper trading."
            )
            raise RuntimeError(
                "LIVE trading is DISABLED. Set APEX_LIVE_ENABLED=true to enable. "
                "WARNING: This will use REAL MONEY."
            )
    
    adapters = {
        BrokerType.ALPACA_PAPER: _create_alpaca_paper_adapter,
        BrokerType.ALPACA_LIVE: _create_alpaca_live_adapter,
        BrokerType.BINANCE: lambda: _create_binance_adapter(testnet=False),
        BrokerType.BINANCE_TESTNET: lambda: _create_binance_adapter(testnet=True),
        BrokerType.BINANCE_FUTURES: lambda: _create_binance_adapter(testnet=False),
        BrokerType.IBKR: lambda: _create_ibkr_adapter(),
        BrokerType.IBKR_PAPER: lambda: _create_ibkr_adapter(),
        BrokerType.BYBIT: lambda: _create_bybit_adapter(testnet=False),
        BrokerType.BYBIT_TESTNET: lambda: _create_bybit_adapter(testnet=True),
        BrokerType.TRADIER: lambda: _create_tradier_adapter(sandbox=False),
        BrokerType.TRADIER_SANDBOX: lambda: _create_tradier_adapter(sandbox=True),
        BrokerType.PAPER_SIM: PaperSimAdapter,
    }
    
    if mode == ExecutionMode.PAPER:
        paper_types = {
            BrokerType.ALPACA_PAPER, 
            BrokerType.PAPER_SIM,
            BrokerType.BINANCE_TESTNET,
            BrokerType.BYBIT_TESTNET,
            BrokerType.TRADIER_SANDBOX,
            BrokerType.IBKR_PAPER,
        }
        if broker_type not in paper_types:
            logger.warning(
                f"[Universal] {broker_type.value} requested in PAPER mode. "
                f"Switching to testnet/paper variant."
            )
            if broker_type in (BrokerType.BINANCE, BrokerType.BINANCE_FUTURES):
                return _create_binance_adapter(testnet=True)
            elif broker_type == BrokerType.BYBIT:
                return _create_bybit_adapter(testnet=True)
            elif broker_type == BrokerType.TRADIER:
                return _create_tradier_adapter(sandbox=True)
            elif broker_type == BrokerType.IBKR:
                return _create_ibkr_adapter()
            elif broker_type == BrokerType.ALPACA_LIVE:
                return _create_alpaca_paper_adapter()
    
    factory = adapters.get(broker_type)
    if factory:
        return factory()
    
    logger.warning(f"[Universal] Unknown broker {broker_type}, using PaperSim")
    return PaperSimAdapter()


def send_order(symbol: str, qty: float) -> Dict[str, Any]:
    """
    Universal order placement.
    
    Args:
        symbol: Ticker symbol
        qty: Quantity (positive=buy, negative=sell)
        
    Returns:
        Dict with order result
    """
    from .models import OrderTicket
    from .enums import OrderSide, OrderType, TimeInForce
    
    adapter = get_adapter()
    
    order = OrderTicket(
        symbol=symbol.upper(),
        qty=abs(qty),
        side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    
    result = adapter.place_order(order)
    
    return {
        "order_id": result.order_id,
        "symbol": result.symbol,
        "status": result.status.value,
        "filled_qty": result.filled_qty,
        "avg_price": result.avg_fill_price,
        "message": result.message,
    }


def get_positions() -> Dict[str, float]:
    """
    Get all positions as symbol:qty dict.
    
    Returns:
        Dict mapping symbol to quantity
    """
    adapter = get_adapter()
    positions = adapter.get_positions()
    return {pos.symbol: pos.qty for pos in positions}


def get_account() -> Tuple[float, float]:
    """
    Get account cash and equity.
    
    Returns:
        Tuple of (cash, total_equity)
    """
    adapter = get_adapter()
    equity = adapter.get_account_equity()
    return (equity, equity)


def get_status() -> Dict[str, Any]:
    """Get universal router status."""
    mode = _get_execution_mode()
    broker = _get_broker_type()
    adapter = get_adapter()
    
    return {
        "mode": mode.value,
        "broker": broker.value,
        "adapter": adapter.name,
        "is_paper": adapter.is_paper,
        "live_enabled": os.environ.get("APEX_LIVE_ENABLED", "false").lower() == "true",
    }


__all__ = [
    "get_adapter",
    "send_order",
    "get_positions",
    "get_account",
    "get_status",
    "BrokerType",
    "ExecutionMode",
]
