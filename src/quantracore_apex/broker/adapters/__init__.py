"""
Broker Adapters for QuantraCore Apex.

Provides pluggable broker implementations for different execution modes.

Universal Broker Support:
- Alpaca (paper + live)
- Binance (spot + futures)
- Interactive Brokers (TWS/Gateway)
- Bybit (spot + derivatives)
- Tradier (equities + options)
"""

from .base_adapter import BrokerAdapter
from .null_adapter import NullAdapter
from .paper_sim_adapter import PaperSimAdapter
from .alpaca_adapter import AlpacaPaperAdapter
from .binance_adapter import BinanceAdapter
from .ibkr_adapter import IBKRAdapter
from .bybit_adapter import BybitAdapter
from .tradier_adapter import TradierAdapter

__all__ = [
    "BrokerAdapter",
    "NullAdapter",
    "PaperSimAdapter",
    "AlpacaPaperAdapter",
    "BinanceAdapter",
    "IBKRAdapter",
    "BybitAdapter",
    "TradierAdapter",
]
