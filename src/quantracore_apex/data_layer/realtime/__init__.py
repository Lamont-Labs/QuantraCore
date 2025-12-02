"""
Real-time data streaming infrastructure for QuantraCore Apex.

Provides WebSocket connections to market data providers for:
- Live price updates
- Real-time trade streams
- Instant quote data

Requires: Alpaca Algo Trader Plus subscription ($99/month) or Elite tier
Falls back gracefully to EOD data on free tier.
"""

from .alpaca_stream import AlpacaRealtimeClient, RealtimeQuote, RealtimeTrade, RealtimeBar

__all__ = [
    "AlpacaRealtimeClient",
    "RealtimeQuote",
    "RealtimeTrade",
    "RealtimeBar",
]
