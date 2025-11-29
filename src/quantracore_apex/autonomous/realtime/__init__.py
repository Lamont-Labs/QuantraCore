"""
Real-time data streaming infrastructure.

Provides WebSocket connections to market data providers
with automatic reconnection, heartbeat monitoring, and
data normalization to OhlcvBar format.
"""

from .polygon_stream import PolygonWebSocketStream
from .rolling_window import RollingWindowManager


__all__ = [
    "PolygonWebSocketStream",
    "RollingWindowManager",
]
