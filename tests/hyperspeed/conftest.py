"""Pytest fixtures for hyperspeed tests."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, date, timedelta

from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar
from src.quantracore_apex.hyperspeed import HyperspeedConfig
from src.quantracore_apex.hyperspeed.models import DataSource


def generate_mock_bars(symbol: str, num_bars: int = 150, start_price: float = 100.0) -> list[OhlcvBar]:
    """Generate realistic mock OHLCV bars for testing."""
    bars = []
    current_price = start_price
    base_time = datetime.utcnow() - timedelta(days=num_bars)
    
    for i in range(num_bars):
        change = (0.5 - (i % 5) / 10) * 0.02
        current_price = current_price * (1 + change)
        
        high = current_price * (1 + abs(change) * 0.5)
        low = current_price * (1 - abs(change) * 0.5)
        open_price = current_price * (1 + change * 0.3)
        close_price = current_price
        volume = int(1_000_000 + (i % 10) * 100_000)
        
        bars.append(OhlcvBar(
            timestamp=base_time + timedelta(days=i),
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
        ))
    
    return bars


@pytest.fixture
def mock_bars():
    """Provide mock OHLCV bars."""
    return generate_mock_bars("AAPL", 150)


@pytest.fixture
def mock_window(mock_bars):
    """Provide a mock OHLCV window."""
    return OhlcvWindow(
        symbol="AAPL",
        timeframe="1d",
        bars=mock_bars,
    )


@pytest.fixture
def test_config():
    """Provide a test hyperspeed configuration."""
    return HyperspeedConfig(
        replay_years=1,
        replay_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
        parallel_simulations=10,
        max_samples_per_cycle=100,
        overnight_training_enabled=False,
    )


@pytest.fixture
def mock_polygon_response():
    """Generate mock Polygon API response."""
    base_time = datetime.utcnow() - timedelta(days=30)
    results = []
    price = 150.0
    
    for i in range(30):
        results.append({
            "t": int((base_time + timedelta(days=i)).timestamp() * 1000),
            "o": price * 0.99,
            "h": price * 1.02,
            "l": price * 0.97,
            "c": price,
            "v": 10_000_000 + i * 100_000,
        })
        price *= 1.005
    
    return {"results": results, "resultsCount": len(results)}


@pytest.fixture
def mock_alpaca_response():
    """Generate mock Alpaca API response."""
    base_time = datetime.utcnow() - timedelta(days=30)
    bars = []
    price = 150.0
    
    for i in range(30):
        bars.append({
            "t": (base_time + timedelta(days=i)).isoformat() + "Z",
            "o": price * 0.99,
            "h": price * 1.02,
            "l": price * 0.97,
            "c": price,
            "v": 10_000_000 + i * 100_000,
        })
        price *= 1.005
    
    return {"bars": {"AAPL": bars}}


@pytest.fixture
def mock_options_flow():
    """Generate mock options flow data."""
    return {
        "symbol": "AAPL",
        "call_volume": 150000,
        "put_volume": 80000,
        "put_call_ratio": 0.53,
        "unusual_activity": True,
        "large_trades": [
            {"type": "call", "strike": 180, "premium": 2500000},
            {"type": "put", "strike": 170, "premium": 1200000},
        ],
    }


@pytest.fixture
def mock_dark_pool():
    """Generate mock dark pool data."""
    return {
        "symbol": "AAPL",
        "total_volume": 5_000_000,
        "percentage_of_total": 35.5,
        "avg_trade_size": 500,
        "block_trades": 25,
        "net_flow": "bullish",
    }


@pytest.fixture
def mock_sentiment():
    """Generate mock sentiment data."""
    return {
        "symbol": "AAPL",
        "overall_score": 0.72,
        "news_sentiment": 0.65,
        "social_sentiment": 0.78,
        "analyst_rating": "buy",
        "mentions_count": 15000,
    }
