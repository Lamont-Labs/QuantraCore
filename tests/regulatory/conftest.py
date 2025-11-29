"""
Regulatory Test Suite Configuration — QuantraCore Apex v9.0-A

This conftest.py provides shared fixtures and configuration for
all regulatory compliance tests.

REGULATORY REFERENCES:
- SEC Market Access Rule (15c3-5)
- FINRA Rule 3110 & 15-09
- MiFID II RTS 6
- Basel Committee Stress Testing Principles
- Federal Reserve SR 11-7 (Model Risk Management)
"""

import pytest
import numpy as np
from typing import List
from datetime import datetime, timedelta

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


@pytest.fixture(scope="session")
def regulatory_engine() -> ApexEngine:
    """
    Session-scoped engine for regulatory tests.
    
    Provides consistent engine instance across test session.
    """
    return ApexEngine(enable_logging=False, auto_load_protocols=True)


@pytest.fixture(scope="function")
def fresh_engine() -> ApexEngine:
    """
    Function-scoped fresh engine for isolation.
    
    Each test gets a new engine instance.
    """
    return ApexEngine(enable_logging=False, auto_load_protocols=True)


@pytest.fixture
def standard_bars() -> List[OhlcvBar]:
    """
    Generate standard market bars for baseline testing.
    
    100 bars of normal market activity.
    """
    np.random.seed(42)
    bars = []
    price = 100.0
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    
    for i in range(100):
        change = np.random.normal(0, 0.01)
        open_p = price
        close_p = price * (1 + change)
        high = max(open_p, close_p) * 1.002
        low = min(open_p, close_p) * 0.998
        volume = int(np.random.uniform(100000, 300000))
        
        bars.append(OhlcvBar(
            timestamp=base_time + timedelta(minutes=i),
            open=round(open_p, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close_p, 4),
            volume=volume
        ))
        price = close_p
    
    return bars


@pytest.fixture
def volatile_bars() -> List[OhlcvBar]:
    """
    Generate high-volatility bars for stress testing.
    
    Simulates market stress conditions.
    """
    np.random.seed(42)
    bars = []
    price = 100.0
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    
    for i in range(100):
        change = np.random.normal(0, 0.05)
        open_p = price
        close_p = price * (1 + change)
        high = max(open_p, close_p) * 1.01
        low = min(open_p, close_p) * 0.99
        volume = int(np.random.uniform(500000, 2000000))
        
        bars.append(OhlcvBar(
            timestamp=base_time + timedelta(minutes=i),
            open=round(open_p, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close_p, 4),
            volume=volume
        ))
        price = close_p
    
    return bars


@pytest.fixture
def standard_window(standard_bars: List[OhlcvBar]) -> OhlcvWindow:
    """Standard OHLCV window for testing."""
    return OhlcvWindow(symbol="AAPL", timeframe="1m", bars=standard_bars)


@pytest.fixture
def volatile_window(volatile_bars: List[OhlcvBar]) -> OhlcvWindow:
    """High-volatility OHLCV window for stress testing."""
    return OhlcvWindow(symbol="AAPL", timeframe="1m", bars=volatile_bars)


def pytest_configure(config):
    """
    Configure pytest for regulatory test suite.
    
    Adds custom markers for regulatory test categories.
    """
    config.addinivalue_line(
        "markers", "determinism: marks tests verifying deterministic behavior"
    )
    config.addinivalue_line(
        "markers", "stress: marks stress tests (4x volume, high volatility)"
    )
    config.addinivalue_line(
        "markers", "latency: marks latency compliance tests (2.5s threshold)"
    )
    config.addinivalue_line(
        "markers", "market_abuse: marks market abuse detection tests"
    )
    config.addinivalue_line(
        "markers", "risk_control: marks risk control validation tests"
    )
    config.addinivalue_line(
        "markers", "backtest: marks backtesting validation tests"
    )
    config.addinivalue_line(
        "markers", "compliance: marks regulatory compliance tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on module location.
    
    Enables filtering by regulatory category.
    """
    for item in items:
        if "determinism" in item.nodeid:
            item.add_marker(pytest.mark.determinism)
        if "stress" in item.nodeid:
            item.add_marker(pytest.mark.stress)
        if "latency" in item.nodeid:
            item.add_marker(pytest.mark.latency)
        if "market_abuse" in item.nodeid:
            item.add_marker(pytest.mark.market_abuse)
        if "risk_control" in item.nodeid:
            item.add_marker(pytest.mark.risk_control)
        if "backtest" in item.nodeid:
            item.add_marker(pytest.mark.backtest)


class RegulatoryTestMetrics:
    """
    Tracks metrics for regulatory compliance reporting.
    
    Used to generate audit-ready test reports.
    """
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.determinism_checks = 0
        self.stress_scenarios_tested = 0
        self.latency_violations = 0
    
    def record_pass(self, test_type: str = "general"):
        self.tests_run += 1
        self.tests_passed += 1
        
        if test_type == "determinism":
            self.determinism_checks += 1
        elif test_type == "stress":
            self.stress_scenarios_tested += 1
    
    def record_fail(self, test_type: str = "general"):
        self.tests_run += 1
        self.tests_failed += 1
        
        if test_type == "latency":
            self.latency_violations += 1
    
    def compliance_rate(self) -> float:
        if self.tests_run == 0:
            return 100.0
        return (self.tests_passed / self.tests_run) * 100


@pytest.fixture(scope="session")
def regulatory_metrics() -> RegulatoryTestMetrics:
    """Session-scoped metrics tracker for compliance reporting."""
    return RegulatoryTestMetrics()
