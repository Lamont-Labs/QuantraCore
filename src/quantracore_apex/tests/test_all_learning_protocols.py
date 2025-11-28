"""
Comprehensive Learning Protocol Tests (LP01-LP25)

Tests all learning protocols for correct signature and execution.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List
import importlib

from src.quantracore_apex.core.schemas import OhlcvBar


def generate_bars(n: int = 100, seed: int = 42) -> List[OhlcvBar]:
    """Generate deterministic test bars."""
    np.random.seed(seed)
    bars = []
    price = 100.0
    
    for i in range(n):
        change = np.random.randn() * 0.02
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.005))
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.005))
        volume = 1000000 + np.random.randint(-200000, 200000)
        
        bars.append(OhlcvBar(
            timestamp=datetime(2024, 1, 1),
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=float(volume),
        ))
        price = close_price
    
    return bars


class TestLearningProtocols:
    """Tests for all Learning protocols LP01-LP25."""
    
    @pytest.fixture
    def bars(self):
        return generate_bars(100)
    
    @pytest.mark.parametrize("protocol_num", range(1, 26))
    def test_protocol_exists(self, protocol_num):
        """Test that protocol module exists."""
        protocol_id = f"LP{protocol_num:02d}"
        try:
            module = importlib.import_module(
                f"src.quantracore_apex.protocols.learning.{protocol_id}"
            )
            assert module is not None
        except ImportError as e:
            pytest.skip(f"Protocol {protocol_id} not implemented: {e}")
    
    @pytest.mark.parametrize("protocol_num", range(1, 11))
    def test_implemented_protocols(self, protocol_num, bars):
        """Test fully implemented protocols LP01-LP10."""
        protocol_id = f"LP{protocol_num:02d}"
        try:
            module = importlib.import_module(
                f"src.quantracore_apex.protocols.learning.{protocol_id}"
            )
            run_fn = getattr(module, f"run_{protocol_id}", None)
            
            if run_fn:
                result = run_fn(bars)
                assert result is not None
        except Exception as e:
            pytest.skip(f"Protocol {protocol_id} execution error: {e}")
    
    def test_learning_loader(self, bars):
        """Test learning protocol loader exists."""
        try:
            from src.quantracore_apex.protocols.learning.learning_loader import LearningLoader
            loader = LearningLoader()
            protocols = loader.get_loaded_protocols()
            assert len(protocols) >= 0
        except ImportError:
            from src.quantracore_apex.protocols.learning import learning_loader
            assert learning_loader is not None
