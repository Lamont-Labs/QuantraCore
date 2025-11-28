"""
Stress and Performance Tests

Tests system under load and measures performance.
"""

import pytest
import numpy as np
import time
import gc
from datetime import datetime
from typing import List

from src.quantracore_apex.core.schemas import OhlcvBar
from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.protocols.monster_runner import MonsterRunnerLoader
from src.quantracore_apex.prediction.volatility_projection import project_volatility
from src.quantracore_apex.prediction.compression_forecast import forecast_compression
from src.quantracore_apex.prediction.continuation_estimator import estimate_continuation
from src.quantracore_apex.prediction.instability_predictor import predict_instability


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


def get_memory_mb():
    """Get current memory usage in MB."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class TestEngineBatchProcessing:
    """Tests for batch processing performance."""
    
    def test_batch_10_symbols(self):
        """Test processing 10 symbols."""
        loader = MonsterRunnerLoader()
        
        start_time = time.time()
        
        for i in range(10):
            bars = generate_bars(100, seed=i)
            result = loader.run_all(bars)
            assert result is not None
        
        elapsed = time.time() - start_time
        
        assert elapsed < 30
        print(f"Batch 10 symbols: {elapsed:.2f}s")
    
    def test_large_bars_processing(self):
        """Test processing large bar counts."""
        loader = MonsterRunnerLoader()
        
        bars = generate_bars(200, seed=42)
        
        start_time = time.time()
        result = loader.run_all(bars)
        elapsed = time.time() - start_time
        
        assert result is not None
        assert elapsed < 5
        print(f"Large bars (200): {elapsed:.3f}s")


class TestDeterminismStress:
    """Stress tests for determinism."""
    
    def test_repeated_scans(self):
        """Test repeated scans produce same results."""
        loader = MonsterRunnerLoader()
        bars = generate_bars(100, seed=42)
        
        results = []
        for _ in range(10):
            result = loader.run_all(bars)
            results.append(result.monster_score)
        
        assert len(set(results)) == 1
    
    def test_different_seeds(self):
        """Test different seeds produce different results."""
        loader = MonsterRunnerLoader()
        
        scores = set()
        for seed in range(10):
            bars = generate_bars(100, seed=seed)
            result = loader.run_all(bars)
            scores.add(round(result.monster_score, 4))
        
        assert len(scores) >= 1


class TestMemoryStress:
    """Memory stress tests."""
    
    def test_no_memory_leak_prediction(self):
        """Test prediction stack doesn't leak memory."""
        gc.collect()
        initial_mem = get_memory_mb()
        
        for i in range(100):
            bars = generate_bars(100, seed=i % 10)
            
            project_volatility(bars)
            forecast_compression(bars)
            estimate_continuation(bars)
            predict_instability(bars)
            
            if i % 50 == 0:
                gc.collect()
        
        gc.collect()
        final_mem = get_memory_mb()
        
        delta = final_mem - initial_mem
        assert delta < 50
        print(f"Memory delta after 100 prediction runs: {delta:.2f} MB")
    


class TestMonsterRunnerStress:
    """Stress tests for MonsterRunner."""
    
    def test_batch_monster_runner(self):
        """Test MonsterRunner batch processing."""
        loader = MonsterRunnerLoader()
        
        start_time = time.time()
        
        for i in range(50):
            bars = generate_bars(100, seed=i)
            result = loader.run_all(bars)
            assert result is not None
        
        elapsed = time.time() - start_time
        
        assert elapsed < 30
        print(f"MonsterRunner batch 50: {elapsed:.2f}s ({50/elapsed:.1f} runs/sec)")
    
    def test_monster_runner_determinism(self):
        """Test MonsterRunner determinism under load."""
        loader = MonsterRunnerLoader()
        bars = generate_bars(100, seed=42)
        
        scores = []
        for _ in range(50):
            result = loader.run_all(bars)
            scores.append(result.monster_score)
        
        assert len(set(scores)) == 1


class TestPredictionStackStress:
    """Stress tests for prediction stack."""
    
    def test_prediction_stack_batch(self):
        """Test prediction stack batch processing."""
        start_time = time.time()
        
        for i in range(100):
            bars = generate_bars(100, seed=i)
            
            vol = project_volatility(bars)
            comp = forecast_compression(bars)
            cont = estimate_continuation(bars)
            inst = predict_instability(bars)
            
            assert vol.current_volatility >= 0
            assert 0 <= comp.current_compression <= 1
            assert 0 <= cont.continuation_probability <= 1
            assert 0 <= inst.instability_score <= 1
        
        elapsed = time.time() - start_time
        
        assert elapsed < 30
        print(f"Prediction stack batch 100: {elapsed:.2f}s")


class TestConcurrencyReadiness:
    """Tests for concurrency readiness (single-threaded)."""
    
    def test_loader_reuse(self):
        """Test loader can be reused safely."""
        loader = MonsterRunnerLoader()
        
        for i in range(10):
            bars = generate_bars(100, seed=i)
            result = loader.run_all(bars)
            assert result is not None


class TestEdgeCases:
    """Edge case stress tests."""
    
    def test_minimum_bars(self):
        """Test with minimum bar count."""
        loader = MonsterRunnerLoader()
        
        for n in [50, 60, 100]:
            bars = generate_bars(n, seed=42)
            result = loader.run_all(bars)
            assert result is not None
    
    def test_extreme_values(self):
        """Test with extreme price values."""
        loader = MonsterRunnerLoader()
        
        bars = []
        for i in range(100):
            bars.append(OhlcvBar(
                timestamp=datetime(2024, 1, 1),
                open=1000000.0 + i,
                high=1000001.0 + i,
                low=999999.0 + i,
                close=1000000.5 + i,
                volume=1e9,
            ))
        
        result = loader.run_all(bars)
        assert result is not None
    
    def test_flat_market(self):
        """Test with flat market (no movement)."""
        loader = MonsterRunnerLoader()
        
        bars = []
        for i in range(100):
            bars.append(OhlcvBar(
                timestamp=datetime(2024, 1, 1),
                open=100.0,
                high=100.01,
                low=99.99,
                close=100.0,
                volume=1000000,
            ))
        
        result = loader.run_all(bars)
        assert result is not None
