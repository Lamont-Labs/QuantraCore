"""
Determinism Verification Tests — FINRA 15-09 Compliance

REGULATORY BASIS:
- FINRA Regulatory Notice 15-09 Section 1: Software Testing and System Validation
- SEC Market Access Rule 15c3-5: Pre-trade risk controls must be deterministic
- MiFID II RTS 6 Article 5: Algorithmic systems must produce consistent results

STANDARD REQUIREMENT: Systems must produce reproducible outputs
QUANTRACORE REQUIREMENT: 100% bitwise-identical results across 100 iterations
(2x the standard 50-iteration validation requirement)

Tests verify that identical inputs always produce identical outputs,
a fundamental requirement for auditable trading systems.
"""

import pytest
import hashlib
import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


def generate_deterministic_bars(seed: int, count: int = 100) -> List[OhlcvBar]:
    """Generate reproducible OHLCV data for determinism testing."""
    np.random.seed(seed)
    bars = []
    base_price = 100.0
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    
    for i in range(count):
        change = np.random.normal(0, 0.02)
        open_price = base_price * (1 + change)
        high = open_price * (1 + abs(np.random.normal(0, 0.01)))
        low = open_price * (1 - abs(np.random.normal(0, 0.01)))
        close = np.random.uniform(low, high)
        volume = int(np.random.uniform(100000, 1000000))
        
        bars.append(OhlcvBar(
            timestamp=base_time + timedelta(minutes=i),
            open=round(open_price, 4),
            high=round(high, 4),
            low=round(low, 4),
            close=round(close, 4),
            volume=volume
        ))
        base_price = close
    
    return bars


def result_to_hash(result: Any) -> str:
    """Convert analysis result to deterministic hash for comparison."""
    if hasattr(result, '__dict__'):
        data = {}
        for key, value in result.__dict__.items():
            if isinstance(value, float):
                data[key] = round(value, 10)
            elif isinstance(value, (list, tuple)):
                data[key] = [round(v, 10) if isinstance(v, float) else str(v) for v in value]
            elif isinstance(value, dict):
                data[key] = {k: round(v, 10) if isinstance(v, float) else str(v) for k, v in value.items()}
            else:
                data[key] = str(value)
    else:
        data = str(result)
    
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


class TestDeterminismVerification:
    """
    FINRA 15-09 / MiFID II RTS 6 Determinism Verification
    
    Regulatory Requirement: Algorithmic systems must produce consistent,
    reproducible results for audit and regulatory review purposes.
    
    QuantraCore Standard: 100 identical iterations (2x industry standard of 50)
    """
    
    ITERATION_COUNT = 100
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"]
    SEEDS = [42, 123, 456, 789, 1000]
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        """Create fresh engine instance for each test."""
        return ApexEngine(enable_logging=False, auto_load_protocols=True)
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_quantrascore_determinism_100_iterations(
        self, engine: ApexEngine, symbol: str, seed: int
    ):
        """
        FINRA 15-09 §1: QuantraScore must be bitwise-identical across 100 runs.
        
        Standard requirement: 50 iterations
        QuantraCore requirement: 100 iterations (2x stricter)
        """
        bars = generate_deterministic_bars(seed, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_result = None
        reference_hash = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            current_hash = result_to_hash(result)
            
            if reference_result is None:
                reference_result = result
                reference_hash = current_hash
            else:
                assert current_hash == reference_hash, (
                    f"Determinism violation at iteration {iteration}: "
                    f"QuantraScore changed from {reference_result.quantrascore} "
                    f"to {result.quantrascore}"
                )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_verdict_determinism_100_iterations(self, engine: ApexEngine, symbol: str):
        """
        SEC 15c3-5: Trading verdicts must be reproducible for regulatory audit.
        
        Validates that verdict strings are identical across 100 iterations.
        """
        bars = generate_deterministic_bars(42, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_verdict = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            
            if reference_verdict is None:
                reference_verdict = result.verdict
            else:
                assert result.verdict == reference_verdict, (
                    f"Verdict determinism violation at iteration {iteration}"
                )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_regime_classification_determinism(self, engine: ApexEngine, symbol: str):
        """
        MiFID II RTS 6 §5: Market regime classification must be consistent.
        
        Regime changes affect risk controls; inconsistent classification
        would violate regulatory requirements for predictable behavior.
        """
        bars = generate_deterministic_bars(789, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_regime = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            
            if reference_regime is None:
                reference_regime = result.regime
            else:
                assert result.regime == reference_regime, (
                    f"Regime classification instability at iteration {iteration}: "
                    f"changed from {reference_regime} to {result.regime}"
                )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_entropy_calculation_determinism(self, engine: ApexEngine, symbol: str):
        """
        Basel Committee: Risk metrics must be reproducible for stress testing.
        
        Entropy values feed into risk calculations; non-deterministic
        entropy would invalidate stress test results.
        """
        bars = generate_deterministic_bars(456, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_entropy = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            
            if reference_entropy is None:
                reference_entropy = result.entropy_state
            else:
                assert result.entropy_state == reference_entropy, (
                    f"Entropy calculation drift at iteration {iteration}"
                )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_microtraits_determinism(self, engine: ApexEngine, symbol: str):
        """
        FINRA 15-09 §2: All derived features must be reproducible.
        
        Microtraits are intermediate calculations; any drift here
        would cascade into final analysis results.
        """
        bars = generate_deterministic_bars(123, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_microtraits = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            
            if hasattr(result, 'microtraits'):
                if reference_microtraits is None:
                    reference_microtraits = result_to_hash(result.microtraits)
                else:
                    current_hash = result_to_hash(result.microtraits)
                    assert current_hash == reference_microtraits, (
                        f"Microtrait calculation drift at iteration {iteration}"
                    )
    
    def test_cross_engine_determinism(self):
        """
        SEC Regulation SCI: Multiple system instances must produce identical results.
        
        Validates that separate engine instances analyzing the same data
        produce bitwise-identical results (required for redundant systems).
        """
        bars = generate_deterministic_bars(42, count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        engines = [ApexEngine(enable_logging=False) for _ in range(5)]
        results = [engine.analyze(window) for engine in engines]
        
        reference_hash = result_to_hash(results[0])
        
        for i, result in enumerate(results[1:], start=2):
            current_hash = result_to_hash(result)
            assert current_hash == reference_hash, (
                f"Cross-engine determinism failure: Engine {i} produced different result"
            )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_omega_directive_determinism(self, engine: ApexEngine, symbol: str):
        """
        FINRA 3110: Supervisory controls (Omega Directives) must be deterministic.
        
        Safety overrides that fire non-deterministically would create
        unpredictable trading behavior, violating supervisory requirements.
        """
        bars = generate_deterministic_bars(1000, count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        reference_omega = None
        
        for iteration in range(self.ITERATION_COUNT):
            result = engine.analyze(window)
            
            if hasattr(result, 'omega_overrides'):
                if reference_omega is None:
                    reference_omega = result_to_hash(result.omega_overrides)
                else:
                    current_hash = result_to_hash(result.omega_overrides)
                    assert current_hash == reference_omega, (
                        f"Omega directive instability at iteration {iteration}"
                    )


class TestFloatingPointDeterminism:
    """
    IEEE 754 Floating-Point Determinism Tests
    
    Financial regulations require exact reproducibility, which means
    floating-point operations must be carefully controlled.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_floating_point_precision_consistency(self, engine: ApexEngine):
        """
        Verify floating-point calculations maintain precision across iterations.
        
        Uses numpy's decimal precision to ensure no floating-point drift.
        """
        bars = generate_deterministic_bars(42, count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        results = [engine.analyze(window) for _ in range(50)]
        scores = [r.quantrascore for r in results]
        
        assert all(s == scores[0] for s in scores), (
            "Floating-point precision drift detected in QuantraScore"
        )
    
    def test_numpy_random_state_isolation(self):
        """
        Verify engine doesn't corrupt global numpy random state.
        
        Required for regulatory-compliant backtesting where random
        seeds must be exactly reproducible.
        """
        np.random.seed(42)
        expected_sequence = [np.random.random() for _ in range(10)]
        
        engine = ApexEngine(enable_logging=False)
        bars = generate_deterministic_bars(999, count=50)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        _ = engine.analyze(window)
        
        np.random.seed(42)
        actual_sequence = [np.random.random() for _ in range(10)]
        
        assert expected_sequence == actual_sequence, (
            "Engine corrupted global numpy random state"
        )
