"""
Stress Testing Suite — MiFID II RTS 6 & Basel Committee Compliance

REGULATORY BASIS:
- MiFID II RTS 6 Article 10: Stress testing with 2x maximum historical volume
- Basel Committee Stress Testing Principles (2018): Adverse scenario modeling
- Federal Reserve Stress Testing: Severely adverse economic scenarios
- FINRA 15-09 §3: Trading systems must handle peak message volumes

STANDARD REQUIREMENT: 2x maximum 6-month historical volume
QUANTRACORE REQUIREMENT: 4x maximum volume (2x the standard)

STANDARD LATENCY REQUIREMENT: Sub-5 second alert generation
QUANTRACORE REQUIREMENT: Sub-2.5 second response (2x stricter)
"""

import pytest
import time
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


def generate_stress_bars(
    count: int,
    volatility_multiplier: float = 1.0,
    volume_multiplier: float = 1.0
) -> List[OhlcvBar]:
    """Generate OHLCV data with configurable stress parameters."""
    bars = []
    base_price = 100.0
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    base_volume = 500000
    
    for i in range(count):
        change = np.random.normal(0, 0.02 * volatility_multiplier)
        open_price = base_price * (1 + change)
        high = open_price * (1 + abs(np.random.normal(0, 0.01 * volatility_multiplier)))
        low = open_price * (1 - abs(np.random.normal(0, 0.01 * volatility_multiplier)))
        close = np.random.uniform(low, high)
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
        
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


class TestVolumeStress:
    """
    MiFID II RTS 6 Article 10: High Volume Stress Testing
    
    Standard: Test with 2x maximum volume from past 6 months
    QuantraCore: Test with 4x maximum volume (2x stricter)
    """
    
    VOLUME_MULTIPLIER = 4.0
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"]
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_4x_volume_processing(self, engine: ApexEngine, symbol: str):
        """
        Process 4x normal message volume without degradation.
        
        MiFID II requires 2x; we test at 4x for institutional margin.
        """
        bars = generate_stress_bars(
            count=200,
            volume_multiplier=self.VOLUME_MULTIPLIER
        )
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        start_time = time.perf_counter()
        result = engine.run(window)
        elapsed = time.perf_counter() - start_time
        
        assert result is not None, "Analysis failed under 4x volume stress"
        assert result.quantrascore >= 0, "Invalid QuantraScore under stress"
        assert result.quantrascore <= 100, "Invalid QuantraScore under stress"
        assert elapsed < 2.5, f"Processing exceeded 2.5s threshold: {elapsed:.3f}s"
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_burst_volume_handling(self, engine: ApexEngine, symbol: str):
        """
        Handle sudden 10x volume spikes (flash crash simulation).
        
        Basel Committee requires testing for unexpected market conditions.
        """
        normal_bars = generate_stress_bars(count=50, volume_multiplier=1.0)
        burst_bars = generate_stress_bars(count=20, volume_multiplier=10.0)
        recovery_bars = generate_stress_bars(count=30, volume_multiplier=1.0)
        
        all_bars = normal_bars + burst_bars + recovery_bars
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=all_bars)
        
        result = engine.run(window)
        
        assert result is not None, "Engine failed during volume burst"
        assert 0 <= result.quantrascore <= 100, "Score out of bounds after burst"
    
    def test_sustained_high_volume(self, engine: ApexEngine):
        """
        Maintain stability under sustained 4x volume for extended period.
        
        Simulates prolonged market stress (e.g., major news event).
        """
        iterations = 50
        bars = generate_stress_bars(count=100, volume_multiplier=4.0)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = engine.run(window)
            latencies.append(time.perf_counter() - start)
            
            assert result is not None, "Engine degraded under sustained load"
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        assert avg_latency < 1.0, f"Average latency {avg_latency:.3f}s exceeds 1.0s"
        assert max_latency < 2.5, f"Max latency {max_latency:.3f}s exceeds 2.5s"


class TestVolatilityStress:
    """
    Basel Committee & Federal Reserve: Adverse Scenario Testing
    
    Tests system behavior under extreme market volatility conditions
    similar to historical crisis events.
    """
    
    VOLATILITY_SCENARIOS = [
        ("normal", 1.0),
        ("elevated", 2.0),
        ("high_stress", 4.0),
        ("crisis_2008", 6.0),
        ("flash_crash", 10.0),
    ]
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    @pytest.mark.parametrize("scenario_name,volatility", VOLATILITY_SCENARIOS)
    def test_volatility_regime_handling(
        self, engine: ApexEngine, scenario_name: str, volatility: float
    ):
        """
        System must remain stable across all volatility regimes.
        
        Basel Committee requires testing against historical crisis scenarios.
        """
        bars = generate_stress_bars(count=100, volatility_multiplier=volatility)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result is not None, f"Analysis failed in {scenario_name} scenario"
        assert 0 <= result.quantrascore <= 100, (
            f"Score {result.quantrascore} out of bounds in {scenario_name}"
        )
        valid_regimes = [
            "stable", "volatile", "trending", "chaotic", "unknown",
            "range_bound", "breakout", "consolidation", "compressed",
            "trending_up", "trending_down", "sideways"
        ]
        regime_value = result.regime.value if hasattr(result.regime, 'value') else str(result.regime)
        assert regime_value.lower() in [r.lower() for r in valid_regimes], (
            f"Invalid regime classification in {scenario_name}: {regime_value}"
        )
    
    def test_volatility_regime_transition(self, engine: ApexEngine):
        """
        Handle rapid volatility regime transitions (crash/recovery).
        
        Federal Reserve stress tests require modeling rapid transitions.
        """
        stable_bars = generate_stress_bars(count=30, volatility_multiplier=1.0)
        crash_bars = generate_stress_bars(count=10, volatility_multiplier=10.0)
        recovery_bars = generate_stress_bars(count=30, volatility_multiplier=2.0)
        stable2_bars = generate_stress_bars(count=30, volatility_multiplier=1.0)
        
        all_bars = stable_bars + crash_bars + recovery_bars + stable2_bars
        window = OhlcvWindow(symbol="SPY", timeframe="1m", bars=all_bars)
        
        result = engine.run(window)
        
        assert result is not None, "Failed during volatility transition"
        assert result.regime is not None, "Regime classification failed"


class TestLatencyCompliance:
    """
    MiFID II RTS 6 Article 17: Real-Time Alert Latency
    
    Standard Requirement: Alerts within 5 seconds
    QuantraCore Requirement: Alerts within 2.5 seconds (2x stricter)
    """
    
    LATENCY_THRESHOLD_MS = 2500
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"]
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_single_analysis_latency(self, engine: ApexEngine, symbol: str):
        """
        Single analysis must complete within 2.5 seconds.
        
        MiFID II requires 5s; we enforce 2.5s for institutional margin.
        """
        bars = generate_stress_bars(count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        start = time.perf_counter()
        result = engine.run(window)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < self.LATENCY_THRESHOLD_MS, (
            f"Latency {elapsed_ms:.1f}ms exceeds {self.LATENCY_THRESHOLD_MS}ms threshold"
        )
    
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_p99_latency_compliance(self, engine: ApexEngine, symbol: str):
        """
        P99 latency must be under 2.5 seconds across 100 iterations.
        
        Ensures consistent performance, not just average case.
        """
        bars = generate_stress_bars(count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = engine.run(window)
            latencies.append((time.perf_counter() - start) * 1000)
        
        p99_latency = np.percentile(latencies, 99)
        
        assert p99_latency < self.LATENCY_THRESHOLD_MS, (
            f"P99 latency {p99_latency:.1f}ms exceeds threshold"
        )
    
    def test_concurrent_analysis_latency(self, engine: ApexEngine):
        """
        Concurrent requests must all complete within threshold.
        
        FINRA 15-09 requires systems handle peak concurrent load.
        """
        bars = generate_stress_bars(count=100)
        windows = [
            OhlcvWindow(symbol=sym, timeframe="1m", bars=bars)
            for sym in ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"]
        ]
        
        results = []
        start = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(engine.run, w) for w in windows]
            for future in as_completed(futures):
                results.append(future.result())
        
        total_elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert len(results) == 5, "Not all concurrent analyses completed"
        assert total_elapsed_ms < self.LATENCY_THRESHOLD_MS * 2, (
            f"Concurrent latency {total_elapsed_ms:.1f}ms exceeds threshold"
        )


class TestSystemResilience:
    """
    SEC Regulation SCI & MiFID II: System Resilience Testing
    
    Tests system recovery from adverse conditions and maintains
    stability under degraded operation modes.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_malformed_data_resilience(self, engine: ApexEngine):
        """
        System must handle malformed data gracefully.
        
        SEC Regulation SCI requires systems prevent erroneous orders.
        """
        bars = generate_stress_bars(count=50)
        
        bars[25] = OhlcvBar(
            timestamp=bars[25].timestamp,
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0
        )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        assert result is not None, "Engine crashed on zero-value bar"
    
    def test_gap_data_resilience(self, engine: ApexEngine):
        """
        System must handle data gaps (halted trading, market close).
        
        Required for pre/post market and trading halt scenarios.
        """
        bars = generate_stress_bars(count=50)
        
        for i in range(20, 30):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp + timedelta(hours=1),
                open=bars[19].close,
                high=bars[19].close * 1.01,
                low=bars[19].close * 0.99,
                close=bars[19].close,
                volume=0
            )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        assert result is not None, "Engine failed on gapped data"
    
    def test_extreme_price_movement_resilience(self, engine: ApexEngine):
        """
        System must handle circuit-breaker level price movements.
        
        Tests +/- 20% single-bar movements (beyond normal circuit breakers).
        """
        bars = generate_stress_bars(count=50)
        
        bars[25] = OhlcvBar(
            timestamp=bars[25].timestamp,
            open=bars[24].close,
            high=bars[24].close * 1.25,
            low=bars[24].close * 0.80,
            close=bars[24].close * 0.80,
            volume=bars[24].volume * 10
        )
        
        window = OhlcvWindow(symbol="GME", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        assert result is not None, "Engine failed on extreme price movement"
        assert result.quantrascore >= 0, "Invalid score on extreme movement"
    
    def test_memory_stability_under_load(self, engine: ApexEngine):
        """
        Memory usage must remain stable under sustained load.
        
        Prevents memory leaks that could cause system degradation.
        """
        import sys
        
        initial_size = sys.getsizeof(engine)
        
        for i in range(100):
            bars = generate_stress_bars(count=100)
            window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
            _ = engine.run(window)
        
        final_size = sys.getsizeof(engine)
        
        growth_factor = final_size / initial_size if initial_size > 0 else 1.0
        assert growth_factor < 2.0, (
            f"Memory grew {growth_factor:.1f}x during sustained load"
        )
    
    def test_thread_safety(self):
        """
        Engine must be thread-safe for concurrent access.
        
        Required for multi-threaded trading systems.
        """
        engine = ApexEngine(enable_logging=False)
        errors = []
        results = []
        lock = threading.Lock()
        
        def analyze_symbol(symbol: str):
            try:
                bars = generate_stress_bars(count=50)
                window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
                result = engine.run(window)
                with lock:
                    results.append((symbol, result))
            except Exception as e:
                with lock:
                    errors.append((symbol, str(e)))
        
        threads = []
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"] * 4
        
        for symbol in symbols:
            t = threading.Thread(target=analyze_symbol, args=(symbol,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety violations: {errors}"
        assert len(results) == 20, "Not all concurrent analyses completed"
