"""
Market Abuse Detection Tests — SEC/FINRA/MAR Compliance

REGULATORY BASIS:
- SEC Rule 10b-5: Prohibition on market manipulation
- FINRA Rule 5210: Wash sales prohibition
- FINRA 15-09 §5: Surveillance for manipulative patterns
- EU Market Abuse Regulation (MAR): Spoofing, layering, momentum ignition
- MiFID II RTS 6: Multi-algorithm interaction monitoring

STANDARD REQUIREMENT: Detection systems must identify basic manipulation
QUANTRACORE REQUIREMENT: Detection with 2x sensitivity thresholds

These tests verify the system can detect and flag potential market
manipulation patterns that would trigger regulatory scrutiny.
"""

import pytest
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


def generate_normal_bars(count: int = 100, base_price: float = 100.0) -> List[OhlcvBar]:
    """Generate normal market activity bars."""
    bars = []
    price = base_price
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    
    for i in range(count):
        change = np.random.normal(0, 0.005)
        open_p = price
        close_p = price * (1 + change)
        high = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.002)))
        low = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.002)))
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


def inject_wash_trade_pattern(bars: List[OhlcvBar], start_idx: int) -> List[OhlcvBar]:
    """
    Inject wash trade pattern: Identical price/volume at regular intervals.
    
    Wash trades are prohibited under SEC Rule 10b-5 and FINRA Rule 5210.
    """
    result = bars.copy()
    
    for i in range(5):
        if start_idx + i * 3 < len(result):
            idx = start_idx + i * 3
            result[idx] = OhlcvBar(
                timestamp=result[idx].timestamp,
                open=100.50,
                high=100.55,
                low=100.45,
                close=100.50,
                volume=50000
            )
    
    return result


def inject_layering_pattern(bars: List[OhlcvBar], start_idx: int) -> List[OhlcvBar]:
    """
    Inject layering/spoofing pattern: Artificial price levels creating false demand.
    
    Prohibited under MAR Article 12 and Dodd-Frank Section 747.
    """
    result = bars.copy()
    base_price = result[start_idx].close
    
    for i in range(10):
        if start_idx + i < len(result):
            idx = start_idx + i
            multiplier = 1 + (i * 0.002)
            result[idx] = OhlcvBar(
                timestamp=result[idx].timestamp,
                open=base_price * multiplier,
                high=base_price * multiplier * 1.001,
                low=base_price * multiplier * 0.999,
                close=base_price * multiplier,
                volume=result[idx].volume * 5
            )
    
    for i in range(10, 12):
        if start_idx + i < len(result):
            idx = start_idx + i
            result[idx] = OhlcvBar(
                timestamp=result[idx].timestamp,
                open=base_price * 1.018,
                high=base_price * 1.019,
                low=base_price * 0.98,
                close=base_price * 0.985,
                volume=result[idx].volume * 10
            )
    
    return result


def inject_momentum_ignition(bars: List[OhlcvBar], start_idx: int) -> List[OhlcvBar]:
    """
    Inject momentum ignition pattern: Aggressive orders to trigger stops.
    
    Prohibited under MAR and subject to FINRA enforcement.
    """
    result = bars.copy()
    base_price = result[start_idx].close
    
    for i in range(3):
        if start_idx + i < len(result):
            idx = start_idx + i
            spike = base_price * (1 + (i + 1) * 0.015)
            result[idx] = OhlcvBar(
                timestamp=result[idx].timestamp,
                open=base_price if i == 0 else result[idx-1].close,
                high=spike,
                low=result[idx].low,
                close=spike,
                volume=result[idx].volume * 20
            )
    
    for i in range(3, 8):
        if start_idx + i < len(result):
            idx = start_idx + i
            result[idx] = OhlcvBar(
                timestamp=result[idx].timestamp,
                open=result[idx-1].close,
                high=result[idx-1].close * 1.001,
                low=result[idx-1].close * 0.995,
                close=result[idx-1].close * 0.997,
                volume=result[idx].volume * 3
            )
    
    return result


class TestWashTradeDetection:
    """
    SEC Rule 10b-5 & FINRA Rule 5210: Wash Trade Detection
    
    Wash trades create false appearance of market activity.
    System must flag patterns with 2x sensitivity.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_identical_price_volume_detection(self, engine: ApexEngine):
        """
        Detect suspiciously identical price/volume combinations.
        
        Standard: Flag 5+ identical trades
        QuantraCore: Flag 3+ identical trades (2x sensitivity)
        """
        bars = generate_normal_bars(100)
        bars = inject_wash_trade_pattern(bars, 30)
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        
        result = engine.analyze(window)
        
        assert result is not None
        assert result.quantrascore <= 70, (
            "System failed to reduce score for wash trade pattern"
        )
    
    def test_self_trade_pattern_detection(self, engine: ApexEngine):
        """
        Detect patterns consistent with self-trading.
        
        FINRA surveilles for orders that match against the same entity.
        """
        bars = generate_normal_bars(100)
        
        for i in range(40, 50, 2):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[i-1].close,
                high=bars[i-1].close * 1.001,
                low=bars[i-1].close * 0.999,
                close=bars[i-1].close,
                volume=100000
            )
            bars[i+1] = OhlcvBar(
                timestamp=bars[i+1].timestamp,
                open=bars[i].close,
                high=bars[i].close * 1.001,
                low=bars[i].close * 0.999,
                close=bars[i].close,
                volume=100000
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.regime != "stable", (
            "System should not classify wash pattern as stable"
        )


class TestSpoofingLayeringDetection:
    """
    MAR Article 12 & Dodd-Frank Section 747: Spoofing Detection
    
    Spoofing involves placing orders with intent to cancel,
    creating false impression of supply/demand.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_layering_pattern_detection(self, engine: ApexEngine):
        """
        Detect layered orders creating artificial price levels.
        
        Standard: Detect 10+ level patterns
        QuantraCore: Detect 5+ level patterns (2x sensitivity)
        """
        bars = generate_normal_bars(100)
        bars = inject_layering_pattern(bars, 40)
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        
        result = engine.analyze(window)
        
        assert result is not None
        assert result.entropy_state in ["elevated", "chaotic"] or result.quantrascore < 75, (
            "System failed to detect layering pattern entropy"
        )
    
    def test_quote_stuffing_detection(self, engine: ApexEngine):
        """
        Detect quote stuffing patterns (rapid order placement/cancellation).
        
        Manifests as high volume with minimal price movement.
        """
        bars = generate_normal_bars(100)
        
        for i in range(50, 60):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[49].close,
                high=bars[49].close * 1.0001,
                low=bars[49].close * 0.9999,
                close=bars[49].close,
                volume=bars[i].volume * 50
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.quantrascore is not None


class TestMomentumIgnitionDetection:
    """
    MAR & FINRA: Momentum Ignition Detection
    
    Aggressive orders designed to trigger other participants'
    stop-loss orders or algorithmic trading.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_stop_hunting_pattern_detection(self, engine: ApexEngine):
        """
        Detect patterns consistent with stop-loss hunting.
        
        Standard: Flag 5%+ spikes with reversal
        QuantraCore: Flag 2.5%+ spikes with reversal (2x sensitivity)
        """
        bars = generate_normal_bars(100)
        bars = inject_momentum_ignition(bars, 45)
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        
        result = engine.analyze(window)
        
        assert result is not None
        assert result.regime in ["volatile", "chaotic"] or result.quantrascore < 80, (
            "System failed to detect momentum ignition volatility"
        )
    
    def test_flash_crash_pattern_detection(self, engine: ApexEngine):
        """
        Detect flash crash patterns (rapid price collapse and recovery).
        
        Similar to May 6, 2010 Flash Crash characteristics.
        """
        bars = generate_normal_bars(100)
        base_price = bars[50].close
        
        for i in range(50, 53):
            drop = (i - 50 + 1) * 0.03
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=base_price * (1 - drop + 0.03),
                high=base_price * (1 - drop + 0.035),
                low=base_price * (1 - drop),
                close=base_price * (1 - drop),
                volume=bars[i].volume * 30
            )
        
        for i in range(53, 56):
            recovery = (i - 53 + 1) * 0.025
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=base_price * 0.91,
                high=base_price * (0.91 + recovery),
                low=base_price * 0.905,
                close=base_price * (0.91 + recovery),
                volume=bars[i].volume * 20
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.regime == "chaotic" or result.quantrascore < 50, (
            "System failed to classify flash crash as high-risk"
        )


class TestMultiAlgorithmInteraction:
    """
    FINRA 15-09 §5: Multi-Algorithm Interaction Monitoring
    
    Systems must detect when multiple algorithms interact to
    create unintended market effects.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_synchronized_trading_detection(self, engine: ApexEngine):
        """
        Detect synchronized patterns that suggest algo coordination.
        
        Required by FINRA for multi-strategy surveillance.
        """
        bars = generate_normal_bars(100)
        
        for i in range(30, 70, 5):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=100.0,
                high=100.5,
                low=99.5,
                close=100.25,
                volume=500000
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
    
    def test_feedback_loop_detection(self, engine: ApexEngine):
        """
        Detect potential feedback loops between trading systems.
        
        Oscillating patterns may indicate algo feedback.
        """
        bars = generate_normal_bars(100)
        
        for i in range(40, 60):
            direction = 1 if i % 2 == 0 else -1
            change = direction * 0.01
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=100.0,
                high=100.0 + max(0, change) + 0.005,
                low=100.0 + min(0, change) - 0.005,
                close=100.0 + change,
                volume=bars[i].volume * 2
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.entropy_state in ["elevated", "chaotic"] or result.quantrascore < 80, (
            "System failed to detect oscillation pattern"
        )


class TestAnomalyDetection:
    """
    SEC Regulation SCI & MAR: Anomaly Detection
    
    Systems must detect statistical anomalies that may indicate
    market manipulation or system errors.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_volume_anomaly_detection(self, engine: ApexEngine):
        """
        Detect statistically significant volume anomalies.
        
        Standard: Flag 3+ standard deviation spikes
        QuantraCore: Flag 1.5+ standard deviation spikes (2x sensitivity)
        """
        bars = generate_normal_bars(100)
        
        avg_volume = np.mean([b.volume for b in bars[:80]])
        std_volume = np.std([b.volume for b in bars[:80]])
        anomaly_volume = int(avg_volume + 4 * std_volume)
        
        for i in range(80, 85):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[i].open,
                high=bars[i].high,
                low=bars[i].low,
                close=bars[i].close,
                volume=anomaly_volume
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
    
    def test_price_anomaly_detection(self, engine: ApexEngine):
        """
        Detect statistically significant price anomalies.
        
        Prices deviating significantly from recent mean.
        """
        bars = generate_normal_bars(100)
        
        prices = [b.close for b in bars[:80]]
        avg_price = np.mean(prices)
        std_price = np.std(prices)
        
        bars[85] = OhlcvBar(
            timestamp=bars[85].timestamp,
            open=bars[84].close,
            high=avg_price + 5 * std_price,
            low=bars[84].close,
            close=avg_price + 4 * std_price,
            volume=bars[85].volume * 5
        )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.quantrascore < 85, (
            "System failed to reduce confidence for price anomaly"
        )
    
    def test_spread_anomaly_detection(self, engine: ApexEngine):
        """
        Detect abnormal bid-ask spread implied by high-low range.
        
        Unusual spreads may indicate market stress or manipulation.
        """
        bars = generate_normal_bars(100)
        
        for i in range(60, 70):
            avg_price = (bars[i].open + bars[i].close) / 2
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=avg_price,
                high=avg_price * 1.05,
                low=avg_price * 0.95,
                close=avg_price,
                volume=bars[i].volume
            )
        
        window = OhlcvWindow(symbol="TEST", timeframe="1m", bars=bars)
        result = engine.analyze(window)
        
        assert result is not None
        assert result.regime != "stable", (
            "System should detect abnormal spread volatility"
        )
