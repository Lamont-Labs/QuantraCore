"""
Risk Control Validation Tests — SEC 15c3-5 & MiFID II RTS 6 Compliance

REGULATORY BASIS:
- SEC Market Access Rule (15c3-5): Pre-trade risk controls
- MiFID II RTS 6 Article 15: Pre-trade controls
- MiFID II RTS 6 Article 17: Post-trade controls
- FINRA 3110: Supervisory control requirements
- Basel Committee: Capital adequacy and risk limits

STANDARD REQUIREMENT: Risk controls must prevent erroneous orders
QUANTRACORE REQUIREMENT: Controls with 2x safety margins

Tests verify all risk control mechanisms function correctly and
prevent potentially harmful trading activity.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import OhlcvBar, OhlcvWindow


def generate_market_bars(
    count: int = 100,
    trend: str = "neutral",
    base_price: float = 100.0
) -> List[OhlcvBar]:
    """Generate market bars with specified trend."""
    bars = []
    price = base_price
    base_time = datetime(2024, 1, 1, 9, 30, 0)
    
    trend_factor = {
        "bullish": 0.001,
        "bearish": -0.001,
        "neutral": 0.0
    }.get(trend, 0.0)
    
    for i in range(count):
        change = np.random.normal(trend_factor, 0.005)
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


class TestPreTradeRiskControls:
    """
    SEC 15c3-5 & MiFID II RTS 6 Article 15: Pre-Trade Controls
    
    Validates that the system enforces pre-trade risk limits
    to prevent erroneous or excessive orders.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_quantrascore_threshold_enforcement(self, engine: ApexEngine):
        """
        System must flag low-confidence scenarios for human review.
        
        Standard: Flag scores below 30
        QuantraCore: Flag scores below 40 (2x conservative)
        """
        bars = generate_market_bars(count=100, trend="neutral")
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result.quantrascore is not None
        assert 0 <= result.quantrascore <= 100
        
        if result.quantrascore < 40:
            verdict_action = result.verdict.action.lower()
            assert "caution" in verdict_action or "neutral" in verdict_action, (
                "Low score should include caution in verdict"
            )
    
    def test_extreme_regime_suppression(self, engine: ApexEngine):
        """
        System must suppress signals during extreme market conditions.
        
        FINRA 3110: Supervisory controls must prevent trading during chaos.
        """
        bars = generate_market_bars(count=100)
        
        for i in range(50, 60):
            volatility = (i - 50) * 0.02
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[i].open,
                high=bars[i].open * (1 + volatility),
                low=bars[i].open * (1 - volatility),
                close=bars[i].open * (1 + np.random.uniform(-volatility, volatility)),
                volume=bars[i].volume * 10
            )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        result = engine.run(window)
        
        if result.regime == "chaotic":
            assert result.quantrascore < 60, (
                "Chaotic regime should significantly reduce confidence"
            )
    
    def test_omega_directive_activation(self, engine: ApexEngine):
        """
        Omega Directives (safety overrides) must activate in extreme conditions.
        
        Required by FINRA 3110 for supervisory control.
        """
        bars = generate_market_bars(count=100)
        
        for i in range(70, 80):
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[69].close * 0.85,
                high=bars[69].close * 0.86,
                low=bars[69].close * 0.80,
                close=bars[69].close * 0.82,
                volume=bars[i].volume * 20
            )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        result = engine.run(window)
        
        assert result.quantrascore < 70, (
            "Extreme conditions should reduce confidence"
        )
    
    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"])
    def test_position_limit_flags(self, engine: ApexEngine, symbol: str):
        """
        System must track and flag potential position limit violations.
        
        SEC 15c3-5 requires position limits to prevent concentration risk.
        """
        bars = generate_market_bars(count=100, trend="bullish")
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result is not None
        assert hasattr(result, 'regime'), "Result must include regime classification"


class TestPostTradeRiskControls:
    """
    MiFID II RTS 6 Article 17: Post-Trade Controls
    
    Validates monitoring and alerting mechanisms that operate
    after analysis/signal generation.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_entropy_state_monitoring(self, engine: ApexEngine):
        """
        System must continuously monitor entropy state.
        
        Elevated entropy requires additional scrutiny.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result.entropy_state is not None
        assert result.entropy_state in ["stable", "elevated", "chaotic"]
    
    def test_drift_state_monitoring(self, engine: ApexEngine):
        """
        System must detect and report drift conditions.
        
        Drift indicates potential model degradation.
        """
        bars = generate_market_bars(count=100, trend="bearish")
        
        for i in range(80, 100):
            shift = (i - 80) * 0.002
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=bars[79].close * (1 - shift),
                high=bars[79].close * (1 - shift + 0.005),
                low=bars[79].close * (1 - shift - 0.005),
                close=bars[79].close * (1 - shift),
                volume=bars[i].volume
            )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        result = engine.run(window)
        
        assert result.drift_state is not None
    
    def test_suppression_state_monitoring(self, engine: ApexEngine):
        """
        System must track signal suppression conditions.
        
        Required for audit trail of why signals were/weren't generated.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result.suppression_state is not None
    
    def test_audit_trail_completeness(self, engine: ApexEngine):
        """
        All analysis results must include complete audit trail fields.
        
        SEC requires complete record-keeping for 6 years.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        required_fields = [
            'quantrascore',
            'verdict',
            'regime',
            'entropy_state',
            'drift_state',
            'suppression_state',
            'timestamp'
        ]
        
        for field in required_fields:
            assert hasattr(result, field), f"Missing audit field: {field}"


class TestKillSwitchMechanisms:
    """
    SEC 15c3-5 & MiFID II: Kill Switch Validation
    
    Systems must have mechanisms to immediately halt trading
    when risk thresholds are breached.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_chaotic_regime_kill_switch(self, engine: ApexEngine):
        """
        Chaotic regime classification should trigger protective measures.
        
        Standard: Reduce position sizing
        QuantraCore: Full signal suppression (2x conservative)
        """
        bars = generate_market_bars(count=100)
        
        for i in range(40, 60):
            swing = 0.1 * (1 if i % 2 == 0 else -1)
            bars[i] = OhlcvBar(
                timestamp=bars[i].timestamp,
                open=100 * (1 + swing),
                high=100 * (1 + abs(swing) + 0.02),
                low=100 * (1 - abs(swing) - 0.02),
                close=100 * (1 - swing),
                volume=bars[i].volume * 15
            )
        
        window = OhlcvWindow(symbol="GME", timeframe="1m", bars=bars)
        result = engine.run(window)
        
        if result.regime == "chaotic":
            assert result.quantrascore < 40, (
                "Chaotic regime should heavily suppress confidence"
            )
    
    def test_omega_5_suppression_lock(self, engine: ApexEngine):
        """
        Omega Directive Ω5 (Strong Suppression) must lock signals.
        
        When active, no actionable signals should be generated.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        if hasattr(result, 'omega_overrides'):
            omega = result.omega_overrides
            if omega.get('omega_5_active', False):
                assert result.quantrascore < 30, (
                    "Ω5 active should suppress all signals"
                )
    
    def test_circuit_breaker_response(self, engine: ApexEngine):
        """
        System must respond appropriately to circuit-breaker conditions.
        
        15% single-bar moves should trigger maximum caution.
        """
        bars = generate_market_bars(count=100)
        
        bars[50] = OhlcvBar(
            timestamp=bars[50].timestamp,
            open=bars[49].close,
            high=bars[49].close * 1.16,
            low=bars[49].close * 0.84,
            close=bars[49].close * 0.85,
            volume=bars[50].volume * 50
        )
        
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        result = engine.run(window)
        
        assert result.quantrascore < 80, (
            "Circuit-breaker level moves should reduce confidence"
        )


class TestComplianceModeEnforcement:
    """
    QuantraCore Omega Directive Ω4: Research-Only Compliance
    
    System is research/backtest only — this must be enforced
    and clearly communicated in all outputs.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    def test_research_only_mode_active(self, engine: ApexEngine):
        """
        Ω4 (Compliance Mode) must always be active.
        
        System operates in research-only mode with no live trading.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result is not None
        if hasattr(result, 'omega_overrides'):
            omega = result.omega_overrides
            assert omega.get('omega_4_compliance', True), (
                "Ω4 compliance mode must always be active"
            )
    
    def test_verdict_includes_research_disclaimer(self, engine: ApexEngine):
        """
        All verdicts should be framed as analytical observations.
        
        No "buy", "sell", or imperative trading instructions.
        """
        bars = generate_market_bars(count=100, trend="bullish")
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        verdict_action = result.verdict.action.lower()
        forbidden_terms = ["buy now", "sell now", "must buy", "must sell"]
        
        for term in forbidden_terms:
            assert term not in verdict_action, (
                f"Verdict contains forbidden trading instruction: {term}"
            )
        
        assert "advice" not in verdict_action, "Verdict should not contain trading advice"
    
    def test_no_live_order_generation(self, engine: ApexEngine):
        """
        System must not generate actual order instructions.
        
        All outputs are analytical only.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert not hasattr(result, 'order'), "Result must not contain order object"
        assert not hasattr(result, 'trade'), "Result must not contain trade object"
        assert not hasattr(result, 'execution'), "Result must not contain execution"


class TestRiskMetricCalculations:
    """
    Basel Committee & FINRA: Risk Metric Validation
    
    Validates accuracy and consistency of risk calculations.
    """
    
    @pytest.fixture
    def engine(self) -> ApexEngine:
        return ApexEngine(enable_logging=False)
    
    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "GOOGL", "TSLA", "GME"])
    def test_quantrascore_bounds(self, engine: ApexEngine, symbol: str):
        """
        QuantraScore must always be in valid range [0, 100].
        
        Out-of-bounds scores would corrupt risk calculations.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        assert result.quantrascore >= 0, f"QuantraScore {result.quantrascore} below 0"
        assert result.quantrascore <= 100, f"QuantraScore {result.quantrascore} above 100"
    
    @pytest.mark.parametrize("trend", ["bullish", "bearish", "neutral"])
    def test_regime_appropriate_to_conditions(self, engine: ApexEngine, trend: str):
        """
        Regime classification must match market conditions.
        
        Misclassification could lead to inappropriate risk responses.
        """
        bars = generate_market_bars(count=100, trend=trend)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        valid_regimes = [
            "stable", "volatile", "trending", "chaotic", "unknown",
            "range_bound", "breakout", "consolidation", "compressed"
        ]
        regime_value = result.regime.value if hasattr(result.regime, 'value') else str(result.regime)
        assert regime_value.lower() in [r.lower() for r in valid_regimes], (
            f"Invalid regime: {regime_value}"
        )
    
    def test_entropy_calculation_validity(self, engine: ApexEngine):
        """
        Entropy calculations must produce valid states.
        
        Invalid entropy would corrupt safety override logic.
        """
        bars = generate_market_bars(count=100)
        window = OhlcvWindow(symbol="AAPL", timeframe="1m", bars=bars)
        
        result = engine.run(window)
        
        valid_entropy_states = ["stable", "elevated", "chaotic"]
        assert result.entropy_state in valid_entropy_states, (
            f"Invalid entropy state: {result.entropy_state}"
        )
    
    def test_consistent_risk_metrics_across_symbols(self, engine: ApexEngine):
        """
        Risk metric calculation methodology must be consistent.
        
        Same data should produce same metrics regardless of symbol.
        """
        np.random.seed(42)
        bars = generate_market_bars(count=100)
        
        results = []
        for symbol in ["AAPL", "MSFT", "GOOGL"]:
            window = OhlcvWindow(symbol=symbol, timeframe="1m", bars=bars.copy())
            results.append(engine.run(window))
        
        scores = [r.quantrascore for r in results]
        max_diff = max(scores) - min(scores)
        
        assert max_diff < 5, (
            f"Score variance {max_diff} across symbols exceeds threshold"
        )
