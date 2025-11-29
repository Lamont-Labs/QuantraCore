"""
Tests for Estimated Move Module.

Tests the core computation logic, safety gates, and output validation.
"""

import pytest
import numpy as np
from datetime import datetime

from src.quantracore_apex.estimated_move import (
    EstimatedMoveEngine,
    EstimatedMoveInput,
    EstimatedMoveOutput,
    HorizonWindow,
    MoveRange,
)
from src.quantracore_apex.estimated_move.schemas import MoveConfidence


class TestEstimatedMoveEngine:
    """Tests for EstimatedMoveEngine core functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return EstimatedMoveEngine(seed=42)
    
    @pytest.fixture
    def basic_input(self):
        """Create basic test input."""
        return EstimatedMoveInput(
            symbol="TEST",
            quantra_score=65.0,
            risk_tier="moderate",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="trending",
            suppression_state="active",
            protocol_vector=[1.0] * 80 + [0.0] * 25,
            runner_prob=0.3,
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.15,
            market_cap_band="mid",
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initializes correctly."""
        assert engine is not None
        assert engine.seed == 42
        assert len(engine.HORIZONS) == 4
    
    def test_compute_returns_output(self, engine, basic_input):
        """Test compute returns EstimatedMoveOutput."""
        output = engine.compute(basic_input)
        
        assert isinstance(output, EstimatedMoveOutput)
        assert output.symbol == "TEST"
        assert output.timestamp is not None
        assert len(output.ranges) == 4
    
    def test_all_horizons_computed(self, engine, basic_input):
        """Test all horizon windows are computed."""
        output = engine.compute(basic_input)
        
        expected_horizons = ["1d", "3d", "5d", "10d"]
        for horizon in expected_horizons:
            assert horizon in output.ranges
            assert isinstance(output.ranges[horizon], MoveRange)
    
    def test_move_ranges_have_valid_percentiles(self, engine, basic_input):
        """Test move ranges have logically ordered percentiles."""
        output = engine.compute(basic_input)
        
        for horizon, move_range in output.ranges.items():
            # Percentiles should be ordered (with some tolerance for skew)
            assert move_range.min_move_pct <= move_range.low_move_pct
            assert move_range.low_move_pct <= move_range.median_move_pct
            assert move_range.median_move_pct <= move_range.high_move_pct
            assert move_range.high_move_pct <= move_range.max_move_pct
    
    def test_uncertainty_score_range(self, engine, basic_input):
        """Test uncertainty score is in valid range."""
        output = engine.compute(basic_input)
        
        assert 0.0 <= output.overall_uncertainty <= 1.0
        for move_range in output.ranges.values():
            assert 0.0 <= move_range.uncertainty_score <= 1.0
    
    def test_determinism(self, engine, basic_input):
        """Test engine produces deterministic results."""
        output1 = engine.compute(basic_input)
        output2 = engine.compute(basic_input)
        
        for horizon in output1.ranges:
            r1 = output1.ranges[horizon]
            r2 = output2.ranges[horizon]
            assert r1.median_move_pct == r2.median_move_pct
            assert r1.max_move_pct == r2.max_move_pct
    
    def test_compliance_note_present(self, engine, basic_input):
        """Test compliance note is always present."""
        output = engine.compute(basic_input)
        
        assert output.compliance_note is not None
        assert len(output.compliance_note) > 0
        assert "research" in output.compliance_note.lower() or "not" in output.compliance_note.lower()


class TestSafetyGates:
    """Tests for safety gate functionality."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    def test_high_avoid_trade_clamps_output(self, engine):
        """Test high avoid_trade_prob triggers safety clamping."""
        input_data = EstimatedMoveInput(
            symbol="RISKY",
            quantra_score=50.0,
            risk_tier="high",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="ranging",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            avoid_trade_prob=0.5,  # Above threshold
            runner_prob=0.1,
            ensemble_disagreement=0.1,
        )
        
        output = engine.compute(input_data)
        
        assert output.safety_clamped == True
    
    def test_suppression_blocked_clamps_output(self, engine):
        """Test blocked suppression state triggers safety clamping."""
        input_data = EstimatedMoveInput(
            symbol="BLOCKED",
            quantra_score=70.0,
            risk_tier="low",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="trending",
            suppression_state="blocked",  # Blocked
            protocol_vector=[0.5] * 105,
            avoid_trade_prob=0.1,
            runner_prob=0.3,
            ensemble_disagreement=0.1,
        )
        
        output = engine.compute(input_data)
        
        assert output.safety_clamped == True
    
    def test_high_uncertainty_returns_neutral(self, engine):
        """Test very high uncertainty returns neutral output."""
        input_data = EstimatedMoveInput(
            symbol="UNCERTAIN",
            quantra_score=30.0,  # Low score
            risk_tier="high",
            volatility_band="extreme",
            entropy_band="high",
            regime_type="volatile",
            suppression_state="active",
            protocol_vector=[0.1] * 105,
            avoid_trade_prob=0.6,  # High
            runner_prob=0.1,
            ensemble_disagreement=0.8,  # Very high
        )
        
        output = engine.compute(input_data)
        
        # Should be uncertain/clamped
        assert output.confidence == MoveConfidence.UNCERTAIN or output.safety_clamped
    
    def test_high_disagreement_uses_deterministic_mode(self, engine):
        """Test high ensemble disagreement switches to deterministic mode."""
        input_data = EstimatedMoveInput(
            symbol="DISAGREE",
            quantra_score=60.0,
            risk_tier="moderate",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="trending",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            avoid_trade_prob=0.1,
            runner_prob=0.3,
            ensemble_disagreement=0.5,  # High disagreement
        )
        
        output = engine.compute(input_data)
        
        assert output.computation_mode == "deterministic"


class TestRunnerBoost:
    """Tests for runner probability boost functionality."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    def test_high_runner_prob_applies_boost(self, engine):
        """Test high runner probability applies boost."""
        input_with_runner = EstimatedMoveInput(
            symbol="RUNNER",
            quantra_score=75.0,
            risk_tier="low",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="breakout",
            suppression_state="active",
            protocol_vector=[0.7] * 105,
            runner_prob=0.75,  # High runner prob
            avoid_trade_prob=0.05,
            ensemble_disagreement=0.1,
        )
        
        output = engine.compute(input_with_runner)
        
        assert output.runner_boost_applied == True
    
    def test_low_runner_prob_no_boost(self, engine):
        """Test low runner probability does not apply boost."""
        input_without_runner = EstimatedMoveInput(
            symbol="NORMAL",
            quantra_score=65.0,
            risk_tier="moderate",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="ranging",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            runner_prob=0.2,  # Low runner prob
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.15,
        )
        
        output = engine.compute(input_without_runner)
        
        assert output.runner_boost_applied == False


class TestMarketCapBands:
    """Tests for market cap band volatility scaling."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    @pytest.mark.parametrize("market_cap,expected_higher", [
        ("mega", False),  # Lower volatility
        ("micro", True),  # Higher volatility
    ])
    def test_market_cap_affects_volatility(self, engine, market_cap, expected_higher):
        """Test different market caps produce different volatility ranges."""
        base_input = EstimatedMoveInput(
            symbol="CAP_TEST",
            quantra_score=60.0,
            risk_tier="moderate",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="trending",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            runner_prob=0.3,
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.1,
            market_cap_band=market_cap,
        )
        
        output = engine.compute(base_input)
        
        # Micro cap should have wider ranges than mega cap
        five_day_range = output.ranges["5d"]
        spread = five_day_range.max_move_pct - five_day_range.min_move_pct
        
        if expected_higher:
            assert spread > 5.0  # Wider spread for small caps
        else:
            assert spread < 10.0  # Narrower for mega caps


class TestRegimeEffects:
    """Tests for regime type effects on estimated move."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    def test_volatile_regime_increases_range(self, engine):
        """Test volatile regime produces wider ranges."""
        volatile_input = EstimatedMoveInput(
            symbol="VOLATILE",
            quantra_score=60.0,
            risk_tier="high",
            volatility_band="high",
            entropy_band="high",
            regime_type="volatile",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            runner_prob=0.3,
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.1,
        )
        
        quiet_input = EstimatedMoveInput(
            symbol="QUIET",
            quantra_score=60.0,
            risk_tier="low",
            volatility_band="low",
            entropy_band="low",
            regime_type="quiet",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            runner_prob=0.3,
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.1,
        )
        
        volatile_output = engine.compute(volatile_input)
        quiet_output = engine.compute(quiet_input)
        
        volatile_spread = volatile_output.ranges["5d"].max_move_pct - volatile_output.ranges["5d"].min_move_pct
        quiet_spread = quiet_output.ranges["5d"].max_move_pct - quiet_output.ranges["5d"].min_move_pct
        
        assert volatile_spread > quiet_spread


class TestBatchComputation:
    """Tests for batch computation functionality."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    def test_batch_compute(self, engine):
        """Test batch computation processes multiple inputs."""
        inputs = [
            EstimatedMoveInput(
                symbol=f"SYM_{i}",
                quantra_score=50.0 + i * 5,
                risk_tier="moderate",
                volatility_band="normal",
                entropy_band="normal",
                regime_type="trending",
                suppression_state="active",
                protocol_vector=[0.5] * 105,
                runner_prob=0.2 + i * 0.1,
                avoid_trade_prob=0.1,
                ensemble_disagreement=0.1,
            )
            for i in range(5)
        ]
        
        outputs = engine.compute_batch(inputs)
        
        assert len(outputs) == 5
        for i, output in enumerate(outputs):
            assert output.symbol == f"SYM_{i}"
            assert isinstance(output, EstimatedMoveOutput)


class TestOutputSerialization:
    """Tests for output serialization."""
    
    @pytest.fixture
    def engine(self):
        return EstimatedMoveEngine(seed=42)
    
    @pytest.fixture
    def basic_input(self):
        return EstimatedMoveInput(
            symbol="SERIALIZE",
            quantra_score=65.0,
            risk_tier="moderate",
            volatility_band="normal",
            entropy_band="normal",
            regime_type="trending",
            suppression_state="active",
            protocol_vector=[0.5] * 105,
            runner_prob=0.3,
            avoid_trade_prob=0.1,
            ensemble_disagreement=0.1,
        )
    
    def test_to_dict(self, engine, basic_input):
        """Test to_dict produces valid dictionary."""
        output = engine.compute(basic_input)
        result = output.to_dict()
        
        assert isinstance(result, dict)
        assert "symbol" in result
        assert "ranges" in result
        assert "compliance_note" in result
        assert "timestamp" in result
    
    def test_move_range_to_dict(self, engine, basic_input):
        """Test MoveRange to_dict produces valid dictionary."""
        output = engine.compute(basic_input)
        move_range = output.ranges["5d"]
        result = move_range.to_dict()
        
        assert isinstance(result, dict)
        assert "horizon" in result
        assert "median_move_pct" in result
        assert "max_move_pct" in result
        assert "uncertainty_score" in result
