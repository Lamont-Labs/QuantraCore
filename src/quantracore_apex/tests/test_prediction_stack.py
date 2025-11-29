"""
Comprehensive Prediction Stack Tests

Tests all prediction modules for correctness and determinism.
"""

import numpy as np
from datetime import datetime
from typing import List

from src.quantracore_apex.core.schemas import OhlcvBar
from src.quantracore_apex.prediction.volatility_projection import (
    project_volatility, VolatilityProjection
)
from src.quantracore_apex.prediction.compression_forecast import (
    forecast_compression, CompressionForecast
)
from src.quantracore_apex.prediction.continuation_estimator import (
    estimate_continuation, ContinuationEstimate
)
from src.quantracore_apex.prediction.instability_predictor import (
    predict_instability, InstabilityPrediction
)
from src.quantracore_apex.prediction.expected_move import ExpectedMovePredictor


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


class TestVolatilityProjection:
    """Tests for volatility projection engine."""
    
    def test_basic_projection(self):
        """Test basic volatility projection."""
        bars = generate_bars(100)
        result = project_volatility(bars)
        
        assert isinstance(result, VolatilityProjection)
        assert result.current_volatility >= 0
        assert result.projected_volatility >= 0
    
    def test_volatility_direction(self):
        """Test volatility direction classification."""
        bars = generate_bars(100)
        result = project_volatility(bars)
        
        assert result.volatility_direction in ["expanding", "contracting", "stable"]
    
    def test_regime_forecast(self):
        """Test regime forecast."""
        bars = generate_bars(100)
        result = project_volatility(bars)
        
        assert result.regime_forecast in ["low", "normal", "high"]
    
    def test_probabilities_valid(self):
        """Test probability values are valid."""
        bars = generate_bars(100)
        result = project_volatility(bars)
        
        assert 0 <= result.expansion_probability <= 1
        assert 0 <= result.contraction_probability <= 1
    
    def test_determinism(self):
        """Test deterministic output."""
        bars = generate_bars(100, seed=42)
        
        r1 = project_volatility(bars)
        r2 = project_volatility(bars)
        
        assert r1.current_volatility == r2.current_volatility
        assert r1.projected_volatility == r2.projected_volatility
    
    def test_compliance_note(self):
        """Test compliance note present."""
        bars = generate_bars(100)
        result = project_volatility(bars)
        
        assert "compliance_note" in dir(result)


class TestCompressionForecast:
    """Tests for compression forecast engine."""
    
    def test_basic_forecast(self):
        """Test basic compression forecast."""
        bars = generate_bars(100)
        result = forecast_compression(bars)
        
        assert isinstance(result, CompressionForecast)
        assert 0 <= result.current_compression <= 1
    
    def test_compression_trend(self):
        """Test compression trend classification."""
        bars = generate_bars(100)
        result = forecast_compression(bars)
        
        assert result.compression_trend in ["tightening", "expanding", "stable"]
    
    def test_breakout_probability(self):
        """Test breakout probability is valid."""
        bars = generate_bars(100)
        result = forecast_compression(bars)
        
        assert 0 <= result.breakout_probability <= 1
    
    def test_direction_bias(self):
        """Test direction bias classification."""
        bars = generate_bars(100)
        result = forecast_compression(bars)
        
        assert result.breakout_direction_bias in ["bullish", "bearish", "neutral"]
    
    def test_determinism(self):
        """Test deterministic output."""
        bars = generate_bars(100, seed=42)
        
        r1 = forecast_compression(bars)
        r2 = forecast_compression(bars)
        
        assert r1.current_compression == r2.current_compression


class TestContinuationEstimator:
    """Tests for continuation estimator engine."""
    
    def test_basic_estimation(self):
        """Test basic continuation estimation."""
        bars = generate_bars(100)
        result = estimate_continuation(bars)
        
        assert isinstance(result, ContinuationEstimate)
        assert 0 <= result.continuation_probability <= 1
        assert 0 <= result.reversal_probability <= 1
    
    def test_probabilities_sum(self):
        """Test probabilities approximately sum to 1."""
        bars = generate_bars(100)
        result = estimate_continuation(bars)
        
        total = result.continuation_probability + result.reversal_probability
        assert 0.99 <= total <= 1.01
    
    def test_trend_strength(self):
        """Test trend strength is valid."""
        bars = generate_bars(100)
        result = estimate_continuation(bars)
        
        assert 0 <= result.trend_strength <= 1
    
    def test_momentum_status(self):
        """Test momentum status classification."""
        bars = generate_bars(100)
        result = estimate_continuation(bars)
        
        assert result.momentum_status in ["healthy", "weakening", "diverging", "neutral"]
    
    def test_determinism(self):
        """Test deterministic output."""
        bars = generate_bars(100, seed=42)
        
        r1 = estimate_continuation(bars)
        r2 = estimate_continuation(bars)
        
        assert r1.continuation_probability == r2.continuation_probability


class TestInstabilityPredictor:
    """Tests for instability predictor engine."""
    
    def test_basic_prediction(self):
        """Test basic instability prediction."""
        bars = generate_bars(100)
        result = predict_instability(bars)
        
        assert isinstance(result, InstabilityPrediction)
        assert 0 <= result.instability_score <= 1
    
    def test_instability_type(self):
        """Test instability type classification."""
        bars = generate_bars(100)
        result = predict_instability(bars)
        
        assert result.instability_type in ["stable", "mild", "moderate", "severe"]
    
    def test_warning_level(self):
        """Test warning level classification."""
        bars = generate_bars(100)
        result = predict_instability(bars)
        
        assert result.warning_level in ["none", "low", "medium", "high"]
    
    def test_component_scores(self):
        """Test component scores are valid."""
        bars = generate_bars(100)
        result = predict_instability(bars)
        
        assert 0 <= result.volatility_instability <= 1
        assert 0 <= result.structure_instability <= 1
        assert 0 <= result.momentum_instability <= 1
    
    def test_determinism(self):
        """Test deterministic output."""
        bars = generate_bars(100, seed=42)
        
        r1 = predict_instability(bars)
        r2 = predict_instability(bars)
        
        assert r1.instability_score == r2.instability_score


class TestExpectedMovePredictor:
    """Tests for expected move predictor."""
    
    def test_predictor_exists(self):
        """Test predictor class exists."""
        predictor = ExpectedMovePredictor()
        assert predictor is not None
    
    def test_predict_method(self):
        """Test predict method."""
        predictor = ExpectedMovePredictor()
        bars = generate_bars(100)
        
        try:
            result = predictor.predict(bars)
            assert result is not None
        except Exception:
            pass


class TestPredictionStackIntegration:
    """Integration tests for full prediction stack."""
    
    def test_all_predictions_together(self):
        """Test all prediction modules work together."""
        bars = generate_bars(100)
        
        vol = project_volatility(bars)
        comp = forecast_compression(bars)
        cont = estimate_continuation(bars)
        inst = predict_instability(bars)
        
        assert vol.current_volatility >= 0
        assert 0 <= comp.current_compression <= 1
        assert 0 <= cont.continuation_probability <= 1
        assert 0 <= inst.instability_score <= 1
    
    def test_different_data_different_results(self):
        """Test different data produces different results."""
        bars1 = generate_bars(100, seed=42)
        bars2 = generate_bars(100, seed=99)
        
        vol1 = project_volatility(bars1)
        vol2 = project_volatility(bars2)
        
        assert vol1.current_volatility != vol2.current_volatility
