"""
Determinism Golden Set Tests for QuantraCore Apex.

Verifies that identical inputs produce identical outputs.
"""

import pytest
from datetime import datetime, timedelta

from src.quantracore_apex.core.engine import ApexEngine
from src.quantracore_apex.core.schemas import ApexContext
from src.quantracore_apex.data_layer.adapters.synthetic_adapter import SyntheticAdapter
from src.quantracore_apex.data_layer.normalization import normalize_ohlcv
from src.quantracore_apex.apexlab.windows import WindowBuilder


class TestDeterminism:
    """Test deterministic behavior of Apex engine."""
    
    @pytest.fixture
    def engine(self):
        return ApexEngine(enable_logging=False)
    
    @pytest.fixture
    def sample_window(self):
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("GOLDEN_TEST", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        return window_builder.build_single(normalized_bars, "GOLDEN_TEST")
    
    def test_identical_outputs_same_seed(self, engine, sample_window):
        """Verify identical outputs when using same seed."""
        context = ApexContext(seed=42, compliance_mode=True)
        
        result1 = engine.run(sample_window, context)
        result2 = engine.run(sample_window, context)
        
        assert result1.quantrascore == result2.quantrascore
        assert result1.regime == result2.regime
        assert result1.risk_tier == result2.risk_tier
        assert result1.entropy_state == result2.entropy_state
        assert result1.suppression_state == result2.suppression_state
        assert result1.window_hash == result2.window_hash
    
    def test_multiple_runs_consistency(self, engine, sample_window):
        """Verify consistency across multiple runs."""
        context = ApexContext(seed=42, compliance_mode=True)
        
        scores = []
        for _ in range(5):
            result = engine.run(sample_window, context)
            scores.append(result.quantrascore)
        
        assert all(s == scores[0] for s in scores), "Scores should be identical across runs"
    
    def test_window_hash_deterministic(self, sample_window):
        """Verify window hash is deterministic."""
        hash1 = sample_window.get_hash()
        hash2 = sample_window.get_hash()
        
        assert hash1 == hash2
    
    def test_microtraits_deterministic(self, engine, sample_window):
        """Verify microtraits are deterministic."""
        from src.quantracore_apex.core.microtraits import compute_microtraits
        
        traits1 = compute_microtraits(sample_window)
        traits2 = compute_microtraits(sample_window)
        
        assert traits1.wick_ratio == traits2.wick_ratio
        assert traits1.body_ratio == traits2.body_ratio
        assert traits1.compression_score == traits2.compression_score
        assert traits1.noise_score == traits2.noise_score


class TestGoldenSetValues:
    """Test against known golden values."""
    
    @pytest.fixture
    def golden_window(self):
        adapter = SyntheticAdapter(seed=42)
        window_builder = WindowBuilder(window_size=100)
        
        end_date = datetime(2024, 1, 1)
        start_date = end_date - timedelta(days=150)
        
        bars = adapter.fetch_ohlcv("GOLDEN", start_date, end_date, "1d")
        normalized_bars, _ = normalize_ohlcv(bars)
        
        return window_builder.build_single(normalized_bars, "GOLDEN")
    
    def test_quantrascore_in_valid_range(self, golden_window):
        """Verify QuantraScore is in valid range."""
        engine = ApexEngine(enable_logging=False)
        result = engine.run(golden_window)
        
        assert 0 <= result.quantrascore <= 100
    
    def test_regime_is_valid(self, golden_window):
        """Verify regime is a valid value."""
        engine = ApexEngine(enable_logging=False)
        result = engine.run(golden_window)
        
        valid_regimes = ["trending_up", "trending_down", "range_bound", 
                        "volatile", "compressed", "unknown"]
        assert result.regime.value in valid_regimes
    
    def test_verdict_has_compliance_note(self, golden_window):
        """Verify verdict includes compliance note."""
        engine = ApexEngine(enable_logging=False)
        result = engine.run(golden_window)
        
        assert result.verdict.compliance_note is not None
        assert "advice" in result.verdict.compliance_note.lower() or \
               "analysis" in result.verdict.compliance_note.lower()
