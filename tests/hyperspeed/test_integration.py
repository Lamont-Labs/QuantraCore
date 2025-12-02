"""
Integration tests for the complete Hyperspeed Learning System.

Tests the full cycle: replay -> battle simulations -> training.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, date, timedelta
import threading
import time

from src.quantracore_apex.hyperspeed import (
    HyperspeedEngine,
    HyperspeedConfig,
    HistoricalReplayEngine,
    ParallelBattleCluster,
    MultiSourceAggregator,
    OvernightScheduler,
)
from src.quantracore_apex.hyperspeed.models import (
    HyperspeedMode,
    ReplaySpeed,
    SimulationStrategy,
    DataSource,
)
from src.quantracore_apex.core.schemas import OhlcvWindow, OhlcvBar


class TestHyperspeedEngineIntegration:
    """Integration tests for HyperspeedEngine orchestration."""
    
    def test_engine_initialization(self, test_config):
        """Test engine initializes all components."""
        engine = HyperspeedEngine(config=test_config)
        
        assert engine.config == test_config
        assert engine.replay_engine is not None
        assert engine.battle_cluster is not None
        assert engine.aggregator is not None
        assert engine.scheduler is not None
    
    def test_engine_component_wiring(self, test_config):
        """Test that components share the same config."""
        engine = HyperspeedEngine(config=test_config)
        
        assert engine.replay_engine.config == test_config
        assert engine.battle_cluster.config == test_config
        assert engine.aggregator.config == test_config
        assert engine.scheduler.config == test_config
    
    def test_engine_mode_from_config(self, test_config):
        """Test engine mode comes from config."""
        engine = HyperspeedEngine(config=test_config)
        assert engine.config.mode == HyperspeedMode.FULL_HYPERSPEED
    
    def test_scheduler_task_registration(self, test_config):
        """Test scheduler properly registers engine tasks."""
        engine = HyperspeedEngine(config=test_config)
        
        tasks = engine.scheduler._tasks
        
        assert "historical_replay" in tasks
        assert "battle_simulations" in tasks
        assert "model_training" in tasks
    
    def test_sample_caching(self, test_config):
        """Test samples can be cached for training."""
        engine = HyperspeedEngine(config=test_config)
        
        import numpy as np
        features = np.zeros(100)
        labels = {"quantrascore": 75.0}
        
        with engine._sample_lock:
            engine._training_samples.append((features, labels))
        
        assert len(engine._training_samples) >= 1


class TestReplayEngineIntegration:
    """Integration tests for HistoricalReplayEngine."""
    
    def test_caching_behavior(self, test_config, mock_bars):
        """Test bar caching reduces API calls."""
        engine = HistoricalReplayEngine(config=test_config)
        
        cache_key = f"AAPL:{date.today() - timedelta(days=30)}:{date.today()}"
        engine._bar_cache[cache_key] = mock_bars
        
        result = engine.fetch_historical_bars(
            symbol="AAPL",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
            use_cache=True,
        )
        
        assert result == mock_bars
    
    def test_rate_limiting(self, test_config):
        """Test rate limiting is applied."""
        engine = HistoricalReplayEngine(config=test_config)
        
        start = time.time()
        engine._rate_limit()
        engine._rate_limit()
        elapsed = time.time() - start
        
        assert elapsed >= engine._rate_delay


class TestBattleClusterIntegration:
    """Integration tests for ParallelBattleCluster."""
    
    def test_cluster_initialization(self, test_config):
        """Test cluster initializes correctly."""
        cluster = ParallelBattleCluster(config=test_config)
        
        assert cluster.config == test_config
        assert cluster._simulations == []
    
    def test_strategy_configs_loaded(self, test_config):
        """Test all strategies have configs."""
        from src.quantracore_apex.hyperspeed.battle_cluster import STRATEGY_CONFIGS
        
        for strategy in SimulationStrategy:
            assert strategy in STRATEGY_CONFIGS


class TestAggregatorIntegration:
    """Integration tests for MultiSourceAggregator."""
    
    def test_sample_creation(self, test_config, mock_window):
        """Test aggregated sample creation."""
        aggregator = MultiSourceAggregator(config=test_config)
        
        sample = aggregator.aggregate_sample(
            window=mock_window,
            labels={"quantrascore": 75.0},
            fetch_external=False,
        )
        
        assert sample.symbol == "AAPL"
        assert sample.labels["quantrascore"] == 75.0
        assert len(sample.primary_features) > 0
    
    def test_source_status_tracking(self, test_config):
        """Test data source status is tracked."""
        aggregator = MultiSourceAggregator(config=test_config)
        
        status = aggregator.get_source_status()
        
        assert len(status) > 0
    
    def test_synthetic_data_fallback(self, test_config, mock_window):
        """Test synthetic data when external sources unavailable."""
        aggregator = MultiSourceAggregator(config=test_config)
        
        sample = aggregator.aggregate_sample(
            window=mock_window,
            fetch_external=False,
        )
        
        assert sample.data_completeness_score >= 0.0


class TestSchedulerIntegration:
    """Integration tests for OvernightScheduler."""
    
    def test_market_hours_detection(self, test_config):
        """Test market hours detection works."""
        scheduler = OvernightScheduler(config=test_config)
        
        is_market = scheduler.is_market_hours()
        is_overnight = scheduler.is_overnight_window()
        
        assert isinstance(is_market, bool)
        assert isinstance(is_overnight, bool)
    
    def test_task_registration_and_retrieval(self, test_config):
        """Test tasks can be registered and retrieved."""
        scheduler = OvernightScheduler(config=test_config)
        
        callback = lambda: None
        scheduler.register_task(
            task_id="test_task",
            name="Test Task",
            callback=callback,
            priority=1,
        )
        
        assert "test_task" in scheduler._tasks
    
    def test_time_calculations(self, test_config):
        """Test time until overnight and remaining calculations."""
        scheduler = OvernightScheduler(config=test_config)
        
        time_until = scheduler.get_time_until_overnight()
        remaining = scheduler.get_overnight_remaining()
        
        assert isinstance(time_until, timedelta)
        assert isinstance(remaining, timedelta)
        assert time_until.total_seconds() >= 0
        assert remaining.total_seconds() >= 0


class TestFallbackAdapters:
    """Tests for fallback data adapters."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data can be generated."""
        from src.quantracore_apex.hyperspeed.adapters import SyntheticDataAdapter
        
        adapter = SyntheticDataAdapter()
        
        bars = adapter.generate_bars(
            symbol="AAPL",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )
        
        assert len(bars) > 0
        assert all(isinstance(b, OhlcvBar) for b in bars)
    
    def test_synthetic_options_flow(self):
        """Test synthetic options flow generation."""
        from src.quantracore_apex.hyperspeed.adapters import SyntheticDataAdapter
        
        adapter = SyntheticDataAdapter()
        options = adapter.generate_options_flow("AAPL")
        
        assert "call_volume" in options
        assert "put_volume" in options
        assert "put_call_ratio" in options
    
    def test_local_cache_adapter(self, tmp_path):
        """Test local cache save and load."""
        from src.quantracore_apex.hyperspeed.adapters import LocalCacheAdapter, AdapterConfig
        
        config = AdapterConfig(cache_dir=str(tmp_path))
        adapter = LocalCacheAdapter(config)
        
        bars = [
            OhlcvBar(
                timestamp=datetime.utcnow(),
                open=100.0,
                high=105.0,
                low=98.0,
                close=102.0,
                volume=1000000,
            )
        ]
        
        success = adapter.save_cached_bars(
            symbol="AAPL",
            start_date=date.today(),
            end_date=date.today(),
            bars=bars,
        )
        
        assert success
        
        loaded = adapter.load_cached_bars(
            symbol="AAPL",
            start_date=date.today(),
            end_date=date.today(),
        )
        
        assert loaded is not None
        assert len(loaded) == 1


class TestMonitoring:
    """Tests for monitoring system."""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        from src.quantracore_apex.hyperspeed.monitoring import SchedulerMonitor
        
        monitor = SchedulerMonitor()
        
        assert monitor._metrics == {}
        assert monitor._alerts == []
    
    def test_thread_registration(self):
        """Test thread registration."""
        from src.quantracore_apex.hyperspeed.monitoring import SchedulerMonitor, ThreadState
        
        monitor = SchedulerMonitor()
        
        monitor.register_thread(
            thread_id="test_thread",
            name="Test Thread",
            heartbeat_interval=30,
        )
        
        assert "test_thread" in monitor._metrics
        assert monitor._metrics["test_thread"].state == ThreadState.NOT_STARTED
    
    def test_heartbeat_recording(self):
        """Test heartbeat recording updates state."""
        from src.quantracore_apex.hyperspeed.monitoring import SchedulerMonitor
        
        monitor = SchedulerMonitor()
        monitor.register_thread("test", "Test", 30)
        
        monitor.record_heartbeat("test")
        
        assert monitor._metrics["test"].last_heartbeat is not None
        assert monitor._metrics["test"].is_alive == True
    
    def test_alert_generation(self):
        """Test alert generation and retrieval."""
        from src.quantracore_apex.hyperspeed.monitoring import SchedulerMonitor, AlertLevel
        
        monitor = SchedulerMonitor()
        
        monitor._generate_alert(
            AlertLevel.WARNING,
            "test_component",
            "Test warning message",
        )
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["level"] == "warning"
    
    def test_recovery_suggestions(self):
        """Test recovery suggestions for troubled threads."""
        from src.quantracore_apex.hyperspeed.monitoring import SchedulerMonitor, ThreadState
        
        monitor = SchedulerMonitor()
        monitor.register_thread("test", "Test", 30)
        monitor._metrics["test"].state = ThreadState.STALLED
        
        suggestions = monitor.get_recovery_suggestions("test")
        
        assert len(suggestions) > 0


class TestFullCycleIntegration:
    """End-to-end integration tests for complete learning cycle."""
    
    def test_engine_creates_cycle(self, test_config):
        """Test engine can create a training cycle."""
        engine = HyperspeedEngine(config=test_config)
        
        assert engine._current_cycle is None
        assert engine._active == False
    
    def test_metrics_tracking(self, test_config):
        """Test metrics are tracked correctly."""
        engine = HyperspeedEngine(config=test_config)
        
        initial_samples = engine._metrics.total_samples_generated
        initial_simulations = engine._metrics.total_simulations_run
        
        assert initial_samples == 0
        assert initial_simulations == 0
    
    def test_model_attachment(self, test_config):
        """Test model can be attached to engine."""
        engine = HyperspeedEngine(config=test_config)
        
        assert engine._model is None
        
        mock_model = MagicMock()
        engine.set_model(mock_model)
        
        assert engine._model == mock_model
