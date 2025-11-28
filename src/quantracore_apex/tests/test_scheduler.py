"""
Comprehensive Scheduler Tests

Tests the scheduler system for correctness.
"""

import pytest
import time
from datetime import datetime, timedelta

from src.quantracore_apex.scheduler.scheduler import (
    ApexScheduler, ScheduledTask, TaskPriority, TaskStatus, TaskResult,
    create_scan_task
)


class TestScheduledTask:
    """Tests for ScheduledTask class."""
    
    def test_task_creation(self):
        """Test task creation."""
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        assert task.task_id == "test_1"
        assert task.name == "Test Task"
        assert task.interval_seconds == 60
        assert task.enabled is True
    
    def test_task_should_run(self):
        """Test should_run logic."""
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        task.next_run = datetime.utcnow() - timedelta(seconds=10)
        assert task.should_run(datetime.utcnow()) is True
        
        task.next_run = datetime.utcnow() + timedelta(seconds=10)
        assert task.should_run(datetime.utcnow()) is False
    
    def test_task_disabled(self):
        """Test disabled task doesn't run."""
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
            enabled=False,
        )
        
        assert task.should_run(datetime.utcnow()) is False
    
    def test_update_schedule(self):
        """Test schedule update after execution."""
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        now = datetime.utcnow()
        task.update_schedule(now)
        
        assert task.last_run == now
        assert task.next_run == now + timedelta(seconds=60)
        assert task.run_count == 1


class TestApexScheduler:
    """Tests for ApexScheduler class."""
    
    def test_scheduler_creation(self):
        """Test scheduler creation."""
        scheduler = ApexScheduler()
        
        assert scheduler is not None
        assert scheduler.running is False
        assert len(scheduler.tasks) == 0
    
    def test_add_task(self):
        """Test adding tasks."""
        scheduler = ApexScheduler()
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        scheduler.add_task(task)
        
        assert "test_1" in scheduler.tasks
    
    def test_remove_task(self):
        """Test removing tasks."""
        scheduler = ApexScheduler()
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        scheduler.add_task(task)
        result = scheduler.remove_task("test_1")
        
        assert result is True
        assert "test_1" not in scheduler.tasks
    
    def test_remove_nonexistent(self):
        """Test removing nonexistent task."""
        scheduler = ApexScheduler()
        
        result = scheduler.remove_task("nonexistent")
        
        assert result is False
    
    def test_enable_disable_task(self):
        """Test enabling/disabling tasks."""
        scheduler = ApexScheduler()
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "done",
            interval_seconds=60,
        )
        
        scheduler.add_task(task)
        
        scheduler.disable_task("test_1")
        assert scheduler.tasks["test_1"].enabled is False
        
        scheduler.enable_task("test_1")
        assert scheduler.tasks["test_1"].enabled is True
    
    def test_execute_task(self):
        """Test task execution."""
        scheduler = ApexScheduler()
        
        executed = []
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: executed.append(True),
            interval_seconds=60,
        )
        
        result = scheduler.execute_task(task)
        
        assert len(executed) == 1
        assert result.success is True
        assert task.status == TaskStatus.COMPLETED
    
    def test_execute_failing_task(self):
        """Test handling of failing task."""
        scheduler = ApexScheduler()
        
        def failing_callback():
            raise ValueError("Test error")
        
        task = ScheduledTask(
            task_id="test_1",
            name="Failing Task",
            callback=failing_callback,
            interval_seconds=60,
        )
        
        result = scheduler.execute_task(task)
        
        assert result.success is False
        assert "Test error" in result.error
        assert task.failure_count == 1
    
    def test_run_once(self):
        """Test run_once executes pending tasks."""
        scheduler = ApexScheduler()
        
        results = []
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: results.append(1),
            interval_seconds=60,
        )
        task.next_run = datetime.utcnow() - timedelta(seconds=10)
        
        scheduler.add_task(task)
        scheduler.run_once()
        
        assert len(results) == 1
    
    def test_get_status(self):
        """Test status retrieval."""
        scheduler = ApexScheduler()
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: None,
            interval_seconds=60,
        )
        
        scheduler.add_task(task)
        
        status = scheduler.get_status()
        
        assert status["running"] is False
        assert status["total_tasks"] == 1
        assert "compliance_note" in status
    
    def test_task_priority(self):
        """Test tasks are sorted by priority."""
        scheduler = ApexScheduler()
        
        low_task = ScheduledTask(
            task_id="low",
            name="Low Priority",
            callback=lambda: None,
            interval_seconds=60,
            priority=TaskPriority.LOW,
        )
        low_task.next_run = datetime.utcnow() - timedelta(seconds=10)
        
        high_task = ScheduledTask(
            task_id="high",
            name="High Priority",
            callback=lambda: None,
            interval_seconds=60,
            priority=TaskPriority.HIGH,
        )
        high_task.next_run = datetime.utcnow() - timedelta(seconds=10)
        
        scheduler.add_task(low_task)
        scheduler.add_task(high_task)
        
        pending = scheduler.get_pending_tasks()
        
        assert pending[0].task_id == "high"
        assert pending[1].task_id == "low"
    
    def test_get_task_history(self):
        """Test task history retrieval."""
        scheduler = ApexScheduler()
        
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            callback=lambda: "result",
            interval_seconds=60,
        )
        task.next_run = datetime.utcnow() - timedelta(seconds=10)
        
        scheduler.add_task(task)
        scheduler.run_once()
        
        history = scheduler.get_task_history()
        
        assert len(history) == 1
        assert history[0].task_id == "test_1"


class TestCreateScanTask:
    """Tests for scan task factory."""
    
    def test_create_scan_task(self):
        """Test scan task creation."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        task = create_scan_task(
            task_id="universe_scan",
            symbols=symbols,
            scan_fn=lambda s: {"symbol": s, "score": 50},
            interval_hours=1,
        )
        
        assert task.task_id == "universe_scan"
        assert task.interval_seconds == 3600
        assert "3 symbols" in task.name
    
    def test_scan_task_execution(self):
        """Test scan task executes correctly."""
        symbols = ["AAPL", "GOOGL"]
        
        task = create_scan_task(
            task_id="universe_scan",
            symbols=symbols,
            scan_fn=lambda s: {"score": 50},
            interval_hours=1,
        )
        
        result = task.callback()
        
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"


class TestSchedulerIntegration:
    """Integration tests for scheduler."""
    
    def test_multiple_tasks(self):
        """Test multiple tasks execute correctly."""
        scheduler = ApexScheduler(max_concurrent=5)
        
        results = []
        
        for i in range(3):
            task = ScheduledTask(
                task_id=f"task_{i}",
                name=f"Task {i}",
                callback=lambda idx=i: results.append(idx),
                interval_seconds=60,
            )
            task.next_run = datetime.utcnow() - timedelta(seconds=10)
            scheduler.add_task(task)
        
        scheduler.run_once()
        
        assert len(results) == 3
    
    def test_concurrent_limit(self):
        """Test concurrent execution limit."""
        scheduler = ApexScheduler(max_concurrent=2)
        
        for i in range(5):
            task = ScheduledTask(
                task_id=f"task_{i}",
                name=f"Task {i}",
                callback=lambda: None,
                interval_seconds=60,
            )
            task.next_run = datetime.utcnow() - timedelta(seconds=10)
            scheduler.add_task(task)
        
        run_results = scheduler.run_once()
        
        assert len(run_results) == 2
