"""
Comprehensive tests for QuantraCore Apex Hardening Infrastructure.

Tests cover:
- Protocol manifest generation and validation
- Config validation
- Mode enforcement
- Incident logging
- Kill switch management
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.quantracore_apex.hardening.manifest import (
    ProtocolManifest,
    ManifestValidator,
    ManifestValidationError,
    ProtocolEntry,
)
from src.quantracore_apex.hardening.config_validator import (
    ConfigValidator,
    ConfigValidationError,
    ValidationResult,
)
from src.quantracore_apex.hardening.mode_enforcer import (
    ModeEnforcer,
    ExecutionMode,
    ModePermissions,
    ModeViolationError,
)
from src.quantracore_apex.hardening.incident_logger import (
    IncidentLogger,
    IncidentClass,
    IncidentSeverity,
    Incident,
)
from src.quantracore_apex.hardening.kill_switch import (
    KillSwitchManager,
    KillSwitchReason,
    KillSwitchState,
)


class TestProtocolManifest:
    """Tests for protocol manifest system."""
    
    def test_generate_default_manifest(self):
        """Test generating default manifest with all protocols."""
        manifest = ProtocolManifest.generate_default()
        
        assert manifest.version == "9.0-A"
        assert manifest.engine_snapshot_id.startswith("apex-v9.0A-")
        assert len(manifest.protocols) == 115
        assert manifest.hash != ""
    
    def test_protocol_counts(self):
        """Test correct protocol counts by category."""
        manifest = ProtocolManifest.generate_default()
        
        tier_count = sum(1 for p in manifest.protocols if p.category == "tier")
        learning_count = sum(1 for p in manifest.protocols if p.category == "learning")
        mr_count = sum(1 for p in manifest.protocols if p.category == "monster_runner")
        omega_count = sum(1 for p in manifest.protocols if p.category == "omega")
        
        assert tier_count == 80
        assert learning_count == 25
        assert mr_count == 5
        assert omega_count == 5
    
    def test_manifest_hash_deterministic(self):
        """Test that manifest hash is deterministic."""
        manifest1 = ProtocolManifest.generate_default()
        manifest2 = ProtocolManifest.generate_default()
        
        assert manifest1.compute_hash() == manifest2.compute_hash()
    
    def test_manifest_hash_changes_with_version(self):
        """Test that hash changes when version changes."""
        manifest = ProtocolManifest.generate_default()
        original_hash = manifest.compute_hash()
        
        manifest.version = "10.0-B"
        modified_hash = manifest.compute_hash()
        
        assert original_hash != modified_hash
    
    def test_manifest_save_and_load(self):
        """Test saving and loading manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_manifest.yaml"
            
            manifest = ProtocolManifest.generate_default()
            manifest.save(path)
            
            assert path.exists()
            assert path.with_suffix(".hash").exists()
            
            loaded = ProtocolManifest.load(path)
            assert loaded.version == manifest.version
            assert loaded.hash == manifest.hash
            assert len(loaded.protocols) == len(manifest.protocols)
    
    def test_protocol_entry_structure(self):
        """Test protocol entry has required fields."""
        manifest = ProtocolManifest.generate_default()
        
        for proto in manifest.protocols:
            assert proto.protocol_id
            assert proto.category in ("tier", "learning", "monster_runner", "omega")
            assert proto.execution_order > 0
            assert isinstance(proto.inputs, list)
            assert isinstance(proto.outputs, list)
            assert isinstance(proto.failure_modes, list)
            assert isinstance(proto.assumptions, list)


class TestManifestValidator:
    """Tests for manifest validation."""
    
    def test_validator_with_valid_manifest(self):
        """Test validation of valid manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            manifest = ProtocolManifest.generate_default()
            manifest.save(path)
            
            validator = ManifestValidator()
            assert validator.validate(path) == True
            assert validator.validated == True
    
    def test_validator_missing_manifest(self):
        """Test validation fails for missing manifest."""
        validator = ManifestValidator()
        
        with pytest.raises(ManifestValidationError, match="not found"):
            validator.validate(Path("/nonexistent/manifest.yaml"))
    
    def test_validator_hash_mismatch(self):
        """Test validation fails for hash mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            manifest = ProtocolManifest.generate_default()
            manifest.save(path)
            
            with open(path) as f:
                data = yaml.safe_load(f)
            data["hash"] = "invalid_hash"
            with open(path, "w") as f:
                yaml.dump(data, f)
            
            validator = ManifestValidator()
            with pytest.raises(ManifestValidationError, match="hash mismatch"):
                validator.validate(path)


class TestConfigValidator:
    """Tests for configuration validation."""
    
    def test_validate_mode_config(self):
        """Test mode.yaml validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config"
            config_path.mkdir()
            
            mode_config = {
                "default_mode": "research",
                "modes": {
                    "research": {"id": "research", "description": "Test"}
                },
                "safety_fence": {
                    "enforce_mode_check": True,
                    "fail_on_missing_config": True,
                }
            }
            with open(config_path / "mode.yaml", "w") as f:
                yaml.dump(mode_config, f)
            
            broker_config = {
                "execution": {"mode": "RESEARCH"},
                "risk": {
                    "max_notional_exposure_usd": 50000,
                    "max_position_notional_per_symbol_usd": 5000,
                    "per_trade_risk_fraction": 0.01,
                    "max_leverage": 2.0,
                },
                "brokers": {"alpaca": {"live": {"enabled": False}}}
            }
            with open(config_path / "broker.yaml", "w") as f:
                yaml.dump(broker_config, f)
            
            validator = ConfigValidator()
            validator.REQUIRED_CONFIGS = [
                str(config_path / "mode.yaml"),
                str(config_path / "broker.yaml"),
            ]
            validator.OPTIONAL_CONFIGS = []
            
            assert validator.validate_all() == True
    
    def test_invalid_mode_value(self):
        """Test validation fails for invalid mode value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config"
            config_path.mkdir()
            
            mode_config = {"default_mode": "INVALID_MODE"}
            with open(config_path / "mode.yaml", "w") as f:
                yaml.dump(mode_config, f)
            
            broker_config = {"execution": {"mode": "RESEARCH"}, "risk": {}}
            with open(config_path / "broker.yaml", "w") as f:
                yaml.dump(broker_config, f)
            
            validator = ConfigValidator()
            validator.REQUIRED_CONFIGS = [
                str(config_path / "mode.yaml"),
                str(config_path / "broker.yaml"),
            ]
            
            with pytest.raises(ConfigValidationError):
                validator.validate_all()
    
    def test_numeric_range_validation(self):
        """Test numeric range validation for risk params."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config"
            config_path.mkdir()
            
            mode_config = {"default_mode": "research", "modes": {}}
            with open(config_path / "mode.yaml", "w") as f:
                yaml.dump(mode_config, f)
            
            broker_config = {
                "execution": {"mode": "RESEARCH"},
                "risk": {
                    "per_trade_risk_fraction": 0.5,
                },
            }
            with open(config_path / "broker.yaml", "w") as f:
                yaml.dump(broker_config, f)
            
            validator = ConfigValidator()
            validator.REQUIRED_CONFIGS = [
                str(config_path / "mode.yaml"),
                str(config_path / "broker.yaml"),
            ]
            
            with pytest.raises(ConfigValidationError, match="out of range"):
                validator.validate_all()


class TestModeEnforcer:
    """Tests for mode enforcement system."""
    
    def test_default_research_mode(self):
        """Test default mode is RESEARCH."""
        enforcer = ModeEnforcer()
        assert enforcer.current_mode == ExecutionMode.RESEARCH
    
    def test_research_mode_permissions(self):
        """Test RESEARCH mode has correct permissions."""
        permissions = ModePermissions.for_mode(ExecutionMode.RESEARCH)
        
        assert permissions.engine_enabled == True
        assert permissions.apexlab_enabled == True
        assert permissions.models_enabled == True
        assert permissions.execution_engine_enabled == False
        assert permissions.broker_router_enabled == False
        assert permissions.live_orders_allowed == False
        assert permissions.paper_orders_allowed == False
    
    def test_paper_mode_permissions(self):
        """Test PAPER mode has correct permissions."""
        permissions = ModePermissions.for_mode(ExecutionMode.PAPER)
        
        assert permissions.engine_enabled == True
        assert permissions.execution_engine_enabled == True
        assert permissions.broker_router_enabled == True
        assert permissions.live_orders_allowed == False
        assert permissions.paper_orders_allowed == True
    
    def test_live_mode_permissions(self):
        """Test LIVE mode has correct permissions."""
        permissions = ModePermissions.for_mode(ExecutionMode.LIVE)
        
        assert permissions.execution_engine_enabled == True
        assert permissions.broker_router_enabled == True
        assert permissions.live_orders_allowed == True
        assert permissions.paper_orders_allowed == True
    
    def test_check_permission(self):
        """Test permission checking."""
        enforcer = ModeEnforcer()
        enforcer.current_mode = ExecutionMode.RESEARCH
        enforcer.permissions = ModePermissions.for_mode(ExecutionMode.RESEARCH)
        
        assert enforcer.check_permission("engine_access") == True
        assert enforcer.check_permission("execution_engine_access") == False
        assert enforcer.check_permission("place_live_order") == False
    
    def test_require_permission_raises(self):
        """Test require_permission raises on violation."""
        enforcer = ModeEnforcer()
        enforcer.current_mode = ExecutionMode.RESEARCH
        enforcer.permissions = ModePermissions.for_mode(ExecutionMode.RESEARCH)
        
        with pytest.raises(ModeViolationError, match="not permitted"):
            enforcer.require_permission("place_live_order", context="test")
    
    def test_mode_from_string(self):
        """Test mode parsing from string."""
        assert ExecutionMode.from_string("RESEARCH") == ExecutionMode.RESEARCH
        assert ExecutionMode.from_string("research") == ExecutionMode.RESEARCH
        assert ExecutionMode.from_string("PAPER") == ExecutionMode.PAPER
        assert ExecutionMode.from_string("LIVE") == ExecutionMode.LIVE
        assert ExecutionMode.from_string("invalid") == ExecutionMode.RESEARCH


class TestIncidentLogger:
    """Tests for incident logging system."""
    
    def test_log_incident(self):
        """Test basic incident logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            incident = logger.log_incident(
                incident_class=IncidentClass.MODEL_MANIFEST_FAILURE,
                severity=IncidentSeverity.HIGH,
                message="Test incident",
                context={"model_path": "/test/model"},
            )
            
            assert incident.incident_id.startswith("INC-")
            assert incident.incident_class == IncidentClass.MODEL_MANIFEST_FAILURE
            assert incident.severity == IncidentSeverity.HIGH
    
    def test_log_data_feed_divergence(self):
        """Test data feed divergence logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            incident = logger.log_data_feed_divergence(
                symbol="AAPL",
                feed1="polygon",
                feed2="yahoo",
                deviation_pct=3.5,
            )
            
            assert incident.incident_class == IncidentClass.DATA_FEED_DIVERGENCE
            assert incident.severity == IncidentSeverity.MEDIUM
            assert incident.context["symbol"] == "AAPL"
    
    def test_high_deviation_escalates_severity(self):
        """Test high deviation triggers higher severity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            incident = logger.log_data_feed_divergence(
                symbol="AAPL",
                feed1="polygon",
                feed2="yahoo",
                deviation_pct=10.0,
            )
            
            assert incident.severity == IncidentSeverity.HIGH
    
    def test_incident_counts(self):
        """Test incident counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            logger.log_incident(IncidentClass.MODEL_MANIFEST_FAILURE, IncidentSeverity.HIGH, "1")
            logger.log_incident(IncidentClass.MODEL_MANIFEST_FAILURE, IncidentSeverity.HIGH, "2")
            logger.log_incident(IncidentClass.DATA_FEED_DIVERGENCE, IncidentSeverity.MEDIUM, "3")
            
            counts = logger.get_incident_counts()
            assert counts["MODEL_MANIFEST_FAILURE"] == 2
            assert counts["DATA_FEED_DIVERGENCE"] == 1
    
    def test_severity_counts(self):
        """Test severity counting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            logger.log_incident(IncidentClass.MODEL_MANIFEST_FAILURE, IncidentSeverity.HIGH, "1")
            logger.log_incident(IncidentClass.DATA_FEED_DIVERGENCE, IncidentSeverity.MEDIUM, "2")
            logger.log_incident(IncidentClass.DATA_FEED_DIVERGENCE, IncidentSeverity.MEDIUM, "3")
            
            counts = logger.get_severity_counts()
            assert counts["HIGH"] == 1
            assert counts["MEDIUM"] == 2


class TestKillSwitch:
    """Tests for kill switch system."""
    
    def test_initial_state_not_engaged(self):
        """Test kill switch starts disengaged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
            )
            
            assert manager.is_engaged() == False
            allowed, reason = manager.check_order_allowed()
            assert allowed == True
    
    def test_manual_engage(self):
        """Test manual kill switch engagement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
            )
            
            manager.engage(
                reason=KillSwitchReason.MANUAL,
                engaged_by="test_operator",
                context={"test": True},
            )
            
            assert manager.is_engaged() == True
            allowed, reason = manager.check_order_allowed()
            assert allowed == False
            assert "MANUAL" in reason
    
    def test_auto_engage_on_drawdown(self):
        """Test auto kill switch on drawdown threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
                daily_drawdown_threshold_pct=5.0,
            )
            
            manager.update_daily_drawdown(3.0)
            assert manager.is_engaged() == False
            
            manager.update_daily_drawdown(6.0)
            assert manager.is_engaged() == True
            assert manager.state.reason == KillSwitchReason.DAILY_DRAWDOWN_EXCEEDED
    
    def test_auto_engage_on_broker_errors(self):
        """Test auto kill switch on broker error rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
                broker_error_rate_threshold_pct=20.0,
            )
            
            manager.update_broker_error_rate(15.0)
            assert manager.is_engaged() == False
            
            manager.update_broker_error_rate(25.0)
            assert manager.is_engaged() == True
            assert manager.state.reason == KillSwitchReason.BROKER_ERROR_RATE_HIGH
    
    def test_reset_kill_switch(self):
        """Test resetting kill switch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
            )
            
            manager.engage(KillSwitchReason.MANUAL, "test")
            assert manager.is_engaged() == True
            
            manager.reset("test_operator")
            assert manager.is_engaged() == False
            allowed, _ = manager.check_order_allowed()
            assert allowed == True
    
    def test_state_persistence(self):
        """Test kill switch state persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "kill_switch.json"
            log_path = Path(tmpdir) / "kill_switch.jsonl"
            
            manager1 = KillSwitchManager(state_path=state_path, log_path=log_path)
            manager1.engage(KillSwitchReason.MANUAL, "test")
            
            manager2 = KillSwitchManager(state_path=state_path, log_path=log_path)
            assert manager2.is_engaged() == True
            assert manager2.state.reason == KillSwitchReason.MANUAL
    
    def test_get_status(self):
        """Test status reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
            )
            
            status = manager.get_status()
            
            assert "engaged" in status
            assert "metrics" in status
            assert "thresholds" in status


class TestHardeningIntegration:
    """Integration tests for hardening components."""
    
    def test_mode_enforcer_blocks_execution_in_research(self):
        """Test mode enforcer blocks execution engine in RESEARCH mode."""
        enforcer = ModeEnforcer()
        enforcer.current_mode = ExecutionMode.RESEARCH
        enforcer.permissions = ModePermissions.for_mode(ExecutionMode.RESEARCH)
        
        assert enforcer.check_permission("execution_engine_access") == False
        
        with pytest.raises(ModeViolationError):
            enforcer.require_permission("execution_engine_access")
    
    def test_kill_switch_blocks_orders(self):
        """Test kill switch blocks order placement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KillSwitchManager(
                state_path=Path(tmpdir) / "kill_switch.json",
                log_path=Path(tmpdir) / "kill_switch.jsonl",
            )
            
            allowed, _ = manager.check_order_allowed()
            assert allowed == True
            
            manager.engage(KillSwitchReason.RISK_VIOLATIONS_SPIKE, "auto")
            
            allowed, reason = manager.check_order_allowed()
            assert allowed == False
            assert "RISK_VIOLATIONS_SPIKE" in reason
    
    def test_incident_triggers_kill_switch_consideration(self):
        """Test high severity incidents can trigger kill switch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = IncidentLogger(
                log_path=Path(tmpdir) / "incidents.jsonl",
                auto_escalate=False,
            )
            
            incident = logger.log_incident(
                IncidentClass.BROKER_ERROR_RATE_SPIKE,
                IncidentSeverity.CRITICAL,
                "Critical broker failure",
            )
            
            assert IncidentSeverity.CRITICAL.should_trigger_kill_switch() == True
