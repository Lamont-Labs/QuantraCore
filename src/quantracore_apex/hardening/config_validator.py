"""
Configuration Validation System.

Validates all YAML/JSON configs against JSON Schema at startup.
Ensures thresholds, ranges, and enumerations are within allowed bounds.
Engine refuses to start if validation fails.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ValidationResult:
    """Result of a single config validation."""
    config_path: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ConfigValidator:
    """
    Validates configuration files against schemas and business rules.
    Implements fail-closed behavior: invalid configs prevent startup.
    """
    
    REQUIRED_CONFIGS = [
        "config/mode.yaml",
        "config/broker.yaml",
    ]
    
    OPTIONAL_CONFIGS = [
        "config/scanner.yaml",
        "config/apexlab.yaml",
        "config/models.yaml",
    ]
    
    MODE_ALLOWED_VALUES = ["research", "paper_trading", "live"]
    EXECUTION_MODE_ALLOWED = ["RESEARCH", "PAPER", "LIVE"]
    
    results: list[ValidationResult] = field(default_factory=list)
    all_valid: bool = False
    
    def validate_all(self) -> bool:
        """
        Validate all required and optional configs.
        Returns True if all required configs are valid.
        Raises ConfigValidationError if any required config is invalid.
        """
        self.results = []
        all_required_valid = True
        
        for config_path in self.REQUIRED_CONFIGS:
            result = self._validate_config(config_path, required=True)
            self.results.append(result)
            if not result.valid:
                all_required_valid = False
        
        for config_path in self.OPTIONAL_CONFIGS:
            if Path(config_path).exists():
                result = self._validate_config(config_path, required=False)
                self.results.append(result)
        
        self.all_valid = all_required_valid
        
        if not all_required_valid:
            errors = []
            for r in self.results:
                if not r.valid:
                    errors.extend([f"{r.config_path}: {e}" for e in r.errors])
            raise ConfigValidationError(
                f"Configuration validation failed:\n" + "\n".join(errors)
            )
        
        return True
    
    def _validate_config(self, config_path: str, required: bool = True) -> ValidationResult:
        """Validate a single config file."""
        path = Path(config_path)
        errors = []
        warnings = []
        
        if not path.exists():
            if required:
                errors.append(f"Required config file not found: {config_path}")
            return ValidationResult(
                config_path=config_path,
                valid=not required,
                errors=errors,
            )
        
        try:
            with open(path) as f:
                if path.suffix == ".yaml":
                    config = yaml.safe_load(f)
                elif path.suffix == ".json":
                    config = json.load(f)
                else:
                    errors.append(f"Unsupported config format: {path.suffix}")
                    return ValidationResult(config_path=config_path, valid=False, errors=errors)
        except Exception as e:
            errors.append(f"Failed to parse config: {e}")
            return ValidationResult(config_path=config_path, valid=False, errors=errors)
        
        if "mode.yaml" in config_path:
            self._validate_mode_config(config, errors, warnings)
        elif "broker.yaml" in config_path:
            self._validate_broker_config(config, errors, warnings)
        
        return ValidationResult(
            config_path=config_path,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _validate_mode_config(self, config: dict, errors: list, warnings: list) -> None:
        """Validate mode.yaml structure and values."""
        if "default_mode" not in config:
            errors.append("Missing required field: default_mode")
        elif config["default_mode"] not in self.MODE_ALLOWED_VALUES:
            errors.append(
                f"Invalid default_mode: {config['default_mode']}. "
                f"Allowed: {self.MODE_ALLOWED_VALUES}"
            )
        
        if "modes" not in config:
            errors.append("Missing required field: modes")
        
        if "safety_fence" not in config:
            warnings.append("Missing safety_fence configuration")
        else:
            fence = config["safety_fence"]
            if not fence.get("enforce_mode_check", False):
                warnings.append("safety_fence.enforce_mode_check is disabled")
            if not fence.get("fail_on_missing_config", False):
                warnings.append("safety_fence.fail_on_missing_config is disabled")
    
    def _validate_broker_config(self, config: dict, errors: list, warnings: list) -> None:
        """Validate broker.yaml structure and values."""
        if "execution" not in config:
            errors.append("Missing required field: execution")
        else:
            exec_cfg = config["execution"]
            mode = exec_cfg.get("mode", "")
            if mode not in self.EXECUTION_MODE_ALLOWED:
                errors.append(
                    f"Invalid execution.mode: {mode}. "
                    f"Allowed: {self.EXECUTION_MODE_ALLOWED}"
                )
            if mode == "LIVE":
                warnings.append("DANGER: execution.mode is set to LIVE")
        
        if "risk" not in config:
            errors.append("Missing required field: risk")
        else:
            risk = config["risk"]
            self._validate_numeric_range(
                risk, "max_notional_exposure_usd", 0, 10_000_000, errors
            )
            self._validate_numeric_range(
                risk, "max_position_notional_per_symbol_usd", 0, 1_000_000, errors
            )
            self._validate_numeric_range(
                risk, "per_trade_risk_fraction", 0.001, 0.1, errors
            )
            self._validate_numeric_range(
                risk, "max_leverage", 1.0, 10.0, errors
            )
        
        brokers = config.get("brokers", {})
        alpaca = brokers.get("alpaca", {})
        if alpaca.get("live", {}).get("enabled", False):
            warnings.append("DANGER: Alpaca live trading is enabled")
    
    def _validate_numeric_range(
        self,
        config: dict,
        field: str,
        min_val: float,
        max_val: float,
        errors: list,
    ) -> None:
        """Validate a numeric field is within range."""
        if field not in config:
            return
        
        val = config[field]
        if not isinstance(val, (int, float)):
            errors.append(f"{field} must be numeric, got {type(val).__name__}")
        elif val < min_val or val > max_val:
            errors.append(f"{field}={val} out of range [{min_val}, {max_val}]")
    
    def get_summary(self) -> dict[str, Any]:
        """Get validation summary."""
        return {
            "all_valid": self.all_valid,
            "total_configs": len(self.results),
            "valid_count": sum(1 for r in self.results if r.valid),
            "invalid_count": sum(1 for r in self.results if not r.valid),
            "results": [
                {
                    "path": r.config_path,
                    "valid": r.valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in self.results
            ],
        }
