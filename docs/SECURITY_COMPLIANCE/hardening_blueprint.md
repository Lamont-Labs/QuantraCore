# QuantraCore Apex — Global Hardening Blueprint v1.1

**Last Updated:** November 2025

## Overview

This document defines the hardening measures implemented across QuantraCore Apex to ensure determinism, safety, security, and compliance without sacrificing capability.

## v9.0-A Security Additions

| Feature | Description |
|---------|-------------|
| **API Authentication** | `X-API-Key` header required for protected endpoints |
| **CORS Restriction** | Regex pattern allowing only localhost and Replit domains |
| **Non-blocking Rate Limiting** | Async-compatible delays in Polygon/Binance adapters |
| **TTL Cache** | 1000 entries max, 5-minute expiration, LRU eviction |
| **Timeframe Validation** | Case-insensitive with warning logs for unknown values |

## Global Principles

1. **Fail-closed always**: On uncertainty, missing manifests, bad configs, or degraded tests, the system refuses to act rather than guessing.
2. **Determinism first**: Any new component must be verifiably deterministic w.r.t inputs.
3. **No silent paths**: Every critical decision (risk, promotion, execution) is logged.
4. **Separation of concerns**: Engine vs ML vs execution vs UI vs broker vs vision are cleanly separated.
5. **Explicit modes**: RESEARCH, PAPER, LIVE; system boots in RESEARCH only.
6. **Minimal trust**: APIs, brokers, feeds, and even model outputs are treated as untrusted inputs until validated and bounded.

## Implementation Status

### 1. Engine Hardening

| Component | Status | Location |
|-----------|--------|----------|
| Protocol Manifest Hashing | ✅ Implemented | `src/quantracore_apex/hardening/manifest.py` |
| Config Validation | ✅ Implemented | `src/quantracore_apex/hardening/config_validator.py` |
| Mode Enforcement | ✅ Implemented | `src/quantracore_apex/hardening/mode_enforcer.py` |
| Incident Logging | ✅ Implemented | `src/quantracore_apex/hardening/incident_logger.py` |
| Kill Switch | ✅ Implemented | `src/quantracore_apex/hardening/kill_switch.py` |

### 2. Mode Permissions Table

| Component | RESEARCH | PAPER | LIVE |
|-----------|----------|-------|------|
| Engine | ENABLED | ENABLED | ENABLED |
| ApexLab | ENABLED | ENABLED | ENABLED |
| Models | ENABLED (advisory) | ENABLED (ranking) | ENABLED |
| Execution Engine | DISABLED | ENABLED (paper only) | ENABLED (with risk) |
| Broker Router | NULL adapter | Paper adapters | Live (with sign-off) |
| QuantraVision | Structural only | Structural only | Structural only |

### 3. Kill Switch System

Automatic triggers:
- Daily drawdown exceeds threshold (default: 5%)
- Broker error rate exceeds threshold (default: 20%)
- Risk violations spike (default: 10 in window)

Manual trigger:
- Operator can engage via API or config flag

Behavior when engaged:
- No new orders allowed
- Open positions optionally flattened
- System remains in SAFE mode until explicitly reset

### 4. Incident Classification

| Class | Description | Default Severity |
|-------|-------------|------------------|
| DATA_FEED_DIVERGENCE | Multi-API price divergence | MEDIUM/HIGH |
| MODEL_MANIFEST_FAILURE | Model validation failed | HIGH |
| RISK_REJECT_SPIKE | Multiple risk rejections | MEDIUM/HIGH |
| BROKER_ERROR_RATE_SPIKE | Broker API errors | HIGH/CRITICAL |
| NUCLEAR_DETERMINISM_FAILURE | Reproducibility broken | CRITICAL |
| UNEXPECTED_PNL_DRAWDOWN | Abnormal losses | HIGH |

### 5. Config Validation

On startup:
1. Validate all YAML/JSON configs against schemas
2. Ensure thresholds and ranges are within allowed bounds
3. If invalid → engine refuses to start with clear error message

Validated files:
- `config/mode.yaml` - System mode settings
- `config/broker.yaml` - Broker and risk configuration
- `config/protocol_manifest.yaml` - Protocol execution order

### 6. Protocol Manifest

Location: `config/protocol_manifest.yaml`

Contains:
- Version and engine snapshot ID
- All 145 protocols with execution order
- SHA-256 hash for integrity verification
- Inputs, outputs, failure modes, and assumptions for each protocol

Verification:
- On startup, hash is computed and compared to stored reference
- If mismatch → engine refuses to run until reconciled

## Usage

### Validate Configuration

```python
from src.quantracore_apex.hardening import ConfigValidator

validator = ConfigValidator()
validator.validate_all()  # Raises ConfigValidationError if invalid
```

### Check Mode Permissions

```python
from src.quantracore_apex.hardening import get_mode_enforcer

enforcer = get_mode_enforcer()
if enforcer.check_permission("execution_engine_access"):
    # Proceed with execution
    pass
else:
    # Blocked - mode doesn't allow this
    pass
```

### Log Incidents

```python
from src.quantracore_apex.hardening import get_incident_logger, IncidentClass, IncidentSeverity

logger = get_incident_logger()
logger.log_incident(
    incident_class=IncidentClass.DATA_FEED_DIVERGENCE,
    severity=IncidentSeverity.MEDIUM,
    message="Price divergence detected",
    context={"symbol": "AAPL", "deviation_pct": 2.5}
)
```

### Use Kill Switch

```python
from src.quantracore_apex.hardening import get_kill_switch_manager, KillSwitchReason

manager = get_kill_switch_manager()

# Check if orders allowed
allowed, reason = manager.check_order_allowed()

# Manual engagement
manager.engage(KillSwitchReason.MANUAL, engaged_by="operator")

# Reset
manager.reset("operator")
```

## ExecutionEngine Integration

The hardening infrastructure is integrated into the ExecutionEngine to enforce safety at the order level:

```python
# In ExecutionEngine.execute_signal():

# 1. Check kill switch
kill_switch = get_kill_switch_manager()
allowed, reason = kill_switch.check_order_allowed()
if not allowed:
    return ExecutionResult(status=OrderStatus.REJECTED, error_message=reason)

# 2. Check mode permissions
mode_enforcer = get_mode_enforcer()
if self.mode == ExecutionMode.PAPER and not mode_enforcer.permissions.paper_orders_allowed:
    return ExecutionResult(status=OrderStatus.REJECTED, error_message="Paper orders not permitted")
if self.mode == ExecutionMode.LIVE and not mode_enforcer.permissions.live_orders_allowed:
    return ExecutionResult(status=OrderStatus.REJECTED, error_message="Live orders not permitted")

# 3. Proceed with risk checks and execution...
```

### Order Flow with Hardening

```
Signal → Kill Switch Check → Mode Permission Check → Risk Engine → Broker Execution
           ↓ FAIL              ↓ FAIL                 ↓ FAIL        ↓ FAIL
        REJECTED            REJECTED               REJECTED      REJECTED
```

## Test Coverage

34 hardening tests + 34 broker tests covering:
- Protocol manifest generation and validation
- Hash computation with execution order
- Config validation
- Mode enforcement and permissions
- Incident logging
- Kill switch management
- ExecutionEngine integration
- Mode-based order blocking

Run tests:
```bash
make test-hardening   # Hardening infrastructure
make test-broker      # Broker layer with hardening
make test-e2e         # End-to-end integration
```

## E2E Verification

26 end-to-end tests verify the complete integration:
- Hardening infrastructure (9 tests)
- Broker layer in RESEARCH mode (2 tests)
- Broker layer in PAPER mode (3 tests)
- EEO Engine (4 tests)
- Estimated Move (3 tests)
- Core Engine (2 tests)
- Compliance (2 tests)
- Determinism (1 test)

See: `logs/e2e_test_results_20251129.log`

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
