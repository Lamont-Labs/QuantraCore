# QuantraCore Apex™ — Broker / OMS Layer

**Version:** 8.0  
**Component:** Execution Envelope  
**Status:** Active (Disabled by Default)

---

## Overview

The Broker/OMS layer is an optional institutional-grade execution envelope with strict compliance controls. It is **disabled by default** and requires explicit manual enabling with full compliance configuration.

---

## Operational Modes

| Mode | Description | Default |
|------|-------------|---------|
| Research | Analysis only, no orders | Yes |
| Simulation | Simulated execution | No |
| Paper | Paper trading with broker | No |
| Live-Ready | Live trading capability | No |

---

## Live Mode Requirements

Live mode requires ALL of the following:

1. Explicit manual enable
2. Compliance configuration signed
3. API credential validation
4. Risk engine in fully armed mode

---

## Supported Integrations

### Alpaca

| Capability | Status |
|------------|--------|
| Paper trading | Supported |
| Live trading | Locked behind compliance |
| Order history | Supported |

### Interactive Brokers

| Capability | Status |
|------------|--------|
| Full OMS support | Supported |
| Portfolio sync | Supported |
| Realtime market link | Supported |

### Custom OMS API

| Capability | Status |
|------------|--------|
| REST order flow | Supported |
| Socket order flow | Supported |
| Institutional OMS compatibility | Supported |

---

## Order Types

| Order Type | Description |
|------------|-------------|
| Market | Execute at market price |
| Limit | Execute at limit price |
| Stop | Stop order |
| Reduce Only | Position reduction only |
| Cancel/Replace | Modify existing order |

**Note:** Order type availability depends on provider.

---

## Fail-Closed Rules

The Broker/OMS layer will fail-closed (deny order) if:

| Condition | Action |
|-----------|--------|
| Risk engine denies permission | Order blocked |
| Omega directive invoked | Order blocked |
| Drift/entropy unsafe state | Order blocked |
| Data integrity mismatch | Order blocked |
| Protocol firing mismatch | Order blocked |
| Broker API error | Order blocked |
| Compliance check failure | Order blocked |

---

## Order Logging

Every order attempt is logged with full context:

```json
{
  "order_id": "ORD-2025-001234",
  "symbol": "AAPL",
  "timestamp": "2025-11-27T14:30:00Z",
  "side": "buy",
  "quantity": 100,
  "order_type": "limit",
  "limit_price": 150.25,
  "decision_reason": "manual_trigger",
  "risk_checks_run": [
    "volatility_check",
    "spread_check",
    "regime_check",
    "entropy_check",
    "drift_check"
  ],
  "risk_check_results": {
    "volatility_check": "pass",
    "spread_check": "pass",
    "regime_check": "pass",
    "entropy_check": "pass",
    "drift_check": "pass"
  },
  "fail_closed_triggers": [],
  "order_status": "submitted",
  "broker_response": {
    "status": "accepted",
    "broker_order_id": "BRK-12345"
  }
}
```

---

## Compliance Safeguards

### Trading Restrictions

- Live trading disabled unless manually unlocked
- Risk engine must approve every order
- Compliance config must be signed
- Invalid trades auto-blocked

### Unlock Procedure

1. Access compliance configuration
2. Sign compliance acknowledgment
3. Configure risk parameters
4. Validate API credentials
5. Enable live mode explicitly

---

## Integration with Risk Engine

The Broker/OMS layer ALWAYS consults the Risk Engine before any order:

```
Order Request
    ↓
Risk Engine Evaluation
    ↓
[Pass] → Submit to Broker
[Fail] → Reject Order + Log Reason
```

See [Risk Engine](RISK_ENGINE.md) for full specification.

---

## Related Documentation

- [Risk Engine](RISK_ENGINE.md)
- [Portfolio System](PORTFOLIO_SYSTEM.md)
- [Security & Compliance](SECURITY_COMPLIANCE.md)
