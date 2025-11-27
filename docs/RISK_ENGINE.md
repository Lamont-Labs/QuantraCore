# QuantraCore Apex™ — Risk Engine

**Version:** 8.0  
**Component:** Final Gatekeeper  
**Status:** Active

---

## Overview

The Risk Engine is the final arbiter of whether any trade operation is allowed. It performs comprehensive checks across volatility, spread, regime, entropy, drift, suppression, structural integrity, and fundamental context.

---

## Role

The Risk Engine serves as the mandatory gatekeeper for all system-level decisions. No order can be placed without passing all risk checks.

---

## Check Categories

### Volatility Checks

| Check | Description |
|-------|-------------|
| ATR bands | Volatility within acceptable range |
| Volatility expansion | Not in extreme expansion |
| Relative volatility vs sector | Within sector norms |

### Spread Checks

| Check | Description |
|-------|-------------|
| Spread threshold | Spread below maximum |
| Slippage model | Expected slippage acceptable |

### Regime Checks

| Check | Description |
|-------|-------------|
| Regime mismatch | Strategy matches regime |
| Regime instability | Regime is stable |

### Entropy Checks

| Check | Description |
|-------|-------------|
| entropy_state == reject | Auto-deny if entropy too high |

### Drift Checks

| Check | Description |
|-------|-------------|
| drift_state == unstable | Auto-deny if drift unstable |

### Suppression Checks

| Check | Description |
|-------|-------------|
| suppression_state == clash | Check for structural clash |

### Structural Integrity Checks

| Check | Description |
|-------|-------------|
| Trend integrity | Trend structure valid |
| Range integrity | Range structure valid |
| Slope/strength alignment | Metrics aligned |

### Fundamentals Context Checks

| Check | Description |
|-------|-------------|
| Earnings proximity | Check upcoming earnings |
| Major macro event proximity | Check macro calendar |

---

## Output

| Field | Type | Values |
|-------|------|--------|
| `risk_tier` | enum | low \| medium \| high |
| `final_permission` | enum | allow \| deny |
| `override_code` | enum | none \| entropy \| drift \| suppression \| compliance \| spread \| volatility |

### Output Example

```json
{
  "risk_tier": "medium",
  "final_permission": "allow",
  "override_code": "none",
  "checks_passed": 8,
  "checks_failed": 0,
  "timestamp": "2025-11-27T14:30:00Z"
}
```

---

## Kill Switches

The Risk Engine maintains several kill switches that halt all activity:

| Kill Switch | Trigger |
|-------------|---------|
| Daily loss limit | Daily loss exceeds threshold |
| Max drawdown threshold | Drawdown exceeds limit |
| Portfolio heat | Total exposure too high |
| Unexpected volatility spike | VIX or sector vol spike |
| API error chain | Multiple consecutive API errors |

### Kill Switch Activation

When a kill switch is triggered:

1. All pending orders cancelled
2. New orders blocked
3. Alert logged
4. Manual intervention required

---

## Decision Flow

```
Order Request
    ↓
Volatility Checks
    ↓ [Pass]
Spread Checks
    ↓ [Pass]
Regime Checks
    ↓ [Pass]
Entropy Checks
    ↓ [Pass]
Drift Checks
    ↓ [Pass]
Suppression Checks
    ↓ [Pass]
Structural Integrity
    ↓ [Pass]
Fundamentals Context
    ↓ [Pass]
Kill Switch Check
    ↓ [Clear]
FINAL_PERMISSION = ALLOW
```

If any check fails:
```
FINAL_PERMISSION = DENY
override_code = [failing_check]
```

---

## Integration Points

### With Broker/OMS

Every order passes through Risk Engine before submission.

### With Omega Directives

Omega directives (Ω1–Ω3) can override Risk Engine decisions:
- Ω1 → Force deny all
- Ω2 → Force deny if entropy-related
- Ω3 → Force deny if drift-related

### With QuantraScore

Low QuantraScore contributes to higher risk_tier assignment.

---

## Logging

All Risk Engine decisions are logged:

```json
{
  "timestamp": "2025-11-27T14:30:00Z",
  "request_type": "order_validation",
  "symbol": "AAPL",
  "checks": {
    "volatility": {"result": "pass", "value": 0.23},
    "spread": {"result": "pass", "value": 0.02},
    "regime": {"result": "pass", "value": "trend_up"},
    "entropy": {"result": "pass", "state": "normal"},
    "drift": {"result": "pass", "state": "stable"},
    "suppression": {"result": "pass", "state": "none"},
    "structural": {"result": "pass", "integrity": 0.89},
    "fundamentals": {"result": "pass", "earnings_days": 45}
  },
  "risk_tier": "low",
  "final_permission": "allow",
  "override_code": "none"
}
```

---

## Configuration

Risk Engine parameters are configurable but have strict defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_daily_loss | 2% | Maximum daily loss |
| max_drawdown | 5% | Maximum drawdown |
| max_portfolio_heat | 50% | Maximum exposure |
| max_spread | 0.5% | Maximum bid-ask spread |
| vol_spike_threshold | 2.0 | VIX spike multiple |

---

## Related Documentation

- [Broker/OMS](BROKER_OMS.md)
- [Portfolio System](PORTFOLIO_SYSTEM.md)
- [Omega Directives](OMEGA_DIRECTIVES.md)
- [Core Engine](CORE_ENGINE.md)
