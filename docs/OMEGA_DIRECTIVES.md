# QuantraCore Apex™ — Omega Directives (Ω1–Ω4)

**Version:** 8.0  
**Component:** System Safety Locks  
**Status:** Active

---

## Overview

Omega directives enforce compliance, safety, and deterministic boundaries. They override **ALL** protocol results, all learning protocol outputs, and all ApexCore model outputs. They are absolute system-level kill switches.

---

## Directive Summary

| Directive | Name | Purpose |
|-----------|------|---------|
| Ω1 | Hard Safety Lock | Force system to safe state |
| Ω2 | Entropy Override | Disable logic under high entropy |
| Ω3 | Drift Override | Invalidate trends under instability |
| Ω4 | Compliance Override | Suppress unsafe language |

---

## Ω1: Hard Safety Lock

**Purpose:** Force the system into a complete safe state when catastrophic conditions are detected.

### Triggers

- Catastrophic entropy
- Drift instability
- Protocol mismatch
- Invalid data input

### Effect

- Force `quantrascore = 0`
- System enters safe state
- All outputs suspended
- Manual intervention required

### Example Scenario

```
Input: Corrupted OHLCV data detected
Trigger: Invalid data input
Result: Ω1 activated
        → QuantraScore forced to 0
        → All trading signals suspended
        → System enters safe mode
```

---

## Ω2: Entropy Override

**Purpose:** Disable continuation and extension logic when signal clarity is compromised.

### Triggers

- `entropy_state = reject`

### Effects

- Disable continuation logic
- Disable extension logic
- Clamp all strength metrics to `low`

### Example Scenario

```
Input: High noise detected (entropy_score > 0.85)
Trigger: entropy_state = reject
Result: Ω2 activated
        → Continuation validator disabled
        → Entry timing disabled
        → All strength outputs clamped to low
```

---

## Ω3: Drift Override

**Purpose:** Invalidate trend classifications when structural drift indicates instability.

### Triggers

- `drift_state = unstable`

### Effects

- Invalidate trend classification
- Force `regime = transition`

### Example Scenario

```
Input: Drift score exceeds instability threshold
Trigger: drift_state = unstable
Result: Ω3 activated
        → Trend classification invalidated
        → Regime forced to 'transition'
        → No directional outputs permitted
```

---

## Ω4: Compliance Override

**Purpose:** Suppress any output that could be interpreted as trading advice.

### Triggers

- Output phrasing risk
- Attempt to construct trade-like language

### Effects

- Suppress output
- Return compliance-safe structural summary

### Prohibited Terms

The following terms trigger Ω4 if they appear in outputs:

| Prohibited | Category |
|------------|----------|
| buy | Trade action |
| sell | Trade action |
| long | Position direction |
| short | Position direction |
| entry | Timing instruction |
| exit | Timing instruction |
| target | Price target |
| stop | Risk instruction |
| scalp | Trading style |

### Example Scenario

```
Input: Model attempts to output "Strong buy signal"
Trigger: Output phrasing risk (contains "buy")
Result: Ω4 activated
        → Original output suppressed
        → Replaced with: "Structural analysis: 
           trend=upward, pressure=buy_bias, 
           strength=high, risk_level=low"
```

---

## Directive Hierarchy

Omega directives form a strict hierarchy:

```
Ω1 (Hard Safety Lock)
 ↓ overrides
Ω2 (Entropy Override)
 ↓ overrides
Ω3 (Drift Override)
 ↓ overrides
Ω4 (Compliance Override)
 ↓ overrides
All Protocol Outputs (T01–T80, LP01–LP25)
 ↓ overrides
All ApexCore Model Outputs
```

If Ω1 is triggered, all other directives and outputs are superseded.

---

## Implementation Requirements

### Enforcement Points

1. **Data Ingestion** — Ω1 checks for invalid data
2. **Apex Execution** — All Ω directives checked post-protocol
3. **ApexCore Inference** — Ω4 scans model outputs
4. **Broker/OMS** — Ω1–Ω3 gate order placement
5. **QuantraVision** — Ω4 scrubs narration

### Logging Requirements

Every Omega activation must be logged:

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "directive": "Ω2",
  "trigger": "entropy_state = reject",
  "entropy_score": 0.91,
  "effects_applied": [
    "continuation_logic_disabled",
    "extension_logic_disabled",
    "strength_clamped_low"
  ],
  "original_quantrascore": 67,
  "modified_quantrascore": 23
}
```

---

## Fail-Closed Behavior

All Omega directives enforce fail-closed behavior:

| Condition | Response |
|-----------|----------|
| Directive check fails | Assume triggered |
| Ambiguous state | Activate most restrictive |
| System error | Activate Ω1 |
| Unknown trigger | Activate Ω1 |

---

## Testing Requirements

Omega directives must pass deterministic tests:

1. **Trigger Tests** — Each trigger condition activates correct directive
2. **Effect Tests** — Each effect is correctly applied
3. **Override Tests** — Higher directives correctly override lower
4. **Logging Tests** — All activations are logged
5. **Fail-Closed Tests** — Ambiguous conditions trigger safe response

---

## Related Documentation

- [Core Engine](CORE_ENGINE.md)
- [Protocols: Tier (T01–T80)](PROTOCOLS_TIER.md)
- [Protocols: Learning (LP01–LP25)](PROTOCOLS_LEARNING.md)
- [Security & Compliance](SECURITY_COMPLIANCE.md)
