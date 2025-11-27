# QuantraCore Apex™ — QuantraVision v1 (Legacy)

**Version:** 8.0  
**Component:** Thin-Client Signal Viewer  
**Status:** Legacy (Maintained)

---

## Overview

QuantraVision v1 (Legacy) is a thin-client viewer that displays upstream structural signals produced by the QuantraCore Apex engine. It provides a lightweight interface for retail-facing signal consumption without on-device inference.

---

## Characteristics

| Feature | Description |
|---------|-------------|
| On-device inference | None |
| Signal source | Upstream Apex engine |
| Purpose | Retail signal consumption |
| UI complexity | Lightweight |
| Cloud narration | Optional |

---

## Displayed Outputs

| Output | Description |
|--------|-------------|
| Trend | upward \| downward \| sideways \| unclear |
| Pressure | buy_bias \| sell_bias \| neutral |
| Risk tier | low \| medium \| high |
| QuantraScore | 0–100 score |
| Regime | Current market regime |
| Structural overlays | Read-only visual overlays |

---

## Safety Guarantees

| Rule | Description |
|------|-------------|
| No trading actions | Cannot place orders |
| No predictions on device | All computation upstream |
| No OHLCV captured via screen | No data extraction |

---

## Architecture

```
┌─────────────────────────────────────┐
│      QUANTRACORE APEX (Server)      │
│  ─────────────────────────────────  │
│  Full deterministic analysis        │
│  QuantraScore computation           │
│  Protocol execution                 │
└───────────────┬─────────────────────┘
                │
                │ Structured Signals
                │ (JSON/WebSocket)
                ↓
┌─────────────────────────────────────┐
│      QUANTRAVISION V1 (Client)      │
│  ─────────────────────────────────  │
│  Display-only                       │
│  No local computation               │
│  Read-only overlays                 │
└─────────────────────────────────────┘
```

---

## Signal Format

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "symbol": "AAPL",
  "signals": {
    "trend": "upward",
    "pressure": "buy_bias",
    "strength": "high",
    "risk_tier": "low",
    "quantrascore": 78,
    "regime": "trend_up",
    "entropy_state": "normal",
    "drift_state": "stable"
  },
  "overlays": {
    "support_levels": [148.50, 145.00],
    "resistance_levels": [155.00, 160.00],
    "trend_channel": {
      "upper": 157.00,
      "lower": 151.00
    }
  }
}
```

---

## UI Components

### Signal Panel

- Trend indicator (color-coded arrow)
- Pressure gauge
- Risk tier badge
- QuantraScore display

### Overlay Panel

- Support/resistance lines
- Trend channel
- Regime indicator
- Entropy/drift status

---

## Cloud Narration (Optional)

When enabled, cloud narration provides:

- Text summary of structural state
- Regime context explanation
- Key level descriptions

**Compliance:** All narration is scrubbed for trade language.

---

## Comparison with v2

| Feature | v1 Legacy | v2 Apex |
|---------|-----------|---------|
| On-device inference | No | Yes (ApexCore Mini) |
| Computation | Server | On-device |
| Latency | Network-dependent | <30ms |
| Offline capability | No | Yes |
| Model required | No | ApexCore Mini |

---

## Use Cases

- Quick signal viewing
- Low-power devices
- Network-available environments
- Simple signal consumption

---

## Related Documentation

- [QuantraVision v2](QUANTRAVISION_V2.md)
- [QuantraVision Remote](QUANTRAVISION_REMOTE.md)
- [Core Engine](CORE_ENGINE.md)
