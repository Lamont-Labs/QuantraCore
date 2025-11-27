# QuantraCore Apex™ — QuantraVision v2 (Apex)

**Version:** 8.0  
**Component:** On-Device Neural Copilot  
**Status:** Active

---

## Overview

QuantraVision v2 (Apex) is a full on-device copilot providing real-time mobile chart analysis using on-device neural inference. It runs ApexCore Mini for instant structural analysis without cloud dependency.

---

## Pipeline

```
MediaProjection Capture
    ↓
Bounding-Box Detection
    ↓
VisionLite Primitives
    ↓
ApexLite Structural Logic
    ↓
ApexCore Mini Inference (<30ms)
    ↓
Overlay Rendering
    ↓
HUD Display
    ↓
Narration (Optional)
    ↓
Compliance Validation
```

---

## Features

| Feature | Description |
|---------|-------------|
| Borderless floating Q icon | Unobtrusive activation |
| Instant scan | One-tap analysis |
| On-device inference | <30ms latency |
| Deterministic overlays | Consistent structural visualization |
| Hybrid QuantraScore fusion | Combined scoring |
| Fail-closed behavior | Safe on detection failure |
| Sector-aware hints | Sector context integration |
| Risk-tier classification | Real-time risk assessment |
| Regime classification | Market regime identification |

---

## Safety Guarantees

| Rule | Description |
|------|-------------|
| No buy/sell/entry/exit outputs | Compliance enforced |
| Omega-4 hard compliance gate | Language scrubbing |
| Narration scrub | Verbs + directional language removed |

---

## Model Requirements

| Component | Specification |
|-----------|---------------|
| Model | ApexCore Mini.tflite |
| Size | 0.5–3 MB |
| Latency | <30ms |
| Validation | MODEL_MANIFEST verified |

---

## UI Components

### Floating Q

- Borderless floating activation button
- Positioned for easy access
- One-tap to scan
- Cinematic neon-blue accent

### HUD Display

- QuantraScore prominently displayed
- Trend indicator
- Risk tier badge
- Regime classification
- Entropy/drift status
- Auto-hide functionality

### Overlay System

- Cinematic neon-blue overlays
- Support/resistance zones
- Trend channels
- Compression indicators
- Structural annotations

---

## Fail-Closed Conditions

QuantraVision v2 enters safe mode if:

| Condition | Action |
|-----------|--------|
| Bounding box <65% confidence | Reject scan |
| Model mismatch | Block inference |
| Entropy or drift contradictions | Show warning |
| VisionLite fallback failure | Display error |

### Safe Mode Display

```
┌────────────────────────────┐
│  SCAN QUALITY INSUFFICIENT │
│  ────────────────────────  │
│  Unable to analyze chart   │
│  Please ensure:            │
│  • Chart is clearly visible│
│  • Minimal obstructions    │
│  • Standard chart format   │
└────────────────────────────┘
```

---

## Output Format

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "scan_quality": 0.87,
  "inference_time_ms": 24,
  "results": {
    "quantrascore": 72,
    "score_bucket": "pass",
    "trend": "upward",
    "pressure": "buy_bias",
    "strength": "medium",
    "risk_tier": "low",
    "regime": "trend_up",
    "entropy_state": "normal",
    "volatility_band": "medium"
  },
  "overlays": {
    "support_zones": [...],
    "resistance_zones": [...],
    "trend_channel": {...},
    "compression_regions": [...]
  },
  "compliance_check": "passed"
}
```

---

## Narration (Optional)

When enabled, narration provides spoken structural summaries:

**Example Output:**
> "Structure shows upward trend with medium strength. Risk tier is low. Regime classified as trending. QuantraScore seventy-two."

**Scrubbed Language:**
All narration removes:
- Buy/sell
- Entry/exit
- Target/stop
- Directional recommendations

---

## Comparison with v1

| Feature | v1 Legacy | v2 Apex |
|---------|-----------|---------|
| On-device inference | No | Yes |
| Computation | Server | On-device |
| Latency | Network-dependent | <30ms |
| Offline capability | No | Yes |
| Model | None | ApexCore Mini |
| Chart capture | No | Yes |
| Real-time overlay | No | Yes |

---

## Related Documentation

- [QuantraVision v1](QUANTRAVISION_V1.md)
- [QuantraVision Remote](QUANTRAVISION_REMOTE.md)
- [ApexCore Models](APEXCORE_MODELS.md)
- [Core Engine](CORE_ENGINE.md)
