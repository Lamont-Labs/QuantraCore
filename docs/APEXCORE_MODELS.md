# ApexCore Models — Model Family Documentation

**Version:** 8.0  
**Components:** ApexCore Full, ApexCore Mini  
**Role:** On-device neural assistant models

---

## 1. Overview

ApexCore is the **on-device neural assistant model** within the QuantraCore Apex ecosystem. These neural network models are trained to approximate the behavior of the deterministic Apex core engine, enabling efficient inference in scenarios where running the full engine is impractical.

The family consists of two versions:
- **ApexCore Full** — Desktop-class model for K6 workstation
- **ApexCore Mini** — Mobile-optimized model for Android/QuantraVision

---

## 2. Core Principles

All ApexCore models adhere to these fundamental principles:

1. **Apex is always the teacher** — Models learn exclusively from the deterministic core
2. **ApexCore never overrides Apex** — When both are available, Apex is authoritative
3. **Fails closed if uncertain** — Conservative behavior under ambiguity

---

## 3. ApexCore Full (Desktop)

### 3.1 Target Platform

| Property | Value |
|----------|-------|
| Target | Desktop (K6) |
| Hardware | GMKtec NucBox K6 |

### 3.2 Specifications

| Property | Value |
|----------|-------|
| Size | 3–20 MB |
| Format | TFLite |
| Inference Speed | <20ms |
| Architecture | Multi-head neural network (8–12 heads) |
| Training | Direct from ApexLab |

### 3.3 Capabilities

ApexCore Full provides:

- High-resolution structural classification
- Regime detection
- Chart quality score
- Volatility banding
- Entropy states
- Suppression detection
- Score band
- **Mandatory QuantraScore** (0–100)

---

## 4. ApexCore Mini (Mobile)

### 4.1 Target Platform

| Property | Value |
|----------|-------|
| Target | Android (QuantraVision) |
| Use Case | Mobile overlay analysis |

### 4.2 Specifications

| Property | Value |
|----------|-------|
| Size | 0.5–3 MB |
| Format | TFLite |
| Inference | <30ms |
| Training | Distilled from Full |

### 4.3 Constraints

- Reduced heads (optimized architecture)
- Optimized for speed
- Same teacher labels as Full

---

## 5. Shared Output Contract

Both ApexCore Full and Mini produce outputs aligned with the QuantraScore system:

### 5.1 QuantraScore Range

| Property | Value |
|----------|-------|
| Range | 0–100 |
| Buckets | fail, wait, pass, strong_pass |

### 5.2 Output Dimensions

| Output | Description |
|--------|-------------|
| Regime | Market regime classification |
| Risk tier | Risk level assessment |
| Chart quality | Structural clarity score |
| Volatility band | Volatility regime |
| Entropy states | Disorder classification |
| Suppression detection | Low-activity zone flag |
| Score band | Discretized QuantraScore range |
| QuantraScore | Composite score (0–100) |

---

## 6. Distillation Relationship

The ApexCore models form a knowledge distillation chain:

```
Apex (Deterministic Core)
       ↓ Teacher Labels
ApexCore Full (Desktop Model)
       ↓ Knowledge Distillation
ApexCore Mini (Mobile Model)
```

### 6.1 Apex → Full

- ApexLab runs 100-bar windows through the Apex engine
- Apex outputs become training labels for Full
- Full learns to approximate Apex behavior
- Validation ensures alignment within tolerance

### 6.2 Full → Mini

- Full's predictions become "soft labels" for Mini
- Mini learns to approximate Full's behavior
- Distillation compresses knowledge into smaller architecture
- Validation ensures Mini meets accuracy thresholds

---

## 7. MODEL_MANIFEST.json

Every exported model includes a manifest file documenting:

```json
{
  "model_name": "apexcore_full",
  "version": "8.0.0",
  "training_timestamp": "2025-10-15T10:00:00Z",
  "apex_version": "8.0",
  "training_data_hash": "sha256:abc123...",
  "deterministic_hash": "sha256:def456...",
  "validation_metrics": {
    "quantrascore_accuracy": 0.95,
    "regime_accuracy": 0.97
  },
  "training_windows": "100-bar OHLCV"
}
```

---

## 8. Deployment Scenarios

### 8.1 Desktop Only (Full)

For K6 workstation deployments:
- Use ApexCore Full for primary analysis
- Fall back to Apex engine for validation
- Best accuracy, higher resource usage

### 8.2 Mobile Only (Mini)

For Android/QuantraVision deployments:
- Use ApexCore Mini for all analysis
- <30ms inference for real-time overlays
- Fail-closed behavior on uncertainty

### 8.3 Hybrid (Full + Mini)

For connected deployments:
- Mini provides immediate on-device results
- Full (via remote connection) provides validation
- Discrepancies flagged for review

---

## 9. Test Coverage

ApexCore models undergo rigorous testing:

| Test Category | Tests |
|---------------|-------|
| Inference | Speed validation (<30ms for Mini) |
| Consistency | Alignment with Apex outputs |
| Safety | Fail-closed path verification |

---

## 10. Summary

The ApexCore model family provides efficient neural approximations of the deterministic Apex engine. Through careful training and distillation from 100-bar OHLCV windows, Full and Mini models deliver structural analysis at varying resource profiles while maintaining alignment with the authoritative Apex core. The shared QuantraScore (0–100) output contract ensures consistency across all deployment scenarios.
