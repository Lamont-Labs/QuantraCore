# ApexLab — Offline Training and Distillation Lab

**Version:** 8.0  
**Component:** ApexLab  
**Role:** Self-contained offline local training environment

---

## 1. Overview

ApexLab is the self-contained offline local training environment within the QuantraCore Apex ecosystem. It is responsible for transforming historical market data into trained neural network models (ApexCore Full and ApexCore Mini) that can approximate the behavior of the deterministic Apex core engine.

ApexLab operates under strict constraints:
- **Offline Only** — No internet connectivity during training
- **Reproducible** — All training runs produce identical results given identical inputs
- **Fail-Closed** — Model promotion is blocked if validation fails

---

## 2. Where ApexLab Runs

ApexLab is designed to run on the **GMKtec NucBox K6** workstation with these specifications:

| Resource | Recommendation |
|----------|----------------|
| CPU | 8-core max recommended |
| RAM | 16GB recommended |
| GPU | Optional — CPU-optimized |
| Storage | Local logs + model store |

The K6 environment ensures that:
- Training data never leaves the secure perimeter
- Model weights are not exposed to external networks
- Reproducibility is maintained through environmental controls

---

## 3. Capabilities

ApexLab provides these core capabilities:

1. **Continuous data ingestion** — Ongoing market data collection
2. **Apex-labeled dataset construction** — Building training sets with Apex labels
3. **Feature generation (Apex-native)** — Computing native Apex features
4. **ApexCore model training** — Training Full and Mini models
5. **Model evaluation + rejection** — Quality gates for model promotion
6. **Automated scheduled learning** — Scheduled retraining cycles

---

## 4. Data Ingestion via Adapters

ApexLab pulls historical market data through standardized adapters supporting:

**Supported Providers:**
- Polygon
- Tiingo
- Alpaca Market Data
- Intrinio
- Finnhub

**Data Types:**
- Historical OHLCV
- Realtime OHLCV
- Corporate actions
- Sector metadata
- Volatility indexes
- Macro indexes
- Alternative data (news/feeds)

**Compliance Note:** Only data ingestion — no trade recommendations.

---

## 5. Training Windows and Features

### 5.1 OHLCV Windows

ApexLab constructs training windows using **100-bar OHLCV** data:

```
Raw OHLCV Data → Timestamp Alignment → Gap Handling →
Window Slicing (100 bars) → Normalization → Training-Ready Tensors
```

### 5.2 Apex Feature Building

For each window, ApexLab computes the full suite of Apex features:

- **All microtraits** — Fine-grained pattern details
- **Entropy packet** — Disorder and unpredictability metrics
- **Suppression vectors** — Low-activity zone detection
- **Drift signature** — Gradual directional bias
- **Sector context** — Sector-aware features
- **Regime/volatility labels** — Market regime classification

---

## 6. Teacher Label Generation

ApexLab runs each training window through the full Apex deterministic engine to generate ground-truth labels:

```
100-bar Window → Apex Core Engine → Teacher Labels
```

Teacher labels include:
- QuantraScore target (0–100)
- Regime classification
- Risk tier
- Chart quality score
- Volatility band
- Entropy states
- Suppression detection
- Score band

---

## 7. Model Training: ApexCore Full

ApexCore Full is the desktop-class structural model trained directly by ApexLab:

### 7.1 Target Platform
- **Desktop (K6)** — GMKtec NucBox K6 workstation

### 7.2 Model Specifications

| Property | Value |
|----------|-------|
| Size | 3–20 MB |
| Format | TFLite |
| Target | Desktop (K6) |

### 7.3 Capabilities

- High-resolution structural classification
- Regime detection
- Chart quality score
- Volatility banding
- Entropy states
- Suppression detection
- Score band
- Mandatory QuantraScore

---

## 8. Distillation to ApexCore Mini

ApexCore Mini is distilled from ApexCore Full for mobile deployment:

### 8.1 Target Platform
- **Android (QuantraVision)** — Mobile devices

### 8.2 Model Specifications

| Property | Value |
|----------|-------|
| Size | 0.5–3 MB |
| Inference | <30ms |
| Target | Android |

### 8.3 Constraints

- Reduced heads
- Optimized for speed
- Same teacher labels as Full

---

## 9. Model Principles

All ApexCore models adhere to these principles:

1. **Apex is always the teacher** — Models learn from deterministic core
2. **ApexCore never overrides Apex** — Deterministic rules take precedence
3. **Fails closed if uncertain** — Conservative behavior under ambiguity

---

## 10. Export Artifacts

Successful training produces the following artifacts:

### 10.1 Model Files

- `ApexCore.tflite` — Trained TFLite model

### 10.2 Manifest

- `MODEL_MANIFEST.json` — Complete training documentation

### 10.3 Verification

- Deterministic hash — For integrity verification
- Training metrics — Performance measurements

---

## 11. Tests

ApexLab includes comprehensive testing:

### 11.1 Dataset Tests
- Dataset integrity
- Label reproducibility

### 11.2 Model Tests
- Model accuracy gates
- Consistency with Apex
- Fail-closed paths

### 11.3 Inference Tests
- Inference speed validation

---

## 12. Summary

ApexLab is the controlled, offline environment where QuantraCore Apex's neural models are born. Using 100-bar OHLCV windows and Apex-native features, ApexLab ensures that ApexCore Full and Mini remain aligned with the authoritative core while providing efficient inference for deployment scenarios. The fail-closed design guarantees that no degraded model ever reaches production.
