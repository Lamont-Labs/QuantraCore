# ApexLab — Offline Training and Distillation Lab

**Version:** 8.0  
**Component:** ApexLab  
**Role:** Model Training Facility for ApexCore Full and Mini

---

## 1. Overview

ApexLab is the offline training and distillation laboratory within the QuantraCore Apex ecosystem. It is responsible for transforming historical market data into trained neural network models (ApexCore Full and ApexCore Mini) that can approximate the behavior of the deterministic Apex core engine.

ApexLab operates under strict constraints:
- **Offline Only** — No internet connectivity during training
- **Reproducible** — All training runs produce identical results given identical inputs
- **Fail-Closed** — Model promotion is blocked if validation fails

---

## 2. Where ApexLab Runs

ApexLab is designed to run on dedicated hardware, designated as **K6** in the system architecture. This hardware provides:

- Sufficient GPU/CPU resources for model training
- Air-gapped network isolation
- Secure storage for training data and model artifacts
- Controlled access for authorized personnel only

The K6 environment ensures that:
- Training data never leaves the secure perimeter
- Model weights are not exposed to external networks
- Reproducibility is maintained through environmental controls

---

## 3. Data Ingestion via Adapters

ApexLab pulls historical market data through a standardized **adapter layer**. Adapters must comply with the deterministic contract:

1. **Cache Raw Payloads** — All API responses are stored in `archives/raw_api_cache/`
2. **Transform Deterministically** — Data normalization produces identical outputs for identical inputs
3. **Log Hashes** — Every transformation step is hashed and logged
4. **Fail Closed on Schema Drift** — If the data schema changes unexpectedly, ingestion halts

Supported adapter types include:
- Market data (IEX, Polygon, Alpaca)
- Fundamentals (financials, earnings calendars)
- Volatility (options chains, volatility indices)
- Macro (macro indicators, sector indices)
- Alternatives (short interest, insider trades)
- News (text-only feeds)

---

## 4. Window and Feature Building

### 4.1 OHLCV Window Generation

ApexLab constructs training windows from historical OHLCV (Open, High, Low, Close, Volume) data:

```
Raw OHLCV Data → Timestamp Alignment → Gap Handling →
Window Slicing → Normalization → Training-Ready Tensors
```

Windows are typically configured as:
- Fixed-length sequences (e.g., 60 bars)
- Multiple timeframes (1m, 5m, 15m, 1h, 1D)
- Overlapping or non-overlapping depending on training mode

### 4.2 Apex Feature Building

For each window, ApexLab computes the full suite of Apex features:

- **Traits** — Structural pattern classifications
- **Microtraits** — Fine-grained pattern details
- **Entropy** — Disorder and unpredictability metrics
- **Suppression** — Low-activity zone detection
- **Drift** — Gradual directional bias
- **Continuation** — Trend persistence signals

These features become the input representation for model training.

---

## 5. Teacher Label Generation

ApexLab runs each training window through the full Apex deterministic engine to generate ground-truth labels:

```
Training Window → Apex Core Engine → Teacher Labels
```

Teacher labels include:
- Regime classification
- Risk tier
- Entropy band
- Volatility band
- Suppression state
- Continuation signals
- QuantraScore (numeric)
- Structural confidence

These labels represent "what Apex would output" for each window, serving as the training target for ApexCore models.

---

## 6. Model Training: ApexCore Full

ApexCore Full is the desktop-class structural model trained directly by ApexLab:

### 6.1 Training Process

```
Apex Features + Teacher Labels → Neural Network Training →
Validation Against Apex Outputs → Model Checkpoint →
Stability Testing → TFLite Export → Manifest Generation
```

### 6.2 Model Specifications

- **Size:** 4–20MB depending on architecture
- **Inputs:** Apex feature tensors
- **Outputs:** Regime, risk tier, chart quality, entropy band, volatility band, suppression state, score band, QuantraScore (numeric)

### 6.3 Training Constraints

- Deterministic weight initialization (seeded)
- Reproducible data shuffling (seeded)
- Identical outputs for identical training runs
- Proof logging of all hyperparameters and checkpoints

---

## 7. Distillation to ApexCore Mini

ApexCore Mini is not trained directly from data—it is **distilled** from ApexCore Full:

### 7.1 Knowledge Distillation

```
ApexCore Full (Teacher) → Soft Labels → ApexCore Mini (Student) →
Validation → TFLite Export → Manifest Generation
```

### 7.2 Model Specifications

- **Size:** 0.5–3MB
- **Inference Time:** <30ms on mobile devices
- **Outputs:** Same contract as Full (regime, risk, entropy, etc.)

### 7.3 Distillation Constraints

- Mini can never exceed Full's output distribution
- Alignment thresholds must be met before promotion
- Fail-closed if distillation degradation is detected

---

## 8. Validation and Safety

Before any model is promoted for deployment, ApexLab runs comprehensive validation:

### 8.1 Alignment Testing

- Compare model outputs to Apex ground truth across validation set
- Compute accuracy, precision, recall for each output dimension
- Flag any outputs that exceed tolerance thresholds

### 8.2 Stability Testing

- Run identical inputs multiple times to confirm deterministic inference
- Test edge cases (extreme values, missing data, unusual patterns)
- Verify fail-closed behavior under uncertainty

### 8.3 Fail-Closed Promotion

If any validation check fails:
- Model promotion is **blocked**
- Error logs are generated with detailed diagnostics
- Manual review is required before retry

---

## 9. Export Artifacts

Successful training produces the following artifacts:

### 9.1 TFLite Models

- `apexcore_full_vX.Y.tflite` — Desktop model
- `apexcore_mini_vX.Y.tflite` — Mobile model

### 9.2 Model Manifests

JSON manifests documenting:
- Model version and training timestamp
- Training data hash
- Hyperparameters
- Validation metrics
- Apex engine version used for labeling

### 9.3 Proof Logs

Complete audit trail of the training run:
- Input data hashes
- Feature computation logs
- Training step logs
- Validation results
- Export confirmation

---

## 10. Summary

ApexLab is the controlled, offline environment where QuantraCore Apex's neural models are born. By using the deterministic Apex engine as a teacher, ApexLab ensures that ApexCore Full and Mini remain aligned with the authoritative core while providing efficient inference for deployment scenarios. The fail-closed design guarantees that no degraded model ever reaches production.
