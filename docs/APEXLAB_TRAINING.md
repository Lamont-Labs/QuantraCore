# QuantraCore Apex™ — ApexLab Training Environment

**Version:** 8.0  
**Component:** Offline Intelligence Distillation  
**Status:** Active

---

## Overview

ApexLab is a fully autonomous offline training environment living on the K6 workstation. It continuously ingests OHLCV data, constructs deterministic datasets, generates teacher labels using QuantraCore Apex, and trains both versions of ApexCore models.

---

## Characteristics

- Runs entirely offline
- Uses local-only datasets
- Nightly scheduled training
- Can run 24/7 on K6
- Reproducible model training
- Zero cloud cost, zero privacy leakage

---

## Data Windows

| Parameter | Value |
|-----------|-------|
| OHLCV Window Size | 100 bars |
| Stride | 1 |
| Normalization | Z-score + volatility-adjusted scaling |

### Additional Metadata

- Timestamp
- Symbol
- Sector
- Volatility index
- Macro volatility state

---

## Feature Generation

### Apex-Native Features

| Feature | Description |
|---------|-------------|
| `wick_ratio` | Wick to body proportion |
| `body_ratio` | Body size relative to range |
| `body_power` | Directional body strength |
| `compression_score` | Range compression level |
| `noise_score` | Signal noise measurement |
| `strength_score` | Trend strength |
| `entropy_score` | Signal clarity |
| `suppression_score` | Structural suppression |
| `drift_score` | Trend deviation |
| `sector_flow` | Sector money flow |
| `sector_strength` | Sector relative strength |
| `volume_pulse` | Volume intensity |
| `volatility_band` | Volatility classification |
| `trend_slope` | Trend direction slope |
| `range_density` | Range utilization |

**Feature Vector Size:** 300–600 (depending on configuration)

---

## Teacher Labeling

**Source:** QuantraCore Apex deterministic execution

### Generated Labels

| Label | Description |
|-------|-------------|
| `regime_class` | Current market regime |
| `volatility_band` | Volatility classification |
| `compression_state` | Compression level |
| `continuation_likelihood` | Continuation probability |
| `risk_tier` | Risk level |
| `pressure_class` | Directional pressure |
| `suppression_state` | Structural suppression |
| `entropy_state` | Signal clarity state |
| `drift_state` | Trend drift state |
| `sector_bias` | Sector pressure |
| `score_bucket` | QuantraScore band |
| `final_quant_score_numeric` | Raw score 0–100 |

### Guarantees

- Labels are deterministic and fully reproducible
- ApexLab stores protocol trace per label

---

## Training Pipeline

### Pipeline Stages

```
1. Dataset Preparation
    ↓
2. Feature Normalization
    ↓
3. Teacher Inference → Deterministic Label Generation
    ↓
4. Multi-Head Neural Model Training
    ↓
5. Cross-Entropy + MSE Hybrid Loss
    ↓
6. Model Validation
    ↓
7. Model Rejection / Promotion
```

### Loss Heads

| Head | Purpose |
|------|---------|
| `regime_head` | Regime classification |
| `risk_head` | Risk tier prediction |
| `quality_head` | Chart quality assessment |
| `entropy_head` | Entropy state prediction |
| `drift_head` | Drift state prediction |
| `suppression_head` | Suppression state prediction |
| `scorehead` | QuantraScore numeric regression |

---

## Model Validation

### Must-Match Apex Outputs

The following outputs must align with Apex teacher:

- Regime
- Risk
- Entropy
- Suppression
- Drift
- Score band
- QuantraScore

### Determinism Checks

- Protocol-map-aligned label sampling
- Hash consistency across datasets
- Stable predictions on golden-set items

---

## Model Rejection Criteria

A model is rejected if any of these conditions are met:

| Criterion | Threshold |
|-----------|-----------|
| Accuracy drop on golden-set | >2% |
| Divergence from Apex | >1.5% |
| Entropy classification drift | Any |
| Suppression misalignment | Any |
| Overfitting indicators | Detected |
| Score head numeric error | >3 points |

---

## Model Export

### Output Files

- `ApexCore_Full.tflite`
- `ApexCore_Mini.tflite`
- `MODEL_MANIFEST.json`

### Model Manifest Contents

```json
{
  "model_hash": "sha256:...",
  "quant_score_alignment": 0.987,
  "protocol_version_map": {
    "T01": "1.0.0",
    "T02": "1.0.0",
    "...": "..."
  },
  "dataset_hash": "sha256:...",
  "training_metrics": {
    "regime_accuracy": 0.94,
    "risk_accuracy": 0.92,
    "score_mae": 2.1
  },
  "rejection_tests_passed": true
}
```

---

## Scheduler Integration

ApexLab runs on a nightly schedule:

| Time | Task |
|------|------|
| 00:00 | Data ingest finalization |
| 01:00 | Dataset preparation |
| 02:00 | Feature normalization |
| 03:00 | Teacher label generation |
| 04:00 | Model training (Full) |
| 05:00 | Model training (Mini) |
| 06:00 | Validation + rejection tests |
| 06:30 | Model export (if passed) |

---

## Directory Structure

```
apexlab/
├── data/
│   ├── raw/
│   ├── normalized/
│   └── labeled/
├── models/
│   ├── training/
│   ├── validation/
│   └── export/
├── logs/
│   └── training_log.json
└── golden_set/
    └── exemplars.json
```

---

## Related Documentation

- [ApexCore Models](APEXCORE_MODELS.md)
- [Core Engine](CORE_ENGINE.md)
- [Protocols: Learning (LP01–LP25)](PROTOCOLS_LEARNING.md)
- [Data Layer](DATA_LAYER.md)
