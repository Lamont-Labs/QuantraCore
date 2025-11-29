# ApexCore V2 — Multi-Head Neural Models

**Version:** 9.0-A  
**Component:** ApexCore V2 (Big/Mini)  
**Role:** Enhanced neural models with 5 output heads

---

## 1. Overview

ApexCore V2 is the enhanced generation of neural models within the QuantraCore Apex ecosystem. These models feature a multi-head architecture with 5 specialized output heads, providing comprehensive structural analysis with fail-closed integration.

---

## 2. Model Variants

### 2.1 ApexCore V2 Big

| Property | Value |
|----------|-------|
| Architecture | scikit-learn RandomForest ensemble |
| Output Heads | 5 |
| Estimators | 100 per head |
| Max Depth | 12 |
| Use Case | Desktop analysis, maximum accuracy |

### 2.2 ApexCore V2 Mini

| Property | Value |
|----------|-------|
| Architecture | scikit-learn RandomForest ensemble |
| Output Heads | 5 |
| Estimators | 50 per head |
| Max Depth | 8 |
| Use Case | Faster inference, lighter footprint |

---

## 3. Output Heads

ApexCore V2 produces 5 specialized outputs:

| Head | Type | Output | Description |
|------|------|--------|-------------|
| `quantra_score` | Regression | 0-100 | QuantraScore prediction |
| `runner_prob` | Classification | 0-1 | Probability of 15%+ move |
| `quality_tier` | Multi-class | A/B/C/D/A_PLUS | Quality grade |
| `avoid_trade` | Classification | 0-1 | Avoid-trade probability |
| `regime` | Multi-class | 5 classes | Market regime |

### 3.1 Regime Classes

| Class | Description |
|-------|-------------|
| `chop` | Range-bound, choppy market |
| `trend_up` | Upward trending |
| `trend_down` | Downward trending |
| `squeeze` | Volatility compression |
| `crash` | High volatility decline |

---

## 4. Core Principles

ApexCore V2 adheres to strict principles:

1. **Apex is always the teacher** — Models learn from deterministic core
2. **ApexCore never overrides Apex** — Engine rules take precedence
3. **Fails closed if uncertain** — Conservative behavior under ambiguity
4. **Manifest verification required** — Hash mismatch disables model

---

## 5. Model Configuration

```python
from src.quantracore_apex.apexcore.apexcore_v2 import (
    ApexCoreV2Config,
    ApexCoreV2Model,
    ModelVariant
)

# Configure V2 Big model
config = ApexCoreV2Config(
    variant=ModelVariant.BIG,
    random_state=42,
    n_estimators_big=100,
    n_estimators_mini=50,
    max_depth_big=12,
    max_depth_mini=8
)

# Create model
model = ApexCoreV2Model(config=config)
```

---

## 6. Training

```python
from src.quantracore_apex.apexlab.training import ApexCoreV2Trainer

trainer = ApexCoreV2Trainer(
    model_dir="./models",
    enable_logging=True
)

# Train with walk-forward splits
manifest = trainer.train(
    X=features,
    targets=target_dict,
    val_size=0.2,
    test_size=0.1
)
```

---

## 7. Inference

```python
# Load model
model = ApexCoreV2Model.load("./models/apexcore_v2_big.pkl")

# Run inference
outputs = model.forward(features)

# Access outputs
quantra_score = outputs["quantra_score"]    # Array of scores
runner_prob = outputs["runner_prob"]        # Array of probabilities
quality_tier = outputs["quality_tier"]      # Array of tier labels
avoid_trade = outputs["avoid_trade"]        # Array of probabilities
regime = outputs["regime"]                  # Array of regime labels
```

---

## 8. Manifest System

Every trained model includes a manifest:

```json
{
  "version": "2.0",
  "model_variant": "big",
  "trained_at": "2025-11-29T00:00:00Z",
  "apex_version": "9.0-A",
  "model_hash": "sha256:abc123def456...",
  "training_samples": 10000,
  "metrics": {
    "val_auc_runner": 0.85,
    "val_mae_quantra": 5.2,
    "val_acc_quality": 0.78,
    "val_auc_avoid": 0.82,
    "val_acc_regime": 0.75
  },
  "thresholds": {
    "min_auc_runner_to_promote": 0.70,
    "min_acc_quality_to_promote": 0.60,
    "max_mae_quantra_to_promote": 10.0
  }
}
```

### 8.1 Hash Verification

```python
from src.quantracore_apex.apexcore.manifest import (
    ApexCoreV2Manifest,
    verify_model_hash
)

manifest = ApexCoreV2Manifest.load("./models/manifest.json")
is_valid = verify_model_hash("./models/model.pkl", manifest.model_hash)
```

---

## 9. Determinism Guarantee

ApexCore V2 ensures deterministic outputs:

- Fixed `random_state` in all estimators
- Controlled initialization
- Reproducible training splits
- Verified via 100-iteration tests

---

## 10. Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_apexcore_v2_heads.py | 21 | Head output validation |
| test_apexcore_v2_determinism.py | 10 | Determinism verification |
| test_apexcore_v2_manifest.py | 23 | Manifest system |

---

## 11. Related Documentation

- [ApexCore Models](APEXCORE_MODELS.md)
- [ApexLab V2](APEXLAB_V2.md)
- [PredictiveAdvisor](PREDICTIVE_ADVISOR.md)
- [Training Pipeline](APEXLAB_TRAINING.md)

---

**QuantraCore Apex v9.0-A** — Lamont Labs | November 2025
