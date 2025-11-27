# ApexCore Models — Model Family Documentation

**Version:** 8.0  
**Components:** ApexCore Full, ApexCore Mini  
**Role:** Neural Approximation of the Apex Deterministic Engine

---

## 1. Overview

ApexCore is a **model family** within the QuantraCore Apex ecosystem. These neural network models are trained to approximate the behavior of the deterministic Apex core engine, enabling efficient inference in scenarios where running the full engine is impractical.

The family consists of two members:
- **ApexCore Full** — Desktop-class model for high-accuracy analysis
- **ApexCore Mini** — Mobile-optimized model for on-device inference

Both models share the same output contract and are derived through a teacher-student training pipeline where Apex is the ultimate authority.

---

## 2. ApexCore Full (Desktop)

### 2.1 Role

ApexCore Full serves as the primary neural model for desktop deployments. It provides:
- Near-Apex accuracy for structural analysis
- Faster inference than running the full deterministic pipeline
- Suitable for scanning large universes in real-time

### 2.2 Specifications

| Property | Value |
|----------|-------|
| Size | 4–20MB |
| Format | TFLite |
| Inference | 50–200ms per sample |
| Training | Direct from ApexLab |
| Accuracy Target | >95% alignment with Apex |

### 2.3 Outputs

ApexCore Full produces the complete Apex output contract:

```yaml
outputs:
  - regime           # trending, ranging, volatile, suppressed
  - risk_tier        # low, medium, high, extreme
  - chart_quality    # structural clarity score
  - entropy_band     # disorder level classification
  - volatility_band  # volatility regime
  - suppression_state # activity suppression detection
  - score_band       # discretized QuantraScore range
  - quantrascore_numeric # 0.0–1.0 composite score
```

### 2.4 Guarantees

- **Apex Alignment** — Outputs are validated against Apex ground truth
- **Fail-Closed** — Uncertain predictions trigger abstention
- **Teacher Priority** — When Apex is available, it takes precedence

---

## 3. ApexCore Mini (Mobile)

### 3.1 Role

ApexCore Mini enables structural analysis on mobile devices and resource-constrained environments. It powers:
- QuantraVision Apex mobile overlays
- Edge device deployments
- Real-time scanning on lightweight hardware

### 3.2 Specifications

| Property | Value |
|----------|-------|
| Size | 0.5–3MB |
| Format | TFLite |
| Inference | <30ms on mobile |
| Training | Distilled from ApexCore Full |
| Accuracy Target | >90% alignment with Apex |

### 3.3 Outputs

ApexCore Mini produces the same output contract as Full:

```yaml
outputs:
  - regime
  - risk_tier
  - chart_quality
  - entropy_band
  - volatility_band
  - suppression_state
  - score_band
  - quantrascore_numeric
```

### 3.4 Restrictions

- **Cannot Override Apex** — When both are available, Apex is authoritative
- **Offline Only** — No cloud inference path
- **Fail-Closed on Uncertainty** — Conservative behavior under ambiguity

---

## 4. Shared Output Contract

Both ApexCore Full and Mini adhere to a strict output contract that mirrors the Apex engine:

### 4.1 Output Dimensions

| Output | Type | Description |
|--------|------|-------------|
| `regime` | enum | Market regime classification |
| `risk_tier` | enum | Risk level assessment |
| `chart_quality` | float | Structural clarity (0.0–1.0) |
| `entropy_band` | enum | Disorder classification |
| `volatility_band` | enum | Volatility regime |
| `suppression_state` | bool | Low-activity zone flag |
| `score_band` | enum | Discretized score range |
| `quantrascore_numeric` | float | Composite score (0.0–1.0) |

### 4.2 Contract Guarantees

1. **Type Stability** — Output types never change between versions
2. **Range Validity** — All values fall within documented ranges
3. **Determinism** — Same input always produces same output
4. **Graceful Degradation** — Uncertainty is explicit, not hidden

---

## 5. Distillation Relationship

The ApexCore models form a knowledge distillation chain:

```
Apex (Deterministic Core)
       ↓ Teacher Labels
ApexCore Full (Desktop Model)
       ↓ Knowledge Distillation
ApexCore Mini (Mobile Model)
```

### 5.1 Apex → Full

- ApexLab runs historical data through the Apex engine
- Apex outputs become training labels for Full
- Full learns to approximate Apex behavior
- Validation ensures alignment within tolerance

### 5.2 Full → Mini

- Full's predictions become "soft labels" for Mini
- Mini learns to approximate Full's behavior
- Distillation compresses knowledge into smaller architecture
- Validation ensures Mini meets accuracy thresholds

### 5.3 Distillation Constraints

- **No Shortcutting** — Mini is never trained directly from Apex
- **Cascade Validation** — Each step validates against its teacher
- **Version Coupling** — Mini version tied to Full version

---

## 6. Versioning and MODEL_MANIFEST

### 6.1 Version Scheme

Models follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** — Breaking changes to output contract
- **MINOR** — Accuracy improvements, architecture changes
- **PATCH** — Bug fixes, retraining with same data

### 6.2 MODEL_MANIFEST.json

Every exported model includes a manifest file:

```json
{
  "model_name": "apexcore_full",
  "version": "8.0.0",
  "training_timestamp": "2025-10-15T14:30:00Z",
  "apex_version": "8.0",
  "training_data_hash": "sha256:abc123...",
  "validation_metrics": {
    "regime_accuracy": 0.967,
    "risk_tier_accuracy": 0.954,
    "quantrascore_mae": 0.023
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100
  },
  "output_contract_version": "1.0"
}
```

### 6.3 Manifest Requirements

- **Immutable** — Once exported, manifest is never modified
- **Signed** — Cryptographic hash of model weights included
- **Traceable** — Links to training run proof logs

---

## 7. Deployment Scenarios

### 7.1 Desktop Only (Full)

For desktop applications with sufficient resources:
- Use ApexCore Full for primary analysis
- Fall back to Apex engine for validation
- Best accuracy, higher resource usage

### 7.2 Mobile Only (Mini)

For mobile/edge deployments:
- Use ApexCore Mini for all analysis
- Accept slightly lower accuracy for speed
- Fail-closed behavior on uncertainty

### 7.3 Hybrid (Full + Mini)

For connected mobile applications:
- Mini provides immediate on-device results
- Full (via QuantraVision Remote) provides validation
- Discrepancies flagged for review

### 7.4 Full Stack (Apex + Full + Mini)

For institutional deployments:
- Apex is authoritative for all decisions
- Full handles high-throughput scanning
- Mini enables mobile analyst tools
- Complete audit trail across all levels

---

## 8. Summary

The ApexCore model family provides efficient neural approximations of the deterministic Apex engine. Through careful training and distillation, Full and Mini models deliver structural analysis at varying resource profiles while maintaining alignment with the authoritative Apex core. The shared output contract, strict versioning, and comprehensive manifests ensure that these models remain trustworthy, auditable, and compliant with institutional requirements.
