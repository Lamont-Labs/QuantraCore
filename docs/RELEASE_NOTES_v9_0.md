# Release Notes — QuantraCore Apex v9.0-A

**Release Date:** November 2025  
**Version:** 9.0-A (Institutional Hardening)  
**Owner:** Lamont Labs

---

## Overview

QuantraCore Apex v9.0-A represents a major upgrade with the Predictive Layer V2 implementation, Regulatory Excellence Framework, and comprehensive institutional hardening. This release adds 142 new tests for a total of 970+ tests, introduces ApexLab V2 and ApexCore V2 models, and implements fail-closed integration via PredictiveAdvisor.

---

## Major Features

### 1. Predictive Layer V2

Complete implementation of the enhanced predictive layer:

- **ApexLab V2** — 40+ field labeling schema with structural inputs, future outcomes, quality labels, runner flags, and safety labels
- **ApexCore V2 Big/Mini** — Multi-head models with 5 output heads (quantra_score, runner_prob, quality_tier, avoid_trade, regime)
- **Model Manifest System** — Version tracking, SHA-256 hash verification, metrics storage, promotion thresholds
- **PredictiveAdvisor** — Fail-closed engine integration with DISABLED/NEUTRAL/UPRANK/DOWNRANK/AVOID recommendations

### 2. Regulatory Excellence Framework

System now EXCEEDS regulatory requirements by 2x-5x margins:

| Regulation | Standard | QuantraCore |
|------------|----------|-------------|
| FINRA 15-09 | 50 iterations | 100 iterations |
| MiFID II RTS 6 | 2x volume | 4x volume |
| MiFID II RTS 6 | 5s latency | 2.5s latency |
| SEC 15c3-5 | Basic detection | 2x sensitivity |
| Basel Committee | Standard scenarios | 10 historical crises |

### 3. Omega Directive Ω5

Added new Omega directive for signal suppression lock:

| Directive | Trigger | Effect |
|-----------|---------|--------|
| Ω5 | Strong suppression | Signal suppression lock |

---

## Test Suite Expansion

**Total: 970+ tests | 100% pass rate**

### New Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| Regulatory | 163 | SEC/FINRA/MiFID II/Basel compliance |
| Predictive Layer | 142 | ApexLab V2, ApexCore V2, manifest, integration |

### Predictive Layer Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_apexlab_v2_schema.py | 38 | Schema validation, field types, bounds |
| test_apexlab_v2_dataset_shapes.py | 17 | Output shapes, protocol vectors |
| test_apexlab_v2_eval_metrics.py | 17 | Calibration, ranking, regime metrics |
| test_apexcore_v2_heads.py | 21 | Model heads, outputs, ensembles |
| test_apexcore_v2_determinism.py | 10 | Determinism, reproducibility |
| test_apexcore_v2_manifest.py | 23 | Manifest system, hash verification |
| test_predictive_integration.py | 16 | PredictiveAdvisor integration |

---

## New Documentation

| Document | Description |
|----------|-------------|
| [ApexLab V2](APEXLAB_V2.md) | Enhanced labeling system |
| [ApexCore V2](APEXCORE_V2.md) | Multi-head neural models |
| [PredictiveAdvisor](PREDICTIVE_ADVISOR.md) | Fail-closed integration |
| [Release Notes v9.0](RELEASE_NOTES_v9_0.md) | This document |

---

## Technical Improvements

### Model Architecture

- scikit-learn RandomForest ensemble (chosen for disk space efficiency)
- 5 specialized output heads with calibrated classifiers
- Deterministic outputs via controlled random_state
- Walk-forward time-aware training splits

### Manifest System

- SHA-256 model hash verification
- Version tracking with timestamps
- Validation metrics storage
- Promotion threshold enforcement

### Fail-Closed Behavior

- Model disabled on hash mismatch
- NEUTRAL recommendation on high disagreement (>0.5)
- AVOID recommendation on high avoid-trade probability (>0.8)
- Engine rules always take precedence

---

## Breaking Changes

None. This release is fully backward compatible with v8.x.

---

## Migration Guide

No migration required. All existing functionality remains unchanged.

To use the new Predictive Layer V2:

```python
from src.quantracore_apex.apexlab.apexlab_v2 import ApexLabV2Builder
from src.quantracore_apex.apexcore.apexcore_v2 import ApexCoreV2Model
from src.quantracore_apex.core.integration_predictive import PredictiveAdvisor

# Build V2 labels
builder = ApexLabV2Builder()
row = builder.build_row(window, future_prices)

# Train V2 model
model = ApexCoreV2Model()
model.fit(X, targets)

# Use PredictiveAdvisor
advisor = PredictiveAdvisor(model_path, manifest_path)
recommendation = advisor.get_recommendation(features, engine_score)
```

---

## Known Issues

None.

---

## Future Roadmap

- Enhanced calibration monitoring dashboards
- Periodic manifest rotation testing
- Compliance logging telemetry integration
- Real-world calibration summary reports

---

## Contributors

- **Jesse J. Lamont** — Founder, Lamont Labs

---

## Related Documentation

- [Master Spec v9.0-A](QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md)
- [Roadmap](ROADMAP.md)
- [Developer Guide](DEVELOPER_GUIDE.md)

---

**QuantraCore Apex v9.0-A** — Deterministic. Reproducible. Research-Ready.

*Lamont Labs | November 2025*
