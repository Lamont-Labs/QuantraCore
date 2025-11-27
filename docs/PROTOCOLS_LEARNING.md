# QuantraCore Apex™ — Learning Protocols (LP01–LP25)

**Version:** 8.0  
**Component:** Protocol System  
**Status:** Active

---

## Overview

Learning protocols transform the deterministic outputs of Apex into high-quality training labels for ApexLab. They enforce bias correction, stabilizing rules, rare-pattern preservation, and long-term learning consistency. Models (ApexCore) must precisely match protocol-derived labels.

---

## Responsibilities

- Capture teacher labels
- Generate multi-head training targets
- Ensure balanced sample distribution
- Preserve rare structures for MonsterRunner
- Reject ambiguous/noisy examples
- Stabilize learning through decay correction

---

## Learning Protocol Reference

### Label Generation Protocols (LP01–LP08)

| Protocol | Function |
|----------|----------|
| LP01 | Label generation: regime |
| LP02 | Label generation: volatility band |
| LP03 | Label generation: risk tier |
| LP04 | Label generation: suppression state |
| LP05 | Label generation: entropy state |
| LP06 | Label generation: continuation result |
| LP07 | Label generation: QuantraScore bucket |
| LP08 | Label generation: sector bias |

---

### Sampling & Alignment Protocols (LP09–LP15)

| Protocol | Function |
|----------|----------|
| LP09 | Decay-corrected window sampling |
| LP10 | Noise-filtered label alignment |
| LP11 | Microstructure alignment gate |
| LP12 | Rare pattern bookmarking |
| LP13 | Continuation reinforcement sampler |
| LP14 | Estimation variance smoothing |
| LP15 | Stability feedback checks |

---

### Reinforcement & Balancing Protocols (LP16–LP20)

| Protocol | Function |
|----------|----------|
| LP16 | Tilt-adjusted class balancing |
| LP17 | Hyper-rare anomaly preservation |
| LP18 | Sector-phase reinforcement |
| LP19 | Entropy-boundary reinforcement |
| LP20 | Suppression-boundary reinforcement |

---

### Synchronization & Validation Protocols (LP21–LP25)

| Protocol | Function |
|----------|----------|
| LP21 | Multi-head label synchronizer |
| LP22 | Drift-range stability |
| LP23 | Momentum-tail reinforcement |
| LP24 | Model-rejection label auditing |
| LP25 | Golden-set exemplar preservation |

---

## Invariants

1. **No Randomness** — Learning protocols may not introduce randomness
2. **Apex Reference** — Every learning decision must reference Apex outputs
3. **Reproducibility** — All label sets must be reproducible
4. **Model Rejection** — Models that drift from Apex are rejected

---

## Integration with ApexLab

Learning protocols are executed within ApexLab's training pipeline:

```
Raw OHLCV Data
    ↓
Apex Deterministic Execution
    ↓
LP01–LP08: Label Generation
    ↓
LP09–LP15: Sampling & Alignment
    ↓
LP16–LP20: Reinforcement & Balancing
    ↓
LP21–LP25: Synchronization & Validation
    ↓
Training Dataset (Hash-Locked)
    ↓
ApexCore Model Training
```

---

## Label Types Generated

| Label | Source Protocol | Description |
|-------|-----------------|-------------|
| `regime_class` | LP01 | Current market regime |
| `volatility_band` | LP02 | Volatility classification |
| `risk_tier` | LP03 | Risk level assignment |
| `suppression_state` | LP04 | Structural suppression |
| `entropy_state` | LP05 | Signal clarity |
| `continuation_likelihood` | LP06 | Continuation probability |
| `score_bucket` | LP07 | QuantraScore band |
| `sector_bias` | LP08 | Sector pressure direction |

---

## Rare Pattern Handling

Learning protocols ensure rare patterns are preserved for MonsterRunner training:

- **LP12** — Bookmarks rare patterns for special handling
- **LP17** — Preserves hyper-rare anomalies
- **LP25** — Maintains golden-set exemplars

These preserved patterns enable ApexCore to recognize extreme-move precursors without over-weighting common patterns.

---

## Model Rejection Criteria

Learning protocols feed into ApexLab's model validation. A model is rejected if:

- Accuracy drop >2% on golden-set
- Divergence from Apex >1.5%
- Entropy classification drift
- Suppression misalignment
- Overfitting indicators
- Score head numeric error >3 points

---

## Related Documentation

- [Core Engine](CORE_ENGINE.md)
- [Protocols: Tier (T01–T80)](PROTOCOLS_TIER.md)
- [Omega Directives](OMEGA_DIRECTIVES.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
