# QuantraCore Apex™ — System Overview

**Version:** 8.0  
**Owner:** Lamont Labs — Jesse J. Lamont  
**Architecture:** Deterministic → Neural → Hybrid Intelligence

---

## 1. What is QuantraCore Apex?

QuantraCore Apex™ is a next-generation hybrid AI trading intelligence system designed for institutional-grade analysis, research, and decision support. Unlike traditional black-box AI trading systems, Apex combines a fully deterministic rule-based core engine with trainable neural network models in a unified architecture that prioritizes:

- **Transparency** — Every output can be traced back through the pipeline
- **Reproducibility** — Identical inputs always produce identical outputs
- **Compliance** — Built-in proof logging satisfies regulatory audit requirements
- **Safety** — Fail-closed design prevents dangerous outputs under uncertainty

The system is architected for offline-first operation, meaning all core logic runs without cloud dependencies. This makes Apex suitable for high-security environments where data sovereignty and air-gapped operation are requirements.

---

## 2. The Deterministic vs Neural Hybrid Design

QuantraCore Apex employs a unique "teacher-student" architecture where the deterministic core engine (Apex) acts as the authoritative source of truth, while neural models (ApexCore Full and ApexCore Mini) learn to approximate its behavior for deployment scenarios where the full engine cannot run.

### 2.1 The Deterministic Core (Apex)

The Apex core engine processes market data through a rigorous pipeline:

```
Market Data → Normalization → Validation → Trait Extraction →
Microtrait Analysis → Entropy/Suppression/Drift Detection →
Tier Protocols (T01–T80) → Learning Protocols (LP01–LP25) →
Omega Safety Directives (Ω1–Ω4) → QuantraScore → Proof Log
```

Every step in this pipeline is:
- **Deterministic** — No randomness or external API calls
- **Logged** — Hashed and recorded for later verification
- **Reproducible** — Can be replayed to confirm outputs

The core engine produces structured outputs including:
- Regime classification (trending, ranging, volatile, suppressed)
- Risk tier assignment
- Entropy and volatility bands
- Suppression state detection
- Continuation signals
- QuantraScore (composite structural quality metric)

### 2.2 The Neural Models (ApexCore)

While the deterministic engine is authoritative, it may be too resource-intensive for certain deployment scenarios (e.g., mobile devices, real-time scanning of large universes). The ApexCore model family addresses this:

- **ApexCore Full** (4–20MB) — Desktop-class model trained directly by ApexLab
- **ApexCore Mini** (0.5–3MB) — Mobile-optimized model distilled from Full

Both models produce the same output contract as the core engine, but with:
- Faster inference times
- Lower resource requirements
- Slight accuracy trade-offs (validated against tolerance thresholds)

**Critical Constraint:** Neural models can never override the deterministic core. When both are available, Apex is authoritative. Models are students, not masters.

---

## 3. How the Components Cooperate

### 3.1 ApexLab — The Training Laboratory

ApexLab is the offline training facility responsible for:

1. **Data Ingestion** — Pulling historical market data via adapters
2. **Window Generation** — Building OHLCV windows for training
3. **Feature Extraction** — Computing Apex features (entropy, suppression, drift, etc.)
4. **Teacher Labeling** — Running data through Apex to generate ground-truth labels
5. **Model Training** — Training ApexCore Full on labeled data
6. **Distillation** — Compressing Full into Mini via knowledge distillation
7. **Validation** — Ensuring models meet alignment thresholds
8. **Export** — Producing TFLite models and manifests for deployment

ApexLab runs on dedicated hardware (K6) in a completely offline environment. No training data or model weights ever leave the secure perimeter.

### 3.2 Signal and Prediction Systems

The **Signal System** handles:
- Universe scanning (which instruments to analyze)
- Watchlist routing (prioritizing candidates)
- Candidate building (assembling analysis packages)

The **Prediction System** provides:
- Regime-aware price movement predictions
- Volatility forecasting
- Expected move estimation

These systems consume Apex outputs and add forward-looking estimates while maintaining the same deterministic, logged, fail-closed principles.

### 3.3 MonsterRunner — Rare Event Detection

MonsterRunner is a specialized engine for identifying rare, high-impact market conditions ("monster moves"). It uses:
- High-resolution volatility analysis
- Apex trait signatures
- Sector momentum
- Options skew
- Short interest
- Macro regime indicators

Outputs include a MonsterScore, expected move percentile, and feature importance rankings.

---

## 4. QuantraVision — Visual Intelligence

### 4.1 QuantraVision Apex v4.2 (Mobile)

QuantraVision Apex is a mobile overlay copilot that provides real-time chart analysis. The pipeline:

```
Screen Capture → Bounding Box Detection → VisionLite (chart parsing) →
CandleLite (candle extraction) → Visual Primitives →
ApexLite (structural scoring) → QuantraScore Light → HUD Overlay
```

When available, QuantraVision uses ApexCore Mini for enhanced structural analysis. The system operates under strict safety constraints:
- **Narration only** — Text overlays, no trade execution
- **Forbidden phrases** — Certain language patterns are blocked
- **No trading features** — Read-only analysis

### 4.2 QuantraVision Remote

QuantraVision Remote enables a desktop Apex engine to stream structural overlays to a mobile device in real-time. This provides:
- Full Apex engine accuracy on mobile
- Read-only operation (no execution path)
- Zero trade execution capability

---

## 5. Design Principles

### 5.1 Deterministic by Default

All core logic produces identical outputs for identical inputs. Random number generators are seeded deterministically. External API responses are cached and replayed.

### 5.2 Offline-First

The system can operate indefinitely without internet connectivity. Cloud services are optional enhancements, not dependencies.

### 5.3 Fail-Closed Safety

When uncertainty exceeds thresholds, the system restricts outputs rather than guessing. The Omega directives enforce:
- **Ω1 — Integrity Lock** — Blocks operation if system integrity is compromised
- **Ω2 — Risk Kill Switch** — Halts activity if risk limits are breached
- **Ω3 — Config Guard** — Prevents unauthorized configuration changes
- **Ω4 — Compliance Gate** — Enforces regulatory constraints

### 5.4 Acquisition-Ready Design

The architecture is modular, well-documented, and suitable for institutional due diligence:
- Clean separation of concerns
- Comprehensive proof logging
- Version-controlled model manifests
- SBOM and provenance tracking

### 5.5 Zero Cloud Dependency for Logic

Business logic never requires cloud connectivity. Only optional features (e.g., QuantraVision cloud narration) use external services, and these are clearly separated.

### 5.6 Apex Teaches; Models Never Override

Neural models learn from the deterministic core. They can approximate but never contradict. When discrepancies arise, Apex is authoritative.

---

## 6. End-to-End Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTRACORE APEX v8.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Market Data → Normalization → Apex Core Engine → Tier Protocols (T01–T80) │
│       ↓                                                                     │
│  Learning Protocols (LP01–LP25) → Omega Safety (Ω1–Ω4) → QuantraScore      │
│       ↓                                                                     │
│  Prediction Engines → Proof Logs → ApexLab Distillation                    │
│       ↓                                                                     │
│  ApexCore Full/Mini → Dashboard + QuantraVision                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Conclusion

QuantraCore Apex™ v8.0 represents a mature, institution-ready hybrid AI trading intelligence system. By combining deterministic rule engines with distilled neural models, Apex achieves the transparency of classical systems with the efficiency of modern machine learning—all while maintaining rigorous safety, compliance, and reproducibility standards.

For detailed specifications, see the [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml).
