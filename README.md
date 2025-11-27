# QuantraCore Apex™ — Hybrid AI Trading Intelligence Engine

**Owner:** Jesse J. Lamont  
**Org:** Lamont Labs  
**Version:** v8.0  
[![QuantraCore CI](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml)  
**Status:** Production-ready, acquisition-grade architecture  
**Repo:** https://github.com/Lamont-Labs/QuantraCore

---

## Overview

QuantraCore Apex™ is an institution-grade hybrid AI trading intelligence system that combines deterministic rule-based engines with neural network models in a unified, fail-closed architecture. The system is designed from the ground up for transparency, reproducibility, and regulatory compliance. Every output is hashed, logged, and replayable—ensuring that analysts, auditors, and acquirers can verify provenance at any step. The platform operates offline-first with zero cloud dependency for core logic, making it suitable for high-security institutional environments.

---

## Major Components

- **QuantraCore Apex** — The deterministic core engine that serves as the "teacher" for all downstream neural models
- **ApexLab** — Offline training and distillation lab for building and validating ApexCore models
- **ApexCore Full** — Desktop-class structural neural model (4–20MB) trained by ApexLab
- **ApexCore Mini** — Mobile-optimized neural model (0.5–3MB) distilled from ApexCore Full
- **MonsterRunner** — Rare-event detection engine for identifying high-impact market conditions
- **QuantraVision Apex v4.2** — Mobile overlay copilot for real-time chart analysis
- **QuantraVision Remote** — Desktop-to-mobile structural overlay streaming
- **Apex Dashboard** — React-based visualization console for signals, entropy, drift, and proof logs
- **Signal System** — Universe scanning, watchlist routing, and candidate building
- **Prediction System** — Regime-aware predictors for volatility and expected move estimation

---

## Why QuantraCore Apex Exists

Traditional AI trading systems suffer from opacity, non-reproducibility, and regulatory uncertainty. QuantraCore Apex solves these problems by:

1. **Deterministic Core as Teacher** — The Apex engine generates reproducible outputs that train neural models, ensuring alignment and auditability
2. **Offline-First, Zero Cloud Dependency** — Core logic never requires internet connectivity; all learning happens locally
3. **Fail-Closed Safety** — When uncertainty is high, the system restricts rather than guesses
4. **Proof Logging Everywhere** — Every pipeline step is hashed and logged for complete provenance
5. **Acquisition-Ready Design** — Clean modular architecture suitable for institutional due diligence

---

## Repository Structure

```
/src/                    — Core engine, protocols, prediction, signal, and vision modules
/cli/                    — Typer CLI for demo and testing
/tests/                  — Reproducibility and filter tests
/docs/                   — Architecture, specs, guides, and compliance documentation
/assets/                 — Branding and screenshots
/SBOM/                   — CycloneDX metadata, provenance JSON, checksums
/proof_logs/             — Determinism verification logs
/archives/               — Raw API cache and transformed data
```

---

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Lamont-Labs/QuantraCore.git
   cd QuantraCore
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the deterministic demo**
   ```bash
   python -m cli.main
   ```

4. **Verify checksums**
   ```bash
   bash verify.sh
   ```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Overview](docs/OVERVIEW_QUANTRACORE_APEX.md) | Full narrative overview of the Apex ecosystem |
| [Master Spec v8.0](docs/QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml) | Canonical YAML specification |
| [ApexLab Overview](docs/APEXLAB_OVERVIEW.md) | Offline training and distillation lab |
| [ApexCore Models](docs/APEXCORE_MODELS.md) | Model family documentation (Full/Mini) |
| [QuantraVision Apex v4.2](docs/QUANTRAVISION_APEX_v4_2.md) | Mobile overlay copilot |
| [QuantraVision Remote](docs/QUANTRAVISION_REMOTE.md) | Desktop-to-mobile streaming |
| [Prediction & MonsterRunner](docs/PREDICTION_AND_MONSTERRUNNER.md) | Prediction system and rare-event detection |
| [API Integration](docs/API_INTEGRATION_FOR_LEARNING_AND_PREDICTION.md) | Adapter layer and data contracts |
| [Compliance Policy](docs/COMPLIANCE_POLICY.md) | Institutional compliance and audit |
| [Security & Hardening](docs/SECURITY_AND_HARDENING.md) | Hash verification, encryption, config guards |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Onboarding for engineers |
| [Release Notes v8.0](docs/RELEASE_NOTES_v8_0.md) | What changed in this version |

---

## What This Repo Does NOT Include

- No live brokerage connections or API keys
- No real financial data or external model feeds
- No user accounts or personal information
- No claims of profit, advice, or market prediction
- Execution is **disabled by default** (research and simulation only)

All data is synthetic and demonstrative.

---

## Contact

**Jesse J. Lamont** — Founder, Lamont Labs  
Email: lamontlabs@proton.me  
GitHub: https://github.com/Lamont-Labs

---

## Disclaimers

This is a demonstration repository only—no trading advice or financial activity.  
All data is synthetic or public domain.  
No production trading systems are connected.

---

**Persistence = Proof.**  
Every build, every log, every checksum—reproducible by design.
