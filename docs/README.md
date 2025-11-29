# QuantraCore Apex v9.0-A — Documentation

Welcome to the QuantraCore Apex documentation. This folder contains comprehensive documentation for the institutional-grade deterministic AI trading intelligence system.

---

## Quick Navigation

### Getting Started

- [Quickstart](QUICKSTART.md) — Installation and first steps
- [Getting Started Desktop](GETTING_STARTED_DESKTOP.md) — Desktop setup guide
- [Developer Guide](DEVELOPER_GUIDE.md) — Engineer onboarding

### System Overview

- [Overview](OVERVIEW_QUANTRACORE_APEX.md) — Full narrative of the Apex ecosystem
- [Master Spec v9.0-A](QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md) — Current system specification
- [Master Spec v8.2](QUANTRACORE_APEX_MASTER_SPEC_v8.2.md) — Previous stable specification
- [Roadmap](ROADMAP.md) — Development roadmap

### Components

- [Core Engine](CORE_ENGINE.md) — Deterministic signal engine
- [Architecture](ARCHITECTURE.md) — System architecture
- [ApexLab Overview](APEXLAB_OVERVIEW.md) — Offline training and distillation lab
- [ApexLab Training](APEXLAB_TRAINING.md) — Training environment guide
- [ApexCore Models](APEXCORE_MODELS.md) — Model family documentation (V1 Full/Mini + V2 Big/Mini)
- [Prediction Stack](PREDICTION_STACK.md) — Forward-looking analysis
- [MonsterRunner](MONSTERRUNNER.md) — Extreme move detection

### Predictive Layer V2 (New)

- [ApexLab V2](APEXLAB_V2.md) — Enhanced labeling with 40+ fields
- [ApexCore V2](APEXCORE_V2.md) — Multi-head models (Big/Mini)
- [PredictiveAdvisor](PREDICTIVE_ADVISOR.md) — Fail-closed engine integration

### Protocols

- [Tier Protocols](PROTOCOLS_TIER.md) — T01-T80 analysis protocols
- [Learning Protocols](PROTOCOLS_LEARNING.md) — LP01-LP25 label generation
- [Omega Directives](OMEGA_DIRECTIVES.md) — Ω1-Ω20 safety overrides
- [Determinism Tests](DETERMINISM_TESTS.md) — Reproducibility verification

### Integration

- [API Reference](API_REFERENCE.md) — REST API documentation
- [API Integration](API_INTEGRATION_FOR_LEARNING_AND_PREDICTION.md) — External data adapters
- [Data Layer](DATA_LAYER.md) — Data provider adapters
- [Broker/OMS](BROKER_OMS.md) — Order management (simulation only)

### Risk & Portfolio

- [Risk Engine](RISK_ENGINE.md) — Risk assessment system
- [Risk Model](RISK_MODEL.md) — Risk model documentation
- [Portfolio System](PORTFOLIO_SYSTEM.md) — Position tracking

### Compliance & Security

- [Compliance & Safety](COMPLIANCE_AND_SAFETY.md) — Research-only constraints
- [Compliance Policy](COMPLIANCE_POLICY.md) — Institutional compliance guide
- [Security](SECURITY.md) — Security summary
- [Security & Hardening](SECURITY_AND_HARDENING.md) — Security documentation

### Audit & Provenance

- [Audit Bundle Guide](AUDIT_BUNDLE_GUIDE.md) — Audit bundle creation
- [SBOM & Provenance](SBOM_PROVENANCE.md) — Software bill of materials
- [Provenance](PROVENANCE.md) — Provenance documentation

### Additional Resources

- [Glossary](GLOSSARY.md) — Terminology definitions
- [Limitations](LIMITATIONS.md) — Known limitations
- [Demo Script](DEMO_SCRIPT.md) — Demo walkthrough

---

## Current System Status

| Component | Status |
|-----------|--------|
| ApexEngine | Operational |
| ApexDesk UI | Operational (React 18 + Vite 5) |
| FastAPI Backend | Operational (27 endpoints) |
| Test Suite | **970+ tests passing** |
| Universal Scanner | Operational |
| ApexLab | Operational (V1 + V2) |
| ApexCore Models | Operational (V1 + V2) |
| **ApexLab V2** | Operational (40+ field schema) |
| **ApexCore V2** | Operational (Big/Mini, 5 heads) |
| **PredictiveAdvisor** | Operational (fail-closed integration) |

---

## Documentation Standards

All documentation follows these principles:
- Clear, professional tone suitable for institutional readers
- Consistent terminology matching the Master Spec v9.0-A
- Self-contained documents with cross-references where appropriate
- Accurate test counts and system status

---

**QuantraCore Apex v9.0-A** — Deterministic. Reproducible. Research-Ready.

*Lamont Labs | November 2025*
