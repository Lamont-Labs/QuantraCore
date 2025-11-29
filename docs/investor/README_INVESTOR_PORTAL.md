# QuantraCore Apex Investor Portal

**Document Classification:** Investor Due Diligence  
**Version:** 9.0-A  
**Date:** November 2025  
**Status:** Production  

---

## Welcome

This `/docs/investor` folder is your portal into QuantraCore Apex. Every document is designed to answer the questions institutional investors, technical diligence teams, and business partners typically ask.

---

## Quick Navigation by Time Available

### If You Have 10 Minutes

Start here for the essential overview:

| Document | Purpose |
|----------|---------|
| [00_EXEC_SUMMARY.md](00_EXEC_SUMMARY.md) | What this is, why it matters, proof so far |
| [01_ONE_PAGER.md](01_ONE_PAGER.md) | Single-page value proposition |

### If You Have 30 Minutes

Add these for deeper understanding:

| Document | Purpose |
|----------|---------|
| [02_FOUNDER_PROFILE.md](02_FOUNDER_PROFILE.md) | Who built this and why |
| [03_PLATFORM_OVERVIEW.md](03_PLATFORM_OVERVIEW.md) | All components and their status |
| [05_DEAL_SUMMARY.md](05_DEAL_SUMMARY.md) | What's on the table |
| [60_INVESTOR_FAQ.md](60_INVESTOR_FAQ.md) | Direct Q&A on common questions |

### If You're Doing Deep Technical Diligence

Follow this path:

1. **Architecture:** [10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md](10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md)
2. **Engine:** [11_ENGINE_AND_PROTOCOLS.md](11_ENGINE_AND_PROTOCOLS.md)
3. **ML Pipeline:** [12_APEXLAB_AND_APEXCORE_MODELS.md](12_APEXLAB_AND_APEXCORE_MODELS.md)
4. **Safety:** [13_PREDICTIVE_LAYER_AND_SAFETY.md](13_PREDICTIVE_LAYER_AND_SAFETY.md)
5. **Testing:** [51_TESTING_AND_COVERAGE_SUMMARY.md](51_TESTING_AND_COVERAGE_SUMMARY.md)
6. **Provenance:** [32_SECURITY_AND_PROVENANCE.md](32_SECURITY_AND_PROVENANCE.md)

### If Legal/Compliance Is Reviewing

Focus on these files:

| Document | Purpose |
|----------|---------|
| [30_RISK_MANAGEMENT_AND_GUARDS.md](30_RISK_MANAGEMENT_AND_GUARDS.md) | Risk controls and kill-switches |
| [31_COMPLIANCE_AND_USAGE_POLICIES.md](31_COMPLIANCE_AND_USAGE_POLICIES.md) | Usage modes and regulatory considerations |
| [32_SECURITY_AND_PROVENANCE.md](32_SECURITY_AND_PROVENANCE.md) | SBOM, provenance, supply chain |
| [33_LIMITATIONS_AND_HONEST_RISKS.md](33_LIMITATIONS_AND_HONEST_RISKS.md) | Explicit limitations and risks |

---

## Document Index

### Top Layer (Strategy & Vision)

| # | File | Answers |
|---|------|---------|
| 00 | [EXEC_SUMMARY.md](00_EXEC_SUMMARY.md) | What is this? Why now? What's the path? |
| 01 | [ONE_PAGER.md](01_ONE_PAGER.md) | Single-page overview |
| 02 | [FOUNDER_PROFILE.md](02_FOUNDER_PROFILE.md) | Who is Jesse? Why trust him? |
| 03 | [PLATFORM_OVERVIEW.md](03_PLATFORM_OVERVIEW.md) | How do all components fit together? |
| 04 | [COMPETITIVE_POSITIONING.md](04_COMPETITIVE_POSITIONING.md) | How is this different? |
| 05 | [DEAL_SUMMARY.md](05_DEAL_SUMMARY.md) | What's the deal shape? |

### Technical Core

| # | File | Answers |
|---|------|---------|
| 10 | [SYSTEM_ARCHITECTURE_DEEP_DIVE.md](10_SYSTEM_ARCHITECTURE_DEEP_DIVE.md) | End-to-end stack walkthrough |
| 11 | [ENGINE_AND_PROTOCOLS.md](11_ENGINE_AND_PROTOCOLS.md) | Deterministic engine and protocol taxonomy |
| 12 | [APEXLAB_AND_APEXCORE_MODELS.md](12_APEXLAB_AND_APEXCORE_MODELS.md) | Labeling, training, model architecture |
| 13 | [PREDICTIVE_LAYER_AND_SAFETY.md](13_PREDICTIVE_LAYER_AND_SAFETY.md) | PredictiveAdvisor safety gating |
| 14 | [BROKER_AND_EXECUTION_ENVELOPE.md](14_BROKER_AND_EXECUTION_ENVELOPE.md) | Execution layer (disabled by default) |
| 15 | [APEXDESK_UI_AND_APIS.md](15_APEXDESK_UI_AND_APIS.md) | Dashboard and API endpoints |
| 16 | [QUANTRAVISION_APEX_OVERVIEW.md](16_QUANTRAVISION_APEX_OVERVIEW.md) | Android structural copilot |

### Data, Training & Metrics

| # | File | Answers |
|---|------|---------|
| 20 | [DATA_SOURCES_AND_UNIVERSE.md](20_DATA_SOURCES_AND_UNIVERSE.md) | Where does data come from? |
| 21 | [LABELING_METHODS_AND_LEAKAGE_GUARDS.md](21_LABELING_METHODS_AND_LEAKAGE_GUARDS.md) | How do you prevent overfitting? |
| 22 | [TRAINING_PROCESS_AND_HYPERPARAMS.md](22_TRAINING_PROCESS_AND_HYPERPARAMS.md) | How are models trained? |
| 23 | [EVALUATION_AND_LIMITATIONS.md](23_EVALUATION_AND_LIMITATIONS.md) | How good is it? Where does it fail? |
| 24 | [MONSTERRUNNER_EXPLAINED.md](24_MONSTERRUNNER_EXPLAINED.md) | Extreme move detection |

### Risk, Compliance & Security

| # | File | Answers |
|---|------|---------|
| 30 | [RISK_MANAGEMENT_AND_GUARDS.md](30_RISK_MANAGEMENT_AND_GUARDS.md) | Risk controls and guardrails |
| 31 | [COMPLIANCE_AND_USAGE_POLICIES.md](31_COMPLIANCE_AND_USAGE_POLICIES.md) | Regulatory considerations |
| 32 | [SECURITY_AND_PROVENANCE.md](32_SECURITY_AND_PROVENANCE.md) | Supply chain and provenance |
| 33 | [LIMITATIONS_AND_HONEST_RISKS.md](33_LIMITATIONS_AND_HONEST_RISKS.md) | Uncomfortable truths |

### Business & Commercial

| # | File | Answers |
|---|------|---------|
| 40 | [COMMERCIAL_MODELS_AND_PATHS.md](40_COMMERCIAL_MODELS_AND_PATHS.md) | How could this make money? |
| 41 | [TARGET_CUSTOMERS_AND_SEGMENTS.md](41_TARGET_CUSTOMERS_AND_SEGMENTS.md) | Who are the buyers? |
| 42 | [ROADMAP_AND_CAPITAL_USE.md](42_ROADMAP_AND_CAPITAL_USE.md) | What happens with investment? |

### Engineering & Operations

| # | File | Answers |
|---|------|---------|
| 50 | [ENGINEERING_OVERVIEW_AND_PRACTICES.md](50_ENGINEERING_OVERVIEW_AND_PRACTICES.md) | Code structure and standards |
| 51 | [TESTING_AND_COVERAGE_SUMMARY.md](51_TESTING_AND_COVERAGE_SUMMARY.md) | Test suite overview |
| 52 | [OPERATIONS_AND_RUNBOOKS.md](52_OPERATIONS_AND_RUNBOOKS.md) | Day-to-day operations |

### FAQ

| # | File | Answers |
|---|------|---------|
| 60 | [INVESTOR_FAQ.md](60_INVESTOR_FAQ.md) | Direct Q&A, no spin |

---

## Supporting Assets

### Architecture Diagrams

Located in `docs/assets/investor/`:

| Asset | Description |
|-------|-------------|
| 01_quantracore_apex_ecosystem.png | Full system ecosystem |
| 02_apex_pipeline_deterministic_to_ml.png | Deterministic to ML pipeline |
| 03_apexlab_v2_labeling_flow.png | ApexLab labeling process |
| 04_apexcore_v2_model_family.png | Model family relationships |
| 05_predictiveadvisor_safety_gating.png | Safety gating architecture |
| 06_broker_safety_envelope.png | Broker layer safety |
| 07_determinism_and_provenance_chain.png | Provenance guarantees |
| 08_apexcore_v2_training_loss.png | Training loss curves |
| 09_apexlab_v2_eval_summary.png | Evaluation metrics |

### Root Repository Assets

| File | Purpose |
|------|---------|
| SBOM.json | Software Bill of Materials |
| PROVENANCE.manifest | SHA256 hashes for critical files |
| CHANGELOG.md | Chronological changes |
| LICENSE | Licensing terms |
| proof_logs/README.md | Proof log explanation |

---

## Verification

All claims in this documentation are backed by:

- **970 automated tests** with 100% pass rate
- **99.25% regulatory compliance score** (verified via API)
- **36 REST endpoints** (documented and operational)
- **145 protocols** (80 Tier + 25 Learning + 20 MonsterRunner + 20 Omega)
- **Production model manifests** with SHA256 verification

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
