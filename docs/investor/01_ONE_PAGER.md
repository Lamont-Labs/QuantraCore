# QuantraCore Apex — One-Pager

**Deterministic AI Trading Intelligence for Institutional Research**

---

## Tagline

> **The deterministic teacher that trains the neural student — audit-ready, fail-closed, offline-first.**

---

## Value Propositions

- **Deterministic Signals:** 145 protocols generate reproducible structural analysis with hash-verified outputs
- **Offline Lab:** ApexLab v2 creates labeled datasets from historical data without cloud dependencies
- **On-Device Models:** ApexCore v2 (Big/Mini) provides ranking assistance that never overrides engine authority
- **Mobile Copilot:** QuantraVision Apex brings structural overlays to retail investors on Android

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        QUANTRACORE APEX v9.0-A                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Market Data │───▶│   Scanner    │───▶│ Deterministic│          │
│  │  (Polygon)   │    │  (8 modes)   │    │   Engine     │          │
│  └──────────────┘    └──────────────┘    │  (115 proto) │          │
│                                           └──────┬───────┘          │
│                                                  │                   │
│                      ┌───────────────────────────┼───────────────┐  │
│                      ▼                           ▼               │  │
│               ┌──────────────┐           ┌──────────────┐       │  │
│               │  ApexLab v2  │           │  Predictive  │       │  │
│               │  (Labeling)  │           │   Advisor    │       │  │
│               └──────┬───────┘           │  (Ranker)    │       │  │
│                      │                   └──────┬───────┘       │  │
│                      ▼                          │               │  │
│               ┌──────────────┐                  │               │  │
│               │ ApexCore v2  │──────────────────┘               │  │
│               │  (Big/Mini)  │                                  │  │
│               └──────────────┘                                  │  │
│                                                                  │  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │  │
│  │  ApexDesk UI │    │QuantraVision │    │ Broker Layer │       │  │
│  │   (React)    │    │   (Android)  │    │  (Optional)  │       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘       │  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

*Full diagram: [docs/assets/investor/01_quantracore_apex_ecosystem.png](../assets/investor/01_quantracore_apex_ecosystem.png)*

---

## Key Differentiators

| vs. Generic Quant Infra | QuantraCore Apex |
|------------------------|------------------|
| Black-box ML signals | Deterministic teacher + neural ranker |
| Cloud-dependent | Offline-first, on-premise capable |
| No audit trail | Full provenance chain with proof logs |
| Regulatory minimum | 3-5x stricter than requirements |
| Execution-focused | Research-only by default |

| vs. Retail Pattern Tools | QuantraCore Apex |
|--------------------------|------------------|
| Subjective pattern matching | Quantified structural protocols |
| No ML component | Neural ranking layer (fail-closed) |
| No institutional features | Compliance, risk controls, audit logs |

| vs. Black-Box ML Funds | QuantraCore Apex |
|-----------------------|------------------|
| Opaque predictions | Explainable protocol traces |
| Overfit to backtest | Walk-forward with leakage guards |
| "Oracle" claims | Honest "ranker/assistant" positioning |

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Deterministic Engine | Implemented | 145 protocols operational |
| ApexLab v2 | Implemented | Labeling and dataset generation |
| ApexCore v2 | Implemented | Big/Mini models with manifests |
| PredictiveAdvisor | Implemented | Fail-closed ranker |
| ApexDesk UI | Implemented | React/Vite dashboard |
| Scanner | Implemented | 8 scan modes |
| Broker Layer | Spec-only | Disabled by default |
| QuantraVision | Separate codebase | v4.x operational |

---

## Where Capital Goes

| Category | Allocation | Purpose |
|----------|------------|---------|
| **Engineering** | 50-60% | Senior engineers to harden and extend |
| **Infrastructure** | 15-20% | Data feeds, compute, testing |
| **Compliance** | 10-15% | Legal counsel, regulatory preparation |
| **Founder Runway** | 10-15% | Continued architecture and direction |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Automated Tests | 970 (100% pass) |
| Compliance Score | 99.25% |
| API Endpoints | 36 |
| Protocols | 115 |
| Model Variants | Big (AUC 0.782) + Mini (AUC 0.754) |

---

## Contact

**Lamont Labs**  
QuantraCore Apex Development Team

*Read more: [Executive Summary](00_EXEC_SUMMARY.md) | [Platform Overview](03_PLATFORM_OVERVIEW.md)*

---

*QuantraCore Apex v9.0-A | November 2025*
