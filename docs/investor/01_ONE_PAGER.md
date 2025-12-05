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
| Strategy Orchestrator | Implemented | 4 strategies (Swing, Scalp, Momentum, MonsterRunner) |
| Universe Scanner | Implemented | 7,952+ symbols → 197 hot stocks |
| Scheduled Automation | Implemented | Every 30 min, 4 AM - 8 PM ET |
| ApexCore Models | Implemented | 42 trained models |
| ApexDesk UI | Implemented | 15 real-time panels |
| Paper Trading | Active | 8 positions, ~$107k equity |
| Stop-Loss Manager | Implemented | -15% hard, trailing at +10% |

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
| Python Codebase | 439 files, 111,884 lines |
| API Endpoints | 293 REST endpoints |
| ML Models | 42 trained models |
| Protocols | 145 (T01-T80, LP01-LP25, MR01-MR20, Ω01-Ω20) |
| Strategies | 4 concurrent (Swing, Scalp, Momentum, MonsterRunner) |
| Universe | 7,952+ symbols → 197 hot stocks |
| Test Coverage | 67 test modules |

---

## Contact

**Lamont Labs**  
QuantraCore Apex Development Team

*Read more: [Executive Summary](00_EXEC_SUMMARY.md) | [Platform Overview](03_PLATFORM_OVERVIEW.md)*

---

*QuantraCore Apex v9.0-A | November 2025*
