# Technical Architecture Documentation

**Document Classification:** Investor Due Diligence  
**Version:** 9.0-A  
**Date:** November 2025  
**Status:** Production  
**Owner:** Lamont Labs

---

## Executive Summary

QuantraCore Apex v9.0-A is an institutional-grade deterministic AI trading intelligence engine designed for desktop deployment. The system features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore), operating as a unified deterministic + neural hybrid stack.

**Key Technical Differentiators:**
- 100% deterministic outputs (identical inputs → identical results)
- Fail-closed architecture (safe state on any error)
- Zero cloud dependencies (complete offline operation)
- 970+ automated tests with 100% pass rate
- 99.25% regulatory compliance excellence score
- 3-5x stricter than SEC/FINRA/MiFID II requirements

---

## Technology Stack

### Backend
| Component | Technology | Version |
|-----------|------------|---------|
| Runtime | Python | 3.11 |
| API Framework | FastAPI | Latest |
| ASGI Server | Uvicorn | Latest |
| ML Framework | scikit-learn | Latest |
| Numerical | NumPy, Pandas | Latest |
| HTTP Client | HTTPX | Latest |

### Frontend
| Component | Technology | Version |
|-----------|------------|---------|
| Framework | React | 18.2 |
| Build Tool | Vite | 5.x |
| Styling | Tailwind CSS | 3.4 |
| Language | TypeScript | Latest |

### Development & Testing
| Component | Technology |
|-----------|------------|
| Backend Testing | pytest |
| Frontend Testing | vitest |
| Type Checking | mypy |
| Linting | ruff |
| Package Manager | pip/npm |

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUANTRACORE APEX v9.0-A                               │
│                   Institutional Hardening Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    DATA LAYER (Unified Ingestion)                   │     │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │     │
│  │  │ Polygon  │ Alpha    │ Yahoo    │ CSV      │Synthetic │          │     │
│  │  │ Adapter  │ Vantage  │ Finance  │ Bundle   │ Adapter  │          │     │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘          │     │
│  │  ────────────────────────────────────────────────────────          │     │
│  │  Hash Verification │ Deterministic Caching │ OHLCV Normalization   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    ↓                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               QUANTRACORE APEX (Deterministic Core)                 │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ ApexEngine │ ZDE │ Microtraits │ Signal Engines │ QuantraScore│   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ Protocols: T01-T80 (Tier) │ LP01-LP25 (Learning) │ MR01-MR05│   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ Safety: Omega Directives Ω1-Ω5 │ Compliance Mode: RESEARCH  │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                          ↓                    ↓                              │
│  ┌───────────────────────────┐    ┌───────────────────────────────┐         │
│  │       APEXLAB V2          │    │      PREDICTION STACK         │         │
│  │    (Training Factory)     │    │   (Structural Forecasting)    │         │
│  │  ───────────────────────  │    │  ───────────────────────────  │         │
│  │  40+ Field Schema         │    │  Expected Move │ Volatility   │         │
│  │  Walk-Forward Splits      │    │  Compression │ Continuation   │         │
│  │  Runner/Monster Labels    │    │  MonsterRunner (Rare Events)  │         │
│  │  Quality Tiers            │    │  Regime Detection             │         │
│  └───────────────────────────┘    └───────────────────────────────┘         │
│              ↓                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │               APEXCORE V2 (Neural Assistant Models)                 │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ ApexCore Big (Desktop)  │  ApexCore Mini (Lightweight)      │   │     │
│  │  │ 5 Prediction Heads: quantra_score, runner_prob, quality_tier│   │     │
│  │  │                         avoid_trade, regime                  │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  │  Manifest Verification │ Hash Matching │ Fail-Closed Rules         │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    ↓                                         │
│  ┌───────────────────────────┐    ┌───────────────────────────────┐         │
│  │      RISK ENGINE          │    │     COMPLIANCE ENGINE         │         │
│  │   (Final Gatekeeper)      │    │  (Regulatory Excellence)      │         │
│  │  ───────────────────────  │    │  ───────────────────────────  │         │
│  │  Volatility Gates         │    │  99.25% Excellence Score      │         │
│  │  Entropy/Drift Detection  │    │  3-5x Regulatory Margins      │         │
│  │  Kill Switches            │    │  SEC/FINRA/MiFID II/Basel     │         │
│  └───────────────────────────┘    └───────────────────────────────┘         │
│                                    ↓                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    API & VISUALIZATION LAYER                        │     │
│  │  ┌─────────────────────────────────────────────────────────────┐   │     │
│  │  │ FastAPI Server (36 Endpoints) │ React Dashboard (ApexDesk)  │   │     │
│  │  │ REST API │ Interactive Docs │ Real-time Analysis Display    │   │     │
│  │  └─────────────────────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Architectural Principles

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Determinism First** | Identical inputs always produce identical outputs via seed control and hash verification | Reproducible analysis, audit compliance |
| **Fail-Closed** | All components default to safe state on any error; no silent failures | Institutional safety |
| **Privacy/Offline** | Zero cloud dependencies; all processing local | Data sovereignty |
| **Compliance Safe** | Research-only mode enforced; OMS disabled by default; Ω4 always active | Regulatory compliance |
| **Modular Design** | File-isolated components for independent upgrade | Maintainability |
| **Reproducibility** | Hash-locked datasets, versioned protocols, model manifests | Auditability |

---

## Component Inventory

### Protocol System (115 Total)

| Type | Count | IDs | Purpose |
|------|-------|-----|---------|
| Tier Protocols | 80 | T01-T80 | Structural analysis rules |
| Learning Protocols | 25 | LP01-LP25 | Label generation for training |
| MonsterRunner | 5 | MR01-MR05 | Rare event detection |
| Omega Directives | 5 | Ω1-Ω5 | Safety overrides |

### Signal Engines

| Engine | Purpose |
|--------|---------|
| ZDE Engine | Zero-divergence equilibrium detection |
| Continuation Estimator | Trend continuation probability |
| Entry Timing | Optimal entry point analysis |
| Volume Spike | Unusual volume detection |
| Microtrait Extractor | Structural pattern extraction |
| Suppression Detector | Signal dampening detection |
| Entropy Engine | Market disorder measurement |
| Drift Engine | Trend drift analysis |
| Regime Detector | Market regime classification |

### Safety Systems (Omega Directives)

| Directive | Trigger | Action |
|-----------|---------|--------|
| Ω1 | Extreme risk tier detected | Hard safety lock |
| Ω2 | Chaotic entropy state | Entropy override |
| Ω3 | Critical drift state | Drift override |
| Ω4 | Always active | Compliance mode (research-only) |
| Ω5 | Strong suppression | Signal suppression lock |

---

## Data Flow Architecture

```
Market Data Input (OHLCV)
         ↓
┌─────────────────────────┐
│  Data Normalization     │  Z-score + volatility-adjusted
│  Hash Verification      │  SHA-256 integrity check
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Window Builder         │  100-bar analysis windows
│  Feature Extraction     │  Microtraits computation
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Protocol Execution     │  T01-T80, LP01-LP25
│  Signal Engines         │  ZDE, Entropy, Drift, etc.
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Omega Directives       │  Ω1-Ω5 safety checks
│  Risk Gates             │  Kill switches, limits
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  QuantraScore Fusion    │  0-100 composite score
│  Verdict Construction   │  trend, pressure, strength
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│  Proof Logging          │  JSON trace with hash
│  API Response           │  Deterministic output
└─────────────────────────┘
```

---

## Directory Structure

```
quantracore-apex/
├── src/quantracore_apex/
│   ├── core/               # ApexEngine, schemas, microtraits, quantrascore
│   ├── protocols/
│   │   ├── tier/           # T01-T80 protocols
│   │   ├── learning/       # LP01-LP25 protocols
│   │   ├── monster_runner/ # MR01-MR05 protocols
│   │   └── omega/          # Omega directives
│   ├── data_layer/         # Adapters: Polygon, Alpha Vantage, CSV, Synthetic
│   ├── apexlab/            # V1 + V2 training pipeline
│   ├── apexcore/           # V1 + V2 neural models
│   ├── prediction/         # Expected move, volatility, continuation
│   ├── compliance/         # Regulatory excellence engine
│   ├── risk/               # Risk assessment engine
│   ├── signal/             # Signal builder
│   ├── broker/             # OMS (disabled by default)
│   ├── portfolio/          # Portfolio management
│   └── server/             # FastAPI application
├── tests/                  # 970+ institutional tests
├── dashboard/              # React frontend
├── config/                 # YAML configurations
├── docs/                   # 50+ documentation files
└── models/                 # Trained model artifacts
```

---

## Scalability Design

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Engine initialization | <500ms |
| Single symbol scan | 1-5 seconds |
| Protocol execution (80 protocols) | <100ms |
| API response latency | <50ms typical |
| Memory footprint | <500MB |
| CPU cores utilized | 1-8 (configurable) |

### Execution Modes

| Mode | Latency | Use Case |
|------|---------|----------|
| Fast Scan | 1-5 seconds | Universe sweeps |
| Deep Scan | 30-90 seconds | High-confidence analysis |
| Micro Scan | <1 second | Real-time recalculation |

---

## Security Architecture

### Encryption
- **At Rest:** AES-256-GCM for sensitive data
- **In Transit:** TLS 1.3 required
- **Archives:** Envelope encryption with key rotation

### Access Controls
- API authentication (configurable)
- Rate limiting (100 req/min default)
- Audit logging with hash chains

### Integrity Verification
- SHA-256 hash verification on all components
- Automatic halt on integrity failure (Ω1)
- Cryptographic provenance chains

---

## Deployment Architecture

### Current Deployment
| Component | Host | Port |
|-----------|------|------|
| FastAPI Backend | 0.0.0.0 | 8000 |
| React Frontend | 0.0.0.0 | 5000 |

### Target Hardware
| Platform | Specification |
|----------|--------------|
| Desktop | 8-core CPU, 16GB RAM, SSD |
| Reference | GMKtec NucBox K6 |

### Deployment Options
- Local desktop installation
- On-premise server deployment
- Air-gapped network operation

---

## Technology Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| scikit-learn over PyTorch | Disk space efficiency (<100MB vs 2GB+), faster cold start, sufficient for tabular prediction tasks |
| FastAPI over Flask | Async support, automatic OpenAPI docs, type validation, better performance |
| React + Vite over Next.js | SPA model fits desktop deployment, faster dev builds, simpler deployment |
| Deterministic seeding | Audit compliance, reproducible research, regulatory requirements |
| Offline-first design | Data sovereignty, no cloud dependency, institutional security requirements |

---

## Appendix: Module Count Summary

| Category | Count |
|----------|-------|
| Core Modules | 12 |
| Tier Protocols | 80 |
| Learning Protocols | 25 |
| MonsterRunner Protocols | 5 |
| Omega Directives | 5 |
| API Endpoints | 36 |
| Signal Engines | 9 |
| Data Adapters | 5 |
| **Total Components** | **168+** |

---

*Document prepared for investor due diligence. Confidential.*
