# QuantraCore Apex™ — System Architecture

**Version:** 8.0  
**Status:** Active  
**Owner:** Lamont Labs

---

## Overview

QuantraCore Apex™ is an institutional-grade deterministic intelligence engine fused with a private offline-learning ecosystem (ApexLab) and dual neural assistant models (ApexCore Full + ApexCore Mini). The system operates as a unified deterministic + neural hybrid stack.

---

## Core Architectural Principles

| Principle | Description |
|-----------|-------------|
| **Determinism First** | Identical inputs always produce identical outputs |
| **Fail-Closed** | All components default to safe state on error |
| **Privacy/Offline** | Local-only datasets, no cloud processing |
| **Compliance Safe** | No trade recommendations, OMS disabled by default |
| **Modular Upgrading** | File-isolated components for independent improvement |
| **Reproducibility** | Hash-locked datasets, versioned protocols, model manifests |

---

## System Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTRACORE APEX ECOSYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  DATA LAYER (Unified Ingestion)              │    │
│  │  Polygon │ Alpaca │ IEX │ Tiingo │ Finnhub │ Intrinio       │    │
│  │  ────────────────────────────────────────────────────────   │    │
│  │  OHLCV │ Fundamentals │ Sector │ Macro │ Alt-Data (text)    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              QUANTRACORE APEX (Deterministic Core)           │    │
│  │  ─────────────────────────────────────────────────────────  │    │
│  │  ZDE Engine │ Microtraits │ Signal Engines │ QuantraScore   │    │
│  │  Protocols T01–T80 │ LP01–LP25 │ Omega Ω1–Ω4               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                    ↓                    ↓                            │
│  ┌──────────────────────┐    ┌──────────────────────────────┐       │
│  │     APEXLAB          │    │     PREDICTION STACK          │       │
│  │  (Training Factory)  │    │  (Structural Forecasting)     │       │
│  │  ─────────────────   │    │  ───────────────────────────  │       │
│  │  Teacher Labels      │    │  Expected Move │ Volatility   │       │
│  │  Model Training      │    │  Compression │ Continuation   │       │
│  │  Validation          │    │  Regime │ Instability         │       │
│  │  Rejection/Promotion │    │  MonsterRunner (Rare Events)  │       │
│  └──────────────────────┘    └──────────────────────────────┘       │
│            ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              APEXCORE MODELS (Neural Assistants)             │    │
│  │  ─────────────────────────────────────────────────────────  │    │
│  │  ApexCore Full (Desktop, 3–20MB, <20ms)                     │    │
│  │  ApexCore Mini (Android, 0.5–3MB, <30ms)                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│  ┌──────────────────────┐    ┌──────────────────────────────┐       │
│  │    RISK ENGINE       │    │     BROKER / OMS              │       │
│  │  (Final Gatekeeper)  │    │  (Execution Envelope)         │       │
│  │  ─────────────────   │    │  ───────────────────────────  │       │
│  │  Volatility Gates    │    │  Research │ Simulation        │       │
│  │  Entropy/Drift       │    │  Paper │ Live-Ready           │       │
│  │  Kill Switches       │    │  (Disabled by Default)        │       │
│  └──────────────────────┘    └──────────────────────────────┘       │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    VISUALIZATION LAYER                        │    │
│  │  ─────────────────────────────────────────────────────────  │    │
│  │  Apex Dashboard (Desktop) │ QuantraVision v1 (Legacy)       │    │
│  │  QuantraVision v2 (On-Device) │ Remote Overlay              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Descriptions

### 1. Data Layer
Unified deterministic ingestion framework for all market data.
- **Providers:** Polygon, Alpaca, IEX Cloud, Tiingo, Finnhub, Intrinio
- **Data Types:** OHLCV, fundamentals, sector/macro, alt-data (text-only)
- **Principles:** Deterministic caching, hash-locked datasets, immutable storage

### 2. QuantraCore Apex (Deterministic Core)
Master deterministic signal engine executing all structural analysis.
- **Signal Engines:** ZDE, Continuation, Entry Timing, Volume Spike, Microtrait, Suppression, Entropy, Drift, Regime, Sector Context
- **Protocols:** T01–T80 (Tier), LP01–LP25 (Learning)
- **Safety:** Omega Directives Ω1–Ω4
- **Output:** QuantraScore (0–100)

### 3. ApexLab (Training Factory)
Fully autonomous offline training environment.
- **Location:** K6 workstation
- **Function:** Generates teacher labels, trains ApexCore models
- **Data Windows:** 100-bar OHLCV
- **Validation:** Strict alignment to Apex outputs

### 4. Prediction Stack
Supervised predictive intelligence (informational only).
- **Engines:** Expected Move, Volatility Projection, Compression Forecaster, Continuation Estimator, Regime Transition, Early Instability
- **MonsterRunner:** Rare-event precursor detection
- **Compliance:** No trade recommendations

### 5. ApexCore Models
On-device neural assistants trained by ApexLab.
- **ApexCore Full:** Desktop (3–20MB, <20ms, 8–12 heads)
- **ApexCore Mini:** Android (0.5–3MB, <30ms, distilled)
- **Rule:** Always subordinate to Apex teacher

### 6. Risk Engine
Final arbiter of all system decisions.
- **Checks:** Volatility, spread, regime, entropy, drift, suppression
- **Kill Switches:** Daily loss, drawdown, portfolio heat
- **Output:** risk_tier, final_permission

### 7. Broker / OMS
Optional execution envelope (disabled by default).
- **Modes:** Research, simulation, paper, live-ready
- **Integrations:** Alpaca, Interactive Brokers, Custom OMS
- **Safety:** Fail-closed on any risk denial

### 8. Visualization Layer
Research-grade visual interfaces.
- **Apex Dashboard:** Signal grid, protocol explorer, entropy/drift console
- **QuantraVision v1:** Thin-client signal viewer
- **QuantraVision v2:** On-device copilot with ApexCore Mini
- **Remote:** Desktop-powered overlay to mobile

---

## Data Flow

```
OHLCV Input
    ↓
Data Normalization (Z-score + volatility-adjusted)
    ↓
Baseline Microtrait Extraction
    ↓
Protocol Execution (T01–T80, LP01–LP25)
    ↓
Signal Engines (ZDE, Entropy, Drift, Suppression, etc.)
    ↓
Risk Gates (Omega Ω1–Ω4)
    ↓
QuantraScore Fusion (0–100)
    ↓
Verdict Construction (trend, pressure, strength, risk_level)
    ↓
Proof Logging (JSON trace)
```

---

## Execution Modes

| Mode | Latency | Usage |
|------|---------|-------|
| Fast Scan | 1–5 seconds | Universe sweeps |
| Deep Scan | 30–90 seconds | High-confidence structural analysis |
| Micro Scan | <1 second | Real-time recalculation |

---

## Cross-System Cohesion Rules

1. **Apex ⇢ ApexLab ⇢ ApexCore** forms a closed learning loop
2. **QuantraVision Mini** uses ApexCore Mini only
3. **QuantraVision Legacy** uses Apex outputs only
4. **Broker/OMS** never activated unless compliance conditions are knowingly enabled
5. **Risk Engine** is always the final authority
6. **Data Layer** feeds Apex and ApexLab in identical normalized format

---

## Hardware Targets

| Platform | Device | Specs |
|----------|--------|-------|
| Workstation | GMKtec NucBox K6 | 8-core, 16GB RAM |
| Mobile | Android | QuantraVision only |

---

## Related Documentation

- [Core Engine](CORE_ENGINE.md)
- [Protocols: Tier](PROTOCOLS_TIER.md)
- [Protocols: Learning](PROTOCOLS_LEARNING.md)
- [Omega Directives](OMEGA_DIRECTIVES.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
- [ApexCore Models](APEXCORE_MODELS.md)
- [Prediction Stack](PREDICTION_STACK.md)
- [MonsterRunner](MONSTERRUNNER.md)
- [Data Layer](DATA_LAYER.md)
- [Risk Engine](RISK_ENGINE.md)
- [Broker/OMS](BROKER_OMS.md)
