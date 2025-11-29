# QuantraCore Apex — Architecture Overview

## System Architecture

QuantraCore Apex v9.0-A is an institutional-grade deterministic AI trading intelligence engine. This document provides a high-level overview of the system architecture.

## Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        ApexDesk (React UI)                       │
│                     Port 5000 | Dashboard                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                             │
│                     Port 8000 | API                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Scanner   │  │   Engine    │  │  PredictiveAdvisor     │  │
│  │   (Data)    │──│  (Core)     │──│  (ML Integration)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│         │               │                    │                   │
│  ┌──────▼───────────────▼────────────────────▼──────────────┐   │
│  │                    Hardening Layer                        │   │
│  │  (Mode Enforcer | Kill Switch | Incident Logger)         │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  EEO Engine │──│  Execution  │──│     Broker Router       │  │
│  │ (Entry/Exit)│  │   Engine    │  │ (Alpaca/PaperSim)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                          │                                       │
│                   ┌──────▼──────┐                               │
│                   │ Risk Engine │                               │
│                   │ (9 Checks)  │                               │
│                   └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Scanner (Data Layer)

**Location**: `src/quantracore_apex/data_layer/`

Responsible for:
- Multi-API data ingestion (Polygon, Alpha Vantage, Yahoo, CSV)
- OHLCV normalization and validation
- Rate limiting and caching
- Data integrity checks

### 2. Engine (Core)

**Location**: `src/quantracore_apex/core/`, `src/quantracore_apex/protocols/`

The deterministic kernel:
- 80 Tier Protocols (T01-T80)
- 25 Learning Protocols (LP01-LP25)
- 5 Monster Runner Protocols (MR01-MR05)
- 5 Omega Directives (Ω1-Ω5)

Outputs: QuantraScore (0-100), protocol traces, regime detection

### 3. ApexLab (Training)

**Location**: `src/quantracore_apex/apexlab/`

Offline training environment:
- Feature engineering (40+ field schema)
- Label generation (ret_1d, ret_3d, quality tiers)
- Dataset building and validation
- Model training pipelines

### 4. ApexCore (Models)

**Location**: `src/quantracore_apex/apexcore/`

On-device neural models:
- Big/Mini variants
- 5 prediction heads
- Manifest verification
- Fail-closed on invalid manifests

### 5. PredictiveAdvisor

**Location**: `src/quantracore_apex/prediction/`, `src/quantracore_apex/core/integration_predictive.py`

ML integration layer:
- Runner probability estimation
- Quality tier prediction
- Avoid trade detection
- Ensemble disagreement handling

### 6. EEO Engine (Entry/Exit Optimization)

**Location**: `src/quantracore_apex/eeo_engine/`

Calculates optimal:
- Entry zones and order types
- Protective stops
- Profit targets
- Trailing stops
- Position sizing

### 7. Execution Engine

**Location**: `src/quantracore_apex/broker/execution_engine.py`

Order management:
- Plan to OrderTicket conversion
- Risk validation
- Broker routing
- Execution logging

### 8. Broker Layer

**Location**: `src/quantracore_apex/broker/`

Pluggable broker adapters:
- Alpaca Paper
- PaperSim (internal)
- Null Adapter (RESEARCH mode)

### 9. Hardening Infrastructure

**Location**: `src/quantracore_apex/hardening/`

Safety systems:
- Protocol manifest verification
- Config validation
- Mode enforcement (RESEARCH/PAPER/LIVE)
- Incident logging
- Kill switch management

## Data Flows

### Signal Generation Flow

```
Market Data → Scanner → Engine → Protocol Execution → QuantraScore
                                      │
                                      ▼
                              PredictiveAdvisor → Enhanced Signal
```

### Execution Flow

```
Signal → EEO Engine → Entry/Exit Plan → Risk Engine → Broker Router
                                              │
                                              ▼
                                        Kill Switch Check
                                              │
                                              ▼
                                        Order Execution
```

## Key Design Decisions

### Determinism

- No RNG in engine path
- Static protocol order (manifest verified)
- Reproducible outputs given inputs

### Fail-Closed

- Invalid config → refuse to start
- Missing manifest → engine disabled
- Kill switch engaged → no new orders

### Mode Boundaries

- RESEARCH: No orders, no broker calls
- PAPER: Paper orders only
- LIVE: Requires explicit enablement + compliance

## File Structure

```
src/quantracore_apex/
├── apexcore/           # Neural models
├── apexlab/            # Training environment
├── broker/             # Execution layer
│   ├── adapters/       # Broker adapters
│   ├── execution_engine.py
│   ├── risk_engine.py
│   └── router.py
├── core/               # Deterministic engine
├── data_layer/         # Data ingestion
│   └── adapters/       # API adapters
├── eeo_engine/         # Entry/Exit optimization
├── hardening/          # Safety infrastructure
├── protocols/          # T/LP/MR/Omega protocols
│   ├── tier/
│   ├── learning/
│   ├── monster_runner/
│   └── omega/
└── server/             # FastAPI backend
```

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
