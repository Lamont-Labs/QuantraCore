# QuantraCore Apex - Hybrid AI Trading Intelligence Engine

**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** v8.2  
[![QuantraCore CI](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml)  
**Status:** Active - Full Protocol System (Desktop-Only)  
**Repo:** https://github.com/Lamont-Labs/QuantraCore

---

## Overview

QuantraCore Apex is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). The system represents a unified deterministic + neural hybrid stack designed for transparency, reproducibility, and regulatory compliance.

**Core Principles:**
- Determinism first
- Fail-closed always
- No cloud dependencies
- Local-only learning
- QuantraScore mandatory everywhere (0-100)
- Rule engine overrides AI always

---

## What Works Today

| Component | Status | Description |
|-----------|--------|-------------|
| Core Engine | Production | Deterministic analysis with 80 protocols |
| ApexLab | Production | Offline training environment |
| ApexCore | Production | Neural models (Full + Mini) |
| Desktop API | Production | FastAPI server with 20+ endpoints |
| ApexDesk UI | Production | React dashboard with branding |
| Risk Engine | Production | Multi-layer risk assessment |
| OMS | Simulation | Order management (paper only) |
| Portfolio | Production | Position and exposure tracking |

---

## What Is Out of Scope

- **Live Trading** - No real brokerage connections or order execution
- **Mobile/Android** - Strictly prohibited (desktop-only architecture)
- **Cloud Dependencies** - Runs entirely locally

---

## Major Components

- **QuantraCore Apex Engine** - Deterministic core with ZDE Engine, QuantraScore (0-100), and 80 Tier Protocols
- **ApexLab** - Self-contained offline local training environment for model development
- **ApexCore Full** - Desktop neural model (3-20MB) for workstation
- **ApexCore Mini** - Lightweight neural model (0.5-3MB) for reduced resource usage
- **MonsterRunner Engine** - Extreme move detection (phase-compression, volume ignition, entropy collapse)
- **ApexDesk** - React-based research dashboard with Lamont Labs branding
- **Universe Scanner** - Fast scan, deep scan, and bulk scan modes
- **Risk Engine** - Multi-layer risk assessment with Omega overrides

---

## Hardware Targets

| Platform | Target |
|----------|--------|
| Workstation | GMKtec NucBox K6 |
| CPU | 8-core recommended |
| RAM | 16GB recommended |
| GPU | Optional (CPU-optimized) |

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
   cd dashboard && npm install && cd ..
   ```

3. **Start the API server**
   ```bash
   uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000
   ```

4. **Start the dashboard**
   ```bash
   cd dashboard && npm run dev
   ```

5. **Open ApexDesk**
   ```
   http://localhost:5000
   ```

See [Getting Started Guide](docs/GETTING_STARTED_DESKTOP.md) for detailed setup instructions.

---

## Documentation

### Core Documentation
| Document | Description |
|----------|-------------|
| [Master Spec v8.2](docs/QUANTRACORE_APEX_MASTER_SPEC_v8.2.md) | Complete system specification |
| [Getting Started](docs/GETTING_STARTED_DESKTOP.md) | Setup and first steps |
| [Compliance & Safety](docs/COMPLIANCE_AND_SAFETY.md) | Research-only constraints |

### Architecture & Core
| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and component relationships |
| [Core Engine](docs/CORE_ENGINE.md) | Deterministic signal engine |
| [Protocols: Tier](docs/PROTOCOLS_TIER.md) | T01-T80 protocol specifications |
| [Protocols: Learning](docs/PROTOCOLS_LEARNING.md) | LP01-LP25 learning protocols |
| [Omega Directives](docs/OMEGA_DIRECTIVES.md) | Ω1-Ω5 system safety locks |

### Intelligence Ecosystem
| Document | Description |
|----------|-------------|
| [ApexLab Training](docs/APEXLAB_TRAINING.md) | Offline training environment |
| [ApexCore Models](docs/APEXCORE_MODELS.md) | Model family documentation |
| [Prediction Stack](docs/PREDICTION_STACK.md) | Prediction engines |
| [MonsterRunner](docs/MONSTERRUNNER.md) | Rare-event detection |

### Data & Execution
| Document | Description |
|----------|-------------|
| [Data Layer](docs/DATA_LAYER.md) | Data provider adapters |
| [Broker/OMS](docs/BROKER_OMS.md) | Order management (simulation) |
| [Risk Engine](docs/RISK_ENGINE.md) | Risk assessment system |
| [Portfolio System](docs/PORTFOLIO_SYSTEM.md) | Position tracking |
| [API Reference](docs/API_REFERENCE.md) | REST API documentation |

---

## Test Coverage

```
349 tests passed, 5 skipped (ApexCore sklearn internals)
- 80 Tier Protocol tests
- 25 Learning Protocol tests
- 12 MonsterRunner tests
- 40+ Core Engine tests
- 23 API Endpoint tests
- 5 Frontend tests
- 7 Determinism tests
```

Run tests:
```bash
pytest src/quantracore_apex/tests/ -v
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/scan_universe` | POST | Multi-symbol scan |
| `/scan_symbol` | POST | Single symbol scan |
| `/trace/{hash}` | GET | Protocol trace |
| `/monster_runner/{symbol}` | POST | MonsterRunner check |
| `/risk/assess/{symbol}` | POST | Risk assessment |
| `/signal/generate/{symbol}` | POST | Signal generation |
| `/portfolio/status` | GET | Portfolio snapshot |
| `/api/stats` | GET | System statistics |

---

## What This Repo Does NOT Include

- No live brokerage connections or API keys
- No real financial data (synthetic data for testing)
- No user accounts or personal information
- No claims of profit, advice, or market prediction
- Execution is **disabled by default** (research and simulation only)

All data is synthetic and demonstrative.

---

## Contact

**Jesse J. Lamont** - Founder, Lamont Labs  
Email: lamontlabs@proton.me  
GitHub: https://github.com/Lamont-Labs

---

## Disclaimers

This is a demonstration repository only - no trading advice or financial activity.  
All data is synthetic or public domain.  
No production trading systems are connected.

---

**Persistence = Proof.**  
Every build, every log, every checksum - reproducible by design.
