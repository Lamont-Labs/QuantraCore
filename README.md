# QuantraCore Apex™ — Hybrid AI Trading Intelligence Engine

**Owner:** Lamont Labs — Jesse J. Lamont  
**Version:** v8.0  
[![QuantraCore CI](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Lamont-Labs/QuantraCore/actions/workflows/ci.yml)  
**Status:** Active — Core Engine  
**Repo:** https://github.com/Lamont-Labs/QuantraCore

---

## Overview

QuantraCore Apex™ is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). The system represents a unified deterministic + neural hybrid stack designed for transparency, reproducibility, and regulatory compliance.

**Core Principles:**
- Determinism first
- Fail-closed always
- No cloud dependencies
- Local-only learning
- QuantraScore mandatory everywhere
- Rule engine overrides AI always

---

## Major Components

- **QuantraCore Apex Engine** — Deterministic core with ZDE Engine, QuantraScore (0–100), and 80 Tier Protocols
- **ApexLab** — Self-contained offline local training environment for model development
- **ApexCore Full** — Desktop neural model (3–20MB) for K6 workstation
- **ApexCore Mini** — Mobile neural model (0.5–3MB) for Android/QuantraVision
- **MonsterRunner Engine** — Extreme move detection (phase-compression, volume ignition, entropy collapse)
- **QuantraVision v2** — Mobile overlay copilot with HUD and ApexCore Mini integration
- **Universe Scanner** — Fast scan (1-5s), deep scan (30-90s), and bulk scan modes
- **Broker Layer** — Paper/Sim/Live modes with Alpaca, Interactive Brokers, Custom OMS support

---

## Hardware Targets

| Platform | Target |
|----------|--------|
| Workstation | GMKtec NucBox K6 |
| Mobile | Android (QuantraVision only) |

**Recommended:** 8-core CPU, 16GB RAM, GPU optional (CPU-optimized)

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

4. **Start the API server**
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 5000
   ```

---

## Documentation

### Architecture & Core
| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and component relationships |
| [Core Engine](docs/CORE_ENGINE.md) | Deterministic signal engine (ZDE, microtraits, QuantraScore) |
| [Protocols: Tier](docs/PROTOCOLS_TIER.md) | T01–T80 protocol specifications |
| [Protocols: Learning](docs/PROTOCOLS_LEARNING.md) | LP01–LP25 learning protocols |
| [Omega Directives](docs/OMEGA_DIRECTIVES.md) | Ω1–Ω4 system safety locks |

### Intelligence Ecosystem
| Document | Description |
|----------|-------------|
| [ApexLab Training](docs/APEXLAB_TRAINING.md) | Offline training environment (100-bar windows) |
| [ApexCore Models](docs/APEXCORE_MODELS.md) | Model family (Full 3–20MB / Mini 0.5–3MB) |
| [Prediction Stack](docs/PREDICTION_STACK.md) | 7 prediction engines |
| [MonsterRunner](docs/MONSTERRUNNER.md) | Rare-event detection engine |

### Data & Execution
| Document | Description |
|----------|-------------|
| [Data Layer](docs/DATA_LAYER.md) | Polygon, Tiingo, Alpaca, Intrinio, Finnhub |
| [Broker/OMS](docs/BROKER_OMS.md) | Execution envelope (disabled by default) |
| [Risk Engine](docs/RISK_ENGINE.md) | Final gatekeeper with kill switches |
| [Portfolio System](docs/PORTFOLIO_SYSTEM.md) | Position and exposure tracking |

### Vision & Mobile
| Document | Description |
|----------|-------------|
| [QuantraVision v1](docs/QUANTRAVISION_V1.md) | Legacy thin-client viewer |
| [QuantraVision v2](docs/QUANTRAVISION_V2.md) | On-device copilot with ApexCore Mini |
| [QuantraVision Remote](docs/QUANTRAVISION_REMOTE.md) | Desktop-powered mobile overlay |

### Security & Compliance
| Document | Description |
|----------|-------------|
| [Security & Compliance](docs/SECURITY_COMPLIANCE.md) | Compliance center and security layer |
| [Determinism Tests](docs/DETERMINISM_TESTS.md) | Reproducibility verification suite |
| [SBOM & Provenance](docs/SBOM_PROVENANCE.md) | Software bill of materials |
| [Roadmap](docs/ROADMAP.md) | Development trajectory |

---

## API Connections

**Market Data Providers:**
- Polygon
- Tiingo
- Alpaca Market Data
- Intrinio
- Finnhub

**Compliance Note:** Only data ingestion — no trade recommendations.

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
