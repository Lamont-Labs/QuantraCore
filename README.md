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

| Document | Description |
|----------|-------------|
| [Overview](docs/OVERVIEW_QUANTRACORE_APEX.md) | Full narrative of the Apex ecosystem |
| [Master Spec v8.0](docs/QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml) | Canonical YAML specification |
| [ApexLab Overview](docs/APEXLAB_OVERVIEW.md) | Offline training lab (100-bar windows) |
| [ApexCore Models](docs/APEXCORE_MODELS.md) | Model family (Full 3–20MB / Mini 0.5–3MB) |
| [QuantraVision](docs/QUANTRAVISION_APEX_v4_2.md) | Mobile overlay (v2_apex / v1_legacy) |
| [MonsterRunner](docs/PREDICTION_AND_MONSTERRUNNER.md) | Extreme move detection |
| [API Integration](docs/API_INTEGRATION_FOR_LEARNING_AND_PREDICTION.md) | Polygon, Tiingo, Alpaca, Intrinio, Finnhub |
| [Compliance Policy](docs/COMPLIANCE_POLICY.md) | Omega directives (Ω1–Ω4) |
| [Security](docs/SECURITY_AND_HARDENING.md) | Defense in depth |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Engineer onboarding |

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
