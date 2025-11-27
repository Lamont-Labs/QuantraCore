# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex™ v8.0 is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). Unified deterministic + neural hybrid stack.

**Owner:** Lamont Labs — Jesse J. Lamont  
**Version:** 8.0  
**Status:** Active — Core Engine  
**Repository:** https://github.com/Lamont-Labs/QuantraCore

---

## Core Principles

- Determinism first
- Fail-closed always
- No cloud dependencies
- Local-only learning
- QuantraScore mandatory everywhere (0–100 range)
- Rule engine overrides AI always

---

## Hardware Targets

- **Workstation:** GMKtec NucBox K6 (8-core, 16GB RAM recommended)
- **Mobile:** Android (for QuantraVision only)

---

## Project Architecture

### Core Components

- **Apex Engine** (`src/core/`) — Deterministic core with ZDE Engine, QuantraScore, 80 Tier Protocols
- **API Layer** (`src/api/`) — FastAPI endpoints for integration
- **CLI** (`cli/`) — Typer-based command-line interface
- **Tests** (`tests/`) — Reproducibility and filter tests
- **Documentation** (`docs/`) — Comprehensive v8.0 documentation

### Key Technologies

- **Language:** Python 3.11
- **Web Framework:** FastAPI with Uvicorn
- **CLI Framework:** Typer
- **Testing:** Pytest
- **HTTP Client:** HTTPX
- **Numerical:** NumPy

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

## Running the Project

### API Server (Primary)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

The FastAPI server runs on port 5000 with endpoints:
- `GET /health` — Health check
- `GET /score?ticker=AAPL&seed=42` — Deterministic score computation
- `GET /risk/hud` — Risk HUD data
- `GET /audit/export` — Export audit data

### CLI Demo

```bash
python -m cli.main
```

### Tests

```bash
pytest -q
```

---

## Omega Directives

- **Ω1:** Hard safety lock
- **Ω2:** Entropy override
- **Ω3:** Drift override
- **Ω4:** Compliance override

---

## Recent Changes

### 2025-11-27 — Full 8-Part Spec Documentation Update

Applied comprehensive 8-part specification to documentation:

**New Documentation Files Created:**
- `docs/ARCHITECTURE.md` — System architecture and component relationships
- `docs/CORE_ENGINE.md` — Deterministic signal engine specification
- `docs/PROTOCOLS_TIER.md` — T01–T80 protocol specifications
- `docs/PROTOCOLS_LEARNING.md` — LP01–LP25 learning protocols
- `docs/OMEGA_DIRECTIVES.md` — Ω1–Ω4 system safety locks
- `docs/APEXLAB_TRAINING.md` — Offline training environment
- `docs/PREDICTION_STACK.md` — 7 prediction engines
- `docs/MONSTERRUNNER.md` — Rare-event detection engine
- `docs/DATA_LAYER.md` — Data providers and pipeline
- `docs/BROKER_OMS.md` — Execution envelope
- `docs/RISK_ENGINE.md` — Final gatekeeper with kill switches
- `docs/PORTFOLIO_SYSTEM.md` — Position and exposure tracking
- `docs/QUANTRAVISION_V1.md` — Legacy thin-client viewer
- `docs/QUANTRAVISION_V2.md` — On-device copilot with ApexCore Mini
- `docs/SECURITY_COMPLIANCE.md` — Compliance center and security layer
- `docs/DETERMINISM_TESTS.md` — Reproducibility verification suite
- `docs/SBOM_PROVENANCE.md` — Software bill of materials
- `docs/ROADMAP.md` — Development trajectory

**Key Updates:**
- README.md updated with new documentation structure
- ApexCore Models updated with inference specs (<20ms Full, <30ms Mini)
- All docs aligned with 8-part institutional spec
- No code changes (DOCS_ONLY mode)

---

## Deployment

- **Type:** Autoscale
- **Run Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 5000`
- **Port:** 5000
