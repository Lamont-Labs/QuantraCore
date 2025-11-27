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

### 2025-11-27 — Documentation Alignment to Corrected Master Spec v8.0

Updated all documentation to align with the corrected master spec:

- **Master Spec:** Updated with correct Omega directives, QuantraScore (0–100), hardware targets
- **API Providers:** Polygon, Tiingo, Alpaca, Intrinio, Finnhub
- **ApexCore Full:** 3–20 MB (corrected from 4–20 MB)
- **Training Windows:** 100-bar OHLCV
- **MonsterRunner Signals:** Phase-compression, volume ignition, range flipping, entropy collapse, sector moves
- **QuantraVision:** v2_apex and v1_legacy versions documented
- **Broker Layer:** Alpaca, Interactive Brokers, Custom OMS API

---

## Deployment

- **Type:** Autoscale
- **Run Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 5000`
- **Port:** 5000
