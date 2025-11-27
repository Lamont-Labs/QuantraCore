# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex™ v8.0 is an institution-grade hybrid AI trading intelligence system that combines deterministic rule-based engines with neural network models. The system is designed for transparency, reproducibility, and regulatory compliance.

**Owner:** Lamont Labs — Jesse J. Lamont  
**Version:** 8.0  
**Repository:** https://github.com/Lamont-Labs/QuantraCore

---

## Project Architecture

### Core Components

- **Apex Engine** (`src/core/`) — Deterministic core producing QuantraScore and structural analysis
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

## Recent Changes

### 2025-10-27 — Documentation Update to v8.0

Updated all documentation to reflect QuantraCore Apex v8.0:

- **README.md** — New high-level product overview
- **docs/OVERVIEW_QUANTRACORE_APEX.md** — Full narrative overview
- **docs/QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml** — Canonical YAML specification
- **docs/APEXLAB_OVERVIEW.md** — Offline training lab
- **docs/APEXCORE_MODELS.md** — Model family (Full/Mini)
- **docs/QUANTRAVISION_APEX_v4_2.md** — Mobile overlay copilot
- **docs/QUANTRAVISION_REMOTE.md** — Desktop-to-mobile streaming
- **docs/PREDICTION_AND_MONSTERRUNNER.md** — Prediction system
- **docs/API_INTEGRATION_FOR_LEARNING_AND_PREDICTION.md** — API adapters
- **docs/COMPLIANCE_POLICY.md** — Institutional compliance
- **docs/SECURITY_AND_HARDENING.md** — Security documentation
- **docs/DEVELOPER_GUIDE.md** — Engineer onboarding
- **docs/RELEASE_NOTES_v8_0.md** — Release notes

Updated existing docs for v8.0 alignment:
- docs/QUICKSTART.md
- docs/SECURITY.md
- docs/API_REFERENCE.md
- docs/docs/ARCHITECTURE.md
- docs/README.md

---

## User Preferences

- Documentation updates only (no code changes in this session)
- Replace old terminology with QuantraCore Apex v8.0 naming
- Professional tone suitable for institutional readers

---

## Deployment

- **Type:** Autoscale
- **Run Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 5000`
- **Port:** 5000
