# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v8.0 is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). Unified deterministic + neural hybrid stack.

**Owner:** Lamont Labs — Jesse J. Lamont  
**Version:** 8.0  
**Status:** Active — Core Engine (MVP+ Desktop-Only)
**Repository:** https://github.com/Lamont-Labs/QuantraCore

---

## Core Principles

- Determinism first
- Fail-closed always
- No cloud dependencies
- Local-only learning
- QuantraScore mandatory everywhere (0–100 range)
- Rule engine overrides AI always
- Desktop-only (STRICT NO Android/mobile builds)

---

## Hardware Targets

- **Workstation:** GMKtec NucBox K6 (8-core, 16GB RAM recommended)
- **Note:** Mobile/Android builds are prohibited per project requirements

---

## Project Architecture

### Directory Structure

```
src/quantracore_apex/
├── core/                    # Deterministic core engine
│   ├── engine.py           # Main ApexEngine class
│   ├── schemas.py          # Pydantic data models
│   ├── microtraits.py      # Microtrait computation
│   ├── entropy.py          # Entropy analysis
│   ├── suppression.py      # Suppression detection
│   ├── drift.py            # Drift analysis
│   ├── continuation.py     # Continuation analysis
│   ├── volume_spike.py     # Volume metrics
│   ├── regime.py           # Regime classification
│   ├── quantrascore.py     # QuantraScore computation
│   ├── verdict.py          # Verdict building
│   ├── sector_context.py   # Sector-aware adjustments
│   └── proof_logger.py     # Proof logging
├── protocols/
│   ├── tier/               # T01-T80 Tier Protocols
│   │   ├── T01-T20.py     # Fully implemented
│   │   └── T21-T80.py     # Stubs
│   ├── learning/           # LP01-LP25 Learning Protocols
│   │   ├── LP01-LP10.py   # Fully implemented
│   │   └── LP11-LP25.py   # Stubs
│   └── omega/             # Omega Directives Ω1-Ω4
├── data_layer/
│   ├── adapters/          # Data provider adapters
│   │   ├── alpha_vantage_adapter.py
│   │   ├── polygon_adapter.py     # Polygon.io real market data
│   │   └── synthetic_adapter.py
│   ├── normalization.py   # Data normalization
│   ├── caching.py         # Disk caching
│   └── hashing.py         # SHA-256 verification
├── apexlab/               # Offline training environment
│   ├── windows.py         # Window building
│   ├── features.py        # Feature extraction (30-dim)
│   ├── labels.py          # Label generation
│   ├── dataset_builder.py # Dataset building
│   ├── train_apexcore_demo.py # Demo training
│   └── validation.py      # Model alignment validation
├── apexcore/              # Neural model interface
│   └── interface.py       # ApexCoreFull, ApexCoreMini
├── prediction/            # Prediction engines
│   ├── expected_move.py   # Expected move predictor
│   └── monster_runner.py  # MonsterRunner rare-event detection
├── server/                # API server
│   └── app.py             # FastAPI application
└── tests/                 # Test suite
    ├── test_determinism_golden_set.py
    ├── test_protocol_signatures.py
    ├── test_data_layer.py
    └── test_apexlab_pipeline.py
```

### Key Technologies

- **Language:** Python 3.11
- **Web Framework:** FastAPI with Uvicorn
- **ML:** scikit-learn (no PyTorch for disk space)
- **Testing:** Pytest
- **HTTP Client:** HTTPX
- **Numerical:** NumPy, Pandas

---

## API Endpoints

**Server runs on port 5000:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and compliance note |
| `/health` | GET | Health check |
| `/score?ticker=AAPL` | GET | Legacy QuantraScore endpoint |
| `/scan/{symbol}` | GET | Full Apex analysis scan |
| `/trace/{window_hash}` | GET | Detailed protocol trace |
| `/monster_runner/{symbol}` | GET | MonsterRunner rare-event check |
| `/risk/hud` | GET | Risk HUD data (legacy) |
| `/audit/export` | GET | Audit export (legacy) |

---

## Running the Project

### API Server (Primary)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

### Demo Scripts

```bash
python scripts/fetch_and_scan_demo.py   # Full scan demo
python scripts/run_apexlab_demo.py      # ApexLab training demo
python scripts/validate_apexcore.py     # ApexCore validation
```

### Backtest & Stress Testing (Requires POLYGON_API_KEY)

```bash
python scripts/backtest_and_prove.py    # 1-year backtest on 500 symbols
python scripts/live_stress_test.py      # 24h live stress test
```

### Tests

```bash
pytest src/quantracore_apex/tests/ -v
```

---

## Compliance

All outputs are framed as **structural probabilities**, NOT trading advice.

- Outputs include mandatory `compliance_note` field
- Omega Ω4 directive enforces compliance mode
- No trade execution or order placement functionality

---

## Omega Directives

- **Ω1:** Hard safety lock (extreme risk tier)
- **Ω2:** Entropy override (chaotic entropy state)
- **Ω3:** Drift override (critical drift state)
- **Ω4:** Compliance override (always active)

---

## Recent Changes

### 2025-11-27 — MVP+ Desktop Codebase Build

**Full MVP+ Implementation:**
- Complete core engine with 10+ analysis modules
- T01-T20 tier protocols fully implemented (T21-T80 stubs)
- LP01-LP10 learning protocols implemented (LP11-LP25 stubs)
- Omega Ω1-Ω4 directives implemented
- Data layer with Alpha Vantage + Synthetic + Polygon adapters
- ApexLab training pipeline with sklearn
- ApexCore model interface (Full/Mini)
- MonsterRunner rare-event detection (Stage 1)
- FastAPI server with scan endpoints
- 28 passing tests (determinism, protocols, data layer, pipeline)

**Backtest & Stress Testing Added:**
- `scripts/backtest_and_prove.py` - 1-year backtest on 500 symbols with SHA-256 proof logging
- `scripts/live_stress_test.py` - 24h continuous stress test with real-time data
- Polygon.io adapter for real market data (requires POLYGON_API_KEY secret)

**Design Decisions:**
- scikit-learn instead of PyTorch (disk space)
- Synthetic adapter for testing without API keys
- Desktop-only (NO Android/mobile per requirements)

---

## Deployment

- **Type:** Autoscale
- **Run Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port 5000`
- **Port:** 5000
