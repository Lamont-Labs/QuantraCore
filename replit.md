# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v8.1 is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). Unified deterministic + neural hybrid stack.

**Owner:** Lamont Labs — Jesse J. Lamont  
**Version:** 8.1  
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
│   ├── monster_runner/    # MR01-MR05 MonsterRunner Protocols
│   └── omega/             # Omega Directives Ω1-Ω5
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
│   ├── interface.py       # ApexCore interface
│   └── models.py          # ApexCoreFull, ApexCoreMini models
├── scheduler/             # Task scheduling system
│   └── scheduler.py       # ApexScheduler, ScheduledTask
├── prediction/            # Prediction engines
│   ├── expected_move.py   # Expected move predictor
│   ├── monster_runner.py  # MonsterRunner rare-event detection
│   ├── volatility_projection.py    # Volatility forecasting
│   ├── compression_forecast.py     # Compression cycle prediction
│   ├── continuation_estimator.py   # Trend continuation analysis
│   └── instability_predictor.py    # Early instability detection
├── server/                # API server
│   └── app.py             # FastAPI application
└── tests/                 # Test suite (44 tests)
    ├── test_determinism_golden_set.py
    ├── test_protocol_signatures.py
    ├── test_data_layer.py
    ├── test_apexlab_pipeline.py
    ├── test_monster_runner.py      # MR01-MR05 tests
    ├── test_server_health.py       # Server endpoint tests
    └── fixtures/                   # Golden test data
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
- **Ω5:** Signal suppression lock (strong suppression detected)

---

## Recent Changes

### 2025-11-28 — v8.1 Ecosystem Upgrade

**New Protocol Systems:**
- MonsterRunner protocols MR01-MR05 with deterministic rare-event detection
  - MR01: Compression Explosion Detector
  - MR02: Volume Anomaly Detector
  - MR03: Volatility Regime Shift Detector
  - MR04: Institutional Footprint Detector
  - MR05: Multi-Timeframe Alignment Detector
- MonsterRunnerLoader for protocol discovery and aggregation

**Prediction Stack:**
- `volatility_projection.py` — Forward volatility state forecasting
- `compression_forecast.py` — Compression/expansion cycle prediction
- `continuation_estimator.py` — Trend continuation probability
- `instability_predictor.py` — Early instability signal detection

**ApexCore Models:**
- `models.py` with ApexCoreFull (desktop, 3-20MB) and ApexCoreMini (lightweight)
- Teacher-student training architecture with sklearn MLP

**Infrastructure:**
- Scheduler module for automated research workflows
- Ω5 directive: Signal Suppression Lock added
- Test suite expanded to 44 passing tests
- Test fixtures with golden bars for determinism verification

**Design Decisions:**
- Desktop-only architecture confirmed (NO Android/mobile)
- All MR protocols use deterministic heuristics (no ML)
- Scheduler is research-only, no live trading automation

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
