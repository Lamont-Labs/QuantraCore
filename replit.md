# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade deterministic AI trading intelligence engine designed for desktop use. It features a complete offline learning ecosystem (ApexLab) and an on-device neural assistant model (ApexCore). This system unifies a deterministic and neural hybrid stack with an 80-protocol tier system, incorporating v9.0-A institutional hardening features for enhanced robustness and compliance. The project emphasizes determinism, fail-closed operations, local-only learning, and strict adherence to a desktop-only architecture, explicitly prohibiting mobile builds. All outputs are framed as structural probabilities, not trading advice, with mandatory compliance notes.

## User Preferences

- **Communication:** I prefer simple language and detailed explanations.
- **Workflow:** I want iterative development. Ask before making major changes.
- **Coding Style:** I prefer functional programming paradigms.
- **Restrictions:** Do not make changes to the folder `Z`. Do not make changes to the file `Y`.

## System Architecture

QuantraCore Apex follows a desktop-only architecture with a strong emphasis on determinism, local processing, and fail-closed mechanisms.

### Core Principles

- Determinism first, fail-closed always.
- No cloud dependencies, local-only learning.
- QuantraScore (0–100 range) is mandatory everywhere.
- Rule engine overrides AI always.
- Desktop-only (STRICT NO Android/mobile builds).

### Directory Structure

The project is organized into `src/quantracore_apex/` with distinct modules:
- **`core/`**: Houses the deterministic engine components (e.g., `ApexEngine`, `schemas`, `microtraits`, `quantrascore`, `verdict`, `entropy`, `drift`, `suppression`, `continuation`).
- **`protocols/`**: Contains 80 Tier Protocols (T01-T80) for various analyses (e.g., core, volatility, momentum, volume, pattern recognition, S/R, market context), 25 Learning Protocols (LP01-LP25) for label generation, MonsterRunner Protocols (MR01-MR05) for rare-event detection, and Omega Directives (Ω1-Ω5) for critical overrides.
- **`data_layer/`**: Manages data acquisition, normalization, caching, and hashing through adapters (e.g., Alpha Vantage, Polygon.io, Yahoo Finance, CSV bundle, synthetic) with multi-provider failover support.
- **`apexlab/`**: The offline training environment for ApexCore, including window building, feature extraction, label generation, and dataset construction.
- **`apexcore/`**: The neural model interface, containing `ApexCoreFull` (desktop) and `ApexCoreMini` (lightweight) models built with scikit-learn.
- **`scheduler/`**: Implements `ApexScheduler` for task scheduling in research workflows.
- **`prediction/`**: Contains various prediction engines for expected move, monster runner events, volatility, compression, continuation, and instability.
- **`server/`**: Hosts the FastAPI application.
- **`tests/`**: Comprehensive test suite covering determinism, protocol signatures, data layer, ApexLab pipeline, MonsterRunner, universal scanner, and server health (414 tests).
- **`config/`**: Configuration files including `symbol_universe.yaml` (7 market cap buckets), `scan_modes.yaml` (8 scan modes), and `mode.yaml` (research-only enforcement).

### Frontend Architecture (ApexDesk)

The user interface, ApexDesk, is built with React 19, Vite 7, and Tailwind CSS v4, utilizing TypeScript. It comprises standard components like a navigation sidebar (`LeftRail`), header, symbol universe table, and detail panel. The frontend integrates with the FastAPI backend via a Vite proxy.

### Key Technologies

- **Backend:** Python 3.11, FastAPI with Uvicorn.
- **Frontend:** React 19, Vite 7, Tailwind CSS v4, TypeScript.
- **ML:** scikit-learn (chosen over PyTorch for disk space efficiency).
- **Testing:** Pytest (backend), Vitest (frontend).
- **HTTP Client:** HTTPX.
- **Numerical:** NumPy, Pandas.

### Validation & Stress Testing Scripts

Located in `/scripts/`:
- **`zero_doubt_verification.sh`** - Full 13-stage validation with double-pass testing
- **`run_random_universe_scan.py`** - Random universe stress scan (up to full universe)
- **`run_nuclear_determinism.py`** - 10-cycle determinism validation
- **`run_scanner_all_modes.py`** - All 8 scan modes validation
- **`validate_apexcore_pipeline.py`** - ApexLab/ApexCore pipeline validation

### API Endpoints

The FastAPI server runs on port 5000 and provides endpoints for:
- System information (`/`, `/health`, `/api/stats`).
- UI dashboard (`/desk`).
- Analysis scans (`/scan_symbol`, `/scan_universe`).
- Detailed tracing (`/trace/{window_hash}`).
- Specific predictions and checks (`/monster_runner/{symbol}`, `/risk/assess/{symbol}`, `/signal/generate/{symbol}`).
- Portfolio management and OMS simulation (`/portfolio/status`, `/portfolio/heat_map`, `/oms/*`).

### Omega Directives

- **Ω1:** Hard safety lock (extreme risk tier).
- **Ω2:** Entropy override (chaotic entropy state).
- **Ω3:** Drift override (critical drift state).
- **Ω4:** Compliance override (always active).
- **Ω5:** Signal suppression lock (strong suppression detected).

## External Dependencies

- **Data Providers:**
    - Alpha Vantage
    - Polygon.io (for real market data, requires `POLYGON_API_KEY`)
    - Synthetic adapter (for testing without live API keys)
- **Frameworks/Libraries (included in Key Technologies but listed here for completeness as external dependencies):**
    - FastAPI
    - Uvicorn
    - React
    - Vite
    - Tailwind CSS
    - scikit-learn
    - Pytest
    - Vitest
    - HTTPX
    - NumPy
    - Pandas