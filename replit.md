# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade deterministic AI trading intelligence engine designed exclusively for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system is strictly for **research and backtesting only**, providing structural probabilities rather than trading advice. Key capabilities include a deterministic core with 80 Tier protocols, a universal scanner, and comprehensive regulatory compliance testing that exceeds industry standards.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Core Principles

The architecture prioritizes determinism, fail-closed operations, and local-only learning with no cloud dependencies. All outputs include a QuantraScore (0–100), and a rule engine consistently overrides AI decisions. The system is strictly desktop-only and enforces a research-only mode via `config/mode.yaml`.

### Key Technologies

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Frontend:** React 18.2, Vite 5, Tailwind CSS 3.4, TypeScript
- **Machine Learning:** scikit-learn (chosen for disk space efficiency)
- **Testing:** pytest (backend), vitest (frontend)
- **Numerical:** NumPy, Pandas

### Feature Specifications

- **ApexEngine:** The deterministic core, implementing 80 Tier protocols (T01-T80), 25 Learning Protocols (LP01-LP25), and 5 MonsterRunner protocols (MR01-MR05).
- **ApexLab (V1 & V2):** An offline training environment supporting 40+ field schema for feature and label generation.
- **ApexCore (V1 & V2):** On-device neural models (Big/Mini) with 5 prediction heads, utilizing scikit-learn. Includes manifest verification for model integrity.
- **PredictiveAdvisor:** A fail-closed engine integrated with ApexCore for predictive insights.
- **Estimated Move Module:** Statistical move range analysis with 4 mandatory safety gates, providing percentile-based distributions for research purposes. See [`docs/ESTIMATED_MOVE_SPEC.md`](docs/ESTIMATED_MOVE_SPEC.md).
- **Universal Scanner:** Supports 7 market cap buckets and 8 scan modes for comprehensive market analysis.
- **Omega Directives (Ω1-Ω5):** Five safety override protocols, with Ω4 enforcing a permanent research-only compliance mode.
- **Regulatory Compliance:** Over 989 tests, including 163+ dedicated regulatory tests that exceed SEC/FINRA/MiFID II/Basel requirements by 2x-5x, ensuring institutional-grade safety margins. This includes determinism verification, stress testing, market abuse detection, and risk controls.

### UI/UX Decisions

The frontend is built with React 18, Vite 5, and Tailwind CSS 3, providing a modern and responsive user interface for the ApexDesk desktop application.

### API Endpoints

The FastAPI backend exposes endpoints for health checks, system statistics, single and batch symbol scanning, protocol tracing, extreme move checks, risk assessments, signal generation, and portfolio status.

## External Dependencies

- **Data Providers:**
    - Polygon.io (requires `POLYGON_API_KEY`)
    - Alpha Vantage (optional, requires `ALPHA_VANTAGE_API_KEY`)
    - Yahoo Finance (backup)
    - CSV Bundle (historical data import)
    - Synthetic (for testing purposes)

- **Environment Variables:**
    - `POLYGON_API_KEY`: Required for Polygon.io API access.
    - `ALPHA_VANTAGE_API_KEY`: Optional, for Alpha Vantage API access.

## Future Roadmap

### ApexVision Multi-Modal Upgrade (v9.x → v10.x)

Full specification: [`docs/APEXVISION_UPGRADE_SPEC.md`](docs/APEXVISION_UPGRADE_SPEC.md)

| Component | Purpose |
|-----------|---------|
| ApexVision | Visual Pattern Intelligence Engine |
| ApexLab_Vision | Chart-Image Label Factory |
| ApexCore_VisionFusion | Multi-Modal Neural Model |
| Vision Pattern Dictionary | 25+ deterministic visual patterns |
| QuantraVision v5 | Android real-time structural copilot |

### ApexLab Continuous Data Engine

Full specification: [`docs/APEXLAB_CONTINUOUS_DATA_ENGINE_SPEC.md`](docs/APEXLAB_CONTINUOUS_DATA_ENGINE_SPEC.md)

| Component | Purpose |
|-----------|---------|
| ApexLabScheduler | Central orchestrator for periodic jobs |
| MarketDataIngestor | Multi-API OHLCV ingestion (Polygon, AlphaVantage, etc.) |
| ChartRendererBatcher | High-volume chart image rendering (10k-100k/day) |
| EventLabeler | Teacher-mode deterministic labeling |
| TrainingLauncher | Weekly/monthly model training scheduler |
| EvaluationAndPromotion | Metric-based model promotion system |

**Volume Targets:** 10k–100k+ labeled events/day depending on hardware tier.

### Investor Due Diligence Requirements

Full specification: [`docs/INVESTOR_DUE_DILIGENCE_REQUIREMENTS.md`](docs/INVESTOR_DUE_DILIGENCE_REQUIREMENTS.md)

| Category | Key Items |
|----------|-----------|
| Technical Infrastructure | 500-700+ tests, CI/CD, API docs, build scripts |
| Documentation | Master spec, architecture diagrams, protocol index |
| Model Artifacts | Manifests, calibration curves, determinism proof |
| Product Proof | Screenshots, demo videos, end-to-end pipeline |
| Compliance | Safety policies, fail-closed guarantees, disclaimers |
| Business | Pitch deck, market analysis, roadmap, exit paths |
| Security & Legal | Ownership, licensing, threat modeling |
| Founder Package | Profile, vision, execution philosophy, progress proof |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*