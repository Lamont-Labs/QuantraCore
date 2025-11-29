# QuantraCore Apex — Replit Project Documentation (Compressed)

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system is strictly for **research and backtesting only**, providing structural probabilities rather than trading advice. Key capabilities include a deterministic core with 80 Tier protocols, a universal scanner, and comprehensive regulatory compliance testing.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Core Principles

The architecture emphasizes determinism, fail-closed operations, and local-only learning with no cloud dependencies. All outputs include a QuantraScore (0–100), and a rule engine consistently overrides AI decisions. The system is desktop-only and enforces a research-only mode.

### UI/UX Decisions

The frontend is built with React 18, Vite 5, and Tailwind CSS 3, providing a modern and responsive user interface for the ApexDesk desktop application.

### Technical Implementations

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Frontend:** React 18.2, Vite 5, Tailwind CSS 3.4, TypeScript
- **Machine Learning:** scikit-learn
- **Testing:** pytest (backend), vitest (frontend)
- **Numerical:** NumPy, Pandas

### Feature Specifications

- **ApexEngine:** The deterministic core, implementing 80 Tier protocols, 25 Learning Protocols, and 20 MonsterRunner protocols.
- **ApexLab (V1 & V2):** An offline training environment supporting 40+ field schema for feature and label generation.
- **ApexCore (V1 & V2):** On-device neural models with 5 prediction heads, utilizing scikit-learn and manifest verification for model integrity.
- **PredictiveAdvisor:** A fail-closed engine integrated with ApexCore for predictive insights.
- **Estimated Move Module:** Statistical move range analysis with 4 mandatory safety gates.
- **Broker Layer v1:** Institutional execution engine with pluggable broker adapters and a 9-check risk engine. LIVE trading is disabled by default.
- **Entry/Exit Optimization Engine (EEO):** Calculates best entry zones, exit targets, stops, and position sizing using deterministic and model-assisted methods.
- **Universal Scanner:** Supports 7 market cap buckets and 8 scan modes.
- **Omega Directives (Ω1-Ω20):** Twenty safety override protocols, including a nuclear killswitch and a permanent research-only compliance mode (Ω4).
- **Regulatory Compliance:** Over 1,145 tests, including 163+ dedicated regulatory tests exceeding industry standards.
- **Hardening Infrastructure:** Global hardening system implementing fail-closed behavior, protocol manifest verification, config validation, mode enforcement, incident logging, and kill switch management.
- **Alpha Factory:** A 24/7 live research engine streaming real-time data, tracking simulated portfolio performance, and integrating a self-learning feedback loop for continuous model improvement.
- **MarketSimulator:** A synthetic market simulation engine for stress-testing and accelerated learning through 8 chaos scenarios.
- **Universal Broker Router:** Manages routing to various paper and live trading brokers via environment variables, with robust safety controls.

## External Dependencies

- **Data Providers:**
    - Polygon.io
    - Alpha Vantage (optional)
    - Yahoo Finance (backup)
    - CSV Bundle (historical data import)
    - Synthetic (for testing)
- **Broker Integration (Paper Trading Only):**
    - Alpaca Paper
    - PaperSim (internal)
- **Google Docs Integration:**
    - Connected via Replit OAuth2 connector
    - Scopes: `docs`, `documents`, `documents.readonly`
    - **Automated Export Pipeline** (NEW):
        - Investor reports (daily/weekly/monthly)
        - Due diligence packages for acquirers
        - Trade log exports
        - Trade journals with research notes
        - Monthly investor updates
    - **API Endpoints:**
        - `GET /google-docs/status` - Check connection
        - `POST /google-docs/export/investor-report` - Generate investor report
        - `POST /google-docs/export/due-diligence` - Generate DD package
        - `POST /google-docs/export/trade-log` - Export trade history
        - `GET /google-docs/documents` - List exported documents
        - `POST /google-docs/journal/entry` - Add journal entry
        - `GET /google-docs/journal/today` - Get today's journal
        - `POST /google-docs/investor-update/monthly` - Monthly update
- **Environment Variables:**
    - `POLYGON_API_KEY`
    - `ALPHA_VANTAGE_API_KEY`
    - `ALPACA_PAPER_API_KEY`
    - `ALPACA_PAPER_API_SECRET`

## Recent Changes

- **2025-11-29:** Added automated Google Docs export pipeline for investor/acquirer reporting
  - Created `automated_pipeline.py` with performance metrics collection
  - Added 9 new API endpoints for document export
  - Exports include live trading data, broker snapshots, ML training progress
  - Successfully tested with live export to Google Docs