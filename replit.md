# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates an offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore) to deliver sophisticated, AI-driven trading. The system supports all trading types (long, short, intraday, swing, scalping) primarily through Alpaca paper trading, emphasizing determinism, accuracy optimization, and self-learning. It aims to provide a robust, self-improving platform with advanced risk management and comprehensive reporting for both retail and institutional users.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### UI/UX Decisions
The frontend, built with React 18.2, Vite 7.2, and Tailwind CSS 4.0, employs an institutional trading terminal aesthetic and a custom design system defined by `@theme` blocks. The ApexDesk Dashboard features 15 real-time panels for monitoring and control, offering Standard (30s), High Velocity (5s), and Turbo (2s) refresh rates.

### Technical Implementations
- **Backend:** Python 3.11, FastAPI, Uvicorn (port 8000).
- **Frontend:** React 18.2, Vite 7.2, Tailwind CSS 4.0, TypeScript (port 5000).
- **Machine Learning:** `scikit-learn` (GradientBoosting), `joblib`, `NumPy`, `Pandas`.
- **Testing:** `pytest` (backend), `vitest` (frontend).
- **API Endpoints:** REST APIs for trading, data, scanning, health, and reporting.
- **Security:** `X-API-Key` authentication and restrictive CORS.
- **Performance:** ORJSONResponse, GZipMiddleware, 4-worker Uvicorn, expanded TTL caches, module-level ML model caching, Alpaca client caching, parallel universe scanning, prediction result caching.

### Feature Specifications
- **Full Trading Capabilities:** Supports various trade types with configurable risk limits and order types (MARKET, LIMIT, STOP, STOP_LIMIT). Includes auto-selected entries and comprehensive exits.
- **Deterministic Core:** Ensures consistent outputs for identical inputs.
- **Offline ML (ApexCore V3/V4):** On-device neural models with 7 (V3) or 16 (V4) prediction heads, trained on real market data with simulation-based data augmentation.
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, and auto-retraining.
- **Self-Learning (Alpha Factory & Continuous Learning System):** 24/7 live research loop via WebSockets for feedback and autonomous retraining.
- **Signal & Alert Services:** Manual Trading Signal Service, Twilio SMS alerts, and browser-based Push Notifications with configurable thresholds.
- **Low-Float Runner Screener:** Real-time scanner for penny stocks with volume surge and momentum detection, integrating ApexCore predictions.
- **Protocol System:** Extensive suite of Tier, Learning, and MonsterRunner protocols, plus Omega Directives.
- **Investor Reporting & Logging:** Comprehensive logging for paper trades, due diligence, performance metrics, and ML model training (all stored in `investor_logs/`). Includes automated daily compliance attestations and an Investor Data Exporter.
- **Automated Swing Trade Execution (AutoTrader):** Autonomous system for setup scanning, position sizing, and market order execution on Alpaca paper trading.
- **Model Management:** Hot Model Reload System (ModelManager) and Dual-Phase Incremental Learning (IncrementalTrainer) for efficient model updates and knowledge retention.
- **Trade Hold Manager:** Continuation probability-based system analyzing active positions for dynamic hold decisions.
- **Extended Market Hours Trading:** Full support for pre-market, regular, and after-hours trading.

### System Design Choices
- **Broker Layer:** Supports `NullAdapter`, `PaperSimAdapter`, and `AlpacaPaperAdapter`.
- **Data Layer:** Polygon.io for market data, Alpaca for execution.
- **Configuration:** Parameters managed via `config/data_sources.yaml`, `config/broker.yaml`, and `config/scan_modes.yaml`.
- **Google Docs Integration:** Uses Replit OAuth2 for automated reporting.
- **Database Model Persistence:** PostgreSQL-backed storage for ML models with GZIP compression, version history, and rollback capability.

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading and order execution.
- **Polygon.io (Developer tier):** Primary data source for market data (OHLCV, ticks, quotes), ML training, and extended hours. Requires `POLYGON_API_KEY`.
- **Binance:** Primary source for crypto data in Alpha Factory.
- **Twilio:** Used for SMS alerts.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting.

### Required Secrets
- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_API_SECRET`
- `POLYGON_API_KEY`
- `POLYGON_TIER` (developer recommended)
- `TWILIO_*` (managed by Replit integration)