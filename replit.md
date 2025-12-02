# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates an offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore V4) to deliver sophisticated, AI-driven trading. The system supports all trading types (long, short, intraday, swing, scalping) primarily through Alpaca paper trading, emphasizing determinism, accuracy optimization, and self-learning. Its purpose is to provide a robust, self-improving platform with advanced risk management, database-backed ML persistence, and comprehensive reporting for both retail and institutional users.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### UI/UX Decisions
The frontend, built with React 18.2, Vite 7.2, and Tailwind CSS 4.0, employs an institutional trading terminal aesthetic and a custom design system. The ApexDesk Dashboard features 15 real-time panels for monitoring and control, offering Standard (30s), High Velocity (5s), and Turbo (2s) refresh rates. Frontend performance is optimized with request throttling and lazy loading.

### Technical Implementations
- **Backend:** Python 3.11, FastAPI, Uvicorn (port 8000, 4 workers).
- **Frontend:** React 18.2, Vite 7.2, Tailwind CSS 4.0, TypeScript (port 5000).
- **Machine Learning:** `scikit-learn` (GradientBoosting), `joblib`, `NumPy`, `Pandas`. ApexCore V4 features 16 prediction heads.
- **Database:** PostgreSQL for ML model persistence with GZIP compression, version history, and rollback.
- **Testing:** `pytest` (backend), `vitest` (frontend).
- **API Endpoints:** REST APIs for trading, data, scanning, health, model management, reporting, and hyperspeed learning control.
- **Security:** `X-API-Key` authentication and restrictive CORS.
- **Performance:** ORJSONResponse, GZipMiddleware, 4-worker Uvicorn, expanded TTL caches, module-level ML model caching, Alpaca client caching, parallel universe scanning, prediction result caching.

### Feature Specifications
- **Full Trading Capabilities:** Supports various trade types with configurable risk limits and order types.
- **Deterministic Core:** Ensures consistent outputs for identical inputs.
- **Offline ML (ApexCore V4):** On-device neural models with 16 prediction heads.
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, and auto-retraining.
- **Self-Learning (Alpha Factory & Continuous Learning System):** 24/7 live research loop via WebSockets.
- **Hyperspeed Learning System:** Accelerates ML training through historical data replay (1000x speed), parallel battle simulations, multi-source data fusion, and overnight intensive training. Includes robust thread monitoring and fallback data adapters.
- **Signal & Alert Services:** Manual Trading Signal Service, Twilio SMS alerts, and browser-based Push Notifications.
- **Low-Float Runner Screener:** Real-time scanner for penny stocks.
- **Protocol System:** Extensive suite of trading protocols.
- **Investor Reporting & Logging:** Comprehensive logging for paper trades, performance metrics, and ML model training (stored in `investor_logs/`). Includes automated daily compliance attestations.
- **Automated Swing Trade Execution (AutoTrader):** Autonomous setup scanning, position sizing, and market order execution.
- **Model Management:** Hot Model Reload System (ModelManager) and Dual-Phase Incremental Learning (IncrementalTrainer).
- **Trade Hold Manager:** Continuation probability-based system for active positions.
- **Extended Market Hours Trading:** Full support for pre-market, regular, and after-hours trading.
- **Multi-Source Data Ingestion:** Options flow, sentiment analysis, Level 2 data, dark pool activity, economic indicators, and alternative data feeds.

### System Design Choices
- **Broker Layer:** Supports `NullAdapter`, `PaperSimAdapter`, and `AlpacaPaperAdapter`.
- **Data Layer:** Polygon.io for market data, Alpaca for execution, Binance for crypto.
- **Configuration:** Parameters managed via `config/data_sources.yaml`, `config/broker.yaml`, and `config/scan_modes.yaml`.
- **Google Docs Integration:** Uses Replit OAuth2 for automated reporting.

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading and order execution.
- **Polygon.io (Developer tier):** Primary data source for market data, ML training, and extended hours.
- **Binance:** Primary source for crypto data in Alpha Factory.
- **Twilio:** Used for SMS alerts.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting.
- **PostgreSQL:** Database for ML model persistence (Replit-managed).