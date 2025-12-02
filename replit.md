# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates an offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore V4) to deliver sophisticated, AI-driven trading. The system supports all trading types (long, short, intraday, swing, scalping) primarily through Alpaca paper trading, emphasizing determinism, accuracy optimization, and self-learning. It provides a robust, self-improving platform with advanced risk management, database-backed ML persistence, and comprehensive reporting for both retail and institutional users.

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
- **Backend:** Python 3.11, FastAPI, Uvicorn (port 8000, 4 workers).
- **Frontend:** React 18.2, Vite 7.2, Tailwind CSS 4.0, TypeScript (port 5000).
- **Machine Learning:** `scikit-learn` (GradientBoosting), `joblib`, `NumPy`, `Pandas`.
- **Database:** PostgreSQL for ML model persistence with GZIP compression.
- **Testing:** `pytest` (backend), `vitest` (frontend).
- **API Endpoints:** REST APIs for trading, data, scanning, health, model management, and reporting.
- **Security:** `X-API-Key` authentication and restrictive CORS.
- **Performance:** ORJSONResponse, GZipMiddleware, 4-worker Uvicorn, expanded TTL caches, module-level ML model caching, Alpaca client caching, parallel universe scanning, prediction result caching.

### Feature Specifications
- **Full Trading Capabilities:** Supports various trade types with configurable risk limits and order types (MARKET, LIMIT, STOP, STOP_LIMIT). Includes auto-selected entries and comprehensive exits.
- **Deterministic Core:** Ensures consistent outputs for identical inputs.
- **Offline ML (ApexCore V4):** On-device neural models with 16 prediction heads, trained on real market data with simulation-based data augmentation.
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, and auto-retraining.
- **Self-Learning (Alpha Factory & Continuous Learning System):** 24/7 live research loop via WebSockets for feedback and autonomous retraining.
- **Signal & Alert Services:** Manual Trading Signal Service, Twilio SMS alerts, and browser-based Push Notifications with configurable thresholds.
- **Low-Float Runner Screener:** Real-time scanner for penny stocks with volume surge and momentum detection, integrating ApexCore predictions.
- **Protocol System:** Extensive suite of Tier, Learning, and MonsterRunner protocols, plus Omega Directives.
- **Investor Reporting & Logging:** Comprehensive logging for paper trades, due diligence, performance metrics, and ML model training (all stored in `investor_logs/`). Includes automated daily compliance attestations and an Investor Data Exporter.
- **Automated Swing Trade Execution (AutoTrader):** Autonomous system for setup scanning, position sizing, and market order execution on Alpaca paper trading.
- **Model Management:** Hot Model Reload System (ModelManager) and Dual-Phase Incremental Learning (IncrementalTrainer) for efficient model updates and knowledge retention.
- **Database Model Persistence:** PostgreSQL-backed storage for ML models with GZIP compression, version history, rollback capability, and atomic active-version management.
- **Trade Hold Manager:** Continuation probability-based system analyzing active positions for dynamic hold decisions.
- **Extended Market Hours Trading:** Full support for pre-market (4:00 AM - 9:30 AM ET), regular (9:30 AM - 4:00 PM ET), and after-hours (4:00 PM - 8:00 PM ET) trading.
- **Multi-Source Data Ingestion:** Options flow, sentiment analysis, Level 2 data, dark pool activity, economic indicators, and alternative data feeds.

### System Design Choices
- **Broker Layer:** Supports `NullAdapter`, `PaperSimAdapter`, and `AlpacaPaperAdapter`.
- **Data Layer:** Polygon.io for market data, Alpaca for execution, Binance for crypto.
- **Configuration:** Parameters managed via `config/data_sources.yaml`, `config/broker.yaml`, and `config/scan_modes.yaml`.
- **Google Docs Integration:** Uses Replit OAuth2 for automated reporting.
- **Database Model Persistence:** PostgreSQL-backed storage for ML models with GZIP compression, version history, and rollback capability. Models automatically load from database on startup and save after training.

## Database Model Persistence System

### Architecture
The DatabaseModelStore provides persistent ML model storage that survives republishes:

| Component | Description |
|-----------|-------------|
| **ml_models table** | Stores compressed model components (pkl files) with version tracking |
| **ml_model_versions table** | Tracks version metadata, training samples, manifests, and active status |
| **GZIP Compression** | All model data compressed before storage (60-80% reduction) |
| **Atomic Version Management** | New versions atomically deactivate old versions |
| **File Fallback** | Graceful fallback to file storage if database unavailable |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/model/storage` | GET | Database connection status and storage statistics |
| `/model/storage/versions` | GET | List all model version history with metadata |
| `/model/storage/restore/{version_id}` | POST | Rollback to a previous model version |
| `/model/storage/migrate` | POST | Migrate models from file storage to database |

### Version Tracking
- Each training run creates a new version with timestamp and sample count
- Previous versions preserved for rollback capability
- Active version flag managed atomically during save operations
- Full manifest (head configurations, training metrics) stored with each version

## Push Notification System

### Web Push Notifications
Browser-based push notifications using Web Push Protocol (RFC 8030):

| Component | Description |
|-----------|-------------|
| **Service Worker** | `dashboard/public/sw.js` handles push events and displays notifications |
| **VAPID Keys** | Server-generated keys stored in config for authentication |
| **Subscription Storage** | Client subscriptions stored for targeted push delivery |
| **pywebpush** | Python library for sending push notifications to subscribed browsers |

### Alert Thresholds
- QuantraScore threshold for signal alerts (default: 60+)
- Monster runner probability alerts (default: 70%+)
- Volume surge notifications for screener symbols

### Dual Alert Channels
1. **Twilio SMS:** Traditional SMS alerts for critical signals
2. **Web Push:** Free browser-based notifications for real-time updates

## ApexCore V4 Neural Model

### 16 Prediction Heads

| Head | Type | Output |
|------|------|--------|
| quantrascore | Regression | 0-100 composite score |
| runner | Binary | Monster runner probability |
| quality | Multiclass | Quality tier (0-4) |
| avoid | Binary | Avoid signal |
| regime | Multiclass | Market regime (0-4) |
| timing | Multiclass | Move timing bucket (0-4) |
| runup | Regression | Expected price appreciation |
| direction | Binary | Next-bar direction |
| volatility | Regression | Expected volatility |
| momentum | Regression | Momentum strength |
| support | Regression | Support level proximity |
| resistance | Regression | Resistance level proximity |
| volume | Regression | Expected volume ratio |
| reversal | Binary | Reversal probability |
| breakout | Binary | Breakout probability |
| continuation | Regression | Trend continuation probability |

## Multi-Source Data Ingestion

### Data Feeds

| Source | Description | Endpoint |
|--------|-------------|----------|
| Options Flow | Premium options activity, unusual volume | `/api/data/options-flow` |
| Sentiment | Social media and news sentiment analysis | `/api/data/sentiment/summary` |
| Dark Pool | Institutional off-exchange activity | `/api/data/dark-pool/summary` |
| Level 2 | Order book depth and bid/ask spread | `/api/data/level2/{symbol}` |
| Economic | Macro indicators and economic calendar | `/api/data/macro/summary` |
| Alternative | Alternative data sources and signals | `/api/data/alternative` |

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading and order execution.
- **Polygon.io (Developer tier):** Primary data source for market data (OHLCV, ticks, quotes), ML training, and extended hours. Requires `POLYGON_API_KEY`.
- **Binance:** Primary source for crypto data in Alpha Factory.
- **Twilio:** Used for SMS alerts.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting.
- **PostgreSQL:** Database for ML model persistence (Replit-managed).

### Required Secrets
- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_API_SECRET`
- `POLYGON_API_KEY`
- `POLYGON_TIER` (developer recommended)
- `TWILIO_*` (managed by Replit integration)
- `DATABASE_URL` (auto-provided by Replit PostgreSQL)

## Hyperspeed Learning System

### Overview
The Hyperspeed Learning System accelerates ML training by 1000x through:
- Historical data replay at maximum speed
- Parallel battle simulations (100+ per sample)
- Multi-source data fusion
- Overnight intensive training cycles

### Components

| Component | Description |
|-----------|-------------|
| **HyperspeedEngine** | Main orchestrator coordinating all learning components |
| **HistoricalReplayEngine** | Replays years of market data at 1000x speed |
| **ParallelBattleCluster** | Runs 100+ simulated trades per sample across 8 strategies |
| **MultiSourceAggregator** | Fuses Polygon, Alpaca, options flow, dark pool, sentiment data |
| **OvernightScheduler** | Coordinates intensive learning during off-market hours |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hyperspeed/status` | GET | Get engine status and metrics |
| `/hyperspeed/metrics` | GET | Get aggregate learning metrics |
| `/hyperspeed/replay` | POST | Start historical replay (5 years at 1000x) |
| `/hyperspeed/battle` | POST | Run parallel battle simulations |
| `/hyperspeed/cycle` | POST | Run full hyperspeed learning cycle |
| `/hyperspeed/overnight/start` | POST | Start overnight intensive mode |
| `/hyperspeed/overnight/stop` | POST | Stop overnight mode |
| `/hyperspeed/strategies` | GET | Get strategy performance metrics |
| `/hyperspeed/samples` | GET | Get training sample statistics |
| `/hyperspeed/train` | POST | Trigger model training |

### Simulation Strategies
8 parallel strategies for comprehensive learning:
- Conservative, Moderate, Aggressive
- Scalping, Swing, Contrarian
- Momentum, Mean Reversion

### Acceleration Metrics
- Days equivalent: Years of market experience compressed
- Acceleration factor: Real-time multiplier (target: 1000x)
- Peak speed: Maximum acceleration achieved

## Recent Changes

### December 2025
- **Hyperspeed Learning System**: Full implementation with historical replay, battle simulations, multi-source aggregation, overnight scheduler
- **16 API endpoints** for hyperspeed learning control
- **HyperspeedPanel** dashboard component for monitoring and control
- Implemented database-backed ML model persistence with PostgreSQL
- Added version history and rollback capability for trained models
- Fixed active version tracking with atomic deactivation of old versions
- Added lazy initialization for DATABASE_URL in multi-worker setups
- Integrated database persistence into UnifiedTrainer pipeline
- Added 4 model management API endpoints
- Implemented web push notification system with VAPID authentication
- Upgraded to ApexCore V4 with 16 prediction heads
- Added multi-source data ingestion (options flow, sentiment, dark pool, etc.)
- Expanded dashboard to 15 real-time panels
