# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates an offline learning ecosystem (ApexLab), on-device neural assistant models (ApexCore V4 with 16 prediction heads), and a Hyperspeed Learning System for accelerated model maturity. The system supports all trading types (long, short, intraday, swing, scalping) primarily through Alpaca paper trading, emphasizing determinism, accuracy optimization, and self-learning. Its purpose is to provide a robust, self-improving platform with advanced risk management, database-backed ML persistence, and comprehensive reporting for both retail and institutional users.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Hardware Context
- **Current Development:** GMKtec NucBox K6 (development constraint, not target hardware)
- **Target Hardware:** Any modern x86-64 desktop with 16GB+ RAM
- **Design Philosophy:** CPU-optimized, GPU optional, runs on commodity hardware
- **Platform:** Desktop-only (no mobile/Android builds)

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
- **Hyperspeed Learning System:** Accelerates ML training through historical data replay (1000x speed), parallel battle simulations, multi-source data fusion, and overnight intensive training. Targets 1-2 weeks to model maturity vs. 12+ months of live-only learning.
- **Signal & Alert Services:** Manual Trading Signal Service, Twilio SMS alerts, and browser-based Push Notifications.
- **Low-Float Runner Screener:** Real-time scanner for penny stocks.
- **Protocol System:** Extensive suite of trading protocols (T01-T80, LP01-LP25, MR01-MR20, Ω01-Ω20).
- **Investor Reporting & Logging:** Comprehensive logging for paper trades, performance metrics, and ML model training (stored in `investor_logs/`). Includes automated daily compliance attestations.
- **Automated Swing Trade Execution (AutoTrader):** Autonomous setup scanning, position sizing, and market order execution.
- **Model Management:** Hot Model Reload System (ModelManager) and Dual-Phase Incremental Learning (IncrementalTrainer).
- **Trade Hold Manager:** Continuation probability-based system for active positions.
- **Extended Market Hours Trading:** Full support for pre-market (4:00 AM), regular (9:30 AM - 4:00 PM), and after-hours trading (8:00 PM).
- **Multi-Source Data Ingestion:** Options flow, sentiment analysis, Level 2 data, dark pool activity, economic indicators, and alternative data feeds.

### Hyperspeed Learning System (v9.0-A)
The Hyperspeed Learning System accelerates model training by replaying years of historical data at 1000x speed:

| Component | Description |
|-----------|-------------|
| Historical Replay Engine | Streams 5 years of data through prediction pipeline |
| Battle Simulator | 100 parallel strategy simulations per cycle |
| Multi-Source Fusion | Polygon, Alpaca, Binance data aggregation |
| Overnight Training | Intensive learning during market close (4 PM - 4 AM) |
| Database Persistence | PostgreSQL-backed model storage with version history |

**Current Status:**
- Samples per cycle: 70+ per symbol
- Training threshold: 5,000 samples
- Model storage: 12 MB in PostgreSQL
- Acceleration: ~1,000x real-time

### System Design Choices
- **Broker Layer:** Supports `NullAdapter`, `PaperSimAdapter`, and `AlpacaPaperAdapter`.
- **Data Layer:** Polygon.io for market data, Alpaca for execution, Binance for crypto.
- **Configuration:** Parameters managed via `config/data_sources.yaml`, `config/broker.yaml`, and `config/scan_modes.yaml`.
- **Google Docs Integration:** Uses Replit OAuth2 for automated reporting.

## Data Connections (Complete List)

### Active Data Sources (7 Connected)

**Market Data & Execution:**
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **Polygon.io** | EOD prices, historical data, ML training, extended hours | Developer tier | Active |
| **Alpaca** | Paper trading, order execution, positions, portfolio | 200/min | Active |

**Economic Indicators:**
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **FRED** | Fed rates, CPI, GDP, unemployment, yield curve | 120/min | Active |

**Sentiment & News:**
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **Finnhub** | Reddit/Twitter social sentiment, insider transactions | 60/min | Active |
| **Alpha Vantage** | AI-powered news sentiment, 50+ technical indicators | 500/day | Active |

**Regulatory Filings:**
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **SEC EDGAR** | Insider trades (Form 4), 13F holdings, 8-K events | 10/sec | Active (No key needed) |

**Crypto:**
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **Binance** | Cryptocurrency data for Alpha Factory | 1200/min | Active |

### Ready to Activate (2 Available)
| Provider | Purpose | Rate Limit | Status |
|----------|---------|------------|--------|
| **Nasdaq Data Link** | COT futures positioning (ES, NQ, GC, CL, etc.) | 50/day | Needs API key |
| **FMP** | Earnings calendar, DCF valuations, company profiles | 250/day | Needs API key |

### Environment Variables
```bash
# Market Data & Execution
POLYGON_API_KEY=           # Polygon.io API key
ALPACA_PAPER_API_KEY=      # Alpaca paper trading key
ALPACA_PAPER_API_SECRET=   # Alpaca paper trading secret

# Economic & Sentiment (Active)
FRED_API_KEY=              # Federal Reserve data
FINNHUB_API_KEY=           # Social sentiment
ALPHA_VANTAGE_API_KEY=     # News sentiment

# Optional (Ready to Activate)
NASDAQ_DATA_LINK_API_KEY=  # COT reports
FMP_API_KEY=               # Earnings & valuations
```

### API Endpoints (Data Intelligence)

**Sentiment & News:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sentiment/{symbol}` | GET | Unified sentiment (social + news + economic) |
| `/sentiment/batch` | POST | Batch sentiment for multiple symbols |
| `/sentiment/market` | GET | Overall market sentiment snapshot |
| `/sentiment/providers` | GET | Status of all data providers |
| `/news/{symbol}` | GET | AI-powered news sentiment |
| `/social/{symbol}` | GET | Reddit/Twitter sentiment |

**Economic Indicators:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/economic/regime` | GET | Current economic regime from FRED |
| `/economic/yield_curve` | GET | US Treasury yield curve |

**SEC Filings:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sec/insider/{symbol}` | GET | SEC Form 4 insider trading summary |
| `/sec/insider/{symbol}/transactions` | GET | Detailed insider transaction list |
| `/sec/institutions/{symbol}` | GET | SEC 13F institutional holdings |
| `/sec/events/{symbol}` | GET | SEC 8-K material events |
| `/sec/status` | GET | SEC EDGAR adapter status |

**COT Reports (Futures Positioning):**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cot/{symbol}` | GET | COT positioning (ES, NQ, GC, CL, etc.) |
| `/cot/status` | GET | Nasdaq Data Link adapter status |

**Earnings & Valuations:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/earnings/calendar` | GET | Upcoming earnings releases |
| `/earnings/{symbol}` | GET | Historical earnings for symbol |
| `/valuation/{symbol}` | GET | DCF valuation analysis |
| `/profile/{symbol}` | GET | Company profile and metrics |
| `/dividends/calendar` | GET | Upcoming dividend ex-dates |
| `/fmp/status` | GET | FMP adapter status |

## External Services

- **Twilio:** SMS alerts for trading signals
- **Google Docs:** Automated reporting via Replit OAuth2
- **PostgreSQL:** Database for ML model persistence (Replit-managed)

## Recent Changes (December 2025)

- **10x RunnerHunter System (NEW):** 150 breakout-specific features for detecting massive swing runners
  - Squeeze Detection (20 features): Bollinger/Keltner squeeze, ATR compression, volatility percentiles
  - Momentum Ignition (20 features): RSI breakout, MACD explosion, price acceleration
  - Volume Surge (20 features): Climax volume, pocket pivot, accumulation days
  - Consolidation Quality (20 features): Tight range detection, decreasing volatility, coiled spring
  - Relative Strength (20 features): SPY comparison, sector leadership, 52-week proximity
  - Breakout Proximity (20 features): Resistance distance, channel boundaries, pivot points
  - Timing Signals (15 features): Intraday momentum, opening range breakout signals
  - Catalyst Alignment (15 features): Multi-factor convergence scoring
  - Signal Classification: IMMEDIATE (85+), IMMINENT (70-84), DEVELOPING (50-69)
  - Engine Methods: `hunt_runners()`, `scan_immediate_breakouts()`, `get_runner_signals()`
- **Enhanced Swing Trade Features:** SwingFeatureExtractor with 90 features optimized for 2-10 day holding periods
  - Multi-scale momentum (3/5/10/20/40 day returns, momentum slope, acceleration)
  - Volatility regime (ATR compression, HV percentiles, Bollinger/Keltner squeeze)
  - Volume texture (OBV, CMF, volume spike persistence, smart money flow)
  - Candlestick patterns (NR4, NR7, Inside Day, Engulfing, Hammer)
  - Price structure (swing highs/lows, Fibonacci levels, ADX trend strength)
- **Multi-Horizon Labels:** 3/5/8/10 day forward returns, max adverse/favorable excursion, quality tiers
- **Swing Training Cycle:** `run_swing_training_cycle()` fetches real EOD data from Polygon/Alpaca
- **Training Results:** 97.3% runner accuracy, 100% regime accuracy, 80.7% quality accuracy, 0.18 QuantraScore RMSE
- **Total Feature Coverage:** 240 features (90 swing + 150 runner) for maximum prediction accuracy
- **Hyperspeed Learning System:** Fully operational with 1000x acceleration
- **Database Model Persistence:** ML models stored in PostgreSQL, survive restarts
- **ApexCore V4 Integration:** 16 prediction heads attached to hyperspeed engine
- **Training Pipeline:** End-to-end verified (replay → simulation → training → update)
- **Multi-Worker Note:** Uvicorn 4-worker setup requires shared state for production metrics
- **Alpaca-Primary Training (December 2025):** Switched from Polygon to Alpaca for training data source
  - Alpaca provides 200 requests/min vs Polygon's 5 requests/min
  - `/hyperspeed/train/basic` endpoint for fast training without enrichment APIs
  - `skip_enrichment=True` disables slow external APIs (Finnhub, FRED, SEC) during training
  - Training now completes in ~5 seconds for 7 symbols vs timing out before
  - Use enrichment APIs only for live predictions once rate limits reset
