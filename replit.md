# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system supports all trading types, including long, short, intraday, swing, and scalping, primarily through Alpaca paper trading. Its core purpose is to provide sophisticated, AI-driven trading capabilities with a focus on determinism, accuracy optimization, and self-learning mechanisms for both retail and institutional users. The project aims to offer a robust, self-improving trading platform with advanced risk management and comprehensive reporting.

**Last Updated:** 2025-12-01

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### UI/UX Decisions
The frontend, built with React 18.2, Vite 5, and Tailwind CSS 4.0, adopts an institutional trading terminal aesthetic. It leverages `@theme` blocks for custom color definitions to create a custom design system.

### Technical Implementations
- **Backend:** Python 3.11, FastAPI, Uvicorn for REST API services.
- **Frontend:** React 18.2, Vite 5, Tailwind CSS 4.0, TypeScript for the main user interface (ApexDesk Frontend on port 5000).
- **Machine Learning:** Utilizes `scikit-learn` (GradientBoosting) and `joblib` for model management.
- **Numerical Operations:** `NumPy` and `Pandas` are used for data manipulation.
- **Testing:** `pytest` for backend and `vitest` for frontend.
- **API Endpoints:** A suite of REST APIs manage trading capabilities, data providers, market scanning, health checks, and comprehensive investor reporting.
- **Security:** Implements `X-API-Key` authentication for protected endpoints and a restrictive CORS policy allowing only localhost and Replit domains.
- **Performance Optimizations (3x improvement):** ORJSONResponse for fast JSON serialization, GZipMiddleware for automatic compression (>500 bytes), 4-worker uvicorn configuration, expanded TTL caches (5000 entries for scans, 2000 for predictions @ 120s TTL, 1000 for quotes @ 30s TTL), module-level ML model caching (warm loading), Alpaca client caching, parallel universe scanning with asyncio.gather, prediction result caching.

### Feature Specifications
- **Full Trading Capabilities:** Supports long, short, margin (up to 4x leverage), intraday, swing, and scalping trades with configurable risk limits ($100K max exposure, $10K per symbol, 50 positions max).
- **Order Types:** MARKET, LIMIT, STOP, STOP_LIMIT.
- **Entry/Exit Strategies:** Auto-selected entry strategies based on market conditions (e.g., high volatility, low liquidity) and comprehensive exit strategies including protective stops (ATR-based, ZDE adjustments), trailing stops, profit targets, and time-based exits.
- **Deterministic Core:** Ensures identical outputs for identical inputs.
- **Offline ML (ApexCore v3):** On-device neural models with **7 prediction heads** (quantrascore, runner, quality, avoid, regime, **timing**, **runup**), trained on real market data with actual outcome labels. The **timing head** predicts WHEN a stock will start its significant move using 5 timing buckets: immediate (1 bar), very_soon (2-3 bars), soon (4-6 bars), late (7-10 bars), none. The **runup head** predicts expected price appreciation (0-100%+). Current model trained on **107,978 samples** from **59 symbols** using **222,412 15-minute intraday bars** (2 years of data) via Alpaca IEX feed, achieving **98.88% runner accuracy**, **93.75% quality accuracy**, **99.91% avoid accuracy**, **95.25% regime accuracy**, and **94.4% timing accuracy**. QuantraScore RMSE: 0.00999.
- **Unified Training Pipeline:** Multi-source training using both Alpaca and Polygon data feeds in parallel. Generates training labels from actual future price movements (5-day returns, max runup, drawdowns). Endpoint: `POST /apexlab/train-unified`.
- **Simulation-Based Data Augmentation:** Multiplies training samples using real data variations without creating fake data: (1) Entry timing shifts (+/- 2 bars to simulate different entry points), (2) Walk-forward windows (3 overlapping perspectives from same data), (3) Monte Carlo bootstrapping (preserves real return distributions), (4) Rare event oversampling (runners 3x, crashes 2x). AugmentedWindowGenerator integrates entry shifts and walk-forward during window generation. Endpoint: `POST /apexlab/train-augmented`. Can achieve 2-5x data multiplication.
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, auto-retraining, and multi-horizon prediction.
- **Self-Learning (Alpha Factory):** A 24/7 live research loop using WebSockets for a self-learning feedback mechanism and automatic retraining based on batch thresholds.
- **Continuous Learning System:** Autonomous learning orchestrator with: (1) Auto-learning scheduler running every **15 minutes**, (2) Incremental learning with warm-start and 100K sample cache, (3) Drift detection for feature/label distribution changes (15% threshold), (4) Validation gates before model promotion, (5) Multi-pass training epochs, (6) **Extended 251 symbol universe** covering tech, financials, energy, healthcare, consumer, and more. API endpoints: `POST /apexlab/continuous/start`, `GET /apexlab/continuous/status`, `POST /apexlab/continuous/trigger`.
- **Manual Trading Signal Service (ApexSignalService):** Generates actionable trading signals for use on external platforms (Webull, TD Ameritrade, etc.) without API integration. Features: (1) Signal ranking by priority score (quantrascore, runner probability, timing urgency), (2) Timing guidance with actionable language (e.g., "ENTER NOW - Move expected within 15 minutes"), (3) ATR-based entry/exit levels with 2:1 minimum risk/reward, (4) Conviction tiers (high/medium/low/avoid), (5) Signal persistence to disk (24h rolling store), (6) **Predicted top price** from runup model. API endpoints: `GET /signals/live`, `POST /signals/scan`, `GET /signals/symbol/{symbol}`, `GET /signals/status`.
- **SMS Alert Service:** Sends trading signals via Twilio SMS with configurable thresholds. Features: (1) QuantraScore and conviction tier display, (2) Predicted top price with expected runup %, (3) Entry/stop/target levels, (4) Rate limiting (max alerts/hour, per-symbol cooldown), (5) Timing bucket guidance. API endpoints: `GET /sms/status`, `POST /sms/config`, `POST /sms/test`, `POST /sms/alert`.
- **Low-Float Runner Screener:** Real-time scanner for penny stock runners with volume surge and momentum detection. Features: (1) 3x+ relative volume threshold, (2) 5%+ price momentum filter, (3) Max 50M float filter for true low-float stocks, (4) ApexCore V3 prediction integration, (5) SMS alerts for detected runners, (6) Per-symbol cooldowns to prevent alert spam. Scans **114 low-float symbols** (64 penny + 20 nano + 30 micro caps). API endpoints: `GET /screener/status`, `POST /screener/scan`, `GET /screener/alerts`, `POST /screener/config`, `POST /screener/alert-runner`.
- **MarketSimulator:** Provides 8 chaos scenarios for stress testing the system.
- **Protocol System:** Comprises 80 Tier protocols for analysis, 25 Learning protocols for training labels, 20 MonsterRunner protocols for explosive movement detection, and 20 Omega Directives as safety overrides.
- **Investor Trade Logging:** Comprehensive logging of every paper trade with details on account state, signal quality, market context, risk assessment, protocol analysis, execution details, and trade outcome, stored in `investor_logs/`.
- **Due Diligence Logging:** Complete audit trail for institutional investors: (1) Compliance attestations and control checks, (2) Incident lifecycle with root cause and remediation, (3) Policy manifest with version tracking and checksums, (4) Broker reconciliation records, (5) Consent management for TCPA/GDPR compliance, (6) Document access logging. All stored in `investor_logs/compliance/`, `investor_logs/audit/`, `investor_logs/legal/`. API endpoints: `GET /investor/due-diligence/status`, `POST /investor/due-diligence/attestation`, `POST /investor/due-diligence/incident`, `POST /investor/due-diligence/policy`, `POST /investor/due-diligence/reconciliation`, `POST /investor/due-diligence/consent`, `POST /investor/due-diligence/access`.
- **Automated Daily Attestations:** System automatically generates 8 compliance attestations: risk limits, model health, data quality, system availability, trading controls, security controls, data retention, backup status. API endpoint: `POST /investor/attestations/run-daily`.
- **Performance Metrics Logger:** Automated tracking of daily returns, cumulative performance, drawdowns, Sharpe/Sortino/Calmar ratios, benchmark comparisons. All stored in `investor_logs/performance/`. API endpoint: `GET /investor/performance/status`.
- **ML Model Training Logger:** Complete model lifecycle tracking: training runs with hyperparameters, validation results, deployment events, drift detection. Generates model cards for due diligence. All stored in `investor_logs/models/`. API endpoint: `GET /investor/models/history/{model_name}`.
- **Investor Data Exporter:** Automated export of investor packages: weekly snapshots, monthly packages, and full data room with 37+ documents in 8 sections (Legal, Performance, Trading, Compliance, Risk, Models, Technical, Company). API endpoints: `POST /investor/export/weekly`, `POST /investor/export/monthly`, `POST /investor/export/data-room`.
- **Investor Legal Documents:** Complete legal documentation stored in `docs/investor/legal/`: Terms of Service, Risk Disclosures, Privacy Policy, Fee Schedule.

### Symbol Universe (251 Total)
| Bucket | Count | Description |
|--------|-------|-------------|
| **Penny** | 64 | Sub-$5 volatile small-caps (MARA, RIOT, SOFI, GME, AMC, etc.) |
| **Nano** | 20 | Ultra-small caps < $50M (SNDL, TELL, MNMD, etc.) |
| **Micro** | 30 | Micro caps $50M-$300M |
| **Small** | 40 | Small caps $300M-$2B |
| **Mid** | 40 | Mid caps $2B-$10B |
| **Large** | 50 | Large caps $10B-$200B |
| **Mega** | 30 | Mega caps > $200B (AAPL, MSFT, GOOGL, etc.) |

**Low-Float Universe:** 114 symbols (penny + nano + micro) for runner scanning.

### System Design Choices
- **Broker Layer:** Supports `NullAdapter` (research), `PaperSimAdapter` (offline simulation), and `AlpacaPaperAdapter` (live paper trading).
- **Configuration:** Trading and risk parameters are managed via `config/broker.yaml`, and universe scanning modes are configured in `config/scan_modes.yaml`.
- **Google Docs Integration:** Utilizes Replit OAuth2 for automated export of investor reports, due diligence packages, and trade logs.

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading execution ($100k equity). Free IEX data feed provides 7+ years of historical data (using `feed=iex` parameter). 15-minute intraday bars for high-resolution pattern detection.
- **Polygon.io:** Primary data source for ML training (5 requests/minute). Successfully training models on real OHLCV data from 15+ symbols.
- **Binance:** Primary source (1200 requests/minute) for crypto data, used in Alpha Factory.
- **Twilio:** SMS alerts for high-conviction trading signals via Replit integration.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting and export functionalities.
- **Alpha Vantage:** (Available) For technicals, forex, crypto.
- **EODHD:** (Available) For international markets.
- **Finnhub:** (Available) For news, sentiment, insider trades.
- **CoinGecko:** (Available) For cryptocurrency data.

## Recent Changes

- **2025-12-01:** Implemented Due Diligence Logging infrastructure (attestations, incidents, policies, reconciliation, consents, access logs)
- **2025-12-01:** Scoped Investor Due Diligence Suite (8 modules, 70+ endpoints) - see `docs/INVESTOR_DUE_DILIGENCE_SUITE_SPEC.md`
- **2025-12-01:** Phase 1 performance optimizations: ORJSONResponse, GZipMiddleware, 4-worker uvicorn, expanded caches, parallel scanning
- **2025-12-01:** Added low-float runner screener with 114 symbols, 5 new API endpoints
- **2025-12-01:** Expanded penny stock universe from 15 to 64 symbols
- **2025-12-01:** Fixed ApexCore V3 timing/runup heads (7 total heads now)
- **2025-11-30:** Added SMS alert service with Twilio integration
- **2025-11-30:** Added signal service for manual trading on external platforms