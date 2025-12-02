# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex is an institutional-grade, deterministic AI trading intelligence engine for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system supports all trading types (long, short, intraday, swing, scalping) primarily via Alpaca paper trading. Its core purpose is to deliver sophisticated, AI-driven trading with a focus on determinism, accuracy optimization, and self-learning, offering a robust, self-improving platform with advanced risk management and comprehensive reporting for both retail and institutional users.

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### UI/UX Decisions
The frontend, built with React 18.2, Vite 7.2, and Tailwind CSS 4.0, uses an institutional trading terminal aesthetic and a custom design system defined by `@theme` blocks.

### ApexDesk Dashboard (15 Panels)
The institutional trading dashboard provides real-time monitoring and control via 15 specialized panels:

| Panel | Description | Key Features |
|-------|-------------|--------------|
| **SystemStatusPanel** | Real-time system health monitoring | Market hours, broker status, compliance score, data feeds, ApexCore model status |
| **PortfolioPanel** | Live Alpaca paper trading portfolio | Total equity, cash, positions with P&L, real-time updates (15s intervals) |
| **TradingSetupsPanel** | Top trading opportunities ranked by QuantraScore | Signal ranking, conviction tiers, entry/stop/target levels, timing guidance |
| **ModelMetricsPanel** | ApexCore V3/V4 model performance | 7/16 prediction heads, training samples, accuracy metrics, model reload |
| **AutoTraderPanel** | Autonomous swing trade execution | Trade status, position hold decisions, continuation probability |
| **SignalsAlertsPanel** | Live signals and SMS alert status | Active signals, Twilio SMS configuration, alert history |
| **PushNotificationPanel** | Browser push notifications | One-click subscribe, test notifications, threshold config, free alerts |
| **RunnerScreenerPanel** | Low-float penny stock scanner | 110 symbols, volume surge detection, momentum alerts |
| **ContinuousLearningPanel** | ML training orchestrator status | Drift detection, incremental learning, training schedule |
| **LogsProvenancePanel** | System logs and audit trail | Real-time logs, provenance tracking, compliance audit |
| **OptionsFlowPanel** | Smart money options tracking | Unusual activity, sweeps, premium flow, put/call ratio |
| **SentimentPanel** | News & social sentiment | Real-time news, social volume, bullish/bearish ratios |
| **DarkPoolPanel** | Institutional flow tracking | Block trades, dark pool prints, short interest, accumulation signals |
| **MacroPanel** | Economic regime analysis | Fed stance, yield curve, CPI, GDP, economic calendar |
| **DataProvidersPanel** | Multi-source integration | Provider status, setup guide, cost summary |

**Velocity Mode System:** Standard (30s), High Velocity (5s), and Turbo (2s) refresh rates for different trading scenarios.

### Technical Implementations
- **Backend:** Python 3.11, FastAPI, Uvicorn for REST API services (port 8000).
- **Frontend:** React 18.2, Vite 7.2, Tailwind CSS 4.0, TypeScript (ApexDesk Frontend on port 5000).
- **Dashboard:** 14 real-time panels with Velocity Mode (Standard/High/Turbo refresh rates).
- **Machine Learning:** `scikit-learn` (GradientBoosting) and `joblib`.
- **Numerical Operations:** `NumPy` and `Pandas`.
- **Testing:** `pytest` (backend), `vitest` (frontend).
- **API Endpoints:** REST APIs for trading, data providers, market scanning, health checks, and investor reporting.
- **Security:** `X-API-Key` authentication and restrictive CORS policy (localhost, Replit domains).
- **Performance Optimizations:** ORJSONResponse, GZipMiddleware, 4-worker Uvicorn, expanded TTL caches, module-level ML model caching, Alpaca client caching, parallel universe scanning, prediction result caching.

### Feature Specifications
- **Full Trading Capabilities:** Supports various trade types with configurable risk limits ($100K max exposure, $10K per symbol, 50 positions max).
- **Order Types:** MARKET, LIMIT, STOP, STOP_LIMIT.
- **Entry/Exit Strategies:** Auto-selected entries based on market conditions, comprehensive exits (ATR-based protective stops, trailing stops, profit targets, time-based).
- **Deterministic Core:** Ensures identical outputs for identical inputs.
- **Offline ML (ApexCore V3):** On-device neural models with 7 prediction heads (quantrascore, runner, quality, avoid, regime, timing, runup). The timing head predicts market moves across 5 buckets, and the runup head predicts price appreciation (0-100%+). Models are trained on real market data from Alpaca with swing-trade focus. Current model: 6,085 training samples, 29 diverse symbols, 32,397 15-minute intraday bars. Accuracy: runner 94.99%, quality 81.68%, avoid 99.34%, regime 85.95%, timing 88.33%.
- **ApexCore V4 (Multi-Data Extension):** Extended model with 16 total prediction heads (7 base + 9 extended). New heads: catalyst_probability, sentiment_composite, squeeze_probability, institutional_signal, momentum_burst, tape_reading, volatility_regime, options_flow, macro_regime. Requires multi-source data connections for full capability with fallback to simulated data when providers unavailable.
- **Unified Training Pipeline:** Multi-source training (Alpaca, Polygon) to generate labels from future price movements.
- **Simulation-Based Data Augmentation:** Multiplies training samples using real data variations (entry timing shifts, walk-forward windows, Monte Carlo bootstrapping, rare event oversampling).
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, auto-retraining, and multi-horizon prediction.
- **Self-Learning (Alpha Factory):** 24/7 live research loop via WebSockets for feedback and automatic retraining.
- **Continuous Learning System:** Autonomous orchestrator with 15-minute scheduling, incremental learning, drift detection, validation gates, multi-pass training, and an extended 251-symbol universe.
- **Manual Trading Signal Service (ApexSignalService):** Generates actionable signals with priority scoring, timing guidance, ATR-based levels, conviction tiers, predicted top price, and persistence to disk.
- **SMS Alert Service:** Sends Twilio SMS alerts for trading signals with configurable thresholds, displaying QuantraScore, conviction, predicted top price, entry/stop/target levels, and timing guidance.
- **Push Notification Service (Free Alternative):** Browser-based push notifications for trading signals without SMS costs. Features: VAPID key-based encryption, service worker background delivery, configurable thresholds (QuantraScore, runner probability, timing), rate limiting, subscription management. Works even when browser is closed (requires HTTPS). API endpoints: GET /push/vapid-key, GET /push/status, POST /push/subscribe, POST /push/unsubscribe, POST /push/config, POST /push/test, POST /push/alert. Dashboard panel allows one-click subscribe/unsubscribe with test notifications.
- **Low-Float Runner Screener:** Real-time scanner for penny stocks with volume surge, momentum detection, float limits, ApexCore V3 prediction integration, and SMS alerts. Scans 110 low-float symbols.
- **MarketSimulator:** Provides 8 chaos scenarios for stress testing.
- **Protocol System:** Includes 80 Tier protocols for analysis, 25 Learning protocols, 20 MonsterRunner protocols, and 20 Omega Directives.
- **Investor Trade Logging:** Comprehensive logging of paper trades to `investor_logs/`.
- **Due Diligence Logging:** Audit trail for institutional investors including compliance attestations, incident lifecycle, policy manifest, broker reconciliation, consent management, and document access logging, stored in `investor_logs/`.
- **Automated Daily Attestations:** Generates 8 compliance attestations daily.
- **Performance Metrics Logger:** Tracks daily returns, cumulative performance, drawdowns, risk ratios, and benchmark comparisons, stored in `investor_logs/performance/`.
- **ML Model Training Logger:** Tracks model lifecycle, training runs, validation, deployment, and drift detection, generating model cards, stored in `investor_logs/models/`.
- **Investor Data Exporter:** Automates export of investor packages (weekly, monthly, full data room) with 37+ documents.
- **Investor Legal Documents:** Legal documentation stored in `docs/investor/legal/`.
- **Automated Swing Trade Execution (AutoTrader):** Autonomous system scanning for setups, picking top QuantraScore candidates, calculating position sizing, and executing market orders on Alpaca paper trading. Logs auto-trades to `investor_logs/auto_trades/`.
- **Hot Model Reload System (ModelManager):** Unified model management with automatic hot-reload, cache clearing, and version tracking upon training completion.
- **Dual-Phase Incremental Learning (IncrementalTrainer):** Efficient knowledge retention system with warm-start capability (builds on previous model weights), dual-buffer architecture (anchor reservoir preserves rare patterns, recency buffer tracks recent samples), time-decay sample weighting (older samples weighted less but preserved), and graceful LightGBM/scikit-learn fallback. API endpoints: POST /apexlab/train-incremental, GET /apexlab/incremental/status.
- **Trade Hold Manager (TradeHoldManager):** Continuation probability-based hold decision system that analyzes active positions in real-time. Features: momentum strength analysis, exhaustion detection, reversal risk calculation, dynamic stop adjustment (ATR-based trailing), target extension for strong trends, and hold extension recommendations. Provides 5 decision types: HOLD_STRONG, HOLD_NORMAL, TRAIL_STOP, REDUCE_PARTIAL, EXIT_NOW. API endpoints: GET /positions/continuation, GET /positions/continuation/{symbol}, POST /positions/hold-config. Integrates with AutoTrader panel in dashboard to display position hold status with color-coded decisions.
- **Extended Market Hours Trading:** Full support for extended market hours across all trading sessions. Pre-market (4:00 AM - 9:30 AM ET), Regular hours (9:30 AM - 4:00 PM ET), After-hours (4:00 PM - 8:00 PM ET). All orders include `extended_hours: true` flag by default. Market hours service provides real-time session detection, holiday awareness, and early close handling. API endpoint: GET /market/hours. Dashboard displays current session, trading status, and extended hours schedule.

### Symbol Universe (251 Total)
Categorized by market cap (Penny, Nano, Micro, Small, Mid, Large, Mega). 110 low-float symbols (penny + nano + micro) are used for runner scanning.

### Hybrid Data Architecture
The system uses a hybrid approach for optimal performance:
- **Trading Execution:** Alpaca paper trading (orders, positions, portfolio management)
- **Market Data:** Polygon.io Developer tier (tick data, OHLCV, quotes, NBBO)
- **Fallback:** Alpaca IEX data if Polygon unavailable

**Polygon Tier Configuration:**
- Environment variable: `POLYGON_TIER` (free, starter, developer, advanced)
- Current setting: `developer` ($249/mo) for real-time tick data
- Features: Tick-by-tick data, NBBO pricing, 15+ years history, extended hours

**Configuration Files:**
- `config/data_sources.yaml` - Data source priority configuration
- `config/broker.yaml` - Trading and execution settings

### System Design Choices
- **Broker Layer:** Supports `NullAdapter`, `PaperSimAdapter`, and `AlpacaPaperAdapter`.
- **Data Layer:** Polygon for market data, Alpaca for execution (hybrid setup).
- **Configuration:** Trading and risk parameters via `config/broker.yaml`; universe scanning modes via `config/scan_modes.yaml`; data sources via `config/data_sources.yaml`.
- **Google Docs Integration:** Uses Replit OAuth2 for automated report exports.

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading and order execution.
- **Polygon.io (Developer tier):** Primary data source for market data (OHLCV, ticks, quotes), ML training, and extended hours coverage. Requires `POLYGON_API_KEY` secret.
- **Binance:** Primary source for crypto data in Alpha Factory.
- **Twilio:** Used for SMS alerts via Replit integration.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting.

### Required Secrets
| Secret | Purpose |
|--------|---------|
| `ALPACA_PAPER_API_KEY` | Alpaca paper trading authentication |
| `ALPACA_PAPER_API_SECRET` | Alpaca paper trading authentication |
| `POLYGON_API_KEY` | Polygon.io market data access |
| `POLYGON_TIER` | Polygon subscription tier (developer recommended) |
| `TWILIO_*` | SMS alert configuration (managed by Replit integration) |