# QuantraCore Apex — Replit Project Documentation

**Last Updated:** 2025-12-05

## Overview

QuantraCore Apex v9.0-A is an autonomous AI trading system designed to detect stocks ready for massive runs (50%+ gains) within 5 trading days using EOD data, targeting 70%+ precision. The system operates through Alpaca paper trading with automatic stop-loss management and forward validation tracking to prove real-world performance.

**Current Status:**
- 8 active paper trading positions (~$107k equity, +2.9% P&L)
- Multi-strategy orchestrator: 4 concurrent strategies (Swing, Scalp, Momentum, MonsterRunner)
- Scheduled automation: Every 30 min during extended hours (4 AM - 8 PM ET)
- Strategy-specific universe filtering: Each strategy gets tailored stock lists
- Expanded universe: 7,952+ symbols → 197 hot stocks per scan
- Primary model: massive_ensemble_v3.pkl.gz
- Intraday model: intraday_moonshot_v1.pkl.gz (2M+ 1-minute bars)
- Stop-loss system active (-15% hard, +10%/8% trailing, 5-day time limit)
- Forward validation tracking all predictions
- All API rate limiting optimized with 24-hour caching

## Recent Updates (2025-12-05)

### Scheduled Autonomous Trading System (NEW)
Fully automated trading with learning capabilities:

| Component | Description |
|-----------|-------------|
| Scheduled Scanner | Every 30 minutes during extended hours (4 AM - 8 PM ET) |
| Extended Hours | Pre-market (4-9:30 AM), Regular (9:30 AM-4 PM), After-hours (4-8 PM) |
| Scans Per Day | 32 scans (every 30 min for 16 hours) |
| Auto-Entry | High-confidence signals → bracket orders automatically |
| Trade Tracker | Records all closed position outcomes |
| Learning Loop | Weekly analysis with improvement recommendations |

**Scheduler API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scheduler/status` | GET | Current scheduler status and next scans |
| `/scheduler/start` | POST | Start automated scheduling |
| `/scheduler/stop` | POST | Stop automated scheduling |
| `/scheduler/scan/now` | POST | Trigger immediate manual scan |
| `/scheduler/history` | GET | Recent scan history |
| `/scheduler/config` | POST | Update scheduler configuration |

**Learning API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/outcomes/stats` | GET | Trade outcome statistics |
| `/learning/analyze` | GET | Performance analysis with recommendations |
| `/learning/start` | POST | Start background learning loop |

**Key Files:**
- `src/quantracore_apex/trading/scheduled_automation.py` - Scheduler service
- `src/quantracore_apex/trading/trade_outcome_tracker.py` - Outcome tracking
- `src/quantracore_apex/trading/learning_loop.py` - Performance analysis

### Position Management System (NEW)
Intelligent position analysis and management using ML models:

| Feature | Description |
|---------|-------------|
| Position Analysis | Re-scores all positions using EOD+intraday models |
| Close Criteria | Exit if P&L < -8% or model score < 0.3 |
| Partial Profit | Take 50% profit if P&L > +25% |
| Momentum Check | Combines model confidence with current P&L |

**Position Management Endpoint:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/autotrader/positions/manage` | POST | Analyze and manage positions (supports dry_run) |

**Decision Logic:**
1. Close position if: `pnl < -8%` OR `(pnl < 0 AND model_score < 0.3)`
2. Take partial profit if: `pnl > +25%` AND `model_score > 0.4`
3. Hold position otherwise

### API Rate Limiting Optimizations (INVESTOR-READY)
Implemented comprehensive caching to prevent log spam and preserve API quotas:

| Adapter | Fix Applied | TTL |
|---------|-------------|-----|
| Alpha Vantage | 24-hour OHLCV cache + daily limit checking | 24h |
| FRED | Class-level cache with series+date key | 24h |
| Finnhub | Disabled social sentiment (premium API) | N/A |

**Key Changes:**
- `fetch_ohlcv()` now checks daily limit before API calls
- All cached data falls back to simulated data when unavailable
- No more rate limit spam in logs
- Database constraint errors resolved (ON CONFLICT fixed)

### Multi-Strategy Orchestrator
Built complete concurrent strategy execution system:
- **4 Concurrent Strategies:** Swing (2-5 days), Scalp (minutes-hours), MonsterRunner (1-7 days), Momentum (4-48 hours)
- **Strategy Orchestrator:** Manages lifecycle, coordinates signal generation across all strategies
- **Risk Arbiter:** Priority-based conflict resolution, per-strategy budgets, symbol exclusivity
- **Budget Allocation:** Swing 40%, Scalp 15%, MonsterRunner 30%, Momentum 15%
- **Position Limits:** Max 15 total positions across all strategies
- **Long-Only Enforcement:** All strategies restricted to buy positions only

#### Multi-Strategy API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/strategies/status` | GET | Status of all registered strategies and arbiter |
| `/strategies/run` | POST | Run all enabled strategies (supports dry_run) |
| `/strategies/{type}/enable` | POST | Enable/disable individual strategy |

### Unified Autonomous Trading System
Built complete unified trading system combining EOD and intraday models:
- **Merged Scoring:** 60% EOD model + 40% intraday model weighted combination
- **Bracket Orders:** Atomic entry + stop-loss (8%) + take-profit (50%) via Alpaca
- **Capital Validation:** Checks available cash before each order, prevents over-allocation
- **Position Duplication Check:** Skips symbols already held in portfolio
- **Quick Scan Mode:** 20 high-volatility stocks (default) to avoid API rate limits

### Unified Trading API Endpoints
- `POST /autotrader/unified` - Full scan→analyze→trade workflow
- `GET /autotrader/unified/status` - Check unified system status

### External API Limitations (Known Issues)
| Provider | Limit | Impact |
|----------|-------|--------|
| Alpha Vantage | 25 calls/day | Intraday data limited |
| Finnhub | 403 Forbidden | Sentiment features unavailable |
| Alpaca Free | 200 calls/min | Quick scan mode required |

**Workaround:** Use `quick_scan=true` to limit universe to 26 stocks (default is now expanded scanning)

### Expanded Universe Scanner (NEW - 2025-12-05)
Dramatically expanded stock scanning from 26 to 7,952+ symbols:

| Stage | Symbols | Description |
|-------|---------|-------------|
| Master List | 10,366 | All available US equities |
| Scanned | 7,952 | Valid tickers (2-5 chars, no warrants) |
| Liquid Universe | 822 | $0.50-$50 price, >$500k daily volume |
| Hot Prefilter | ~150-300 | Stocks with momentum/volume surges |
| ML Analysis | ~150-300 | All 4 strategies run on hot stocks |

**How it works:**
1. `build_liquid_universe()` - Filters 7,952 symbols to 822 liquid stocks (cached 24h)
2. `prefilter_for_momentum()` - Uses Alpaca bulk bars API to find hot stocks
3. `get_scan_universe()` - Returns prefiltered list for ML analysis

**Key File:** `src/quantracore_apex/trading/universe_scanner.py`

### Strategy-Specific Universe Filtering (NEW - 2025-12-05)
Each strategy now receives a tailored stock list optimized for its characteristics:

| Strategy | Symbols | Filter Criteria |
|----------|---------|-----------------|
| Swing | ~47 | >$1M volume, 50%+ of weekly range, 3-15% volatility |
| Scalp | ~29 | >$2M volume, 1.5x+ volume surge (high liquidity) |
| Momentum | ~18 | 3%+ day change, 1.3x+ volume surge, 60%+ of range |
| MonsterRunner | ~14 | 5%+ volatility, 4%+ day change, <$20 price (explosive) |

**Key Method:** `get_strategy_universe(strategy_type, max_symbols)` in universe_scanner.py

### Quick Scan Universe (Fallback)
Curated 26 high-volatility stocks across 5 hot sectors for moonshot detection:

| Sector | Stocks | Why They Run |
|--------|--------|--------------|
| **Quantum/AI** | QUBT, RGTI, QBTS, IONQ, BBAI, SOUN | Hottest sector 2025 - 1000%+ runners |
| **Crypto Miners** | MARA, RIOT, BITF, CLSK, HIVE | Move 50%+ with Bitcoin rallies |
| **High Short Interest** | BYND, LCID, HIMS, NVTS, SYM | 20-40% SI - Squeeze candidates |
| **EV/Clean Energy** | PLUG, FCEL, QS, BLNK, CHPT | Catalyst-driven moves |
| **Momentum Small-Caps** | SOFI, FUBO, OPEN, SMCI, COIN | Breakout potential |

**Removed Dead Tickers:** BBBY (bankrupt), VLDR (delisted), WISH, GOEV, WKHS, FFIE (near bankruptcy)

## Earlier Updates (2025-12-05)

### 1-Minute Intraday Training Pipeline
Built complete infrastructure for training on 1-minute bar data:
- **Data Source:** Kaggle S&P 500 dataset (2008-2021, 2M+ bars for SPY)
- **Feature Extractor:** 120 features optimized for intraday patterns
- **Training Samples:** 50,000 windows from SPY data
- **Model Performance:**
  - Precision: 59.4%
  - Precision @ 70% threshold: 76.9%
  - Precision @ 80% threshold: 100%

### Key Files Added
- `src/quantracore_apex/data/intraday_pipeline.py` - Data loading/merging
- `src/quantracore_apex/data/intraday_features.py` - 120 intraday features
- `src/quantracore_apex/training/intraday_trainer.py` - Training pipeline
- `src/quantracore_apex/ml/intraday_predictor.py` - Live prediction
- `models/intraday_moonshot_v1.pkl.gz` - Trained model

### Top Predictive Features (Intraday Model)
1. avg_bar_range (15.0%) - Volatility
2. open_range_size (3.8%) - Morning breakout patterns
3. avg_gap_size (3.6%) - Gap patterns
4. multi_regime_score (3.5%) - Market regime alignment
5. afternoon_trend (2.4%) - End-of-day momentum

## User Preferences
- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data

## System Architecture

### Codebase Metrics (Verified 2025-12-05)
| Metric | Value |
|--------|-------|
| Python Source Files | 439 |
| Python Lines of Code | 111,884 |
| TypeScript/React Files | 38 |
| API Endpoints | 293 |
| ML Model Files | 42 |
| Test Modules | 67 |

### Technical Stack
- **Backend:** Python 3.11, FastAPI, Uvicorn (port 8000, 4 workers)
- **Frontend:** React 18.2, Vite 7.2, Tailwind CSS 3.4, TypeScript (port 5000)
- **Machine Learning:** scikit-learn (GradientBoosting), joblib, NumPy, Pandas
- **Database:** PostgreSQL for ML model persistence
- **Testing:** pytest (backend), vitest (frontend)
- **Security:** X-API-Key authentication, restrictive CORS

### Core Features

#### Moonshot Detection System
- **Goal:** Detect stocks ready for 50%+ gains within 5 trading days
- **Target Precision:** 70%+
- **Primary Model:** massive_ensemble_v3.pkl.gz
- **Data Source:** EOD (End of Day) prices for reliable signals

#### Automatic Stop-Loss Management
| Rule | Configuration |
|------|--------------|
| Hard Stop | -15% from entry price |
| Trailing Stop | Activates at +10% gain, trails 8% below highest price |
| Time Stop | Exit after 5 days if gain < 5% |

#### Forward Validation System
- Records predictions before outcomes are known
- Tracks actual results to calculate true precision
- Proves model accuracy with real data

#### AutoTrader
- Autonomous position entry on high-confidence signals
- Automatic position sizing based on risk rules
- Full paper trading integration via Alpaca

### API Endpoints (Key)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/status` | GET | Current positions and P&L |
| `/stops/status` | GET | Stop-loss status for all positions |
| `/stops/exit-signals` | GET | Positions needing exit |
| `/validation/status` | GET | Forward validation metrics |
| `/moonshot/detect` | POST | Moonshot candidate detection |
| `/scan_universe_mode` | POST | Universe scan |

### Protocol System
- **Tier Protocols (T01-T80):** Market structure analysis
- **Learning Protocols (LP01-LP25):** Label generation for training
- **MonsterRunner Protocols (MR01-MR20):** Extreme move detection
- **Omega Directives (Ω01-Ω20):** Safety and compliance rules

## External Dependencies
- **Alpaca:** Paper trading, order execution, positions (Free tier: 200 calls/min, IEX-only)
- **Polygon.io:** EOD prices, historical data (Free tier: 5 calls/min)
- **FRED:** Federal Reserve economic data
- **Finnhub:** Social sentiment, insider transactions
- **Alpha Vantage:** News sentiment, technical indicators
- **Twilio:** SMS alerts for trading signals
- **Google Docs:** Automated reporting via Replit OAuth2
- **PostgreSQL:** ML model persistence

## Data Plans
| Provider | Current Plan | Rate Limit | Notes |
|----------|--------------|------------|-------|
| Alpaca | Free | 200 calls/min | IEX-only data |
| Polygon | Free | 5 calls/min | EOD prices |
| Alpaca Algo Trader Plus | Available ($99/mo) | 10k calls/min | Full SIP feed |

## Current Portfolio (Paper Trading)
- 11 active positions
- All positions monitored by stop-loss system
- Breakout timing estimates: 2-6 days based on position age and P&L
