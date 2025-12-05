# QuantraCore Apex — Replit Project Documentation

**Last Updated:** 2025-12-05

## Overview

QuantraCore Apex v9.0-A is an autonomous AI trading system designed to detect stocks ready for massive runs (50%+ gains) within 5 trading days using EOD data, targeting 70%+ precision. The system operates through Alpaca paper trading with automatic stop-loss management and forward validation tracking to prove real-world performance.

**Current Status:**
- 10 active paper trading positions
- Primary model: massive_ensemble_v3.pkl.gz
- NEW: Intraday model trained on 2M+ 1-minute bars (intraday_moonshot_v1.pkl.gz)
- Stop-loss system active (-15% hard, +10%/8% trailing, 5-day time limit)
- Forward validation tracking all predictions

## Recent Updates (2025-12-05)

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

### Codebase Metrics (Verified 2025-12-04)
| Metric | Value |
|--------|-------|
| Python Source Files | 423 |
| Python Lines of Code | 104,903 |
| TypeScript/React Files | 38 |
| API Endpoints | 263 |
| ML Model Files | 21 |
| Test Modules | 38 |

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
