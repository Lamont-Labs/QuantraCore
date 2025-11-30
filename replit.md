# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system is strictly for **research and backtesting only**, providing structural probabilities rather than trading advice.

### Key Capabilities
- **145+ Protocols:** 80 Tier + 25 Learning + 20 MonsterRunner + 20 Omega
- **QuantraScore:** 0-100 probability-weighted composite score
- **Deterministic Core:** Same inputs always produce identical outputs
- **Offline ML:** On-device ApexCore v2 neural models (scikit-learn)
- **Paper Trading:** Alpaca integration (LIVE mode disabled)
- **Self-Learning:** Alpha Factory feedback loop with automatic retraining
- **Google Docs Export:** Automated investor/acquirer reporting pipeline

## Recent Changes (November 2025)

### Security Hardening (v9.0-A)
- **API Authentication:** Added `X-API-Key` header verification for protected endpoints
- **CORS Restriction:** Changed from wildcard (`*`) to regex pattern allowing only localhost and Replit domains
- **Non-blocking Rate Limiting:** Updated Polygon and Binance adapters to use async-compatible delays
- **Timeframe Validation:** Added case-insensitive matching with warning logs for unknown timeframes
- **Cache Limits:** Implemented TTL cache with 1000 entry limit and 5-minute expiration

### Frontend Updates
- **Tailwind CSS v4:** Migrated to `@theme` blocks for custom color definitions
- **Custom Design System:** Institutional trading terminal aesthetic with apex/lamont color palette

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Technology Stack
| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | React 18.2, Vite 5, Tailwind CSS 4.0, TypeScript |
| Machine Learning | scikit-learn (GradientBoosting), joblib |
| Numerical | NumPy, Pandas |
| Testing | pytest (backend), vitest (frontend) |

### Running Services
| Service | Port | Description |
|---------|------|-------------|
| ApexDesk Frontend | 5000 | React dashboard (main UI) |
| FastAPI Backend | 8000 | REST API server |
| Alpha Factory Dashboard | 8080 | Static HTML dashboard |

### Security Configuration

#### API Authentication
Protected endpoints require an `X-API-Key` header. Configure via environment variables:
- `APEX_API_KEY` - Primary API key
- `APEX_API_KEY_2` - Secondary API key (optional)
- `APEX_AUTH_DISABLED=true` - Bypass authentication (development only)

#### CORS Policy
Allowed origins (regex pattern):
- `http://localhost:*` and `http://127.0.0.1:*`
- `https://*.replit.dev` and `https://*.repl.co`

#### Cache Configuration
- **Max Entries:** 1000
- **TTL:** 300 seconds (5 minutes)
- **Eviction:** LRU with expired entry cleanup

### Core Systems

#### ApexEngine
The primary analysis engine processes 100-bar OHLCV windows to compute microtraits, entropy, suppression, drift, and continuation metrics. It classifies regimes, generates a QuantraScore (0-100), builds a verdict with a risk tier, runs 80 Tier protocols, and applies 20 Omega directives.

#### ApexCore v2
Multi-head neural models with 5 prediction heads for QuantraScore regression, runner probability classification, quality tier, avoid-trade probability, and regime classification.

#### ApexLab v2
A training pipeline featuring walk-forward validation, bootstrap ensembles, and a 40+ field schema.

#### EEO Engine
Manages Entry/Exit Optimization with various entry strategies (baseline, high-vol, low-liquidity, runner, ZDE-aware), exit optimization (stops, targets, trailing, time-based), and position sizing using a fixed-fraction risk model.

#### Alpha Factory
A 24/7 live research loop utilizing Polygon (equities) and Binance (crypto) WebSockets for a self-learning feedback loop and automatic retraining based on batch thresholds.

#### MarketSimulator
Provides 8 chaos scenarios (flash crash, volatility spike, gap event, liquidity void, momentum exhaustion, squeeze, correlation breakdown, black swan) for stress testing.

#### Broker Layer
Supports a NullAdapter (research mode), PaperSimAdapter (offline simulation), and AlpacaPaperAdapter for Alpaca paper trading. Live mode is disabled.

### Protocol System
- **Tier Protocols (T01-T80):** 80 deterministic analysis protocols covering trend, volatility, continuation, reversal, momentum, volume, patterns, and support/resistance levels.
- **Learning Protocols (LP01-LP25):** Generate labels for supervised training (e.g., regime, volatility, risk tier).
- **MonsterRunner Protocols (MR01-MR20):** Detect explosive movements like compression explosions and volume anomalies.
- **Omega Directives (Ω1-Ω20):** Safety overrides including hard locks, entropy/drift/compliance overrides, and kill switches.

## External Dependencies

### Data Providers (15 Total via UnifiedDataManager)
**OHLCV & Market Data:**
- **Polygon.io:** Primary provider ($29/mo) - US equities, options, crypto
- **Alpha Vantage:** Secondary ($49/mo) - technicals, forex, crypto
- **EODHD:** International markets ($20/mo) - 70+ global exchanges
- **Interactive Brokers:** Broker-integrated (free with account) - requires IB Gateway

**Fundamentals & Filings:**
- **Financial Modeling Prep:** Fundamentals, SEC filings ($20/mo)
- **Nasdaq Data Link:** Economic indicators, Fed data (free tier)

**Options Flow & Dark Pool:**
- **Unusual Whales:** Unusual activity, congressional trades ($35/mo)
- **FlowAlgo:** Sweeps, blocks, institutional flow ($150/mo)
- **InsiderFinance:** Correlated flow analysis ($49/mo)

**Alternative Data & Sentiment:**
- **Finnhub:** News, sentiment, insider trades (free tier)
- **AltIndex:** AI stock scores, social sentiment ($29/mo)
- **Stocktwits:** Social sentiment (free, no API key)

**Cryptocurrency:**
- **Binance:** Free crypto data (no key for public endpoints)
- **CoinGecko:** 10,000+ coins (free tier)

**Testing:**
- **Synthetic:** Deterministic data for testing (free)

**Current Active:** Polygon, Alternative Data (Stocktwits), Crypto (Binance), Synthetic

### Environment Variables

#### Required Secrets
| Secret | Description |
|--------|-------------|
| `POLYGON_API_KEY` | Polygon.io API access |
| `ALPACA_PAPER_API_KEY` | Alpaca paper trading key |
| `ALPACA_PAPER_API_SECRET` | Alpaca paper trading secret |

#### Optional Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `APEX_AUTH_DISABLED` | `false` | Disable API authentication |
| `APEX_API_KEY` | - | Primary API key for authentication |

### Broker Integration
- **Alpaca Paper:** For paper trading.

### Google Docs Integration
Connected via Replit OAuth2 for an automated export pipeline supporting:
- Investor reports (daily/weekly/monthly)
- Due diligence packages
- Trade log exports
- Trade journals with research notes
- Monthly investor updates

## File Structure

```
quantracore-apex/
├── src/quantracore_apex/     # Backend Python source
│   ├── core/                 # Engine, schemas, types
│   ├── data_layer/           # Data adapters and normalization
│   ├── server/               # FastAPI application
│   ├── protocols/            # Tier, Learning, Omega protocols
│   ├── prediction/           # MonsterRunner, ApexCore models
│   ├── risk/                 # Risk engine
│   ├── broker/               # OMS, broker adapters
│   └── portfolio/            # Portfolio management
├── dashboard/                # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   └── index.css         # Tailwind v4 theme
│   └── vite.config.ts        # Vite configuration
├── static/                   # Alpha Factory static dashboard
└── tests/                    # Test suites
```
