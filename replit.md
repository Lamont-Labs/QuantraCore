# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system supports **all trading types** including long, short, intraday, swing, and scalping through Alpaca paper trading.

### Key Capabilities
- **145+ Protocols:** 80 Tier + 25 Learning + 20 MonsterRunner + 20 Omega
- **QuantraScore:** 0-100 probability-weighted composite score
- **Deterministic Core:** Same inputs always produce identical outputs
- **Offline ML:** On-device ApexCore v3 neural models with accuracy optimization
- **Full Paper Trading:** Alpaca integration with all position types enabled
- **Self-Learning:** Alpha Factory feedback loop with automatic retraining
- **Accuracy Optimization:** 8-module suite for calibration, regime-gating, uncertainty
- **Google Docs Export:** Automated investor/acquirer reporting pipeline
- **Dual Data Sources:** Alpaca (200 req/min) + Polygon (5 req/min) for reliability

## Recent Changes (November 2025)

### Full Trading Capabilities (v9.0-A)
- **All Position Types Enabled:** Long, short, margin, intraday, swing, scalping
- **Execution Mode:** Upgraded from RESEARCH to PAPER (live paper trading)
- **Short Selling:** Unblocked for full directional flexibility
- **Margin Trading:** Enabled with 4x max leverage
- **Risk Limits Updated:** $100K max exposure, $10K per symbol, 50 positions max

### Alpaca Data Adapter
- **New Primary Data Source:** Alpaca Markets API (200 requests/minute - 40x faster than Polygon)
- **Retry Logic:** Exponential backoff with 3 retries for production resilience
- **Rate Limiting:** Smart delays prevent API throttling
- **Error Handling:** Graceful recovery from timeouts, connection errors, server errors

### Swing Trading Scanner
- **Real Scan Modes API:** Uses `/scan_universe_mode` endpoint with `config/scan_modes.yaml`
- **4 Pre-configured Modes:** Momentum Runners, Mid-Cap Focus, Blue Chips, High Vol Small Caps
- **Data Provider Validation:** Verifies real data sources before scanning
- **No Hardcoded Symbols:** All universes configured via YAML

### Security Hardening
- **API Authentication:** Added `X-API-Key` header verification for protected endpoints
- **CORS Restriction:** Regex pattern allowing only localhost and Replit domains
- **Non-blocking Rate Limiting:** Async-compatible delays for Polygon and Binance
- **Cache Limits:** TTL cache with 1000 entry limit and 5-minute expiration

### Frontend Updates
- **Tailwind CSS v4:** Migrated to `@theme` blocks for custom color definitions
- **Custom Design System:** Institutional trading terminal aesthetic

### Accuracy Optimization System (v9.0-A)
- **Protocol Telemetry:** Tracks which protocols contribute to winning trades vs noise
- **Feature Store:** Centralized feature management with data quality audits
- **Calibration Layer:** Platt/isotonic calibration for reliable confidence scores
- **Regime-Gated Ensemble:** Different models for trending, choppy, volatile conditions
- **Uncertainty Head:** Conformal prediction for valid confidence bounds
- **Auto-Retraining:** Drift detection and automatic retraining triggers
- **Multi-Horizon Prediction:** Separate models for 1d, 3d, 5d, 10d forecasts
- **Cross-Asset Features:** VIX regime, sector rotation, market breadth context
- **ApexCore V3:** Unified model integrating all accuracy optimizations

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Data Policy:** Absolutely no placeholders - everything must use real data
- **Restrictions:** Do not modify folder `Z` or file `Y`

## Trading Capabilities

### Position Types
| Type | Status | Description |
|------|--------|-------------|
| Long | Enabled | Buy to open, sell to close |
| Short | Enabled | Sell to open, buy to close |
| Margin | Enabled | Up to 4x leverage |
| Intraday | Enabled | Same-day entry/exit |
| Swing | Enabled | 2-10+ day holds |
| Scalping | Enabled | Sub-5 minute trades |

### Order Types
- **MARKET** - Immediate execution
- **LIMIT** - Price-controlled entry
- **STOP** - Triggered at price level
- **STOP_LIMIT** - Stop with limit protection

### Entry Strategies (Auto-Selected)
1. **Baseline Long/Short** - Standard entries for normal conditions
2. **High Volatility** - Conservative limits, scaled entries
3. **Low Liquidity** - Careful entries inside spread
4. **Runner Anticipation** - Aggressive for monster runners
5. **ZDE-Aware** - Tight stops from Zero-Depth Excursion research

### Exit Strategies
- **Protective Stops** - ATR-based with ZDE adjustments
- **Trailing Stops** - ATR, percentage, or structural
- **Profit Targets** - Multiple scaled targets
- **Time-Based Exits** - Auto-close after X bars
- **EOD Exits** - End-of-day flat for day trades

### Risk Configuration
| Parameter | Value |
|-----------|-------|
| Max Exposure | $100,000 |
| Max Per Symbol | $10,000 |
| Max Positions | 50 |
| Max Leverage | 4x |
| Risk Per Trade | 2% |

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

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/trading_capabilities` | GET | Current trading config and limits |
| `/data_providers` | GET | Status of all data sources |
| `/scan_universe_mode` | POST | Run swing scan with mode config |
| `/scan_symbol` | POST | Analyze single symbol |
| `/health` | GET | System health check |

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

#### ApexCore v3
Enhanced multi-head neural models with 5 prediction heads plus accuracy optimization. Includes calibrated probabilities, regime-gated ensemble routing, uncertainty quantification, protocol telemetry weighting, and cross-asset context integration. V3 provides calibrated confidence scores, valid prediction intervals, and knows when to abstain from trading.

#### ApexLab v2
A training pipeline featuring walk-forward validation, bootstrap ensembles, and a 40+ field schema.

#### EEO Engine
Manages Entry/Exit Optimization with various entry strategies (baseline, high-vol, low-liquidity, runner, ZDE-aware), exit optimization (stops, targets, trailing, time-based), and position sizing using a fixed-fraction risk model.

#### Alpha Factory
A 24/7 live research loop utilizing Polygon (equities) and Binance (crypto) WebSockets for a self-learning feedback loop and automatic retraining based on batch thresholds.

#### MarketSimulator
Provides 8 chaos scenarios (flash crash, volatility spike, gap event, liquidity void, momentum exhaustion, squeeze, correlation breakdown, black swan) for stress testing.

#### Broker Layer
- **NullAdapter:** Research mode (signals only)
- **PaperSimAdapter:** Offline simulation
- **AlpacaPaperAdapter:** Live paper trading via Alpaca

### Protocol System
- **Tier Protocols (T01-T80):** 80 deterministic analysis protocols covering trend, volatility, continuation, reversal, momentum, volume, patterns, and support/resistance levels.
- **Learning Protocols (LP01-LP25):** Generate labels for supervised training (e.g., regime, volatility, risk tier).
- **MonsterRunner Protocols (MR01-MR20):** Detect explosive movements like compression explosions and volume anomalies.
- **Omega Directives (Ω1-Ω20):** Safety overrides including hard locks, entropy/drift/compliance overrides, and kill switches.

## Data Providers

### Primary Sources (Active)
| Provider | Rate Limit | Use Case |
|----------|------------|----------|
| **Alpaca** | 200/min | Primary OHLCV, quotes (FREE) |
| **Polygon** | 5/min | Backup OHLCV, options data |
| **Binance** | 1200/min | Crypto data (FREE) |

### Secondary Sources (Available)
- **Alpha Vantage:** Technicals, forex, crypto
- **EODHD:** International markets (70+ exchanges)
- **Finnhub:** News, sentiment, insider trades
- **CoinGecko:** 10,000+ crypto coins

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

### Google Docs Integration
Connected via Replit OAuth2 for an automated export pipeline supporting:
- Investor reports (daily/weekly/monthly)
- Due diligence packages
- Trade log exports
- Trade journals with research notes
- Monthly investor updates

## Configuration Files

### Broker Configuration (`config/broker.yaml`)
Controls execution mode, risk limits, and trading capabilities:
- Execution mode (RESEARCH/PAPER/LIVE)
- Position type toggles (long/short/margin)
- Risk parameters and limits
- EEO profile selection

### Scan Modes (`config/scan_modes.yaml`)
Pre-configured universe scanning modes:
- `momentum_runners` - High momentum across all caps
- `mid_cap_focus` - Growth potential, moderate risk
- `mega_large_focus` - Blue chips, high liquidity
- `high_vol_small_caps` - Volatile small caps
- `demo` - Testing mode (20 symbols)

## File Structure

```
quantracore-apex/
├── src/quantracore_apex/     # Backend Python source
│   ├── core/                 # Engine, schemas, types
│   ├── data_layer/           # Data adapters and normalization
│   │   └── adapters/         # Alpaca, Polygon, Binance adapters
│   ├── server/               # FastAPI application
│   ├── protocols/            # Tier, Learning, Omega protocols
│   ├── prediction/           # MonsterRunner, ApexCore models
│   ├── accuracy/             # Accuracy optimization modules
│   ├── eeo_engine/           # Entry/Exit Optimization
│   ├── risk/                 # Risk engine
│   ├── broker/               # OMS, broker adapters, execution
│   └── portfolio/            # Portfolio management
├── dashboard/                # React frontend
│   ├── src/
│   │   ├── components/       # React components (SwingTradePage, etc.)
│   │   └── index.css         # Tailwind v4 theme
│   └── vite.config.ts        # Vite configuration
├── config/                   # Configuration files
│   ├── broker.yaml           # Trading & risk config
│   └── scan_modes.yaml       # Universe scanning modes
├── static/                   # Alpha Factory static dashboard
└── tests/                    # Test suites
```

## Deployment Notes

- **Execution Mode:** Currently set to PAPER for safe paper trading
- **Live Trading:** Disabled by design - requires explicit institution approval
- **Data Sources:** Alpaca primary (200/min), Polygon backup (5/min)
- **All Workflows:** Running and healthy
