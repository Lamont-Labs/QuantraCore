# QuantraCore Apex — Replit Project Documentation

**Version:** 9.0-A | **Updated:** 2025-11-29

> For complete technical specifications, see [MASTER_SPEC.md](./MASTER_SPEC.md)

## Overview

QuantraCore Apex v9.0-A is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It features a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system is strictly for **research and backtesting only**, providing structural probabilities rather than trading advice.

### Key Capabilities
- **145+ Protocols:** 80 Tier + 25 Learning + 20 MonsterRunner + 20 Omega
- **QuantraScore:** 0-100 probability-weighted composite score
- **Deterministic Core:** Same inputs always produce identical outputs
- **Offline ML:** On-device ApexCore v2 neural models (scikit-learn)
- **Paper Trading:** Alpaca integration (LIVE mode disabled)
- **Self-Learning:** Alpha Factory feedback loop with automatic retraining
- **Google Docs Export:** Automated investor/acquirer reporting pipeline

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Directory Structure
```
quantracore-apex/
├── src/quantracore_apex/    # Main source code
│   ├── core/               # ApexEngine core modules
│   ├── protocols/          # 145+ protocol implementations
│   ├── apexcore/           # Neural models (v2)
│   ├── apexlab/            # Training pipeline (v2)
│   ├── broker/             # Broker adapters
│   ├── eeo_engine/         # Entry/Exit Optimization
│   ├── alpha_factory/      # 24/7 live research loop
│   ├── simulator/          # MarketSimulator chaos engine
│   ├── data_layer/         # Data providers
│   ├── hardening/          # Safety infrastructure
│   ├── integrations/       # External integrations
│   └── server/             # FastAPI server
├── config/                  # Configuration files
├── data/                    # Training data and caches
├── dashboard/               # React frontend (ApexDesk)
├── tests/                   # 1,145+ tests
├── MASTER_SPEC.md           # Complete technical specification
└── replit.md                # This file
```

### Technology Stack
| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | React 18.2, Vite 5, Tailwind CSS 3.4, TypeScript |
| Machine Learning | scikit-learn (GradientBoosting), joblib |
| Numerical | NumPy, Pandas |
| Testing | pytest (backend), vitest (frontend) |

## Protocol System

### Tier Protocols (T01-T80)
80 deterministic analysis protocols covering:
- **Trend Analysis (T01-T05):** Direction, strength, momentum, channels, crossovers
- **Volatility (T06-T10, T21-T30):** Expansion, compression, squeezes
- **Continuation (T11-T15):** Bull/bear flags, pennants, wedges
- **Reversal (T16-T20):** Double tops/bottoms, H&S patterns
- **Momentum (T31-T40):** RSI, MACD, stochastics, divergences
- **Volume (T41-T50):** Spikes, OBV, VWAP, climax detection
- **Patterns (T51-T60):** Candlesticks, triangles, cup/handle
- **S/R Levels (T61-T70):** Support/resistance, Fibonacci, pivots
- **Assessment (T71-T80):** Regime, trend maturity, composite

### Learning Protocols (LP01-LP25)
Label generation for supervised training: regime, volatility, risk tier, momentum, quality, runner detection.

### MonsterRunner Protocols (MR01-MR20)
Explosive move detection: compression explosion, volume anomaly, squeeze detection, gamma ramp, nuclear runners.

### Omega Directives (Ω1-Ω20)
Safety overrides: hard locks, entropy/drift/compliance overrides, kill switches.

## Core Systems

### ApexEngine
Primary analysis engine processing 100-bar OHLCV windows:
1. Compute microtraits (15+ features)
2. Calculate entropy, suppression, drift, continuation metrics
3. Classify regime
4. Generate QuantraScore (0-100)
5. Build verdict with risk tier
6. Run 80 Tier protocols
7. Apply 20 Omega directives

### ApexCore v2
Multi-head neural models with 5 prediction heads:
- QuantraScore regression (0-100)
- Runner probability classification
- Quality tier (A+, A, B, C, D)
- Avoid-trade probability
- Regime classification

### ApexLab v2
Training pipeline with walk-forward validation, bootstrap ensembles, and 40+ field schema.

### EEO Engine
Entry/Exit Optimization with:
- Entry strategies (baseline, high-vol, low-liquidity, runner, ZDE-aware)
- Exit optimization (stops, targets, trailing, time-based)
- Position sizing (fixed-fraction risk model)

### Alpha Factory
24/7 live research loop with:
- Polygon (equities) and Binance (crypto) WebSockets
- Self-learning feedback loop
- Automatic retraining on batch threshold

### MarketSimulator
8 chaos scenarios for stress testing: flash crash, volatility spike, gap event, liquidity void, momentum exhaustion, squeeze, correlation breakdown, black swan.

### Broker Layer
- **NullAdapter:** Research mode (logs only)
- **PaperSimAdapter:** Offline simulation
- **AlpacaPaperAdapter:** Alpaca paper trading
- **LIVE mode:** Disabled

## External Dependencies

### Data Providers
- **Polygon.io:** Primary (requires `POLYGON_API_KEY`)
- **Alpha Vantage:** Backup (optional)
- **Yahoo Finance:** Fallback
- **Synthetic:** Testing/demo

### Broker Integration
- **Alpaca Paper:** Requires `ALPACA_PAPER_API_KEY`, `ALPACA_PAPER_API_SECRET`
- **PaperSim:** Internal simulator

### Google Docs Integration
Connected via Replit OAuth2 connector for automated export pipeline:

**Export Types:**
- Investor reports (daily/weekly/monthly)
- Due diligence packages for acquirers
- Trade log exports
- Trade journals with research notes
- Monthly investor updates

**API Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/google-docs/status` | GET | Connection status |
| `/google-docs/export/investor-report` | POST | Generate investor report |
| `/google-docs/export/due-diligence` | POST | Generate DD package |
| `/google-docs/export/trade-log` | POST | Export trade history |
| `/google-docs/documents` | GET | List exported documents |
| `/google-docs/journal/entry` | POST | Add journal entry |
| `/google-docs/journal/today` | GET | Get today's journal |
| `/google-docs/journal/list` | GET | List all journals |
| `/google-docs/investor-update/monthly` | POST | Monthly update |

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `POLYGON_API_KEY` | Polygon.io data | For live data |
| `ALPHA_VANTAGE_API_KEY` | Backup data | Optional |
| `ALPACA_PAPER_API_KEY` | Alpaca paper trading | For paper mode |
| `ALPACA_PAPER_API_SECRET` | Alpaca auth | For paper mode |

## Testing

- **Total Tests:** 1,145+
- **Regulatory Tests:** 163+
- **Run:** `pytest tests/ -v`

## Workflows

| Workflow | Command | Port |
|----------|---------|------|
| FastAPI Backend | `uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000` | 8000 |
| ApexDesk Frontend | `cd dashboard && npm run dev` | 5000 |
| Alpha Factory Dashboard | `cd static && python -m http.server 8080` | 8080 |

## Recent Changes

- **2025-11-29:** Added HyperLearner — Hyper-Velocity Learning System
  - **EventBus**: Universal event capture with priority queuing
  - **OutcomeTracker**: Links events to outcomes, creates learning pairs
  - **PatternMiner**: Discovers win/loss patterns from historical data
  - **ContinuousTrainer**: Prioritized retraining with batch processing
  - **MetaLearner**: Optimizes the learning process itself
  - Added 12 new API endpoints for learning management
  - Decorator hooks for automatic learning capture (@learn_from_scan, etc.)
  - Every action, outcome, win, loss, pass, fail feeds the learning loop
  - Updated MASTER_SPEC.md with Section 23: HyperLearner

- **2025-11-29:** Added Battle Simulator — Competitive Intelligence
  - **100% Legal & Compliant**: Uses only public SEC EDGAR filings
  - **SECEdgarClient**: Fetches 13F institutional holdings from SEC
  - **StrategyAnalyzer**: Fingerprints institutional trading strategies
  - **BattleEngine**: Compares our signals against institutional actions
  - **AdversarialLearner**: Learns from institutions to improve
  - **AcquirerAdapter**: Adapts to Bloomberg, Refinitiv, or custom infrastructure
  - Added 8 new API endpoints for battle simulation
  - Updated MASTER_SPEC.md with Section 22: Battle Simulator

- **2025-11-29:** Added Autonomous Trading System
  - Created complete `autonomous/` package with institutional-grade components
  - **TradingOrchestrator**: Main async loop coordinating all subsystems
  - **SignalQualityFilter**: Enforces QuantraScore ≥75, A+/A tiers only
  - **PositionMonitor**: Real-time position tracking with stop-loss/targets
  - **TradeOutcomeTracker**: Feedback loop integration for self-learning
  - **PolygonWebSocketStream**: Real-time data with reconnection handling
  - **RollingWindowManager**: 100-bar windows for ApexEngine analysis
  - Added `scripts/run_autonomous.py` runner with CLI args
  - Updated MASTER_SPEC.md with Section 21: Autonomous Trading System

- **2025-11-29:** Created MASTER_SPEC.md with complete system documentation
  - 80 Tier protocols documented with descriptions
  - 25 Learning Protocols with output types
  - 20 MonsterRunner protocols with risk multipliers
  - 20 Omega Directives with trigger conditions
  - Full broker layer, EEO engine, ApexCore/Lab documentation
  - API reference with all endpoints
  - Configuration and deployment guides

- **2025-11-29:** Added automated Google Docs export pipeline
  - Created `automated_pipeline.py` with performance metrics collection
  - Added 9 new API endpoints for document export
  - Exports include live trading data, broker snapshots, ML training progress
  - Successfully tested with live export to Google Docs

## Documentation

- **MASTER_SPEC.md:** Complete technical specification (update on every change)
- **replit.md:** Quick reference and project overview
- **config/*.yaml:** Configuration files with inline documentation

---

**Classification:** Research Only | **Mode:** Paper Trading | **Live Trading:** Disabled
