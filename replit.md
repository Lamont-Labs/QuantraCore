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

## User Preferences

- **Communication:** Simple language with detailed explanations
- **Workflow:** Iterative development with confirmation before major changes
- **Coding Style:** Functional programming paradigms preferred
- **Restrictions:** Do not modify folder `Z` or file `Y`

## System Architecture

### Technology Stack
| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| Frontend | React 18.2, Vite 5, Tailwind CSS 3.4, TypeScript |
| Machine Learning | scikit-learn (GradientBoosting), joblib |
| Numerical | NumPy, Pandas |
| Testing | pytest (backend), vitest (frontend) |

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

### Data Providers
- **Polygon.io:** Primary market data provider.
- **Alpha Vantage:** Backup market data provider (optional).
- **Yahoo Finance:** Fallback market data provider.
- **Financial Modeling Prep:** Fundamental data.
- **Nasdaq Data Link:** Economic data.
- **Unusual Whales, FlowAlgo, InsiderFinance:** Options flow data.
- **Finnhub, AltIndex, Stocktwits:** Alternative data (news, sentiment).
- **Binance, CoinGecko:** Cryptocurrency data.
- **Synthetic:** For testing and demos.

### Broker Integration
- **Alpaca Paper:** For paper trading.

### Google Docs Integration
Connected via Replit OAuth2 for an automated export pipeline supporting:
- Investor reports (daily/weekly/monthly)
- Due diligence packages
- Trade log exports
- Trade journals with research notes
- Monthly investor updates