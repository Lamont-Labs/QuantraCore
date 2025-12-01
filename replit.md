# QuantraCore Apex — Replit Project Documentation

## Overview

QuantraCore Apex is an institutional-grade, deterministic AI trading intelligence engine designed for desktop use. It integrates a complete offline learning ecosystem (ApexLab) and on-device neural assistant models (ApexCore). The system supports all trading types, including long, short, intraday, swing, and scalping, primarily through Alpaca paper trading. Its core purpose is to provide sophisticated, AI-driven trading capabilities with a focus on determinism, accuracy optimization, and self-learning mechanisms for both retail and institutional users. The project aims to offer a robust, self-improving trading platform with advanced risk management and comprehensive reporting.

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
- **Security:** Implements `X-API-Key` authentication for protected endpoints and a restrictive CORS policy allowing only localhost and Replit domains. A TTL cache with a 1000-entry limit and 5-minute expiration is used.

### Feature Specifications
- **Full Trading Capabilities:** Supports long, short, margin (up to 4x leverage), intraday, swing, and scalping trades with configurable risk limits ($100K max exposure, $10K per symbol, 50 positions max).
- **Order Types:** MARKET, LIMIT, STOP, STOP_LIMIT.
- **Entry/Exit Strategies:** Auto-selected entry strategies based on market conditions (e.g., high volatility, low liquidity) and comprehensive exit strategies including protective stops (ATR-based, ZDE adjustments), trailing stops, profit targets, and time-based exits.
- **Deterministic Core:** Ensures identical outputs for identical inputs.
- **Offline ML (ApexCore v3):** On-device neural models with 5 prediction heads (quantrascore, runner, quality, avoid, regime), trained on real market data with actual outcome labels. Current model trained on **42,632 samples** from **56 symbols** using **91,396 15-minute intraday bars** via Alpaca IEX feed, achieving **98.73% runner accuracy**, **92.94% quality accuracy**, **99.88% avoid accuracy**, and **94.16% regime accuracy**. QuantraScore RMSE: 0.011.
- **Unified Training Pipeline:** Multi-source training using both Alpaca and Polygon data feeds in parallel. Generates training labels from actual future price movements (5-day returns, max runup, drawdowns). Endpoint: `POST /apexlab/train-unified`.
- **Accuracy Optimization System:** 8-module suite for calibration, regime-gating, uncertainty quantification, auto-retraining, and multi-horizon prediction.
- **Self-Learning (Alpha Factory):** A 24/7 live research loop using WebSockets for a self-learning feedback mechanism and automatic retraining based on batch thresholds.
- **MarketSimulator:** Provides 8 chaos scenarios for stress testing the system.
- **Protocol System:** Comprises 80 Tier protocols for analysis, 25 Learning protocols for training labels, 20 MonsterRunner protocols for explosive movement detection, and 20 Omega Directives as safety overrides.
- **Investor Trade Logging:** Comprehensive logging of every paper trade with details on account state, signal quality, market context, risk assessment, protocol analysis, execution details, and trade outcome, stored in `investor_logs/`.

### System Design Choices
- **Broker Layer:** Supports `NullAdapter` (research), `PaperSimAdapter` (offline simulation), and `AlpacaPaperAdapter` (live paper trading).
- **Configuration:** Trading and risk parameters are managed via `config/broker.yaml`, and universe scanning modes are configured in `config/scan_modes.yaml`.
- **Google Docs Integration:** Utilizes Replit OAuth2 for automated export of investor reports, due diligence packages, and trade logs.

## External Dependencies

- **Alpaca Markets API:** Primary broker for paper trading execution ($100k equity). Free IEX data feed provides 7+ years of historical data (using `feed=iex` parameter). 15-minute intraday bars for high-resolution pattern detection.
- **Polygon.io:** Primary data source for ML training (5 requests/minute). Successfully training models on real OHLCV data from 15+ symbols.
- **Binance:** Primary source (1200 requests/minute) for crypto data, used in Alpha Factory.
- **Google Docs:** Integrated via Replit OAuth2 for automated reporting and export functionalities.
- **Alpha Vantage:** (Available) For technicals, forex, crypto.
- **EODHD:** (Available) For international markets.
- **Finnhub:** (Available) For news, sentiment, insider trades.
- **CoinGecko:** (Available) For cryptocurrency data.