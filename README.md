# QuantraCore Apex

**Institutional-Grade Autonomous AI Trading System for Moonshot Detection**

[![Mode](https://img.shields.io/badge/mode-PAPER%20TRADING-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)]()
[![React](https://img.shields.io/badge/React-18.2-61DAFB)]()
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6)]()
[![Vite](https://img.shields.io/badge/Vite-7.2-646CFF)]()
[![Tailwind](https://img.shields.io/badge/Tailwind-4.0-06B6D4)]()
[![Desktop](https://img.shields.io/badge/platform-desktop%20only-orange)]()
[![License](https://img.shields.io/badge/license-proprietary-red)]()
[![Status](https://img.shields.io/badge/status-production--ready-success)]()
[![Code](https://img.shields.io/badge/lines-105K-informational)]()
[![Endpoints](https://img.shields.io/badge/API-263%20endpoints-blueviolet)]()

**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** v9.0-A (Production-Ready Paper Trading)  
**Architecture:** Desktop-Only | Mode: PAPER (Alpaca Connected)  
**Stage:** Beta / Production-Ready  
**Last Updated:** 2025-12-04  

---

## For Investors

| Document | Description |
|----------|-------------|
| **[INVESTOR_CONCERNS_ADDRESSED.md](INVESTOR_CONCERNS_ADDRESSED.md)** | Direct response to due diligence concerns |
| **[docs/investor/](docs/investor/)** | Complete investor portal (30+ documents) |
| **[MASTER_SPEC.md](MASTER_SPEC.md)** | Full technical specification (3,400+ lines) |

### Quick Facts for Investors

| Metric | Value |
|--------|-------|
| **Codebase** | 104,903 lines Python, 423 source files |
| **Frontend** | 38 TypeScript/React files |
| **API Endpoints** | 263 REST endpoints |
| **ML Models** | 21 trained models (primary: massive_ensemble_v3) |
| **Protocols** | 145 (T01-T80, LP01-LP25, MR01-MR20, Ω01-Ω20) |
| **Target** | 50%+ gains in 5 days, 70%+ precision |
| **Current Status** | 11 active paper trading positions |
| **Commercial Paths** | IP Acquisition, Licensing, SaaS |

---

## System Status

| Component | Status | Details |
|-----------|--------|---------|
| **ApexEngine** | Operational | Deterministic core with 80 Tier protocols |
| **ApexDesk UI** | Operational | React 18.2 + Vite 7.2 + Tailwind CSS 4 |
| **FastAPI Backend** | Operational | **263 REST endpoints** on port 8000 |
| **Universal Scanner** | Operational | 7 market cap buckets, 4 scan modes |
| **Alpaca Paper Trading** | **Connected** | 11 active positions |
| **Moonshot Detection** | Operational | massive_ensemble_v3 model (50%+ gain detection) |
| **Stop-Loss System** | Operational | -15% hard, +10%/8% trailing, 5-day time stop |
| **Forward Validation** | Operational | Real-time prediction tracking |
| **Data Layer** | Operational | Polygon, Alpaca, FRED, Finnhub, Alpha Vantage |
| **ApexCore Models** | Operational | 21 trained models |
| **Broker Layer** | Operational | Paper trading with risk engine |
| **Velocity Mode** | Operational | Standard (30s), High (5s), Turbo (2s) |

### Trading Capabilities (All Enabled)

| Type | Status | Description |
|------|--------|-------------|
| Long | Enabled | Standard long positions |
| Short | Enabled | Short selling enabled |
| Margin | Enabled | Up to 4x leverage |
| Intraday | Enabled | Same-day trades |
| Swing | Enabled | Multi-day holds |
| Scalping | Enabled | Sub-5 minute trades |

### Codebase Statistics

| Metric | Value |
|--------|-------|
| **Python Source Files** | 423 files |
| **Python Lines** | 104,903 lines |
| **TypeScript/React** | 38 files |
| **API Endpoints** | 263 REST endpoints |
| **ML Models** | 21 trained models |
| **Test Modules** | 38 test files |

---

## What This System Does

QuantraCore Apex is an **autonomous moonshot detection and trading system** that:

- **Detects Moonshot Candidates**: Identifies stocks ready for 50%+ gains within 5 trading days
- **Autonomous Execution**: AutoTrader automatically enters positions on high-confidence signals
- **Automatic Stop-Loss Management**: -15% hard stop, trailing stops (activates at +10%, trails 8%), 5-day time limit
- **Forward Validation**: Tracks every prediction to prove real-world accuracy (70%+ precision target)
- **EOD-Based Analysis**: Uses end-of-day data for reliable, non-noisy signals
- Analyzes market structure using **80 Tier Protocols** (T01-T80)
- Computes **QuantraScore** (0-100) for structural probability assessment
- Identifies potential extreme moves via **20 MonsterRunner protocols** (MR01-MR20)
- **ApexCore V4**: Multi-head models with 16 output heads
- Provides deterministic, reproducible analysis with cryptographic hashing

### Core Metrics Computed

| Metric | Description |
|--------|-------------|
| **QuantraScore** | Structural probability score (0-100) |
| **Regime** | Market state (trending_up, trending_down, range_bound, unknown) |
| **Risk Tier** | Risk classification (low, medium, high, extreme) |
| **Entropy State** | Market chaos level (stable, elevated, chaotic) |
| **Suppression State** | Coil/compression detection (none, light, moderate, heavy) |
| **Drift State** | Statistical distribution shift (normal, warning, critical) |
| **Microtraits** | 10+ computed features (wick_ratio, volatility, compression, noise, etc.) |

---

## What This System Does NOT Do

| Exclusion | Reason |
|-----------|--------|
| **Live Trading** | Paper trading only (LIVE mode disabled) |
| **Financial Advice** | All outputs are structural probabilities only |
| **Mobile/Android** | Desktop-only architecture (strictly prohibited) |
| **Cloud Dependencies** | Runs entirely locally |
| **Guaranteed Returns** | Past analysis does not predict future results |

---

## Testing

The system includes 38 test modules covering:

- **Core Engine**: ApexEngine instantiation, execution, result validation
- **Broker Layer**: Order routing, risk engine, adapters
- **Protocols**: Tier protocol loading, execution, Omega/MR
- **Scanner**: Universe scanner, volatility tags, regime detection
- **Model**: ApexCore model loading, inference, validation
- **Regulatory**: SEC/FINRA/MiFID II compliance

### Run Tests

```bash
pytest tests/
```

---

## Protocol System

### 80 Tier Protocols (T01-T80)

| Range | Category | Description |
|-------|----------|-------------|
| T01-T10 | Core | Trend, ADX, momentum alignment |
| T11-T20 | Volatility | Bollinger, ATR, range analysis |
| T21-T30 | Momentum | RSI, MACD, ROC divergence |
| T31-T40 | Volume | OBV, volume spikes, accumulation |
| T41-T50 | Pattern | Wedges, triangles, channels |
| T51-T60 | Support/Resistance | Key levels, breakout detection |
| T61-T70 | Market Context | Sector correlation, breadth |
| T71-T80 | Advanced | Multi-timeframe, rare events |

### 25 Learning Protocols (LP01-LP25)

Label generation for ApexLab training pipelines.

### 20 MonsterRunner Protocols (MR01-MR20)

| Range | Category | Description |
|-------|----------|-------------|
| MR01-MR05 | Core | Compression, volume, regime, institutional, alignment |
| MR06-MR10 | Breakout | Bollinger, volume explosion, gap, VWAP, NR7 |
| MR11-MR15 | Extreme | Short squeeze, pump, catalyst, fractal, 100% day |
| MR16-MR20 | Parabolic | Phase 3, meme frenzy, gamma ramp, FOMO, nuclear |

### 20 Omega Directives (Ω1-Ω20)

| Range | Category | Description |
|-------|----------|-------------|
| Ω1-Ω5 | Core Safety | Risk lock, entropy, drift, compliance, suppression |
| Ω6-Ω10 | Volatility | Vol cap, divergence, squeeze, MACD, fear spike |
| Ω11-Ω15 | Indicators | RSI extreme, volume spike, trend, gap, tail risk |
| Ω16-Ω20 | Advanced | Overnight, fractal, liquidity, correlation, nuclear |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd dashboard && npm install && cd ..
```

### 2. Start Backend

```bash
uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000
```

### 3. Start Frontend

```bash
cd dashboard && npm run dev
```

### 4. Open ApexDesk

```
http://localhost:5000
```

---

## API Endpoints (263 Total)

### Core Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/trading_capabilities` | GET | Trading config and limits |
| `/data_providers` | GET | Data source status |
| `/scan_symbol` | POST | Single symbol analysis |
| `/scan_universe_mode` | POST | Universe scan with mode |
| `/trace/{hash}` | GET | Full protocol trace |

### Trading & Execution
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/broker/status` | GET | Broker connection status |
| `/broker/execute` | POST | Execute trade |
| `/portfolio/status` | GET | Current portfolio & positions |
| `/oms/orders` | GET | Order history |

### Stop-Loss Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stops/status` | GET | Stop-loss status for all positions |
| `/stops/check/{symbol}` | GET | Check stop for specific symbol |
| `/stops/config` | POST | Update stop-loss configuration |
| `/stops/exit-signals` | GET | Get positions needing exit |

### Predictions & Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictive/advise` | POST | Get prediction advisory |
| `/moonshot/detect` | POST | Moonshot candidate detection |
| `/validation/status` | GET | Forward validation status |
| `/validation/record` | POST | Record new prediction |

### Example API Call

```bash
curl -X POST http://localhost:8000/scan_symbol \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

### Example Response

```json
{
  "symbol": "AAPL",
  "quantrascore": 38.46,
  "score_bucket": "low",
  "regime": "range_bound",
  "risk_tier": "high",
  "entropy_state": "stable",
  "suppression_state": "none",
  "drift_state": "critical",
  "verdict_action": "structural_weakness",
  "verdict_confidence": 0.823,
  "omega_alerts": ["omega_3_drift", "omega_4_compliance"],
  "protocol_fired_count": 35,
  "window_hash": "d680e6cc41aabd1c",
  "timestamp": "2025-11-29T00:27:19.991188"
}
```

---

## Architecture

```
src/quantracore_apex/
├── core/           # ApexEngine, schemas, microtraits, quantrascore
├── protocols/      # Tier (T01-T80), Learning (LP01-LP25), MonsterRunner
├── data_layer/     # Adapters (Polygon, Alpha Vantage, Synthetic, CSV)
├── apexlab/        # Offline training: windows, features, labels
├── apexcore/       # Neural models (Full + Mini)
├── prediction/     # Prediction engines (expected move, volatility, etc.)
├── hardening/      # Safety infrastructure: manifest, mode, kill switch
├── broker/         # Execution engine, risk, adapters
├── eeo_engine/     # Entry/exit optimization
├── server/         # FastAPI application
└── compliance/     # Regulatory excellence engine

tests/              # 1,099 institutional-grade tests

dashboard/          # React 18.2 + Vite 7.2 + Tailwind CSS 4.0 frontend (9 panels)
```

---

## Hardware Targets

| Platform | Specification |
|----------|---------------|
| Target Device | GMKtec NucBox K6 |
| CPU | 8-core recommended |
| RAM | 16GB recommended |
| GPU | Optional (CPU-optimized) |
| OS | Linux, Windows, macOS |

---

## Data Providers

| Provider | Rate Limit | Usage |
|----------|------------|-------|
| **Alpaca** | 200/min | Primary OHLCV data (FREE) |
| **Polygon.io** | 5/min | Backup data, options |
| **Binance** | 1200/min | Crypto data (FREE) |
| **Synthetic** | Unlimited | Testing without API keys |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Master Spec v9.0-A](docs/QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md) | Complete system specification |
| [Getting Started](docs/GETTING_STARTED_DESKTOP.md) | Setup and first steps |
| [Compliance & Safety](docs/COMPLIANCE_AND_SAFETY.md) | Research-only constraints |
| [Architecture](docs/ARCHITECTURE.md) | System design |
| [API Reference](docs/API_REFERENCE.md) | REST API documentation |
| [ApexLab Training](docs/APEXLAB_TRAINING.md) | Offline training guide |
| [ApexCore Models](docs/APEXCORE_MODELS.md) | Neural model documentation |

---

## Compliance Statement

This software is for **educational and research purposes only**.

- All outputs are **structural probability assessments**, not trading signals
- **No financial advice** is provided or implied
- **Live trading is disabled** by default (research mode enforced)
- **Omega Directive Ω4** ensures compliance mode is always active
- Users assume **full responsibility** for any financial decisions

---

## Contact

**Jesse J. Lamont** - Founder, Lamont Labs  
Email: lamontlabs@proton.me  
GitHub: https://github.com/Lamont-Labs

---

## Disclaimer

This is a demonstration and research repository only. No trading advice or financial activity is provided. All data is synthetic or public domain. No production trading systems are connected. Past analysis does not guarantee future results.

---

**Persistence = Proof.**  
Every build, every log, every checksum - reproducible by design.

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
