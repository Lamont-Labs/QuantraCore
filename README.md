# QuantraCore Apex

**Institutional-Grade Deterministic AI Trading Intelligence Engine**

[![Tests](https://img.shields.io/badge/tests-1145%20passed-brightgreen)]()
[![Compliance](https://img.shields.io/badge/regulatory-163%2B%20tests-blueviolet)]()
[![Security](https://img.shields.io/badge/security-fail--closed-critical)]()
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
[![Code](https://img.shields.io/badge/lines-121K+-informational)]()
[![Endpoints](https://img.shields.io/badge/API-148%20endpoints-blueviolet)]()
[![Dashboard](https://img.shields.io/badge/Dashboard-9%20panels-cyan)]()

**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** v9.0-A (Production-Ready Paper Trading)  
**Architecture:** Desktop-Only | Mode: PAPER (Alpaca Connected)  
**Stage:** Beta / Production-Ready  

---

## System Status

| Component | Status | Details |
|-----------|--------|---------|
| **ApexEngine** | Operational | Deterministic core with 80 Tier protocols |
| **ApexDesk UI** | Operational | React 18.2 + Vite 7.2 + Tailwind CSS 4 (9 panels) |
| **FastAPI Backend** | Operational | **148 REST endpoints** on port 8000 |
| **Test Suite** | **1,145 tests passing** | Regulatory excellence + hardening + institutional |
| **Universal Scanner** | Operational | 7 market cap buckets, 4 scan modes |
| **Alpaca Paper Trading** | **Connected** | All position types enabled |
| **Data Layer** | Operational | Alpaca (200/min) + Polygon (5/min) |
| **ApexCore Models** | Operational | V3 neural models with 7 prediction heads |
| **Broker Layer** | Operational | Paper trading with 9-check risk engine |
| **EEO Engine** | Operational | Entry/exit optimization, 6 strategies |
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
| **Total Files** | 516 source files |
| **Total Lines** | 121,207 lines of code |
| **Python** | 459 files / 88,921 lines |
| **TypeScript/React** | 18 files / 3,123 lines |
| **API Endpoints** | 123 REST endpoints |

---

## What This System Does

QuantraCore Apex is a **research and backtesting platform** that:

- Analyzes market structure using **80 Tier Protocols** (T01-T80)
- Computes **QuantraScore** (0-100) for structural probability assessment
- Detects regime states, entropy levels, drift conditions, and suppression patterns
- Identifies potential extreme moves via **20 MonsterRunner protocols** (MR01-MR20)
- Trains local neural models via **ApexLab** offline learning environment (V1 + V2)
- **ApexLab V2**: 40+ field labeling with runner/monster/safety labels
- **ApexCore V2**: Multi-head models with 5 output heads (quantra_score, runner_prob, quality_tier, avoid_trade, regime)
- **PredictiveAdvisor**: Fail-closed integration with manifest verification
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

## Test Coverage

```
1,145 tests | 100% pass rate
```

### Test Breakdown

| Category | Tests | Description |
|----------|-------|-------------|
| **Hardening** | 34 | Protocol manifest, mode enforcement, kill switch |
| **Broker Layer** | 34 | Order routing, risk engine, adapters |
| **EEO Engine** | 42 | Entry/exit optimization, profiles |
| **Core Engine** | 78 | ApexEngine instantiation, execution, result validation |
| **Protocols** | 123 | Tier protocol loading, execution, extended Omega/MR |
| **Scanner** | 78 | Universe scanner, volatility tags, regime detection |
| **Model** | 68 | ApexCore model loading, inference, validation |
| **Lab** | 97 | ApexLab label generation, feature extraction |
| **Performance** | 19 | Protocol latency, scan speed benchmarks |
| **Matrix** | 39 | Cross-symbol protocol matrix validation |
| **Extreme** | 71 | Edge cases, boundary conditions |
| **Nuclear** | 106 | Determinism verification, bit-identical reproducibility |
| **Regulatory** | 163+ | SEC/FINRA/MiFID II/Basel compliance (2x-5x stricter) |
| **Predictive** | 16 | ApexCore V2 integration |
| **API/CLI** | 7 | REST endpoints, CLI commands |
| **E2E Integration** | 26 | End-to-end system validation |

### Run Tests

```bash
# Full suite (1,145 tests)
make test

# End-to-end integration (26 tests)
make test-e2e

# Quick smoke test
make test-smoke
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

## API Endpoints (123 Total)

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
| `/broker/positions` | GET | Current positions |
| `/oms/orders` | GET | Order history |
| `/eeo/plan` | POST | Entry/exit optimization |

### Predictions & Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictive/advise` | POST | Get prediction advisory |
| `/monster_runner/{symbol}` | POST | Monster runner detection |
| `/estimated_move/{symbol}` | GET | Expected move calculation |

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

dashboard/          # React 18 + Vite 5 + Tailwind CSS 4 frontend
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
