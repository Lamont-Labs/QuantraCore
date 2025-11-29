# QuantraCore Apex

**Institutional-Grade Deterministic AI Trading Intelligence Engine**

[![Tests](https://img.shields.io/badge/tests-1099%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)]()
[![React](https://img.shields.io/badge/React-18.2-61DAFB)]()
[![License](https://img.shields.io/badge/license-proprietary-red)]()
[![Status](https://img.shields.io/badge/status-operational-success)]()

**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** v9.0-A (Institutional Hardening)  
**Architecture:** Desktop-Only | Research/Backtest Mode  

---

## System Status

| Component | Status | Details |
|-----------|--------|---------|
| **ApexEngine** | Operational | Deterministic core with 80 Tier protocols |
| **ApexDesk UI** | Operational | React 18 + Vite 5 + Tailwind CSS 3 |
| **FastAPI Backend** | Operational | 27 REST endpoints on port 8000 |
| **Test Suite** | **1,099 tests passing** | Regulatory excellence + hardening + institutional |
| **Universal Scanner** | Operational | 7 market cap buckets, 8 scan modes |
| **ApexLab** | Operational | V1 + V2 offline training environment |
| **ApexCore Models** | Operational | V1 + V2 neural models (scikit-learn) |
| **ApexLab V2** | Operational | 40+ field schema, runner/monster/safety labels |
| **ApexCore V2** | Operational | Big/Mini models, 5 heads, manifest verification |
| **PredictiveAdvisor** | Operational | Fail-closed engine integration |
| **Hardening** | Operational | Protocol manifest, mode enforcement, kill switch |
| **Broker Layer** | Operational | Paper trading with 9-check risk engine |
| **EEO Engine** | Operational | Entry/exit optimization, 3 policy profiles |

---

## What This System Does

QuantraCore Apex is a **research and backtesting platform** that:

- Analyzes market structure using **80 Tier Protocols** (T01-T80)
- Computes **QuantraScore** (0-100) for structural probability assessment
- Detects regime states, entropy levels, drift conditions, and suppression patterns
- Identifies potential extreme moves via **MonsterRunner** detection
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
| **Live Trading** | No brokerage connections or order execution |
| **Financial Advice** | All outputs are structural probabilities only |
| **Mobile/Android** | Desktop-only architecture (strictly prohibited) |
| **Cloud Dependencies** | Runs entirely locally |
| **Guaranteed Returns** | Past analysis does not predict future results |

---

## Test Coverage

```
1,099 tests | 100% pass rate
```

### Test Breakdown

| Category | Tests | Description |
|----------|-------|-------------|
| **Hardening** | 34 | Protocol manifest, mode enforcement, kill switch |
| **Broker Layer** | 34 | Order routing, risk engine, adapters |
| **EEO Engine** | 42 | Entry/exit optimization, profiles |
| **Core Engine** | 78 | ApexEngine instantiation, execution, result validation |
| **Protocols** | 77 | Tier protocol loading, execution, results |
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
# Full suite (1,099 tests)
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

### 5 MonsterRunner Protocols (MR01-MR05)

Extreme move detection: phase compression, volume ignition, entropy collapse.

### 5 Omega Directives (Ω1-Ω5)

| Directive | Trigger | Effect |
|-----------|---------|--------|
| Ω1 | Extreme risk tier | Hard safety lock |
| Ω2 | Chaotic entropy | Entropy override |
| Ω3 | Critical drift | Drift override |
| Ω4 | Always active | Compliance mode |
| Ω5 | Strong suppression | Signal suppression lock |

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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | System statistics |
| `/scan_symbol` | POST | Single symbol analysis |
| `/scan_universe` | POST | Multi-symbol batch scan |
| `/trace/{hash}` | GET | Full protocol trace |
| `/monster_runner/{symbol}` | POST | Extreme move check |
| `/risk/assess/{symbol}` | POST | Risk assessment |
| `/signal/generate/{symbol}` | POST | Signal generation |
| `/portfolio/status` | GET | Portfolio snapshot |

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

dashboard/          # React 18 + Vite 5 + Tailwind CSS 3 frontend
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

| Provider | Usage |
|----------|-------|
| **Polygon.io** | Real market data (requires API key) |
| **Alpha Vantage** | Alternative data source |
| **Yahoo Finance** | Backup provider |
| **CSV Bundle** | Historical data import |
| **Synthetic** | Testing without API keys |

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
