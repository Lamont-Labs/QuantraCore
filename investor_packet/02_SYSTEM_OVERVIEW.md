# QuantraCore Apex
## System Technical Overview

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  Polygon (EOD) │ Finnhub (Sentiment) │ FRED (Macro)        │
│  Alpha Vantage (News) │ Alpaca (Execution)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  PROTOCOL ENGINE                            │
│  80 Tier Protocols │ 25 Learning Protocols │ 20 Omega Rules│
│  Monster Runner Detection │ Signal Aggregation             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 ML PREDICTION LAYER                         │
│  Ensemble Models │ Forward Validation │ Confidence Scoring │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               AUTONOMOUS EXECUTION                          │
│  AutoTrader │ Position Sizing │ Stop-Loss Manager          │
│  Trade Journal │ Due Diligence Logger                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Codebase Metrics

| Metric | Count |
|--------|-------|
| Python Source Files | 423 |
| Lines of Code | 104,903 |
| API Endpoints | 263 |
| ML Model Files | 21 |
| Test Modules | 38 |
| TypeScript/React Files | 38 |

---

## Core Components

### 1. Moonshot Detection
- **Goal:** Identify stocks with 50%+ upside in 5 days
- **Target Precision:** 70%+
- **Model:** Gradient Boosting Ensemble
- **Features:** 200+ technical, fundamental, and sentiment indicators

### 2. Protocol System
- **Tier Protocols (T01-T80):** Market structure analysis
- **Learning Protocols (LP01-LP25):** Training label generation
- **Monster Runner (MR01-MR20):** Extreme move detection
- **Omega Directives (Ω01-Ω20):** Safety and compliance

### 3. Risk Management
| Rule | Configuration |
|------|---------------|
| Hard Stop | -15% from entry |
| Trailing Stop | Activates at +10%, trails 8% |
| Time Stop | Exit after 5 days if <5% gain |
| Position Sizing | Risk-based allocation |

### 4. Audit Infrastructure
- Trade Journal with full entry/exit logging
- Performance snapshots (daily)
- Forward validation (predictions recorded before outcomes)
- Compliance attestations (12 automated checks)

---

## Data Sources

| Provider | Data Type | Current Plan |
|----------|-----------|--------------|
| Polygon.io | EOD prices, historical | Free (5 calls/min) |
| Alpaca | Execution, positions | Free (200 calls/min) |
| Finnhub | Sentiment, insider data | Free |
| FRED | Macro indicators | Free |
| Alpha Vantage | News sentiment | Free |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | scikit-learn, NumPy, Pandas |
| Database | PostgreSQL |
| Frontend | React 18, TypeScript, Vite, Tailwind |
| Testing | pytest, vitest |
| Auth | X-API-Key header |

---

## Deployment

- Backend: Port 8000 (4 workers)
- Frontend: Port 5000
- Database: Managed PostgreSQL
- All running on Replit infrastructure

---

## Key Files

| Purpose | Location |
|---------|----------|
| API Server | `src/quantracore_apex/server/app.py` |
| Stop-Loss Manager | `src/quantracore_apex/broker/stop_loss_manager.py` |
| Trade Journal | `src/quantracore_apex/investor/trade_journal.py` |
| Forward Validator | `src/quantracore_apex/validation/forward_validator.py` |
| ML Models | `models/massive_ensemble_v3.pkl.gz` |
