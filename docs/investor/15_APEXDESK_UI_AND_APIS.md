# ApexDesk UI and APIs

**Document Classification:** Investor Due Diligence — Technical  
**Version:** 9.0-A  
**Date:** November 2025  

---

## How Do PMs/Analysts Interact With the System?

ApexDesk is the React-based dashboard that provides a visual interface for research analysts, portfolio managers, and quantitative researchers to interact with QuantraCore Apex.

---

## ApexDesk Overview

### Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | React 18.2 |
| Build Tool | Vite 7.2 |
| Language | TypeScript |
| Styling | Tailwind CSS 4.0 |
| State | React hooks + Context |
| Port | 5000 (Frontend) |

### Dashboard Panels (9 Total)

The institutional trading dashboard provides real-time monitoring and control via 9 specialized panels:

| Panel | Description | Key Features |
|-------|-------------|--------------|
| **SystemStatusPanel** | Real-time system health monitoring | Market hours, broker status, compliance score, data feeds, ApexCore model status |
| **PortfolioPanel** | Live Alpaca paper trading portfolio | Total equity, cash, positions with P&L, real-time updates (15s intervals) |
| **TradingSetupsPanel** | Top trading opportunities ranked by QuantraScore | Signal ranking, conviction tiers, entry/stop/target levels, timing guidance |
| **ModelMetricsPanel** | ApexCore V3 model performance | 7 prediction heads, training samples, accuracy metrics, model reload |
| **AutoTraderPanel** | Autonomous swing trade execution | Trade status, position hold decisions, continuation probability |
| **SignalsAlertsPanel** | Live signals and SMS alert status | Active signals, Twilio SMS configuration, alert history |
| **RunnerScreenerPanel** | Low-float penny stock scanner | 110 symbols, volume surge detection, momentum alerts |
| **ContinuousLearningPanel** | ML training orchestrator status | Drift detection, incremental learning, training schedule |
| **LogsProvenancePanel** | System logs and audit trail | Real-time logs, provenance tracking, compliance audit |

### Velocity Mode System

| Mode | Refresh Rate | Use Case |
|------|--------------|----------|
| Standard | 30s | Research and analysis |
| High Velocity | 5s | Active trading sessions |
| Turbo | 2s | Scalping and rapid execution |

---

## Scanner View

**Purpose:** Browse scan results with filtering and sorting.

**Features:**
- Multi-mode scan selection (momentum, breakout, etc.)
- Market cap bucket filtering
- Sort by QuantraScore, runner_prob, quality
- Batch selection for detailed analysis
- Export capabilities

**Example Workflow:**
1. Select scan mode (e.g., "Momentum Breakout")
2. Filter by market cap (e.g., "Mid Cap")
3. Sort by QuantraScore descending
4. Click symbol for detailed view

---

## Signal Details View

**Purpose:** Deep analysis of individual symbols.

**Displays:**
- QuantraScore with component breakdown
- Quality tier and confidence
- Runner probability with historical context
- Protocol trace (which protocols fired)
- Risk assessment
- Model predictions (if available)

**Example Workflow:**
1. Click symbol from scanner
2. Review QuantraScore components
3. Examine protocol traces
4. Check model predictions
5. Note any safety warnings

---

## Proof Explorer View

**Purpose:** Audit and verify decision traces.

**Features:**
- Search by symbol, date, or hash
- View full protocol execution log
- Verify output determinism
- Export proof for external audit
- Replay capability

**Example Workflow:**
1. Search for specific analysis
2. View input/output hashes
3. Examine protocol contributions
4. Export proof log for compliance

---

## API Endpoints

The FastAPI backend exposes 148 REST endpoints organized by category (running on port 8000).

### Health & System

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | System health check |
| `/stats` | GET | System statistics |
| `/version` | GET | Version information |

### Scanning

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/scan/single/{symbol}` | GET | Single symbol scan |
| `/scan/batch` | POST | Batch symbol scan |
| `/scan/universe` | POST | Full universe scan |
| `/scan/modes` | GET | Available scan modes |

### Analysis

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze/{symbol}` | GET | Detailed analysis |
| `/trace/{symbol}` | GET | Protocol trace |
| `/protocols/list` | GET | Protocol inventory |
| `/protocols/{id}` | GET | Protocol details |

### Models

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/models/status` | GET | Model availability |
| `/models/predict` | POST | Model inference |
| `/models/metrics` | GET | Model performance |
| `/models/manifest/{variant}` | GET | Model manifest |

### Risk & Compliance

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/risk/assess` | POST | Risk assessment |
| `/risk/extreme/{symbol}` | GET | Extreme move check |
| `/compliance/score` | GET | Compliance score |
| `/compliance/excellence` | GET | Excellence multipliers |
| `/compliance/report` | GET | Full compliance report |
| `/compliance/standards` | GET | Regulatory standards |
| `/compliance/omega` | GET | Omega directive status |

### Signals & Portfolio

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/signals/generate` | POST | Signal generation |
| `/signals/history` | GET | Signal history |
| `/portfolio/status` | GET | Portfolio status |
| `/portfolio/positions` | GET | Position list |

### Lab & Data

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/lab/status` | GET | ApexLab status |
| `/lab/datasets` | GET | Available datasets |
| `/data/symbols` | GET | Symbol universe |
| `/data/status` | GET | Data availability |

---

## Example API Responses

### GET /health

```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T08:07:17.004470",
  "engine": "operational",
  "data_layer": "operational"
}
```

### GET /compliance/score

```json
{
  "overall_score": 99.25,
  "excellence_level": "exceptional",
  "timestamp": "2025-11-29T08:07:15.659746",
  "metrics": {
    "determinism_iterations": 150,
    "stress_test_multiplier": 5.0,
    "latency_margin_ms": 50.0
  },
  "standards_exceeded": [
    "MiFID II RTS 6 Article 17",
    "Basel Committee BCBS 239"
  ],
  "compliance_mode": "RESEARCH_ONLY"
}
```

### GET /scan/single/AAPL

```json
{
  "symbol": "AAPL",
  "quantra_score": 72.4,
  "quality_tier": "B",
  "runner_prob": 0.35,
  "regime": "TRENDING",
  "risk_level": "MODERATE",
  "timestamp": "2025-11-29T10:15:00Z",
  "compliance_note": "Structural detection only - not a trade signal"
}
```

---

## Example Analyst Workflow

### Morning Research Session

1. **Check System Health**
   - API: `GET /health`
   - Verify all components operational

2. **Run Universe Scan**
   - API: `POST /scan/universe`
   - Parameters: mode=momentum, cap=mid

3. **Review Top Candidates**
   - UI: Scanner view sorted by QuantraScore
   - Select top 10 for detailed analysis

4. **Deep Dive Analysis**
   - API: `GET /analyze/{symbol}` for each
   - Review protocol traces and model predictions

5. **Document Findings**
   - UI: Proof Explorer to export traces
   - Note candidates with high runner_prob

6. **Check Compliance**
   - API: `GET /compliance/score`
   - Verify research-only mode active

---

## Authentication (When Enabled)

For production deployments, API authentication can be enabled:

- Bearer token authentication
- Role-based access control
- Audit logging of all requests
- Rate limiting

Currently disabled for research/demo use.

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
