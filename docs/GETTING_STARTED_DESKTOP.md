# Getting Started - Desktop Development

This guide walks you through setting up and running QuantraCore Apex on your desktop workstation.

---

## Prerequisites

- **Python 3.11+** (recommended: 3.11.x)
- **Node.js 20+** (for ApexDesk frontend)
- **Git** for version control

### Recommended Hardware
- 8-core CPU
- 16GB RAM
- SSD storage (for logs and models)
- GMKtec NucBox K6 or equivalent

---

## 1. Clone the Repository

```bash
git clone https://github.com/Lamont-Labs/QuantraCore.git
cd QuantraCore
```

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI + Uvicorn (API server)
- NumPy, Pandas (numerical computation)
- scikit-learn (ML models)
- Pydantic (data validation)
- HTTPX (HTTP client)
- pytest (testing)

---

## 3. Install Frontend Dependencies

```bash
cd dashboard
npm install
cd ..
```

This installs React 19, Vite 7, Tailwind CSS v4, and TypeScript.

---

## 4. Configure Data Provider (Optional)

QuantraCore Apex supports multiple data providers. For real market data, set up environment variables:

### Polygon.io (Recommended)
```bash
export POLYGON_API_KEY="your_polygon_api_key"
```

### Without API Keys
The system includes a synthetic data adapter for testing. No configuration needed.

---

## 5. Start the Backend Server

```bash
uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/health
# Expected: {"status": "operational", ...}
```

---

## 6. Start the Frontend (ApexDesk)

In a new terminal:

```bash
cd dashboard
npm run dev
```

Open your browser to: **http://localhost:5000**

---

## 7. Run a Sample Scan

### Using the API directly:

```bash
curl -X POST http://localhost:8000/scan_universe \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT", "GOOGL"]}'
```

### Using the ApexDesk UI:

1. Open http://localhost:5000
2. Click "RUN SCAN" button
3. View results in the Universe Scanner table
4. Click a symbol to see detailed analysis

---

## 8. Run the Test Suite

```bash
# All tests
pytest src/quantracore_apex/tests/ -v

# Determinism tests only
pytest src/quantracore_apex/tests/test_determinism_golden_set.py -v

# Protocol tests only
pytest src/quantracore_apex/tests/test_all_tier_protocols.py -v
```

---

## 9. Demo Scripts

### Fetch and Scan Demo
```bash
python scripts/fetch_and_scan_demo.py
```

### ApexLab Training Demo
```bash
python scripts/run_apexlab_demo.py
```

### ApexCore Validation
```bash
python scripts/validate_apexcore.py
```

---

## 10. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ApexDesk UI (React)                      │
│                      Port 5000                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI Server                           │
│                      Port 8000                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │  Risk   │  │   OMS   │  │Portfolio│  │  Signal Builder │ │
│  │ Engine  │  │  (Sim)  │  │ Tracker │  │                 │ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
│       └────────────┼───────────┼─────────────────┘          │
│                    ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ApexEngine (Deterministic Core)         │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐ │   │
│  │  │Entropy │ │Suppress│ │ Drift  │ │  QuantraScore  │ │   │
│  │  └────────┘ └────────┘ └────────┘ └────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Protocol Stack                     │   │
│  │  T01-T80 (Tier) │ LP01-LP25 (Learning) │ MR01-MR20   │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Data Layer                        │   │
│  │   Polygon │ Alpha Vantage │ Synthetic (Default)      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8000
# Kill it
kill -9 <PID>
```

### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/QuantraCore
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Not Loading
```bash
# Check Vite is running
cd dashboard
npm run dev
# Verify proxy configuration in vite.config.ts
```

---

## Next Steps

1. **Explore the API** - See [API Reference](API_REFERENCE.md)
2. **Understand Protocols** - See [Protocols: Tier](PROTOCOLS_TIER.md)
3. **Train Models** - See [ApexLab Training](APEXLAB_TRAINING.md)
4. **Run Backtests** - See [Backtest Guide](DEMO_SCRIPT.md)

---

**Note:** This system is for research and backtesting only. No live trading functionality is enabled.
