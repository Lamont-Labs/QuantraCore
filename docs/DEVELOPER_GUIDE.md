# Developer Guide

**Version:** 8.0  
**Audience:** Engineers onboarding to QuantraCore Apex development

---

## 1. Welcome

This guide provides engineers with the essential information needed to understand, navigate, and contribute to the QuantraCore Apex codebase.

---

## 2. System Description

From the master spec:

> QuantraCore Apex™ is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). Unified deterministic + neural hybrid stack.

---

## 3. Core Principles

All development must adhere to these principles:

1. **Determinism first** — All outputs must be reproducible
2. **Fail-closed always** — System restricts on uncertainty
3. **No cloud dependencies** — Core logic runs locally
4. **Local-only learning** — Training happens in ApexLab
5. **QuantraScore mandatory everywhere** — Unified scoring (0–100)
6. **Rule engine overrides AI always** — Apex is authoritative

---

## 4. Hardware Targets

| Platform | Target | Use Case |
|----------|--------|----------|
| Workstation | GMKtec NucBox K6 | Development, analysis |
| Mobile | Android | QuantraVision only |

**Recommended Constraints:**
- CPU: 8-core max recommended
- RAM: 16GB recommended
- GPU: Optional — CPU-optimized
- Storage: Local logs + model store

---

## 5. Directory Structure

```
QuantraCore/
├── core/                    # Core engine modules
│   ├── engine.py            # Main engine logic
│   ├── quant_score.py       # QuantraScore computation
│   ├── microtraits.py       # Microtrait extraction
│   ├── entropy.py           # Entropy analysis
│   ├── suppression.py       # Suppression detection
│   ├── drift.py             # Drift tracking
│   ├── continuation.py      # Continuation validation
│   ├── volume_spike.py      # Volume spike mapping
│   ├── entry_timing.py      # Entry timing optimization
│   ├── pattern_index.py     # Pattern indexing
│   ├── universe_scanner.py  # Universe scanning
│   ├── proof_logger.py      # Proof logging
│   └── restart_logger.py    # Restart tracking
│
├── apexlab/                 # Offline training environment
├── apexcore/                # Neural model artifacts
├── protocols/               # Tier protocols (T01–T80)
├── learning_protocols/      # Learning protocols (LP01–LP25)
├── models/                  # Trained models
├── data_ingest/             # Data ingestion adapters
├── broker/                  # Broker integration
├── risk/                    # Risk engine
├── api_adapters/            # External API adapters
├── docs/                    # Documentation
├── tests/                   # Test suite
├── logs/                    # Proof logs and restarts
│   ├── proof/               # Proof log JSON files
│   └── restarts/            # Restart log JSON files
└── sbom/                    # Software Bill of Materials
```

---

## 6. Core Engine Components

The deterministic rule engine includes:

| Component | Purpose |
|-----------|---------|
| ZDE Engine | Zero-Drift Engine for baseline |
| Continuation Validator | Trend persistence validation |
| Entry Timing Optimizer | Optimal entry analysis |
| Volume Spike Mapper | Unusual volume detection |
| QuantraScore Engine | Final scoring (0–100) |
| Signal Classifier | Signal categorization |
| Trend/Regime Engine | Market regime detection |
| Microtrait Generator | Fine-grained patterns |
| Suppression & Clash Engine | Low-activity/conflict detection |
| Entropy Range Engine | Disorder measurement |
| Risk Tier Classifier | Risk level assignment |
| Drift Engine | Directional bias tracking |
| Sector Context Engine | Sector-aware analysis |
| Score Fusion Engine | Multi-factor combination |

---

## 7. QuantraScore

The unified scoring system:

| Property | Value |
|----------|-------|
| Range | 0–100 |
| Buckets | fail, wait, pass, strong_pass |

**Construction factors:**
- Trend alignment
- Regime classification
- Strength distribution
- Volume/volatility factors
- Structural density
- Compression/noise
- Sector coupling
- Microtraits
- Suppression/entropy
- Omega gates

---

## 8. Protocols

### 8.1 Tier Protocols (T01–T80)

80 protocols with categories:
- Trend, Continuation, Reversal
- Microstructure, Volume, Volatility
- Compression/Expansion, Momentum/Slope
- Outlier Detection, Multi-frame Consensus
- Sector/Correlation, State Rejection

**Development rules:**
- Each protocol in its own .py file
- Auto-registered by loader
- Deterministic execution order

### 8.2 Learning Protocols (LP01–LP25)

25 protocols handling:
- Adaptive decay correction
- Sample bias balancing
- Rare pattern bookmarking
- Continuation reinforcement
- Regime stability training
- Entropy boundary reinforcement

### 8.3 Omega Directives

| Directive | Purpose |
|-----------|---------|
| Ω1 | Hard safety lock |
| Ω2 | Entropy override |
| Ω3 | Drift override |
| Ω4 | Compliance override |

---

## 9. Running Tests

### 9.1 Engine Tests

```bash
pytest tests/test_engine.py
```

Test categories:
- Protocol firing
- Regression baseline
- Score stability
- Failure rate
- Drift/entropy detection

### 9.2 ApexLab Tests

- Dataset integrity
- Label reproducibility
- Model accuracy gates

### 9.3 ApexCore Tests

- Inference speed
- Consistency with Apex
- Fail-closed paths

---

## 10. Universe Scanner

Three scanning modes:

| Mode | Speed | Use Case |
|------|-------|----------|
| Fast scan | 1-5 sec/symbol | Quick screening |
| Deep scan | 30-90 sec/symbol | Detailed analysis |
| Bulk scan | Variable | Full universe sweeps |

**Optimization features:**
- Segmented symbol batching
- Priority queue
- Volatility-aware scheduling

---

## 11. Build Targets

### 11.1 Desktop

- QuantraCore Engine
- ApexLab Training Environment
- ApexCore Full Model

### 11.2 Mobile

- ApexCore Mini
- QuantraVision v2

---

## 12. Proof Logging

All operations must be logged:

**Log locations:**
- `logs/proof/*.json` — Proof logs
- `logs/restarts/*.json` — Restart logs

**Required data:**
- Timestamp
- All indicators
- All protocols fired
- QuantraScore
- Final verdict
- Omega overrides
- Drift state
- Entropy signature

---

## 13. API Connections

Supported market data providers:
- Polygon
- Tiingo
- Alpaca Market Data
- Intrinio
- Finnhub

**Compliance:** Only data ingestion — no trade recommendations.

---

## 14. Getting Help

- Review the [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml)
- Check the [Overview](OVERVIEW_QUANTRACORE_APEX.md)
- Review component-specific documentation
