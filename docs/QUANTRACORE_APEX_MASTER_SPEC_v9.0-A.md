# QUANTRACORE APEX - MASTER SYSTEM SPECIFICATION v9.0-A

**System Name:** QuantraCore Apex  
**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** 9.0-A (Institutional Hardening Upgrade)  
**Base Version:** 8.2  
**Status:** Active - Full Protocol System (Desktop-Only)  
**Repository:** https://github.com/Lamont-Labs/QuantraCore

---

## 1. Core Principles

1. **Determinism first** - All computations must be reproducible given the same inputs
2. **Fail-closed always** - System defaults to safe/no-action state on any error
3. **No cloud dependencies** - Entire system runs locally without external services
4. **Local-only learning** - ApexLab trains models on local data only
5. **QuantraScore mandatory everywhere** - Every analysis produces a 0-100 score
6. **Rule engine overrides AI always** - Deterministic rules take precedence over neural outputs
7. **Redundant verification** - Critical decisions require dual-path confirmation (v9.0-A)
8. **Drift awareness** - Monitor statistical properties and detect distribution shifts (v9.0-A)

---

## 2. v9.0-A Hardening Goals

This upgrade builds upon the stable v8.2 foundation with institutional-grade hardening:

| Goal | Description |
|------|-------------|
| Redundant Scoring | Dual-path QuantraScore with shadow scorer cross-check |
| Drift Detection | Monitor distributions, detect shifts, auto-guard mode |
| Data Failover | Multi-provider with integrity verification |
| Model Gating | Validation pipeline with acceptance criteria |
| Audit Trail | Comprehensive logging for reproducibility |
| Research Fence | Config-enforced research-only mode |

---

## 3. Hardware Targets

| Platform | Target | Notes |
|----------|--------|-------|
| Workstation | GMKtec NucBox K6 | Primary development and execution platform |
| CPU | 8-core max recommended | Intel/AMD x86-64 |
| RAM | 16-32GB recommended | 16GB minimum for full operation |
| GPU | Optional | CPU-optimized by design |
| Storage | Local SSD | For logs, models, cache, and proof storage |

**CRITICAL:** Mobile/Android builds are strictly prohibited. Desktop-only architecture.

---

## 4. System Architecture

### 4.1 Directory Structure (v9.0-A)

```
quantracore-apex/
├── config/                     # Configuration files (NEW)
│   ├── mode.yaml              # Research/paper-trading mode
│   └── symbol_universe.yaml   # Symbol registry
├── data/
│   ├── cache/                 # Cached market data
│   │   └── MANIFEST.json      # Cache integrity manifest
│   └── universes/             # Universe CSV files
├── provenance/                # Audit and proof logs
│   ├── drift_baselines/       # Drift detection baselines
│   ├── score_consistency_log.jsonl
│   ├── drift_events_log.jsonl
│   ├── model_promotion_log.jsonl
│   └── replay_runs_log.jsonl
├── src/quantracore_apex/
│   ├── core/                  # Deterministic core engine
│   │   ├── engine.py         # ApexEngine main class
│   │   ├── redundant_scorer.py  # Shadow scorer (NEW)
│   │   ├── drift_detector.py    # Drift framework (NEW)
│   │   ├── decision_gates.py    # Fail-closed gates (NEW)
│   │   └── ... (existing v8.2 modules)
│   ├── protocols/             # T01-T80, LP01-LP25, MR01-MR20, Ω01-Ω20
│   ├── data_layer/
│   │   ├── adapters/          # Multi-provider adapters
│   │   ├── failover.py        # Failover logic (NEW)
│   │   ├── cache_manager.py   # Cache with integrity (NEW)
│   │   └── symbol_registry.py # Symbol mapping (NEW)
│   ├── apexlab/               # Offline training environment
│   │   ├── temporal_features.py   # Temporal encodings (NEW)
│   │   └── ... (existing modules)
│   ├── apexcore/              # Neural model interface
│   │   ├── prediction_heads.py    # Multi-head outputs (NEW)
│   │   ├── model_validator.py     # Validation pipeline (NEW)
│   │   └── ... (existing modules)
│   ├── replay/                # Sandbox replay engine (NEW)
│   │   └── replay_engine.py
│   ├── cli/                   # CLI tools (NEW)
│   │   └── qapex.py
│   ├── server/                # FastAPI application
│   └── tests/                 # Test suite
├── dashboard/                 # ApexDesk React UI
└── docs/                      # Documentation
```

---

## 5. Engine Hardening

### 5.1 Redundant Scoring Architecture

The v9.0-A engine implements dual-path scoring for anomaly detection:

| Component | Role |
|-----------|------|
| Primary QuantraScore | Canonical v8.2 implementation, unchanged |
| Shadow QuantraScore | Independent recomputation with different aggregation |
| Consistency Checker | Compares scores, flags deviations |

**Thresholds:**
- Allowed absolute difference: 5.0 points
- Band mismatch: Not allowed by default

**Actions:**
- Within threshold: `score_consistency='ok'`
- Minor deviation (5-10 pts): `score_consistency='warning'`, log event
- Major deviation (>10 pts): `score_consistency='fail'`, flag for investigation

### 5.2 Drift Detection Framework

Monitor statistical properties over time and detect distribution shifts:

**Tracked Metrics:**
- QuantraScore distribution per symbol and universe
- Regime frequency (`trending_up`, `range_bound`, etc.)
- MonsterRunner primed/idle ratio
- Score consistency failure rate

**Detection Logic:**
- KS-test style distribution comparison
- Z-score deviation for key counts
- Check period: per universe scan or daily rollup

**Drift Modes:**
- `NORMAL`: Standard operation
- `DRIFT_GUARDED`: Reduced aggressiveness, higher thresholds, learning frozen

### 5.3 Fail-Closed Decision Gates

| Gate | Checks | On Fail |
|------|--------|---------|
| Data Integrity | No missing bars, no duplicates, no negative values | Reject signal |
| Model Integrity | Hash matches manifest, version compatible | Fallback to deterministic |
| Risk Guard | Metrics within limits | Block signal |

### 5.4 Sandbox Replay Engine

Reusable module for historical replay and regression testing:

**Outputs:**
- Equity curve
- Signal frequency stats
- Drift flags

**Usage:**
- CLI: `qapex replay-demo`
- API: `POST /replay/run_demo`

---

## 6. Data Layer & Failover

### 6.1 Multi-Provider Adapters

| Provider | Role |
|----------|------|
| Polygon | Primary US equities |
| Alpha Vantage | Optional secondary |
| CSV Bundle | Offline cached data |

**Unified Interface:**
```python
class DataClient:
    def get_ohlcv(symbol, timeframe, start, end) -> DataFrame
    def get_universe_snapshot(universe, timeframe, lookback_bars) -> Dict
    def get_metadata(symbol) -> SymbolMetadata
```

### 6.2 Failover Logic

1. Try primary provider
2. If rate-limited or partial failure: try cached bundle or secondary
3. If no valid data: mark symbol as `data_unavailable`

### 6.3 Dataset Caching & Integrity

**Cache Format:** Parquet files in `data/cache/`

**Manifest:** `data/cache/MANIFEST.json`
- Symbol, timeframe, date range
- Row count, SHA-256 hash

**Integrity Checks:**
- Verify hash matches manifest
- If mismatch: invalidate and refetch

### 6.4 Symbol Universe Mapping

Centralized registry in `config/symbol_universe.yaml`:
- Symbol, name, asset class, sector
- Active flag, notes
- Engine only scans active symbols

---

## 7. ApexLab & ApexCore Upgrade

### 7.1 Temporal Feature Upgrade

Extended feature builder with temporal awareness:

- Rolling returns over multiple horizons (1, 5, 20 bars)
- Volatility clusters and local HV percentile
- Run-length encodings of trends/ranges
- Compressed T01-T80 protocol activations over time

### 7.2 Prediction Chain Heads

| Head | Outputs |
|------|---------|
| Structure Head | regime, volatility_band, suppression_state, entropy_state |
| Score Head | QuantraScore numeric + band |
| Runner Head | monster_runner_state, bias, confidence |
| Forward Bar Head | next-bar direction, normalized move magnitude |

### 7.3 Model Validation & Gating

**Validation Steps:**
1. Hold-out evaluation (structural accuracy, score band accuracy)
2. Replay evaluation on fixed benchmark set
3. Stability evaluation (deterministic outputs)

**Acceptance Criteria:**
- All primary metrics >= baselines
- No new drift flags over baseline

**Gating Logic:**
- Fail: Keep existing model, log rejection
- Pass: Promote to ACTIVE, archive previous as ROLLBACK

### 7.4 ApexCore Variants

| Variant | Target | Size | Purpose |
|---------|--------|------|---------|
| ApexCore Full | K6 Desktop | 3-20MB | High-precision research, rich analysis |
| ApexCore Mini | Desktop (light) | 0.5-3MB | Low-resource mode, distilled from Full |

---

## 8. Tier Protocols (T01-T80)

All 80 tier protocols fully implemented (unchanged from v8.2):

### T01-T20: Core Protocols
Compression, Momentum, Risk, Entropy, Drift, Volume, Pattern

### T21-T30: Support/Resistance & Gap Analysis
Support/Resistance levels, Pivot points, Fibonacci, Gap patterns

### T31-T40: Breakout & Divergence
Breakout/Breakdown detection, Divergence analysis

### T41-T50: Candlestick Patterns
Engulfing, Doji, Hammer, Morning/Evening Star, etc.

### T51-T60: Moving Average & Trend
MA crossovers, Trend strength, ADX-based analysis

### T61-T70: Oscillators & Momentum
RSI, MACD, Stochastic, CCI, Williams %R

### T71-T80: Market Context
Sector rotation, Correlation, Market breadth, VIX context

---

## 9. Learning Protocols (LP01-LP25)

All 25 learning protocols fully implemented for ApexLab labels.

---

## 10. MonsterRunner Protocols (MR01-MR20)

| Range | Category | Description |
|-------|----------|-------------|
| MR01-MR05 | Core | Compression, volume, regime, institutional, alignment |
| MR06-MR10 | Breakout | Bollinger, volume explosion, gap, VWAP, NR7 |
| MR11-MR15 | Extreme | Short squeeze, pump, catalyst, fractal, 100% day |
| MR16-MR20 | Parabolic | Phase 3, meme frenzy, gamma ramp, FOMO, nuclear |

---

## 11. Omega Directives (Ω1-Ω20)

| Range | Category | Description |
|-------|----------|-------------|
| Ω1-Ω5 | Core Safety | Risk lock, entropy, drift, compliance, suppression |
| Ω6-Ω10 | Volatility | Vol cap, divergence, squeeze, MACD, fear spike |
| Ω11-Ω15 | Indicators | RSI extreme, volume spike, trend, gap, tail risk |
| Ω16-Ω20 | Advanced | Overnight, fractal, liquidity, correlation, nuclear |

---

## 12. API Endpoints

### Existing Endpoints (v8.2)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and compliance note |
| `/health` | GET | Health check |
| `/desk` | GET | ApexDesk dashboard |
| `/scan_symbol` | POST | Full Apex analysis |
| `/scan_universe` | POST | Multi-symbol batch scan |
| `/trace/{window_hash}` | GET | Protocol trace |
| `/monster_runner/{symbol}` | POST | MonsterRunner check |
| `/risk/assess/{symbol}` | POST | Risk assessment |
| `/signal/generate/{symbol}` | POST | Signal generation |
| `/portfolio/status` | GET | Portfolio snapshot |
| `/oms/*` | Various | Order management (simulation) |

### New Endpoints (v9.0-A)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/engine/health_extended` | GET | Extended health with drift status |
| `/drift/status` | GET | Current drift metrics |
| `/replay/run_demo` | POST | Run sandbox replay |
| `/score/consistency/{symbol}` | GET | Score consistency check |

---

## 13. ApexDesk UI Panels

| Panel | Description |
|-------|-------------|
| Universe Scanner | Symbol scanning with score display |
| Signal Detail + Trace | Redundant scores, drift flags, gates |
| Risk & Exposure | Portfolio snapshot, sector exposure |
| Drift & Health | Drift metrics, consistency, model status |
| Replay & Research | Trigger replays, view equity curves |

---

## 14. CLI Tools

```bash
qapex health              # Check engine, data, models
qapex scan-universe       # Run configured universe scan
qapex replay-demo         # Small demo replay
qapex lab-train-mini      # Demo ApexCoreMini training
```

---

## 15. Compliance, Safety & Auditability

### 15.1 Research-Only Safety Fence

**Config:** `config/mode.yaml`
- Default mode: `research`
- Live trading connectors must check mode and refuse in research mode

### 15.2 Audit Logs

| Log | Purpose |
|-----|---------|
| `score_consistency_log.jsonl` | Score deviation events |
| `drift_events_log.jsonl` | Drift detection events |
| `model_promotion_log.jsonl` | Model validation results |
| `replay_runs_log.jsonl` | Replay execution records |

### 15.3 Proof Bundle

`dist/quantracore_apex_proof_bundle_v9_0a/`:
- Master spec document
- Engine test summary
- Drift baselines
- Model manifest
- Sample replay report
- Compliance documentation

---

## 16. Testing Strategy

### Unit & Protocol Tests
- All T01-T80, LP01-LP25 protocols
- Shadow scorer consistency
- Fail-closed gate behavior

### Integration Tests
- Universe scan with demo universe
- ApexLab demo training
- ApexDesk end-to-end smoke test

### Drift & Replay Tests
- Small replay consistency
- Artificial drift detection

### CI Expectations
- Lint/type-check
- Pytest (355+ tests)
- Frontend tests
- No Android builds or heavy jobs

---

## 17. Migration from v8.2

All v8.2 behavior is preserved. v9.0-A is an additive hardening layer:

1. **Phase 1:** Spec, docs, config files
2. **Phase 2:** Engine hardening (scoring, drift, gates)
3. **Phase 3:** ApexLab/ApexCore upgrade
4. **Phase 4:** UI/API extensions
5. **Phase 5:** Compliance and audit

---

## 18. Compliance Note

All outputs from QuantraCore Apex are framed as **structural probabilities** for research purposes only. This system does NOT provide investment advice and does NOT execute real trades by default.

---

**Document Version:** 9.0-A  
**Last Updated:** 2025-12-01  
**Status:** Active

---

## 19. Signal & Alert Services (New in v9.0-A)

### 19.1 Manual Trading Signal Service

Generates actionable trading signals for manual execution on external platforms.

| Feature | Description |
|---------|-------------|
| Signal Ranking | Priority score (QuantraScore, runner probability, timing) |
| Timing Guidance | Actionable entry timing language |
| Entry/Exit Levels | ATR-based with 2:1 minimum R:R |
| Conviction Tiers | high / medium / low / avoid |
| Predicted Top | Expected runup from model |

**Endpoints:** `/signals/live`, `/signals/scan`, `/signals/symbol/{symbol}`, `/signals/status`

### 19.2 SMS Alert Service

Twilio-powered SMS alerts for trading signals.

**Endpoints:** `/sms/status`, `/sms/config`, `/sms/test`, `/sms/alert`

### 19.3 Low-Float Runner Screener

Real-time scanner for penny stock runners (114 symbols).

| Criteria | Threshold |
|----------|-----------|
| Relative Volume | 3x+ |
| Price Momentum | 5%+ |
| Max Float | 50M shares |
| Symbol Universe | 64 penny + 20 nano + 30 micro |

**Endpoints:** `/screener/status`, `/screener/scan`, `/screener/alerts`, `/screener/config`, `/screener/alert-runner`

### 19.4 AutoTrader — Automatic Swing Trade Execution

Fully autonomous swing trade executor with Alpaca paper trading integration.

| Feature | Description |
|---------|-------------|
| Universe Scan | Real-time analysis via ApexSignalService |
| Qualification | QuantraScore >= 60.0 threshold |
| Position Sizing | 10% of account equity per trade |
| Execution | Market orders via AlpacaPaperAdapter |
| Audit Trail | Full trade logging to `investor_logs/auto_trades/` |

**Workflow:**
1. Scan universe for qualified setups
2. Rank by QuantraScore descending
3. Select top N (default 3) excluding existing positions
4. Calculate position size based on account equity
5. Execute market orders on Alpaca paper trading
6. Log all trades for audit compliance

**Endpoints:** `/trading/account`, `/trading/setups`, `/trading/execute`

**Safety Controls:**
- Paper trading only (Alpaca paper-api)
- Excludes existing positions
- ATR-based stop-loss levels
- Position size limits enforced
