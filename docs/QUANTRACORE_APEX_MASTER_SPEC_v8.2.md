# QUANTRACORE APEX - MASTER SYSTEM SPECIFICATION v8.2

**System Name:** QuantraCore Apex  
**Owner:** Lamont Labs - Jesse J. Lamont  
**Version:** 8.2  
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

---

## 2. Hardware Targets

| Platform | Target | Notes |
|----------|--------|-------|
| Workstation | GMKtec NucBox K6 | Primary development and execution platform |
| CPU | 8-core max recommended | Intel/AMD x86-64 |
| RAM | 16GB recommended | Minimum 8GB for basic operation |
| GPU | Optional | CPU-optimized by design |
| Storage | Local SSD | For logs, models, and proof storage |

**CRITICAL:** Mobile/Android builds are strictly prohibited. Desktop-only architecture.

---

## 3. System Architecture

### 3.1 Directory Structure

```
src/quantracore_apex/
├── core/                    # Deterministic core engine
│   ├── engine.py           # ApexEngine main class
│   ├── schemas.py          # Pydantic data models
│   ├── microtraits.py      # Microtrait computation
│   ├── entropy.py          # Entropy analysis
│   ├── suppression.py      # Suppression detection
│   ├── drift.py            # Drift analysis
│   ├── continuation.py     # Continuation analysis
│   ├── volume_spike.py     # Volume metrics
│   ├── regime.py           # Regime classification
│   ├── quantrascore.py     # QuantraScore computation
│   ├── verdict.py          # Verdict building
│   ├── sector_context.py   # Sector-aware adjustments
│   └── proof_logger.py     # Proof logging
├── protocols/
│   ├── tier/               # T01-T80 Tier Protocols
│   ├── learning/           # LP01-LP25 Learning Protocols
│   ├── monster_runner/     # MR01-MR05 MonsterRunner Protocols
│   └── omega/              # Omega Directives Ω1-Ω5
├── data_layer/
│   └── adapters/           # Data provider adapters (Polygon, Synthetic)
├── apexlab/                # Offline training environment
├── apexcore/               # Neural model interface
├── prediction/             # Prediction engines
├── risk/                   # Risk engine
├── broker/                 # Order Management System (simulation only)
├── portfolio/              # Portfolio tracking
├── signal/                 # Signal builder
├── scheduler/              # Task scheduling
├── server/                 # FastAPI application
└── tests/                  # Test suite
```

---

## 4. Tier Protocols (T01-T80)

All 80 tier protocols are fully implemented with deterministic logic.

### T01-T20: Core Protocols
| ID | Name | Category |
|----|------|----------|
| T01 | Compression Scanner | Compression |
| T02 | Compression Depth | Compression |
| T03 | Compression Duration | Compression |
| T04 | Trend Momentum Core | Momentum |
| T05 | Momentum Acceleration | Momentum |
| T06 | Momentum Divergence | Momentum |
| T07 | Risk Assessment Basic | Risk |
| T08 | Volatility State | Risk |
| T09 | Risk Tier Classifier | Risk |
| T10 | Entropy Basic | Entropy |
| T11 | Entropy Trend | Entropy |
| T12 | Entropy Extremes | Entropy |
| T13 | Drift Basic | Drift |
| T14 | Drift Momentum | Drift |
| T15 | Drift Reversal | Drift |
| T16 | Volume Basic | Volume |
| T17 | Volume Trend | Volume |
| T18 | Volume Spike | Volume |
| T19 | Pattern Basic | Pattern |
| T20 | Pattern Sequence | Pattern |

### T21-T30: Support/Resistance & Gap Analysis
| ID | Name | Category |
|----|------|----------|
| T21 | Support Level Detection | Support/Resistance |
| T22 | Resistance Level Detection | Support/Resistance |
| T23 | Pivot Point Analysis | Support/Resistance |
| T24 | Fibonacci Retracement | Support/Resistance |
| T25 | Fibonacci Extension | Support/Resistance |
| T26 | Gap Analysis - Up | Gap Patterns |
| T27 | Gap Analysis - Down | Gap Patterns |
| T28 | Gap Fill Detection | Gap Patterns |
| T29 | Island Reversal | Gap Patterns |
| T30 | Exhaustion Gap | Gap Patterns |

### T31-T40: Breakout & Divergence
| ID | Name | Category |
|----|------|----------|
| T31 | Breakout Detection | Breakout |
| T32 | Breakdown Detection | Breakout |
| T33 | False Breakout | Breakout |
| T34 | Range Breakout | Breakout |
| T35 | Consolidation Break | Breakout |
| T36 | RSI Divergence Bullish | Divergence |
| T37 | RSI Divergence Bearish | Divergence |
| T38 | MACD Divergence | Divergence |
| T39 | Volume Divergence | Divergence |
| T40 | Price-Momentum Divergence | Divergence |

### T41-T50: Candlestick & Market Structure
| ID | Name | Category |
|----|------|----------|
| T41 | Bullish Engulfing | Candlestick |
| T42 | Bearish Engulfing | Candlestick |
| T43 | Doji Star | Candlestick |
| T44 | Hammer/Hanging Man | Candlestick |
| T45 | Morning/Evening Star | Candlestick |
| T46 | Accumulation Detection | Market Structure |
| T47 | Distribution Detection | Market Structure |
| T48 | Wyckoff Spring | Market Structure |
| T49 | Wyckoff Upthrust | Market Structure |
| T50 | Market Structure Break | Market Structure |

### T51-T60: Advanced Volume & Momentum
| ID | Name | Category |
|----|------|----------|
| T51 | Volume Profile Analysis | Advanced Volume |
| T52 | VWAP Deviation | Advanced Volume |
| T53 | OBV Trend | Advanced Volume |
| T54 | Money Flow Index | Advanced Volume |
| T55 | Volume Weighted Trend | Advanced Volume |
| T56 | Momentum Oscillator | Momentum Analysis |
| T57 | Rate of Change Extreme | Momentum Analysis |
| T58 | Stochastic Extreme | Momentum Analysis |
| T59 | Williams %R Signal | Momentum Analysis |
| T60 | CCI Signal | Momentum Analysis |

### T61-T70: Sector & MonsterRunner Precursors
| ID | Name | Category |
|----|------|----------|
| T61 | Sector Rotation | Sector Analysis |
| T62 | Relative Strength vs Sector | Sector Analysis |
| T63 | Sector Leadership | Sector Analysis |
| T64 | Sector Divergence | Sector Analysis |
| T65 | Sector Momentum | Sector Analysis |
| T66 | Phase Compression Precursor | MonsterRunner |
| T67 | Volume Ignition Setup | MonsterRunner |
| T68 | Range Flip Potential | MonsterRunner |
| T69 | Entropy Collapse Signal | MonsterRunner |
| T70 | Sector Cascade Risk | MonsterRunner |

### T71-T80: Regime Detection & Multi-Timeframe
| ID | Name | Category |
|----|------|----------|
| T71 | Volatility Regime Shift | Regime |
| T72 | Trend Regime Change | Regime |
| T73 | Range to Trend Transition | Regime |
| T74 | Trend to Range Transition | Regime |
| T75 | Regime Uncertainty | Regime |
| T76 | Multi-Timeframe Alignment | Multi-Timeframe |
| T77 | Higher Timeframe Trend | Multi-Timeframe |
| T78 | Lower Timeframe Entry | Multi-Timeframe |
| T79 | Timeframe Divergence | Multi-Timeframe |
| T80 | Composite Timeframe Score | Multi-Timeframe |

---

## 5. Learning Protocols (LP01-LP25)

Learning protocols generate labels for ApexLab training.

### LP01-LP10: Core Labels
| ID | Name | Output |
|----|------|--------|
| LP01 | Regime Label | regime_class (5 classes) |
| LP02 | Volatility State Label | volatility_state |
| LP03 | Risk Tier Label | risk_tier (1-5) |
| LP04 | Entropy State Label | entropy_state |
| LP05 | Drift State Label | drift_state |
| LP06 | Suppression Label | suppression_state |
| LP07 | Continuation Label | continuation_probability |
| LP08 | Volume State Label | volume_state |
| LP09 | Score Bucket Label | score_bucket |
| LP10 | Verdict Action Label | verdict_action |

### LP11-LP25: Advanced Labels
| ID | Name | Output |
|----|------|--------|
| LP11 | Future Direction | direction_5d |
| LP12 | Momentum Persistence | momentum_state |
| LP13 | Breakout Prediction | breakout_probability |
| LP14 | Reversal Prediction | reversal_probability |
| LP15 | Trend Strength | trend_strength_score |
| LP16 | Volatility Forecast | vol_forecast_bucket |
| LP17 | Institutional Activity | institutional_score |
| LP18 | Market Phase | market_phase |
| LP19 | Composite Conviction | conviction_score |
| LP20 | Risk-Adjusted Score | risk_adj_score |
| LP21 | Sector Relative | sector_relative |
| LP22 | MonsterRunner Label | monster_probability |
| LP23 | Entry Quality | entry_quality |
| LP24 | Exit Timing | exit_timing |
| LP25 | Composite Multi-Label | multi_label_vector |

---

## 6. MonsterRunner Protocols (MR01-MR05)

Rare-event detection for extreme market moves.

| ID | Name | Detection |
|----|------|-----------|
| MR01 | Compression Explosion | Phase compression leading to explosive move |
| MR02 | Volume Anomaly | Unusual volume patterns preceding major moves |
| MR03 | Volatility Regime Shift | Sudden volatility state changes |
| MR04 | Institutional Footprint | Large player accumulation/distribution |
| MR05 | Multi-Timeframe Alignment | Cross-timeframe convergence signals |

---

## 7. Omega Directives (Ω1-Ω5)

Hard safety overrides that cannot be bypassed.

| Directive | Name | Trigger |
|-----------|------|---------|
| Ω1 | Hard Safety Lock | Extreme risk tier detected |
| Ω2 | Entropy Override | Chaotic entropy state |
| Ω3 | Drift Override | Critical drift state |
| Ω4 | Compliance Override | Always active (research-only mode) |
| Ω5 | Signal Suppression Lock | Strong suppression detected |

---

## 8. ApexLab Training Environment

### 8.1 Components
- **Window Builder** - Creates 100-bar OHLCV windows
- **Feature Builder** - Extracts 30-dimensional feature vectors
- **Label Builder** - Generates multi-head labels from engine
- **Dataset Builder** - Constructs (X, y) training datasets
- **Validation Module** - Model alignment verification

### 8.2 Training Pipeline
1. Fetch historical OHLCV data
2. Build sliding windows (100 bars)
3. Extract features from each window
4. Generate labels using full engine analysis
5. Train ApexCore model(s)
6. Validate alignment with deterministic engine

---

## 9. ApexCore Models

### 9.1 ApexCoreFull
- **Target:** Desktop workstation
- **Size:** 3-20MB
- **Architecture:** Multi-layer perceptron with skip connections
- **Outputs:** Multi-head predictions (regime, risk, score, etc.)

### 9.2 ApexCoreMini
- **Target:** Lightweight desktop deployment (reduced resource usage)
- **Size:** 0.5-3MB
- **Architecture:** Compact MLP
- **Outputs:** Core predictions only
- **Note:** Desktop-only - no mobile/Android deployment

---

## 10. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/desk` | GET | ApexDesk dashboard |
| `/scan_symbol` | POST | Single symbol scan |
| `/scan_universe` | POST | Multi-symbol batch scan |
| `/trace/{hash}` | GET | Detailed protocol trace |
| `/monster_runner/{symbol}` | POST | MonsterRunner check |
| `/risk/assess/{symbol}` | POST | Risk assessment |
| `/signal/generate/{symbol}` | POST | Signal generation |
| `/portfolio/status` | GET | Portfolio snapshot |
| `/portfolio/heat_map` | GET | Sector heat map |
| `/oms/orders` | GET | Order book (simulation) |
| `/oms/positions` | GET | Current positions |
| `/oms/place` | POST | Place order (simulation) |
| `/oms/submit/{id}` | POST | Submit order (simulation) |
| `/oms/fill` | POST | Simulate fill |
| `/oms/cancel/{id}` | POST | Cancel order |
| `/oms/reset` | POST | Reset OMS |
| `/api/stats` | GET | System statistics |

---

## 11. ApexDesk UI

React-based dashboard with:
- **Left Rail** - Brand logos (Lamont Labs, QuantraCore) and navigation
- **Header** - System status, mode badges, action buttons
- **Universe Table** - Scan results with score coloring
- **Detail Panel** - Symbol analysis with tabs (Overview, Trace, Monster, Signal)

---

## 12. QuantraVision Apex v4.2 (Spec Only)

Mobile overlay copilot specification (NOT IMPLEMENTED in this repo):
- HUD-style market overlay
- ApexCore Mini integration
- Real-time score display
- Voice alerts for Omega triggers

**NOTE:** This spec is documentation only. No mobile builds in this repo.

---

## 13. Compliance & Safety

### 13.1 Research-Only Mode
- All outputs are structural probabilities, NOT trading advice
- Omega Ω4 directive enforces compliance mode always
- No live trading execution paths

### 13.2 Data Handling
- API keys read from environment variables only
- No secrets committed to repository
- Synthetic data available for testing without API keys

### 13.3 Audit Trail
- All scans logged with SHA-256 window hashes
- Proof logs stored locally
- Full determinism verification via golden set tests

---

## 14. Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Tier Protocols | 80 | All passing |
| Learning Protocols | 25 | All passing |
| MonsterRunner | 12 | All passing |
| Core Engine | 40+ | All passing |
| API Endpoints | 23 | All passing |
| Frontend | 5 | All passing |
| Determinism | 7 | All passing |
| **Total** | **349+** | **All passing** |

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 8.0 | 2025-11-27 | Initial MVP+ implementation |
| 8.1 | 2025-11-27 | Full protocol system (T01-T80, LP01-LP25) |
| 8.2 | 2025-11-28 | ApexDesk UI, Risk/OMS/Portfolio, complete validation |

---

**Persistence = Proof.**  
Every build, every log, every checksum - reproducible by design.
