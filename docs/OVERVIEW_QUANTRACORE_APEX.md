# QuantraCore Apex™ — System Overview

**Version:** 8.0  
**Owner:** Lamont Labs — Jesse J. Lamont  
**Status:** Active — Core Engine

---

## 1. What is QuantraCore Apex?

QuantraCore Apex™ is an institutional-grade deterministic AI trading intelligence engine with a complete offline learning ecosystem (ApexLab) and on-device neural assistant model (ApexCore). It represents a unified deterministic + neural hybrid stack designed for transparency, reproducibility, and regulatory compliance.

The system operates on these core principles:

- **Determinism first** — Every output can be traced and reproduced
- **Fail-closed always** — System restricts rather than guesses under uncertainty
- **No cloud dependencies** — All core logic runs locally
- **Local-only learning** — Training happens offline in ApexLab
- **QuantraScore mandatory everywhere** — Unified scoring across all outputs
- **Rule engine overrides AI always** — Apex is authoritative over neural models

---

## 2. Hardware Targets

QuantraCore Apex is optimized for specific hardware configurations:

| Platform | Target | Notes |
|----------|--------|-------|
| Workstation | GMKtec NucBox K6 | Primary development/analysis platform |
| Mobile | Android | For QuantraVision only |

**Recommended Constraints:**
- CPU: 8-core max recommended
- RAM: 16GB recommended
- GPU: Optional — CPU-optimized
- Storage: Local logs + model store

---

## 3. The Deterministic vs Neural Hybrid Design

QuantraCore Apex employs a unique "teacher-student" architecture where the deterministic core engine (Apex) acts as the authoritative source of truth, while neural models (ApexCore Full and ApexCore Mini) learn to approximate its behavior.

### 3.1 The Deterministic Core (Apex)

The Apex core engine includes these major components:

- **ZDE Engine** — Zero-Drift Engine for baseline stability
- **Continuation Validator** — Trend persistence validation
- **Entry Timing Optimizer** — Optimal entry point analysis
- **Volume Spike Mapper** — Unusual volume detection
- **QuantraScore Engine** — Final fused scoring (0–100)
- **Signal Classifier** — Signal categorization
- **Trend/Regime Engine** — Market regime detection
- **Microtrait Generator** — Fine-grained pattern extraction
- **Suppression & Clash Engine** — Low-activity and conflict detection
- **Entropy Range Engine** — Disorder measurement
- **Risk Tier Classifier** — Risk level assignment
- **Drift Engine** — Directional bias tracking
- **Sector Context Engine** — Sector-aware analysis
- **Score Fusion Engine** — Multi-factor score combination

### 3.2 The Neural Models (ApexCore)

The ApexCore model family addresses deployment scenarios where the full engine is impractical:

- **ApexCore Full** (3–20MB) — Desktop-class model for K6 workstation
- **ApexCore Mini** (0.5–3MB) — Mobile-optimized for Android/QuantraVision

**Critical Constraint:** Neural models can never override the deterministic core. Apex is always the teacher; ApexCore never overrides Apex.

---

## 4. QuantraScore

QuantraScore is the final fused scoring system at the heart of all analysis:

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

## 5. Protocols

### 5.1 Tier Protocols (T01–T80)

80 tier protocols organized by category:

- Trend
- Continuation
- Reversal
- Microstructure
- Volume
- Volatility
- Compression/Expansion
- Momentum/Slope
- Outlier Detection
- Multi-frame Consensus
- Sector/Correlation
- State Rejection

Each protocol resides in its own .py file, is auto-registered by loader, and executes in deterministic order.

### 5.2 Learning Protocols (LP01–LP25)

25 learning protocols handling:
- Adaptive decay correction
- Sample bias balancing
- Rare pattern bookmarking
- Continuation reinforcement
- Regime stability training
- Entropy boundary reinforcement

### 5.3 Omega Directives

Four safety override directives:

| Directive | Purpose |
|-----------|---------|
| **Ω1** | Hard safety lock |
| **Ω2** | Entropy override |
| **Ω3** | Drift override |
| **Ω4** | Compliance override |

---

## 6. ApexLab — The Training Laboratory

ApexLab is the self-contained offline local training environment:

**Capabilities:**
- Continuous data ingestion
- Apex-labeled dataset construction
- Feature generation (Apex-native)
- ApexCore model training
- Model evaluation + rejection
- Automated scheduled learning

**Training Data:**
- Windows: 100-bar OHLCV
- Features: All microtraits, entropy packet, suppression vectors, drift signature, sector context, regime/volatility labels

**Outputs:**
- ApexCore.tflite
- MODEL_MANIFEST.json
- Deterministic hash
- Training metrics

---

## 7. MonsterRunner Engine

MonsterRunner detects early signatures of extreme moves ("monster moves"):

**Signals:**
- Phase-compression pre-break
- Volume-engine ignition
- Range flipping
- Entropy collapse
- Sector-wide sympathetic moves

**Outputs:**
- runner_probability_0_1
- runner_state

**Compliance:** Not a trading signal — informational only.

---

## 8. QuantraVision Integration

QuantraVision provides mobile visual intelligence in two versions:

### 8.1 v2_apex (Current)
- On-device scan
- ApexLite processing
- ApexCore Mini integration
- HUD overlays
- Cloud narration optional

### 8.2 v1_legacy
- Signal viewer only
- Thin client for retail
- Upstream Apex signals

**Safety:** No trading, no recommendations.

---

## 9. Broker Layer

Institutional-grade execution capability (optional):

**Modes:**
- Paper
- Sim
- Live (locked behind compliance gates)

**Supported Brokers:**
- Alpaca
- Interactive Brokers
- Custom OMS API

**Fail-Closed Triggers:**
- Risk model rejection
- High drift
- High entropy
- Compliance failure

---

## 10. API Connections

### Market Data Providers:
- Polygon
- Tiingo
- Alpaca Market Data
- Intrinio
- Finnhub

**Capabilities:**
- Historical OHLCV
- Realtime OHLCV
- Full universe scanning
- High-frequency optional

**Optional Enhancements:**
- Corporate actions
- Sector metadata
- Volatility indexes
- Macro indexes
- Alternative data (news/feeds)

**Compliance Note:** Only data ingestion — no trade recommendations.

---

## 11. Universe Scanner

Three scanning modes:

| Mode | Speed | Use Case |
|------|-------|----------|
| Fast Scan | 1-5 sec/symbol | Quick screening |
| Deep Scan | 30-90 sec/symbol | Detailed analysis |
| Bulk Scan | Variable | Full universe sweeps |

**Optimization:**
- Segmented symbol batching
- Priority queue
- Volatility-aware scheduling

---

## 12. End-to-End Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUANTRACORE APEX v8.0                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Market Data → Core Engine Components → Tier Protocols (T01–T80)           │
│       ↓                                                                     │
│  Learning Protocols (LP01–LP25) → Omega Directives (Ω1–Ω4)                 │
│       ↓                                                                     │
│  QuantraScore (0–100) → Proof Logs → ApexLab Training                      │
│       ↓                                                                     │
│  ApexCore Full/Mini → QuantraVision + Dashboard                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Conclusion

QuantraCore Apex™ v8.0 represents a mature, institution-ready hybrid AI trading intelligence system. By combining a deterministic rule engine with distilled neural models, Apex achieves transparency and reproducibility while providing efficient on-device inference. The rule engine always overrides AI, ensuring safety and compliance.

For detailed specifications, see the [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml).
