# Prediction System and MonsterRunner Engine

**Version:** 8.0  
**Components:** Prediction System, MonsterRunner Engine  
**Role:** Forward-looking analysis and extreme move detection

---

## 1. Overview

The Prediction System and MonsterRunner Engine are complementary components within QuantraCore Apex that provide forward-looking analysis capabilities:

- **Prediction System** — General-purpose regime-aware prediction for volatility and expected moves
- **MonsterRunner Engine** — Specialized engine for detecting early signatures of extreme moves

Both systems consume Apex outputs and market data, producing probabilistic estimates that inform—but never control—trading decisions.

---

## 2. MonsterRunner Engine

### 2.1 Purpose

MonsterRunner detects early signatures of extreme moves ("monster moves"). These are statistically unusual events that can significantly impact portfolios.

### 2.2 Signals

MonsterRunner monitors for these specific patterns:

| Signal | Description |
|--------|-------------|
| Phase-compression pre-break | Tightening range before breakout |
| Volume-engine ignition | Unusual volume surge patterns |
| Range flipping | Rapid high-low inversions |
| Entropy collapse | Sudden order emergence from chaos |
| Sector-wide sympathetic moves | Cross-sector momentum alignment |

### 2.3 Outputs

| Output | Type | Description |
|--------|------|-------------|
| `runner_probability_0_1` | float | Probability of extreme move (0.0–1.0) |
| `runner_state` | enum | Current detection state |

### 2.4 Compliance

**Critical:** MonsterRunner output is **not a trading signal**. It is informational only and must not be used for automated execution.

---

## 3. Prediction System

### 3.1 Purpose

The Prediction System generates forward-looking estimates for:
- Price movement direction and magnitude
- Volatility regime changes
- Expected move ranges
- Regime transition probabilities

### 3.2 Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTION SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │ Feature Builder │───►│ Regime-Aware Predictor          ││
│  └─────────────────┘    └─────────────────────────────────┘│
│           │                          │                      │
│           ▼                          ▼                      │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │ Volatility      │    │ Expected Move Model             ││
│  │ Predictor       │    │                                 ││
│  └─────────────────┘    └─────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Feature Categories

**Apex Features:**
- Entropy (disorder level)
- Suppression (low-activity detection)
- Drift (gradual directional bias)
- Continuation tags (trend persistence)
- Microtraits (fine-grained patterns)

**Volume/Volatility Features:**
- Volume spikes
- Microvolatility
- ATR cycles

**Context Features:**
- Sector coupling
- Regime classification
- Trend alignment

---

## 4. Integration with Core Engine

### 4.1 Data Flow

```
Market Data + Core Engine Outputs → Prediction System → Forecasts
                                  → MonsterRunner → Extreme Move Alerts
```

### 4.2 QuantraScore Integration

Both systems respect the QuantraScore (0–100) output:
- High scores (70–100): Strong structural quality
- Medium scores (50–69): Moderate quality
- Low scores (0–49): Caution required

---

## 5. Safety Considerations

### 5.1 No Direct Trading

Predictions and MonsterRunner alerts **never directly control trades**:

```
ALLOWED:
  - Display forecasts to analysts
  - Feed into dashboards
  - Trigger human review workflows
  - Log for backtesting

FORBIDDEN:
  - Automatic order placement
  - Position sizing decisions
  - Stop/limit modifications
  - Any execution without human approval
```

### 5.2 Compliance Statement

From the master spec:
> "MonsterRunner output is not a trading signal"

This constraint is enforced at the system level and cannot be bypassed.

### 5.3 Fail-Closed Behavior

When confidence is low or data is missing:
- Predictions abstain rather than guess
- MonsterRunner alerts require high confidence
- Missing features result in conservative outputs

---

## 6. Proof Logging

All predictions and alerts are logged:

```yaml
proof_log_entry:
  timestamp: "2025-10-15T14:30:00Z"
  component: "monster_runner"
  symbol: "AAPL"
  runner_probability: 0.78
  runner_state: "elevated"
  signals_detected:
    - "phase_compression_pre_break"
    - "volume_engine_ignition"
  quantrascore: 72
```

---

## 7. Test Coverage

| Test Category | Tests |
|---------------|-------|
| Engine | Drift/entropy detection |
| Prediction | Score stability, failure rate |
| MonsterRunner | Signal detection accuracy |

---

## 8. Summary

The Prediction System and MonsterRunner Engine extend QuantraCore Apex with forward-looking capabilities while maintaining the core principles of transparency, reproducibility, and safety. MonsterRunner specifically watches for phase-compression, volume ignition, range flipping, entropy collapse, and sector-wide moves—providing early warning of potential extreme moves without crossing into trading signal territory.
