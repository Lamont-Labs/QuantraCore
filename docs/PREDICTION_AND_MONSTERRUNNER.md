# Prediction System and MonsterRunner

**Version:** 8.0  
**Components:** Prediction System, MonsterRunner  
**Role:** Forward-looking analysis and rare-event detection

---

## 1. Overview

The Prediction System and MonsterRunner are complementary components within QuantraCore Apex that provide forward-looking analysis capabilities:

- **Prediction System** — General-purpose regime-aware prediction for volatility and expected moves
- **MonsterRunner** — Specialized engine for detecting rare, high-impact market events ("monster moves")

Both systems consume Apex outputs and market data, producing probabilistic estimates that inform—but never control—trading decisions.

---

## 2. Prediction System

### 2.1 Purpose

The Prediction System generates forward-looking estimates for:
- Price movement direction and magnitude
- Volatility regime changes
- Expected move ranges
- Regime transition probabilities

### 2.2 Components

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

### 2.3 Feature Categories

**Intraday Features:**
- Volume spikes (abnormal volume detection)
- Microvolatility (tick-level variance)
- Intraday range (high-low spread)

**Swing Features:**
- Multi-day momentum (trend strength)
- ATR cycles (volatility rhythm)
- Gap statistics (overnight moves)

**Apex Features:**
- Entropy (disorder level)
- Suppression (low-activity detection)
- Drift (gradual directional bias)
- Continuation tags (trend persistence)

### 2.4 Outputs

| Output | Type | Description |
|--------|------|-------------|
| Direction probability | float | Probability of up/down move |
| Expected magnitude | float | Predicted move size |
| Volatility forecast | enum | Expected volatility regime |
| Confidence | float | Model confidence (0.0–1.0) |

---

## 3. MonsterRunner

### 3.1 Purpose

MonsterRunner is a specialized engine designed to detect conditions that precede rare, high-impact market moves. These "monster moves" are statistically unusual events that can significantly impact portfolios.

### 3.2 Design Philosophy

- **Rare Event Focus** — Optimized for low-frequency, high-impact detection
- **Multi-Factor** — Combines diverse signal sources
- **Conservative** — High specificity preferred over sensitivity
- **Explainable** — Feature attribution for every alert

### 3.3 Features Used

MonsterRunner consumes a broad feature set:

**Volatility Features:**
- High-resolution volatility (tick-level)
- Implied volatility surfaces
- Volatility term structure
- Historical volatility ratios

**Apex Trait Signatures:**
- Extreme entropy readings
- Unusual suppression patterns
- Drift acceleration
- Continuation breakdowns

**Sector and Macro:**
- Sector momentum divergences
- Cross-sector correlations
- Macro regime indicators
- Risk-on/risk-off signals

**Options and Flow:**
- Options skew (put/call imbalance)
- Unusual options activity
- Gamma exposure estimates

**Alternative Data:**
- Short interest levels and changes
- Insider trade patterns
- Institutional flow indicators

### 3.4 Outputs

| Output | Type | Description |
|--------|------|-------------|
| MonsterScore | float | Probability of monster move (0.0–1.0) |
| Expected move percentile | int | Percentile rank of expected move |
| Feature importance | dict | Attribution scores by feature |
| Alert level | enum | None, Watch, Warning, Critical |

### 3.5 Feature Attribution

Every MonsterRunner alert includes explainability:

```yaml
alert:
  monster_score: 0.87
  expected_move_percentile: 98
  level: "Warning"
  top_features:
    - feature: "options_skew"
      contribution: 0.32
    - feature: "apex_entropy"
      contribution: 0.28
    - feature: "short_interest_change"
      contribution: 0.19
    - feature: "sector_divergence"
      contribution: 0.08
```

---

## 4. Integration with Apex

### 4.1 Data Flow

```
Market Data + Apex Outputs → Prediction System → Forecasts
                          → MonsterRunner → Rare Event Alerts
```

### 4.2 Non-Controlling Role

Both systems provide **informational outputs only**:

- No automatic trade execution
- No position modifications
- No order generation

Outputs inform human decision-making or feed into higher-level strategy systems that have their own controls and approvals.

### 4.3 Proof Logging

All predictions and alerts are logged:
- Input feature snapshots
- Model outputs with timestamps
- Confidence intervals
- Later outcome for backtesting

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

### 5.2 Uncertainty Handling

Both systems explicitly represent uncertainty:
- Confidence intervals on all predictions
- Calibrated probability outputs
- "Unknown" state when data is insufficient

### 5.3 Fail-Closed Behavior

When confidence is low or data is missing:
- Predictions abstain rather than guess
- MonsterRunner alerts require high confidence
- Missing features result in conservative outputs

---

## 6. Performance Metrics

| Metric | Prediction System | MonsterRunner |
|--------|-------------------|---------------|
| Latency | <50ms | <100ms |
| Update frequency | Per bar | Per bar |
| Historical accuracy | Tracked daily | Tracked per event |
| Calibration error | <5% | <10% |

---

## 7. Summary

The Prediction System and MonsterRunner extend QuantraCore Apex with forward-looking capabilities while maintaining the core principles of transparency, reproducibility, and safety. By providing probabilistic estimates with full feature attribution, these systems enhance human decision-making without overstepping into autonomous execution—keeping analysts informed and in control.
