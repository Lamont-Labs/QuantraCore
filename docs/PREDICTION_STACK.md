# QuantraCore Apex™ — Prediction Stack

**Version:** 8.0  
**Component:** Structural Forecasting  
**Status:** Active

---

## Overview

The Prediction Stack extends Apex's structural understanding into supervised predictive intelligence while remaining fully compliant. Predictions are strictly informational and are never tied to trade recommendations or actionable signals.

---

## Compliance Rules

- No prediction may imply entry/exit/buy/sell
- All predictions must be phrased as structural probabilities
- All models must remain subordinate to Apex deterministic output
- Predictions must be logged and version-controlled
- Models are trained offline via ApexLab only

---

## Prediction Components

| Component | Purpose |
|-----------|---------|
| Expected Move Engine | Estimate normalized expected move |
| Volatility Projection Engine | Predict volatility direction |
| Compression/Expansion Forecaster | Forecast compression state |
| Continuation Probability Estimator | Predict continuation likelihood |
| Regime Transition Predictor | Forecast regime shifts |
| Early Instability Predictor | Detect instability signatures |
| MonsterRunner Engine | Identify rare-event precursors |

---

## 1. Expected Move Engine

**Purpose:** Estimate the normalized expected move over the next X-bars.

**Compliance:** Strictly informational — NOT a trade signal.

### Features Used

- Volatility band
- Compression score
- Continuation state
- Sector volatility
- Volume ignition patterns

### Outputs

- `expected_move_normalized`
- `confidence_score`

### Invariants

- Never directional by itself
- Never used for trading decisions

---

## 2. Volatility Projection Engine

**Purpose:** Predict volatility cluster direction: normalization, expansion, or contraction.

### Inputs

- ATR bands
- Entropy state
- Sector volatility
- Drift score

### Outputs

- `vol_change`: up | down | neutral
- `vol_confidence`

### Notes

Used for structural awareness only.

---

## 3. Compression/Expansion Forecaster

**Purpose:** Estimate the likelihood of compression holding or expanding.

### Signals

- Compression score
- Strength slope
- Volume contraction
- Volatility floor alignment

### Outputs

- `compression_state_future`: hold | expand | break
- `forecast_confidence`

---

## 4. Continuation Probability Estimator

**Purpose:** Predict the probability of structural continuation (non-directional).

### Features

- Continuation state
- ZDE alignment
- Strength distribution
- Trend integrity

### Output

- `continuation_probability_0_1`

---

## 5. Regime Transition Predictor

**Purpose:** Forecast the likelihood of regime shifts (trend ↔ range).

### Features

- Regime
- Entropy
- Drift
- Sector flow

### Output

- `transition_probability`

---

## 6. Early Instability Predictor

**Purpose:** Detect early signatures of instability in structure.

### Signals

- Microtrend fragmentation
- Entropy uptrend
- Suppression onset
- Drift tilt

### Outputs

- `instability_flag`: yes | no
- `instability_intensity`

---

## 7. MonsterRunner Engine

See [MonsterRunner Documentation](MONSTERRUNNER.md) for full specification.

**Purpose:** Identify the earliest detectable signatures that historically precede extreme moves ("monster runners").

---

## Prediction Compliance Framework

### Rules

- Predictions must be statistical, not actionable
- No directional trade language
- No implied timing windows
- Omega-4 scans for unsafe phrasing
- Prediction outputs may not reference specific price levels
- Prediction outputs must remain high-level structural assessments

### Prohibited Language

Predictions may NOT contain:

| Term | Reason |
|------|--------|
| buy | Trade action |
| sell | Trade action |
| entry | Timing instruction |
| exit | Timing instruction |
| target | Price target |
| stop | Risk instruction |

---

## Output Format

All prediction outputs follow this structure:

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "symbol": "AAPL",
  "predictions": {
    "expected_move": {
      "value_normalized": 0.023,
      "confidence": 0.78
    },
    "volatility_projection": {
      "direction": "up",
      "confidence": 0.65
    },
    "compression_forecast": {
      "state": "break",
      "confidence": 0.82
    },
    "continuation_probability": 0.71,
    "regime_transition_probability": 0.15,
    "instability": {
      "flag": false,
      "intensity": 0.12
    }
  },
  "compliance_check": "passed",
  "apex_version": "8.0"
}
```

---

## Integration with ApexLab

All prediction models are trained via ApexLab:

1. Apex generates structural labels
2. ApexLab creates training targets
3. Prediction heads trained alongside ApexCore
4. Validation against golden-set
5. Rejection if divergence detected

---

## Related Documentation

- [MonsterRunner](MONSTERRUNNER.md)
- [Core Engine](CORE_ENGINE.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
- [ApexCore Models](APEXCORE_MODELS.md)
