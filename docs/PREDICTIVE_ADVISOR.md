# PredictiveAdvisor — Fail-Closed Engine Integration

**Version:** 9.0-A  
**Component:** PredictiveAdvisor  
**Role:** Fail-closed integration between ApexCore V2 and Apex Engine

---

## 1. Overview

PredictiveAdvisor provides fail-closed integration between ApexCore V2 model predictions and the deterministic Apex engine. It translates model outputs into advisory recommendations while enforcing strict safety rules.

---

## 2. Core Principle

**The deterministic Apex engine always has final authority.**

PredictiveAdvisor is strictly advisory — it can uprank, downrank, or avoid, but it cannot override engine safety rules or Omega directives.

---

## 3. Recommendations

| Recommendation | Code | Description |
|----------------|------|-------------|
| DISABLED | 0 | Model disabled due to failure |
| NEUTRAL | 1 | No adjustment (default safe state) |
| UPRANK | 2 | Positive signal, consider upranking |
| DOWNRANK | 3 | Negative signal, consider downranking |
| AVOID | 4 | Strong avoid signal |

---

## 4. Fail-Closed Rules

PredictiveAdvisor enforces strict fail-closed behavior:

### 4.1 Model Disabled (DISABLED)

Triggered when:
- Model hash mismatch detected
- Manifest verification fails
- Model loading error occurs

### 4.2 Neutral State (NEUTRAL)

Triggered when:
- Disagreement > 0.5 between model and engine
- Model confidence below threshold
- Ambiguous signals detected

### 4.3 Avoid Trade (AVOID)

Triggered when:
- `avoid_trade` probability > 0.8
- `quality_tier` is D with high confidence
- Multiple safety flags active

### 4.4 Uprank (UPRANK)

Triggered when:
- `runner_prob` > 0.6
- `quality_tier` in (A, A_PLUS)
- No safety flags active

### 4.5 Downrank (DOWNRANK)

Triggered when:
- `quality_tier` in (C, D)
- `avoid_trade` > 0.3 but < 0.8
- Weak structural signals

---

## 5. Usage

```python
from src.quantracore_apex.core.integration_predictive import (
    PredictiveAdvisor,
    PredictiveRecommendation
)

# Initialize advisor
advisor = PredictiveAdvisor(
    model_path="./models/apexcore_v2_big.pkl",
    manifest_path="./models/manifest.json",
    disagreement_threshold=0.5,
    avoid_trade_cap=0.8
)

# Get recommendation
recommendation = advisor.get_recommendation(
    features=features,
    engine_quantra_score=65.0
)

# Check recommendation
if recommendation == PredictiveRecommendation.AVOID:
    print("Model advises avoiding this setup")
elif recommendation == PredictiveRecommendation.UPRANK:
    print("Model suggests positive outlook")
elif recommendation == PredictiveRecommendation.DISABLED:
    print("Model is disabled - using engine only")
```

---

## 6. Disagreement Calculation

Disagreement measures the difference between model prediction and engine output:

```
disagreement = abs(model_quantra_score - engine_quantra_score) / 100
```

If `disagreement > threshold`, the advisor returns NEUTRAL to avoid conflicting signals.

---

## 7. Integration with Engine

```
┌─────────────────┐     ┌──────────────────┐
│   Apex Engine   │────▶│ PredictiveAdvisor│
│ (Deterministic) │     │ (Advisory Only)  │
└─────────────────┘     └──────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │  Recommendation  │
         │              │ DISABLED/NEUTRAL/│
         │              │ UPRANK/DOWNRANK/ │
         │              │     AVOID        │
         │              └──────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Final Decision                  │
│  Engine rules ALWAYS take precedence    │
│  Omega directives NEVER overridden      │
└─────────────────────────────────────────┘
```

---

## 8. Safety Guarantees

| Guarantee | Enforcement |
|-----------|-------------|
| Model cannot override Omega | Omega checked after advisor |
| Hash mismatch disables model | DISABLED returned on mismatch |
| High disagreement = neutral | NEUTRAL on disagreement > 0.5 |
| High avoid = avoid | AVOID on avoid_trade > 0.8 |
| No model = engine only | Graceful fallback |

---

## 9. Configuration

```python
# PredictiveAdvisor configuration
config = {
    "disagreement_threshold": 0.5,  # Max allowed disagreement
    "avoid_trade_cap": 0.8,         # Threshold for AVOID
    "uprank_runner_threshold": 0.6,  # Min runner_prob for UPRANK
    "uprank_quality_tiers": ["A", "A_PLUS"],  # Required tiers
    "downrank_quality_tiers": ["C", "D"]      # Trigger tiers
}
```

---

## 10. Test Coverage

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_predictive_integration.py | 16 | Integration tests |

Test cases include:
- Hash mismatch detection
- Disagreement threshold enforcement
- Avoid-trade cap enforcement
- Uprank/downrank logic
- Disabled state handling

---

## 11. Related Documentation

- [ApexCore V2](APEXCORE_V2.md)
- [ApexCore Models](APEXCORE_MODELS.md)
- [Omega Directives](OMEGA_DIRECTIVES.md)
- [Core Engine](CORE_ENGINE.md)

---

**QuantraCore Apex v9.0-A** — Lamont Labs | November 2025
