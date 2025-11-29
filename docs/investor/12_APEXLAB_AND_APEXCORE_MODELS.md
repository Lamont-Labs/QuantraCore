# ApexLab and ApexCore Models

**Document Classification:** Investor Due Diligence — Technical  
**Version:** 9.0-A  
**Date:** November 2025  

---

## ApexLab v2: The Label Factory

### Where Do Labels Come From?

ApexLab v2 is the offline labeling environment that generates training datasets. It uses the deterministic engine as a "teacher" to create labels from historical data.

![ApexLab v2 Labeling Flow](../assets/investor/03_apexlab_v2_labeling_flow.png)

---

### ApexLab Pipeline

#### Step 1: Historical Data Loading

```
Raw OHLCV → ApexFeed → Historical bars (years of data)
```

#### Step 2: Window Building

```
Historical bars → Window Builder → 100-300 bar windows
```

Each window represents a point-in-time snapshot that could be analyzed.

#### Step 3: Teacher Labeling

```
Window → ApexEngine (Teacher) → Structural labels
```

The deterministic engine generates:
- QuantraScore (0-100)
- Quality tier (A+/A/B/C/D)
- Regime classification
- Risk assessment
- Protocol traces

#### Step 4: Future Outcome Calculation

```
Window end → Future window (t+N bars) → Outcome labels
```

**Critical:** Future outcomes are calculated from data *after* the window end, preventing lookahead.

Outcomes measured:
- Forward returns (1d, 5d, 20d)
- Maximum drawdown
- Maximum runup
- Runner achievement (>20% gain)

#### Step 5: Quality Tier Assignment

```
Structural labels + Future outcomes → Quality tier
```

Quality tiers combine:
- Teacher's structural assessment
- Actual future performance
- Risk-adjusted outcomes

#### Step 6: Dataset Export

```
All windows → Parquet/Arrow datasets
```

Final datasets contain:
- Input features (40+ fields)
- Teacher labels
- Future outcomes
- Quality tiers

---

### Label Schema (40+ Fields)

| Category | Fields | Description |
|----------|--------|-------------|
| Price | open, high, low, close, volume | OHLCV data |
| Technical | sma_20, rsi_14, atr_14, etc. | Indicator values |
| Structural | quantra_score, tier_flags, etc. | Engine outputs |
| Relative | sector_rank, market_rank, etc. | Comparative metrics |
| Outcome | fwd_return_5d, max_dd, is_runner | Future labels |
| Quality | quality_tier, regime, risk_level | Aggregated labels |

---

## ApexCore v2: The Neural Student

### What Does ApexCore Learn?

ApexCore v2 models learn to predict the teacher's structural assessments and future outcomes from input features.

![ApexCore v2 Model Family](../assets/investor/04_apexcore_v2_model_family.png)

---

### Model Architecture

#### Input Layer
- 40+ features from ApexLab schema
- Normalized and scaled
- Window-based representation

#### Prediction Heads

| Head | Output | Purpose |
|------|--------|---------|
| `quantra_score` | Float (0-100) | Predicted structural score |
| `runner_prob` | Float (0-1) | Probability of significant move |
| `quality_tier` | Categorical | Predicted quality tier |
| `avoid_trade` | Float (0-1) | Warning probability |
| `regime` | Categorical | Market regime classification |

#### Output Integration
- Multi-task learning (shared backbone)
- Head-specific loss weighting
- Ensemble averaging for robustness

---

### Model Variants

#### ApexCore v2 Big

| Property | Value |
|----------|-------|
| **Target** | Desktop / Server |
| **Ensemble Size** | 5 models |
| **AUC (Runner)** | 0.782 |
| **Brier Score** | 0.085 |
| **Calibration Error** | 0.072 |

**Use Cases:**
- Full-featured analysis
- Research workstations
- Server deployments

#### ApexCore v2 Mini

| Property | Value |
|----------|-------|
| **Target** | Mobile / Lightweight |
| **Ensemble Size** | 3 models |
| **AUC (Runner)** | 0.754 |
| **Brier Score** | 0.092 |
| **Calibration Error** | 0.081 |

**Use Cases:**
- Android deployment (QuantraVision)
- Resource-constrained environments
- Real-time inference

---

### Head Outputs Explained

#### QuantraScore Head
- **Output:** Predicted structural score (0-100)
- **Interpretation:** Higher = stronger structural setup
- **Loss:** Mean Squared Error
- **Note:** Approximates teacher's deterministic score

#### Runner Probability Head
- **Output:** Probability of >20% move
- **Interpretation:** Higher = more likely significant move
- **Loss:** Binary Cross-Entropy
- **Threshold:** 0.7 for A+ flag consideration

#### Quality Tier Head
- **Output:** A+, A, B, C, D classification
- **Interpretation:** Overall setup quality
- **Loss:** Cross-Entropy
- **Note:** Integrates structural and outcome factors

#### Avoid Trade Head
- **Output:** Warning probability (0-1)
- **Interpretation:** Higher = more caution warranted
- **Loss:** Binary Cross-Entropy
- **Threshold:** 0.3 max to allow candidate through

#### Regime Head
- **Output:** Market regime classification
- **Classes:** TRENDING, RANGING, VOLATILE, QUIET
- **Loss:** Cross-Entropy
- **Note:** Helps contextualize other predictions

---

### Why This Is a Ranker, Not an Oracle

**Critical Positioning:** ApexCore v2 is explicitly designed as a ranking assistant, not a prediction oracle.

| What It Does | What It Doesn't Do |
|--------------|-------------------|
| Ranks candidates | Generate candidates |
| Suggests caution | Override engine |
| Provides probabilities | Guarantee outcomes |
| Assists research | Replace judgment |

**Architectural Enforcement:**
- Cannot add candidates (only reorder)
- Engine authority always preserved
- Omega directives always override
- Fail-closed on any integrity issue

---

### Model Manifests

Every model has a manifest with:

```json
{
  "model_family": "apexcore_v2",
  "variant": "big",
  "ensemble_size": 5,
  "hashes": {
    "model": "sha256:abc123...",
    "config": "sha256:def456..."
  },
  "metrics": {
    "val_auc_runner": 0.782,
    "val_brier_runner": 0.085,
    "val_calibration_error_runner": 0.072
  },
  "thresholds": {
    "min_auc_runner_to_promote": 0.6,
    "max_calibration_error_to_promote": 0.15
  }
}
```

**Verification:** Before loading, manifests are checked for:
- Hash integrity (model file matches manifest)
- Metric thresholds (meets promotion criteria)
- Version compatibility

---

### Training Process Overview

| Step | Description |
|------|-------------|
| 1 | Generate datasets via ApexLab v2 |
| 2 | Split data (walk-forward, no lookahead) |
| 3 | Train ensemble members independently |
| 4 | Evaluate on validation set |
| 5 | Check promotion thresholds |
| 6 | Generate manifest with hashes |
| 7 | Promote to production if passing |

See [Training Process and Hyperparams](22_TRAINING_PROCESS_AND_HYPERPARAMS.md) for details.

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
