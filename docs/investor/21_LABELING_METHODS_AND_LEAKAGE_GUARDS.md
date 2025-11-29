# Labeling Methods and Leakage Guards

**Document Classification:** Investor Due Diligence — Data/Training  
**Version:** 9.0-A  
**Date:** November 2025  

---

## How Do You Prevent Lookahead/Overfitting?

This document explains the labeling methodology and the strict guards in place to prevent data leakage and overfitting.

---

## Time-Split Regime (Walk-Forward)

### The Problem with Naive Splits

Random train/test splits in time-series data cause leakage:

```
❌ WRONG: Random split allows future data to leak into training
[2018 bar]─┬─ Train
[2019 bar]─┼─ Test   ← Can see future patterns in training
[2020 bar]─┼─ Train
[2021 bar]─┼─ Test
[2022 bar]─┴─ Train
```

### Walk-Forward Solution

Strict temporal ordering prevents any future information in training:

```
✓ CORRECT: Temporal split maintains time ordering
[2018]──┬
[2019]──┤ Train (no future data visible)
[2020]──┤
[2021]──┘
[2022]──── Validation (separated in time)
[2023]──┬
[2024]──┘ Test (held out until final evaluation)
```

### Implementation

```python
def create_walk_forward_splits(data, train_end, val_end):
    train = data[data.date <= train_end]
    val = data[(data.date > train_end) & (data.date <= val_end)]
    test = data[data.date > val_end]
    return train, val, test
```

---

## Label Definition

### Teacher Labels

The deterministic engine (teacher) generates structural labels from each window:

| Label | Source | Description |
|-------|--------|-------------|
| `quantra_score` | Engine | Structural score (0-100) |
| `quality_tier` | Engine | Quality classification |
| `regime` | Engine | Market condition |
| `risk_level` | Engine | Risk assessment |

**Key Property:** Teacher labels use only data within the window — no future information.

### Outcome Labels

Future outcomes are calculated from data AFTER the window end:

| Label | Definition | Lookahead Window |
|-------|------------|------------------|
| `fwd_return_1d` | Return over next 1 day | t+1 |
| `fwd_return_5d` | Return over next 5 days | t+1 to t+5 |
| `fwd_return_20d` | Return over next 20 days | t+1 to t+20 |
| `max_drawdown` | Maximum pullback in window | t+1 to t+20 |
| `max_runup` | Maximum gain in window | t+1 to t+20 |
| `is_runner` | Achieved >20% gain | t+1 to t+60 |

---

## Leakage Prevention

### Rule 1: Strict Time Boundary

```
Window: [t-299, t-298, ..., t]
Future: [t+1, t+2, ..., t+N]

Teacher sees: Window only
Outcome calculated from: Future only
```

No bar from the future window ever enters the feature calculation.

### Rule 2: No Feature Engineering on Future

Features are calculated only from window data:

```
✓ ALLOWED: moving_avg_20 = mean(close[t-19:t+1])
✓ ALLOWED: rsi_14 = calc_rsi(close[t-13:t+1])
✗ FORBIDDEN: future_min = min(close[t+1:t+20])
```

### Rule 3: Gap Between Window and Outcome

A gap can be introduced to prevent immediate future leakage:

```
Window: [t-299, t]
Gap: [t+1] (not used)
Outcome: [t+2, t+N]
```

This prevents the model from learning patterns that only work because of next-bar noise.

### Rule 4: No Overlapping Windows in Train/Val

When creating windows, we ensure train and validation sets don't share overlapping periods:

```
Train windows: end date <= train_cutoff
Val windows: start date >= train_cutoff + gap_days
```

---

## Quality Tier Methodology

Quality tiers combine structural assessment with actual outcomes:

| Tier | Criteria |
|------|----------|
| **A+** | High QuantraScore + actual runner (>20% gain) |
| **A** | High QuantraScore + strong return (>10% gain) |
| **B** | Moderate QuantraScore + positive return |
| **C** | Low QuantraScore or flat return |
| **D** | Negative outcome (loss) |

**Important:** Tiers are assigned retrospectively using future outcomes — they are training labels, not predictions.

---

## Validation of No Leakage

### Test 1: Future Feature Check

```python
def verify_no_future_features(features, window_end_idx, data):
    for col in features.columns:
        if requires_future_data(col, window_end_idx, data):
            raise LeakageError(f"Feature {col} uses future data")
```

### Test 2: Temporal Ordering Check

```python
def verify_temporal_order(train, val, test):
    assert train.date.max() < val.date.min()
    assert val.date.max() < test.date.min()
```

### Test 3: Window Overlap Check

```python
def verify_no_overlap(windows_train, windows_val):
    train_dates = set(w.end_date for w in windows_train)
    val_dates = set(w.start_date for w in windows_val)
    overlap = train_dates.intersection(val_dates)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping dates"
```

---

## Overfitting Guards

### Guard 1: Regularization

- L2 regularization on model weights
- Dropout during training
- Early stopping based on validation loss

### Guard 2: Cross-Validation

Walk-forward cross-validation with multiple folds:

```
Fold 1: Train [2018-2020], Val [2021]
Fold 2: Train [2018-2021], Val [2022]
Fold 3: Train [2018-2022], Val [2023]
Final:  Train [2018-2023], Test [2024]
```

### Guard 3: Ensemble Disagreement

High disagreement across ensemble members suggests overfitting to noise:

```
If std(predictions) > threshold:
    Warning: possible overfit, use engine-only
```

### Guard 4: Regime-Specific Evaluation

Evaluate separately on different market regimes:
- Trending markets
- Ranging markets
- High volatility periods
- Low volatility periods

Consistent performance across regimes suggests generalization.

---

## Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Limited history | May miss rare events | Conservative predictions |
| Survivorship bias | Only sees survivors | Track delisted stocks |
| Regime shift | Future may differ | Regime-aware evaluation |
| Feature selection | May be overfit | Simple features preferred |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
