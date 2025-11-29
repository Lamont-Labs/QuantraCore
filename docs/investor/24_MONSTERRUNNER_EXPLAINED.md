# MonsterRunner Explained

**Document Classification:** Investor Due Diligence — Data/Training  
**Version:** 9.0-A  
**Date:** November 2025  

---

## What Is a MonsterRunner?

A **MonsterRunner** is a stock that achieves an extreme positive move — typically defined as a gain of 20% or more within a specified time window (default: 60 trading days).

---

## Definition and Thresholds

### Standard Definition

| Term | Definition |
|------|------------|
| **MonsterRunner** | Stock achieving ≥20% gain within 60 days |
| **Strong Runner** | Stock achieving ≥15% gain within 60 days |
| **Moderate Runner** | Stock achieving ≥10% gain within 60 days |

### Why 20%?

- Represents significant alpha potential
- Rare enough to be meaningful (~3% of setups)
- Common enough to have sufficient training examples
- Exceeds typical market noise by large margin

### Time Window

| Window | Use Case |
|--------|----------|
| 20 days | Short-term runners |
| 40 days | Medium-term runners |
| 60 days | Standard MonsterRunner (default) |
| 120 days | Long-term positioning |

---

## MonsterRunner Protocols (MR01-MR05)

### MR01: Volume Surge Detection

**Purpose:** Identify unusual volume accumulation that often precedes large moves.

**Features Analyzed:**
- Volume vs 20-day average
- Volume rate of change
- On-balance volume trend
- Volume at key price levels

**Signal:** High volume surge relative to baseline suggests accumulation.

### MR02: Breakout Pattern Recognition

**Purpose:** Detect price patterns associated with breakout potential.

**Features Analyzed:**
- Consolidation pattern (tight range)
- Prior resistance levels
- Price relative to 52-week high/low
- Breakout velocity when triggered

**Signal:** Tight consolidation near resistance with volume uptick.

### MR03: Relative Strength Extremes

**Purpose:** Identify stocks showing unusual strength vs market/sector.

**Features Analyzed:**
- Relative strength vs S&P 500
- Relative strength vs sector
- Rate of RS improvement
- RS rank in universe

**Signal:** Accelerating relative strength, especially from low base.

### MR04: Accumulation Signals

**Purpose:** Detect institutional accumulation behavior.

**Features Analyzed:**
- Accumulation/distribution line
- Money flow index
- Price-volume divergences
- Buying pressure indicators

**Signal:** Consistent accumulation pattern over multiple days/weeks.

### MR05: Multi-Factor Confirmation

**Purpose:** Combine signals from MR01-MR04 for higher confidence.

**Logic:**
```
runner_score = w1*MR01 + w2*MR02 + w3*MR03 + w4*MR04
runner_prob = sigmoid(runner_score)
is_runner_candidate = runner_prob > threshold
```

**Signal:** Multiple factors aligning increases runner probability.

---

## Feature Overview for MonsterRunner

### Price Features

| Feature | Description |
|---------|-------------|
| `pct_from_52w_high` | Distance from 52-week high |
| `pct_from_52w_low` | Distance from 52-week low |
| `consolidation_tightness` | Range compression metric |
| `breakout_velocity` | Speed of recent price move |

### Volume Features

| Feature | Description |
|---------|-------------|
| `volume_surge_ratio` | Current vs average volume |
| `volume_trend` | Volume direction over time |
| `obv_divergence` | Price vs OBV divergence |
| `accumulation_score` | A/D line strength |

### Relative Features

| Feature | Description |
|---------|-------------|
| `rs_vs_spy` | Relative strength vs S&P 500 |
| `rs_vs_sector` | Relative strength vs sector |
| `rs_rank_percentile` | RS rank in universe |
| `rs_momentum` | RS rate of change |

### Technical Features

| Feature | Description |
|---------|-------------|
| `above_sma_20` | Price above 20-day SMA |
| `above_sma_50` | Price above 50-day SMA |
| `macd_histogram` | MACD momentum |
| `rsi_14` | Relative strength index |

---

## How Labels Feed Into Training

### Label Generation

For each historical window, ApexLab calculates:

```python
future_returns = calculate_returns(
    data,
    window_end=t,
    horizon=60
)

is_runner = future_returns.max_gain >= 0.20
```

### Distribution

| Category | Prevalence |
|----------|------------|
| MonsterRunner (≥20%) | ~3% |
| Strong Runner (≥15%) | ~8% |
| Moderate Runner (≥10%) | ~18% |
| Below threshold | ~82% |

### Class Imbalance Handling

Due to rare occurrence, training uses:
- Class weighting (~16x for runners)
- Stratified sampling
- Threshold optimization for F1

---

## ApexCore Runner Probability Head

### What It Outputs

| Output | Range | Interpretation |
|--------|-------|----------------|
| `runner_prob` | 0.0 - 1.0 | Probability of ≥20% gain |

### Interpretation Guide

| Probability | Interpretation | Action |
|-------------|----------------|--------|
| < 0.3 | Low runner likelihood | Standard treatment |
| 0.3 - 0.5 | Moderate possibility | Worth monitoring |
| 0.5 - 0.7 | Elevated probability | Prioritize for research |
| > 0.7 | High runner probability | Strong candidate |

### Important Caveat

High runner probability does NOT mean:
- Guaranteed 20% gain
- Should buy immediately
- No downside risk

It means:
- Historical patterns similar to past runners
- Worth deeper research attention
- Higher ranking in candidate list

---

## Integration With Engine

### Protocol Flow

```
Engine analysis → MR01-MR05 protocols → runner_prob estimate
```

### Quality Tier Impact

Runner probability affects quality tier assignment:

| Base Tier | High Runner Prob | Adjusted Tier |
|-----------|------------------|---------------|
| B | >0.7 | → A |
| A | >0.7 | → A+ |
| C | >0.7 | → B |

### PredictiveAdvisor Use

The advisory layer uses runner probability for:
- Candidate ranking (higher prob = higher rank)
- Research prioritization
- Watchlist suggestions

---

## Validation Results

### Hit Rate at Thresholds

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.5 | 0.22 | 0.45 | 0.30 |
| 0.6 | 0.28 | 0.32 | 0.30 |
| 0.7 | 0.32 | 0.18 | 0.23 |
| 0.8 | 0.38 | 0.08 | 0.13 |

**Interpretation:** Higher thresholds improve precision but sacrifice recall. The 0.7 threshold balances both for research prioritization.

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
