# QuantraCore Apex™ — Core Engine Specification

**Version:** 8.0  
**Component:** Deterministic Signal Logic  
**Status:** Active

---

## Overview

The Core Engine is the master deterministic logic engine. All intelligence and all models originate from it. Every decision is explainable via protocol trace, and QuantraScore is required for every evaluation.

---

## Guarantees

- Deterministic outputs for identical inputs
- Protocol versioning guarantees reproducibility
- Every decision is explainable via protocol trace
- Compliance-first: no buy/sell/entry recommendations
- QuantraScore required for every evaluation
- ApexCore models must align with Apex outputs

---

## Execution Flow

```
OHLCV Input
    ↓
Data Normalization
    ↓
Baseline Microtrait Extraction
    ↓
Protocol Execution (T01–T80, LP01–LP25)
    ↓
Signal Engines
    ↓
Risk Gates
    ↓
QuantraScore Fusion
    ↓
Verdict Construction
    ↓
Proof Logging
```

---

## Deterministic Invariants

- No randomness allowed anywhere
- All protocol results stored in trace
- Same bars = same outputs
- All thresholds version-locked
- All state machines (entropy/drift) deterministic

---

## Signal Engine Components

### 1. ZDE Engine (Zero Deviation Engine)

**Purpose:** Detect micro deviations from ideal structural alignment.

| Signal | Description |
|--------|-------------|
| Microtrend shift | Change in micro-level trend direction |
| Slope break | Deviation from expected slope |
| Deviation collapse | Structural collapse detection |

**Outputs:**
- `zde_score`
- `zde_microtrend`

**Invariants:**
- ZDE never produces direction by itself
- Used to confirm or reject continuation

---

### 2. Continuation Validator

**Purpose:** Validate structural continuation probability.

**Logic:**
- Compare last 3–12 swings
- Detect trend maturity vs exhaustion
- Check volume pattern alignment
- Integrate ZDE + microtraits

**Outputs:**
- `continuation_state`: continue | stall | reverse
- `continuation_strength`: 0–1

---

### 3. Entry Timing Optimizer

**Purpose:** Determine quality of timing window (NOT a buy signal).

**Signals:**
- Micro compression release
- Slope harmony
- Volatility contraction
- Volume ignition delta

**Output:**
- `timing_quality`: low | medium | high

**Compliance:** Never implies actual entry locations.

---

### 4. Volume Spike Mapper

**Purpose:** Detect meaningful abnormal volume events.

**Signals:**
- Ignition spikes
- Climax events
- Dry-up zones

**Outputs:**
- `volume_spike_state`
- `volume_spike_intensity`

---

### 5. Microtrait Engine

**Purpose:** Extract structural micro-features used system-wide.

**Traits:**
| Trait | Description |
|-------|-------------|
| `wick_ratio` | Wick to body ratio |
| `body_ratio` | Body size relative to range |
| `bullish_pct_last20` | Bullish candle percentage (20-bar) |
| `volatility_cluster` | Volatility clustering metric |
| `range_density` | Range utilization density |
| `tail_fatigue` | Exhaustion signal from tails |
| `slope_micro` | Micro-level slope measurement |
| `compression_pulse` | Compression intensity |
| `noise_floor` | Baseline noise level |
| `strength_slope` | Strength trend measurement |

**Output:** `microtraits_packet`

**Notes:**
- Used by all downstream logic
- Deterministic feature extraction required

---

### 6. Suppression Engine

**Purpose:** Detect structural invalidation + trend clash zones.

**Detection:**
- Cross-trend wicks
- Overlapping ranges
- Multi-timeframe clashes
- Sector pressure mismatch

**Outputs:**
- `suppression_state`: none | active | clash
- `suppression_intensity`: 0–1

---

### 7. Entropy Engine

**Purpose:** Quantify disorder vs signal clarity.

**Signals:**
- Candle randomness
- Noise above baseline
- Microtrend fragmentation

**Outputs:**
- `entropy_state`: normal | elevated | reject
- `entropy_score`: 0–1

---

### 8. Drift Engine

**Purpose:** Track divergence from ideal structural behavior.

**Detection:**
- Deviation from trend slope
- Deviation from volatility bands
- Deviation from sector alignment

**Outputs:**
- `drift_state`: stable | tilt | unstable
- `drift_score`: 0–1

---

### 9. Regime Classifier

**Regime Types:**
- `trend_up`
- `trend_down`
- `range_bound`
- `compression`
- `transition`

**Signals Used:**
- Slope
- Volatility band
- Microtraits
- Volume patterns
- Compression score

**Output:** `regime`

---

### 10. Sector Context Engine

**Purpose:** Measure sector-aligned pressure + correlation.

**Features:**
- `sector_strength`
- `sector_volatility`
- `sector_flow`

**Output:** `sector_bias`

---

### 11. Signal Classifier

**Purpose:** Combine all engines into a structured signal snapshot.

**Outputs:**
- `trend`
- `pressure`
- `strength`
- `risk_tier`

**Note:** Not trade instructions—only structural labeling.

---

## QuantraScore

**Description:** Final deterministic score 0–100.

**Purpose:** Fuse all structural information into one normalized metric.

### Contributing Factors

- Trend alignment
- Regime classification
- Volume synergy
- Volatility banding
- Continuation signal
- Compression vs noise
- Entropy penalty
- Suppression penalty
- Drift penalty
- Microtrait fusion
- Omega overrides

### Score Buckets

| Bucket | Range | Interpretation |
|--------|-------|----------------|
| Fail | 0–24 | Structural rejection |
| Wait | 25–49 | Insufficient clarity |
| Pass | 50–74 | Acceptable structure |
| Strong Pass | 75–100 | High structural quality |

### Invariants

- No randomness
- Every component traceable
- Same input = same score
- Score must be logged

---

## Verdict System

### Construction

| Field | Values |
|-------|--------|
| `trend` | upward | downward | sideways | unclear |
| `pressure` | buy_bias | sell_bias | neutral |
| `strength` | low | medium | high |
| `risk_level` | low | medium | high |

**Prohibition:** No buy/sell recommendations.

---

## Proof Logging

### Stored Data

- OHLCV input window
- All microtraits
- All protocols fired
- All states (entropy, suppression, drift)
- QuantraScore
- Final verdict summary

**Format:** JSON

### Visualizers

- Protocol map
- Score timeline
- Entropy/drift charts

---

## Execution Modes

| Mode | Latency | Usage |
|------|---------|-------|
| Fast Scan | 1–5 seconds | Universe sweeps |
| Deep Scan | 30–90 seconds | High-confidence structural analysis |
| Micro Scan | <1 second | Real-time recalculation |

---

## Related Documentation

- [Protocols: Tier (T01–T80)](PROTOCOLS_TIER.md)
- [Protocols: Learning (LP01–LP25)](PROTOCOLS_LEARNING.md)
- [Omega Directives](OMEGA_DIRECTIVES.md)
- [Architecture](ARCHITECTURE.md)
