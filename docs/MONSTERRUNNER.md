# QuantraCore Apex™ — MonsterRunner Engine

**Version:** 8.0  
**Component:** Rare-Event Detection  
**Status:** Active

---

## Overview

MonsterRunner identifies the earliest detectable signatures that historically precede extreme moves ("monster runners"). This engine is NOT a trade system—it provides structural awareness for research, filtering, and long-horizon pattern understanding.

---

## Compliance Statement

| Rule | Description |
|------|-------------|
| Not a buy/sell signal | No trade recommendations |
| Not a recommendation | No actionable advice |
| Structural probabilities only | Statistical outputs |
| Omega-4 compliance | All outputs scrubbed |

---

## Signatures Detected

MonsterRunner scans for these precursor patterns:

| Signature | Description |
|-----------|-------------|
| Phase-compression coil | Extreme range compression |
| Entropy collapse with volume ignition | Clarity emergence with volume |
| Sector synchronization | Cross-sector alignment |
| Microtrend ignition pulse | Micro-level acceleration |
| Multi-swing continuation cascade | Sequential continuation patterns |
| Volatility supercluster | Extreme volatility clustering |
| Break-structure anticipatory signal | Pre-breakout formation |
| Sector-phase sympathetic ignition | Sector-driven ignition |
| Hyper-rejection → expansion rotation | Rejection leading to expansion |
| Ignition-through-resistance patterns | Resistance break patterns |
| Phase-transition supercycle | Major phase transition |

---

## Rare Event Features

| Feature | Description |
|---------|-------------|
| `compression_trace` | Compression history |
| `entropy_floor` | Minimum entropy level |
| `volume_superpulse` | Extreme volume event |
| `ZDE_microburst` | ZDE acceleration |
| `sector_phase_alignment` | Sector synchronization |
| `drift_instability_slope` | Drift acceleration |
| `strength_hyper_slope` | Strength acceleration |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `runner_probability` | 0–1 | Likelihood of extreme move |
| `runner_state` | enum | idle \| forming \| primed |
| `rare_event_class` | string | Pattern classification |
| `confidence` | 0–1 | Detection confidence |

### Runner States

| State | Description |
|-------|-------------|
| `idle` | No precursor patterns detected |
| `forming` | Early signatures emerging |
| `primed` | Multiple signatures aligned |

---

## Protocol Integration

MonsterRunner leverages Tier Protocols T66–T80:

| Protocol | Function |
|----------|----------|
| T66 | Extreme compression coil |
| T67 | Volatility supercluster |
| T68 | Pre-breakout ignition |
| T69 | Sector-phase sympathetic ignition |
| T70 | Microstructure hyper-acceleration |
| T71 | Phase-transition root test |
| T72 | Compression → expansion supercycle |
| T73 | Microtrend ignition microburst |
| T74 | Anti-structure detector |
| T75 | Hyper-rejection structure |
| T76 | Rare continuation cascade |
| T77 | Break-structure anticipatory signal |
| T78 | Sector superposition structure |
| T79 | Impending instability signature |
| T80 | MonsterRunner precursor pattern |

---

## Logging

All MonsterRunner activity is logged for research:

- All runner flags logged
- All rare patterns saved as exemplars
- Golden data set stored for ApexLab training

### Log Format

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "symbol": "TSLA",
  "runner_probability": 0.73,
  "runner_state": "forming",
  "rare_event_class": "phase_compression_coil",
  "confidence": 0.81,
  "signatures_detected": [
    "compression_trace",
    "entropy_floor",
    "sector_phase_alignment"
  ],
  "protocols_fired": ["T66", "T71", "T72"],
  "apex_quantrascore": 78
}
```

---

## ApexLab Integration

MonsterRunner patterns are preserved for model training:

| Protocol | Function |
|----------|----------|
| LP12 | Rare pattern bookmarking |
| LP17 | Hyper-rare anomaly preservation |
| LP25 | Golden-set exemplar preservation |

### Weighted Sampling

- Rare patterns receive higher sampling weight
- Prevents model from ignoring low-frequency events
- Maintains pattern recognition for extreme moves

### Reproducibility

- Strict reproducibility for cross-version comparisons
- All patterns hash-locked
- Version tracking for pattern detection

---

## Dashboard Console

MonsterRunner has a dedicated console in the Apex Dashboard:

### Display Elements

- Runner probability gauge
- Runner state indicator
- Rare event classification
- Historical exemplar viewer
- Signature timeline
- Protocol firing heatmap

### Console Features

- Real-time probability updates
- Historical pattern comparison
- Signature correlation analysis
- Research mode filtering

---

## Use Cases

### Research

- Pattern discovery
- Historical backtesting
- Structural research

### Filtering

- Universe narrowing
- Attention prioritization
- Volatility awareness

### Understanding

- Market structure analysis
- Regime transition study
- Extreme event preparation

---

## Limitations

MonsterRunner does NOT:

- Predict direction
- Recommend trades
- Provide timing signals
- Guarantee outcomes

MonsterRunner ONLY:

- Detects structural patterns
- Provides probability estimates
- Logs for research purposes

---

## Related Documentation

- [Prediction Stack](PREDICTION_STACK.md)
- [Protocols: Tier (T66–T80)](PROTOCOLS_TIER.md)
- [Core Engine](CORE_ENGINE.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
