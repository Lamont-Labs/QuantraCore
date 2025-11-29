# Engine and Protocols

**Document Classification:** Investor Due Diligence — Technical  
**Version:** 9.0-A  
**Date:** November 2025  

---

## What Does the Deterministic Engine Do?

The ApexEngine is the core analysis component of QuantraCore Apex. It applies 145 protocols to market data to generate structural signals with guaranteed reproducibility.

---

## Engine Responsibilities

| Responsibility | Description |
|----------------|-------------|
| **Protocol Execution** | Run 145 protocols in deterministic order |
| **Signal Generation** | Produce QuantraScore and structural labels |
| **Trace Logging** | Record every decision for audit |
| **Determinism Guarantee** | Same inputs always produce same outputs |
| **Safety Override** | Omega directives can halt or modify any signal |

---

## Protocol Taxonomy

### Tier Protocols (T01-T80)

**Purpose:** Structural analysis rules that evaluate market conditions.

| Group | Protocols | Focus |
|-------|-----------|-------|
| T01-T10 | Foundation | Price structure, basic patterns |
| T11-T20 | Momentum | Trend strength, directional bias |
| T21-T30 | Volatility | Range expansion, compression |
| T31-T40 | Volume | Participation, accumulation |
| T41-T50 | Relative | Sector/market comparison |
| T51-T60 | Technical | Indicator-based signals |
| T61-T70 | Risk | Downside assessment |
| T71-T80 | Integration | Multi-factor combination |

**Output:** Each protocol contributes to the aggregate QuantraScore and structural flags.

---

### Learning Protocols (LP01-LP25)

**Purpose:** Generate labels for ApexLab training datasets.

| Group | Protocols | Focus |
|-------|-----------|-------|
| LP01-LP05 | Outcome | Future return classification |
| LP06-LP10 | Quality | Setup quality tier assignment |
| LP11-LP15 | Risk | Drawdown and risk labeling |
| LP16-LP20 | Regime | Market condition classification |
| LP21-LP25 | Special | MonsterRunner and edge cases |

**Output:** Labels used to train ApexCore v2 models.

---

### MonsterRunner Protocols (MR01-MR20)

**Purpose:** Detect potential extreme moves (>20% gains).

| Protocol | Focus |
|----------|-------|
| MR01 | Volume surge detection |
| MR02 | Breakout pattern recognition |
| MR03 | Relative strength extremes |
| MR04 | Accumulation signals |
| MR05 | Multi-factor confirmation |

**Output:** `runner_prob` and MonsterRunner classification.

---

### Omega Directives (Ω1-Ω20)

**Purpose:** Safety override protocols that can halt or modify any signal.

| Directive | Function | Status |
|-----------|----------|--------|
| Ω1 | Hard safety lock (emergency halt) | Available |
| Ω2 | Entropy override (volatility kill-switch) | Available |
| Ω3 | Drift override (model drift detection) | Available |
| Ω4 | Compliance mode (research-only lock) | **Always Active** |
| Ω5 | Suppression lock (signal suppression) | Available |

**Key Property:** Omega directives always override engine and model outputs. Ω4 is permanently active, enforcing research-only mode.

---

## Determinism Guarantees

### What Determinism Means

**Guarantee:** Identical inputs always produce identical outputs.

This is verified through:

1. **Seed Control:** All randomness is seeded (though none is used by default)
2. **Fixed Ordering:** Protocols execute in deterministic order
3. **Hash Verification:** Outputs can be hashed and compared across runs
4. **Proof Logs:** Every decision is logged with reproducible traces

### Verification Method

```
Run 1: input_hash → engine → output_hash_A
Run 2: input_hash → engine → output_hash_B

Assert: output_hash_A == output_hash_B
```

This is tested with 150 iterations (3x FINRA 15-09 requirement).

---

## Proof Logs and Replay

### What Gets Logged

| Field | Description |
|-------|-------------|
| `timestamp` | Execution time (UTC) |
| `symbol` | Symbol analyzed |
| `input_hash` | SHA256 of input data |
| `output_hash` | SHA256 of output |
| `quantra_score` | Final score (0-100) |
| `protocol_traces` | Per-protocol contribution |
| `omega_status` | Omega directive states |

### Replay Capability

Proof logs can be replayed to:
- Verify output correctness
- Debug specific decisions
- Audit historical analysis
- Demonstrate determinism to regulators

---

## Example Signal Walkthrough

**Scenario:** Analyzing symbol XYZ on date 2025-11-15

### Step 1: Data Preparation
```
Input: 300 bars of OHLCV for XYZ
Input hash: sha256:abc123...
```

### Step 2: Protocol Execution
```
T01 (Price Structure): +3.2 contribution
T05 (Base Pattern): +2.1 contribution
T15 (Momentum): +4.5 contribution
...
T78 (Risk Adjustment): -1.2 contribution
```

### Step 3: Aggregation
```
Raw score: 72.4
Risk adjustment: -5.2
Final QuantraScore: 67.2
```

### Step 4: Label Generation
```
LP03 (Quality): B tier
LP08 (Regime): TRENDING
LP15 (Risk): MODERATE
```

### Step 5: MonsterRunner Check
```
MR01-MR20: runner_prob = 0.23
Classification: NOT_RUNNER (threshold 0.7)
```

### Step 6: Omega Check
```
Ω1: INACTIVE
Ω2: INACTIVE
Ω3: INACTIVE
Ω4: ACTIVE (research-only mode)
Ω5: INACTIVE
```

### Step 7: Output
```
{
  "symbol": "XYZ",
  "quantra_score": 67.2,
  "quality_tier": "B",
  "regime": "TRENDING",
  "runner_prob": 0.23,
  "output_hash": "sha256:def456...",
  "omega_status": "Ω4_ACTIVE"
}
```

---

## Protocol Dependencies

Protocols are organized in dependency layers:

```
Layer 1: Foundation (T01-T10, LP01-LP05)
    ↓
Layer 2: Analysis (T11-T50, LP06-LP15)
    ↓
Layer 3: Integration (T51-T80, LP16-LP25, MR01-MR20)
    ↓
Layer 4: Override (Ω1-Ω20)
```

Each layer completes before the next begins, ensuring deterministic ordering.

---

## Code References

| Component | Path |
|-----------|------|
| Engine core | `src/quantracore_apex/engine/` |
| Tier protocols | `src/quantracore_apex/protocols/tier/` |
| Learning protocols | `src/quantracore_apex/protocols/learning/` |
| MonsterRunner | `src/quantracore_apex/protocols/monster_runner/` |
| Omega directives | `src/quantracore_apex/protocols/omega/` |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
