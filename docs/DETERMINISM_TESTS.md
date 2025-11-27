# QuantraCore Apex™ — Determinism Test Suite

**Version:** 8.0  
**Component:** Reproducibility Verification  
**Status:** Active

---

## Overview

The Determinism Suite guarantees that Apex outputs are identical across runs. This is fundamental to the system's reproducibility, compliance, and auditability requirements.

---

## Core Principle

```
Same Input → Same Output (Always)
```

No randomness, no non-deterministic operations, no environmental dependencies.

---

## Test Categories

### 1. Golden Set Tests

**Description:** Fixed set of OHLCV windows for deterministic testing.

**Checked Outputs:**

| Output | Requirement |
|--------|-------------|
| Trend | Exact match |
| Regime | Exact match |
| Risk | Exact match |
| Volatility | Exact match |
| Continuation | Exact match |
| Entropy | Exact match |
| Drift | Exact match |
| Suppression | Exact match |
| QuantraScore | Exact match (byte-for-byte) |

**Rule:** Outputs must match exactly byte-for-byte.

### Golden Set Structure

```json
{
  "golden_set_version": "8.0",
  "created": "2025-01-01T00:00:00Z",
  "items": [
    {
      "id": "GS-001",
      "ohlcv_hash": "sha256:...",
      "expected_outputs": {
        "trend": "upward",
        "regime": "trend_up",
        "risk": "low",
        "volatility_band": "medium",
        "continuation_state": "continue",
        "entropy_state": "normal",
        "drift_state": "stable",
        "suppression_state": "none",
        "quantrascore": 78
      }
    }
  ]
}
```

---

### 2. Protocol Signature Tests

**Description:** Validate each protocol's output against known signatures.

**Rule:** No protocol drift allowed.

**Test Structure:**

```python
def test_protocol_T01():
    input_data = load_test_fixture("T01_input")
    expected = load_expected("T01_expected")
    actual = T01.execute(input_data)
    assert actual == expected, "Protocol T01 drift detected"
```

**Coverage:**

- All 80 Tier Protocols (T01–T80)
- All 25 Learning Protocols (LP01–LP25)
- All 4 Omega Directives (Ω1–Ω4)

---

### 3. Hash Lock Tests

**Description:** Dataset hashes must match MODEL_MANIFEST.

**Rule:** Mismatch → full abort.

**Verification Flow:**

```
Load Dataset
    ↓
Compute SHA-256 Hash
    ↓
Compare to MODEL_MANIFEST
    ↓
[Match] → Continue
[Mismatch] → ABORT
```

---

### 4. ApexCore Alignment Tests

**Description:** Ensure ApexCore model aligns with Apex teacher outputs.

**Rule:** Drift > 1.5% → model rejection.

**Measured Metrics:**

| Metric | Threshold |
|--------|-----------|
| Regime accuracy | >98% |
| Risk accuracy | >98% |
| Entropy accuracy | >98% |
| Score MAE | <3 points |
| Overall alignment | >98.5% |

---

### 5. Cross-Version Tests

**Description:** Ensure new Apex versions maintain consistent behavior.

**Rule:** Cross-version drift must be documented.

**Test Process:**

1. Run golden set on version N
2. Run golden set on version N+1
3. Compare outputs
4. Document any differences
5. Approve or reject release

---

## Test Execution

### Continuous Integration

Tests run on every commit:

```yaml
determinism_tests:
  - golden_set_test
  - protocol_signature_test
  - hash_lock_test
  - apexcore_alignment_test
```

### Release Gate

Before any release:

```yaml
release_gate:
  - all_determinism_tests: pass
  - cross_version_test: documented
  - golden_set_verification: byte_match
```

---

## Failure Handling

### On Golden Set Failure

```
1. Halt release
2. Identify changed outputs
3. Trace to responsible protocol
4. Fix or document intentional change
5. Update golden set if intentional
6. Re-run all tests
```

### On Protocol Drift

```
1. Identify drifting protocol
2. Compare against version-locked reference
3. Fix implementation
4. Verify determinism restored
```

### On Hash Mismatch

```
1. Abort immediately
2. Investigate data source
3. Regenerate dataset if corrupted
4. Verify hash integrity
5. Resume only on match
```

---

## Test Reports

```json
{
  "test_run_id": "DET-2025-001234",
  "timestamp": "2025-11-27T12:00:00Z",
  "apex_version": "8.0.0",
  "results": {
    "golden_set": {
      "status": "passed",
      "items_tested": 1000,
      "items_passed": 1000
    },
    "protocol_signatures": {
      "status": "passed",
      "protocols_tested": 109,
      "protocols_passed": 109
    },
    "hash_locks": {
      "status": "passed",
      "datasets_verified": 50
    },
    "apexcore_alignment": {
      "status": "passed",
      "alignment_score": 0.991
    }
  },
  "overall_status": "PASSED"
}
```

---

## Related Documentation

- [Core Engine](CORE_ENGINE.md)
- [Protocols: Tier](PROTOCOLS_TIER.md)
- [Security & Compliance](SECURITY_COMPLIANCE.md)
- [SBOM & Provenance](SBOM_PROVENANCE.md)
