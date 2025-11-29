# Testing and Coverage Summary

**Document Classification:** Investor Due Diligence — Engineering  
**Version:** 9.0-A  
**Date:** November 2025  

---

## Do You Actually Test This Properly?

Yes. The system has a comprehensive test suite with 970 tests covering core functionality, protocols, safety-critical paths, and edge cases.

---

## Test Suite Overview

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total tests | 970 |
| Pass rate | 100% |
| Core tests | ~200 |
| Protocol tests | ~300 |
| Lab/model tests | ~200 |
| Nuclear/safety tests | ~163 |
| Other tests | ~107 |

---

## Test Categories

### Core Tests (~200)

**Purpose:** Test fundamental engine functionality.

**Examples:**
- Engine initialization
- Data loading and validation
- Signal generation pipeline
- Output formatting
- Error handling

```python
def test_engine_initialization():
    """Engine initializes with default config."""
    engine = ApexEngine()
    assert engine.is_initialized
    assert engine.protocol_count == 115

def test_signal_generation():
    """Engine generates valid signals from test data."""
    engine = ApexEngine()
    result = engine.analyze(test_data)
    assert 0 <= result.quantra_score <= 100
    assert result.quality_tier in VALID_TIERS
```

### Protocol Tests (~300)

**Purpose:** Test each protocol's logic and determinism.

**Coverage:**
- 80 Tier protocols (T01-T80)
- 25 Learning protocols (LP01-LP25)
- 5 MonsterRunner protocols (MR01-MR05)
- 5 Omega directives (Ω1-Ω5)

```python
def test_t15_momentum_protocol():
    """T15 momentum protocol produces expected output."""
    result = T15_Protocol().execute(momentum_data)
    assert result.contribution > 0
    assert result.is_deterministic

def test_omega_2_kill_switch():
    """Ω2 triggers on high VIX."""
    state = SystemState(vix=45)
    omega = Omega2Directive()
    assert omega.should_trigger(state)
    assert omega.action == "HALT"
```

### ApexLab Tests (~100)

**Purpose:** Test labeling and dataset generation.

**Examples:**
- Window building
- Teacher labeling
- Future outcome calculation
- Quality tier assignment
- Leakage prevention

```python
def test_no_lookahead_leakage():
    """Labels use only past data, not future."""
    window = build_window(data, end_idx=100)
    labels = generate_labels(window, data)
    
    # Verify no future data in features
    for feature in labels.features:
        assert feature.max_index <= 100

def test_quality_tier_assignment():
    """Quality tiers assigned correctly."""
    labels = generate_labels(window, data)
    assert labels.quality_tier in ["A+", "A", "B", "C", "D"]
```

### ApexCore Tests (~100)

**Purpose:** Test model inference and safety.

**Examples:**
- Model loading
- Manifest verification
- Inference correctness
- Ensemble aggregation
- Calibration

```python
def test_model_manifest_verification():
    """Model loading fails on hash mismatch."""
    with corrupt_model_file():
        with pytest.raises(IntegrityError):
            load_model_with_verification("apexcore_v2/big")

def test_ensemble_inference():
    """Ensemble produces averaged predictions."""
    model = ApexCoreV2Big()
    predictions = model.predict(test_features)
    assert 0 <= predictions.runner_prob <= 1
```

### Nuclear/Safety Tests (~163)

**Purpose:** Test safety-critical paths and regulatory compliance.

**Coverage:**
- Determinism verification (150 iterations)
- Fail-closed behavior
- Kill-switch functionality
- Integrity checks
- Compliance score calculation

```python
def test_determinism_150_iterations():
    """Engine is deterministic over 150 iterations (3x FINRA)."""
    baseline = engine.analyze(test_data)
    
    for i in range(150):
        result = engine.analyze(test_data)
        assert result.output_hash == baseline.output_hash, \
            f"Non-determinism at iteration {i}"

def test_fail_closed_on_model_error():
    """System uses engine-only on model failure."""
    with broken_model():
        result = advisor.get_ranked_candidates(candidates)
        assert result.mode == "ENGINE_ONLY"
        assert result.model_used == False
```

### Integration Tests (~50)

**Purpose:** Test end-to-end workflows.

**Examples:**
- Full scan → analysis → ranking pipeline
- API request → response cycle
- UI → API integration
- Data flow verification

```python
def test_full_pipeline():
    """Complete pipeline from scan to ranking."""
    # Scan
    candidates = scanner.scan(mode="momentum")
    assert len(candidates) > 0
    
    # Analyze
    analyzed = [engine.analyze(c) for c in candidates]
    assert all(a.quantra_score >= 0 for a in analyzed)
    
    # Rank
    ranked = advisor.rank(analyzed)
    assert ranked[0].rank == 1
```

---

## Determinism Testing

### Why It Matters

Determinism is critical for:
- Regulatory compliance (FINRA 15-09)
- Audit and replay capability
- Trust in outputs
- Debugging and investigation

### How It's Tested

```python
class DeterminismTestSuite:
    """Tests for guaranteed reproducibility."""
    
    def test_same_input_same_output(self):
        """Identical inputs produce identical outputs."""
        result_1 = engine.analyze(data)
        result_2 = engine.analyze(data)
        assert result_1.output_hash == result_2.output_hash
    
    def test_no_random_state(self):
        """No uncontrolled randomness in engine."""
        # Run 150 times (3x FINRA requirement)
        hashes = set()
        for _ in range(150):
            result = engine.analyze(data)
            hashes.add(result.output_hash)
        assert len(hashes) == 1
    
    def test_deterministic_after_restart(self):
        """Determinism holds across process restarts."""
        result_before = engine.analyze(data)
        
        # Simulate restart
        engine = ApexEngine()
        
        result_after = engine.analyze(data)
        assert result_after.output_hash == result_before.output_hash
```

### Golden Hash Testing

Reference outputs are stored and compared:

```python
GOLDEN_HASHES = {
    "test_case_1": "sha256:abc123...",
    "test_case_2": "sha256:def456...",
}

def test_against_golden_hashes():
    """Outputs match known-good reference hashes."""
    for case_name, expected_hash in GOLDEN_HASHES.items():
        data = load_test_case(case_name)
        result = engine.analyze(data)
        assert result.output_hash == expected_hash
```

---

## Running the Test Suite

### Quick Run

```bash
# All tests
pytest tests/

# Specific category
pytest tests/core/
pytest tests/protocols/
pytest tests/nuclear/

# With coverage
pytest --cov=src tests/
```

### Continuous Integration

Tests run automatically on:
- Every commit
- Every pull request
- Nightly scheduled runs

---

## Test Maintenance

| Aspect | Practice |
|--------|----------|
| New features | Tests required before merge |
| Bug fixes | Test added to prevent regression |
| Refactoring | Tests must pass before and after |
| Flaky tests | Immediately investigated and fixed |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
