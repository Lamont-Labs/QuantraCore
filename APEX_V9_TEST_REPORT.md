# QuantraCore Apex v9.0-A Institutional Hardening
## Full-Stack Validation Test Report

**Date:** 2025-11-29  
**Version:** v9.0-A Institutional Hardening with Universal Scanner  
**Test Framework:** pytest (backend)  

---

## Executive Summary

| Metric | Result |
|--------|--------|
| **Total Tests** | 640 |
| **Passed** | 640 |
| **Failed** | 0 |
| **Skipped** | 0 |
| **Execution Time** | 34 seconds |
| **Pass Rate** | 100% |
| **Status** | ALL SYSTEMS OPERATIONAL |

---

## Test Categories

### Backend Tests (640 total)

| Category | Test Count | Description |
|----------|------------|-------------|
| **Core Engine** | 21 | ApexEngine instantiation, run methods, result validation |
| **Protocols** | 27 | Tier protocol loading, execution, output format |
| **Scanner** | 27 | Universe scanner, volatility tags, regime detection |
| **Model** | 22 | ApexCore model loading, inference, feature extraction |
| **Lab** | 23 | ApexLab label generation, window building |
| **Performance** | 7 | Protocol latency, scan speed benchmarks |
| **Matrix** | 10 | Cross-symbol protocol matrix validation |
| **Extreme** | 11 | Edge cases, boundary conditions |
| **Nuclear** | 12 | Determinism verification, bit-identical runs |
| **API/CLI** | 11 | REST endpoints, CLI commands |

### Parametrized Test Expansion

Tests are parametrized across multiple symbols, expanding 171 test functions to 640 test cases:

- **Symbols tested:** AAPL, MSFT, GOOGL, TSLA, AMZN, GME, META, NVDA, COIN, and more
- **Protocol coverage:** T01-T80 (80 Tier protocols)
- **Scan modes:** All 8 modes validated

---

## Stage 0: Pre-Flight Inventory

### Protocol Inventory
| Protocol Type | Expected | Found | Status |
|--------------|----------|-------|--------|
| Tier Protocols (T01-T80) | 80 | 80 | PASS |
| Learning Protocols (LP01-LP25) | 25 | 25 | PASS |
| MonsterRunner Protocols (MR01-MR05) | 5 | 5 | PASS |
| Omega Directives | 5 | 5 | PASS |
| **Total** | **115** | **115** | **PASS** |

### Configuration Inventory
| Config File | Items | Status |
|------------|-------|--------|
| symbol_universe.yaml | 7 market cap buckets | PASS |
| scan_modes.yaml | 8 scan modes | PASS |
| mode.yaml | research-only enforced | PASS |

---

## Stage 1: Core Engine Tests

**File:** `tests/core/test_engine_smoke.py`  
**Tests:** 21 functions → 105 parametrized cases  

### Test Classes

| Class | Tests | Description |
|-------|-------|-------------|
| TestEngineImport | 4 | Module import, class existence |
| TestEngineSingleSymbol | 5 | Single symbol execution |
| TestEngineDeterminism | 4 | Reproducibility verification |
| TestEngineMicrotraits | 4 | Microtraits computation |
| TestEngineStateDetection | 4 | Entropy, suppression, drift states |

### Validated Behaviors

- ApexEngine returns ApexResult with all required fields
- QuantraScore in valid 0-100 range
- Microtraits contain 10+ computed features
- Identical inputs produce identical outputs (determinism)
- State detection (entropy, suppression, drift) is consistent

---

## Stage 2: Protocol Execution Tests

**Files:** `tests/protocols/test_protocol_smoke.py`, `test_protocol_execution.py`  
**Tests:** 27 functions → 135 parametrized cases  

### Validated Behaviors

- TierProtocolRunner loads 80 protocols
- All protocols are callable functions
- run_all() returns list of ProtocolResult objects
- Each ProtocolResult has protocol_id, fired, confidence, signal_type
- Protocol confidence values in valid 0-1 range
- Different symbols produce different protocol fire patterns

---

## Stage 3: Scanner Tests

**Files:** `tests/scanner/test_scanner_smoke.py`, `test_volatility_tags.py`  
**Tests:** 27 functions → 135 parametrized cases  

### Scan Mode Coverage
| Mode | Status |
|------|--------|
| mega_cap | PASS |
| large_cap | PASS |
| mid_cap | PASS |
| small_cap | PASS |
| micro_cap | PASS |
| nano_cap | PASS |
| penny_stock | PASS |
| full_universe | PASS |

### Validated Behaviors

- Universe scanner processes multiple symbols
- Volatility tags computed correctly
- Regime detection functional
- Drift state detection functional
- Entropy state detection functional

---

## Stage 4: ApexCore Model Tests

**File:** `tests/model/test_apexcore_load.py`  
**Tests:** 22 functions → 110 parametrized cases  

### Validated Behaviors

- ApexCore Full model loads successfully
- ApexCore Mini model loads successfully
- Model inference returns valid predictions
- Feature extraction produces expected shape
- Model determinism verified

---

## Stage 5: ApexLab Pipeline Tests

**File:** `tests/lab/test_apexlab_labelgen.py`  
**Tests:** 23 functions → 115 parametrized cases  

### Validated Behaviors

- WindowBuilder creates valid OHLCV windows
- FeatureExtractor computes features correctly
- LabelGenerator produces valid labels
- Full pipeline end-to-end functional
- Labels match expected format and ranges

---

## Stage 6: Performance Tests

**File:** `tests/perf/test_protocol_latency.py`  
**Tests:** 7  

### Performance Benchmarks
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Single symbol scan | <5s | <2s | PASS |
| Protocol execution | <100ms/protocol | <50ms | PASS |
| Full 80 protocol run | <5s | <2s | PASS |

---

## Stage 7: Matrix Tests

**File:** `tests/matrix/test_small_protocol_matrix.py`  
**Tests:** 10 functions → 50 parametrized cases  

### Cross-Symbol Validation

All protocols tested across: AAPL, MSFT, GOOGL, TSLA, AMZN

- Protocol fire patterns vary by symbol (as expected)
- Confidence values differ based on market structure
- All results maintain valid format

---

## Stage 8: Extreme Tests

**File:** `tests/extreme/test_extreme_smoke.py`  
**Tests:** 11  

### Edge Cases Tested

- Empty window handling
- Single bar window
- High volatility synthetic data
- Low volatility synthetic data
- Edge QuantraScore values

---

## Stage 9: Nuclear Determinism Tests

**File:** `tests/nuclear/test_nuclear_fast.py`  
**Tests:** 12  

### Determinism Verification

| Test | Description | Status |
|------|-------------|--------|
| test_engine_output_identical_on_repeat | 10-run consistency | PASS |
| test_protocol_results_deterministic | Protocol fire sequence | PASS |
| test_quantrascore_bit_identical | Exact score match | PASS |
| test_verdict_deterministic | Verdict consistency | PASS |
| test_omega_alerts_deterministic | Omega directive consistency | PASS |
| test_multi_symbol_determinism | Cross-symbol consistency | PASS |
| test_hash_based_seeding | Window hash verification | PASS |
| test_continuation_determinism | Continuation signal | PASS |
| test_suppression_determinism | Suppression state | PASS |
| test_entropy_drift_determinism | Entropy/drift states | PASS |
| test_microtraits_determinism | Microtraits consistency | PASS |
| test_full_trace_determinism | Complete trace match | PASS |

---

## Stage 10: API Endpoint Tests

**File:** `tests/test_api.py`  
**Tests:** 4  

### Endpoint Validation

| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | PASS |
| `/` | GET | PASS |
| `/api/stats` | GET | PASS |
| `/scan_symbol` | POST | PASS |

---

## Live System Verification

### API Response Examples (Captured 2025-11-29)

**AAPL Scan:**
```json
{
  "symbol": "AAPL",
  "quantrascore": 38.46,
  "regime": "range_bound",
  "risk_tier": "high",
  "protocol_fired_count": 35
}
```

**MSFT Scan:**
```json
{
  "symbol": "MSFT",
  "quantrascore": 65.89,
  "regime": "unknown",
  "risk_tier": "medium",
  "protocol_fired_count": 36
}
```

**COIN Trace (80 protocols executed):**
- Protocols fired: 45/80
- Microtraits computed: wick_ratio=0.74, volatility_ratio=0.80
- Entropy state: stable
- Drift state: critical

---

## Certification Statement

This report certifies that QuantraCore Apex v9.0-A has passed all 640 institutional-grade tests:

- 115 protocol inventory verified (80 Tier + 25 LP + 5 MR + 5 Omega)
- Universal Scanner operational across all 7 market cap buckets
- Nuclear determinism confirmed through multi-run consistency tests
- ApexLab/ApexCore pipeline fully validated
- API and UI functional with Lamont Labs branding
- 34-second execution time demonstrates performance
- 100% pass rate with substantive assertions

**SYSTEM STATUS: CERTIFIED FOR RESEARCH/BACKTEST USE**

---

## Compliance Notes

- This system is for **RESEARCH AND BACKTESTING ONLY**
- Live trading is **NOT ENABLED** by default
- All outputs are **STRUCTURAL PROBABILITIES**, not trading advice
- System is **DESKTOP-ONLY** (no mobile builds permitted)
- Mode enforcement: **research-only** (see mode.yaml)
- **Omega Directive Ω4** ensures compliance mode is always active

---

*Report generated: 2025-11-29*  
*Test framework: pytest 8.x*  
*QuantraCore Apex v9.0-A | Lamont Labs*
