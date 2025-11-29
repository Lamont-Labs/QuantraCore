# ApexLab V2 — Enhanced Labeling System

**Version:** 9.0-A  
**Component:** ApexLab V2  
**Role:** Advanced labeling with 40+ fields for ApexCore V2 training

---

## 1. Overview

ApexLab V2 is the enhanced labeling system that generates comprehensive training data for ApexCore V2 models. It extends the original ApexLab with 40+ fields capturing structural inputs, future outcomes, quality labels, runner flags, and safety indicators.

---

## 2. ApexLabV2Row Schema

The core schema contains these field categories:

### 2.1 Identification Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Ticker symbol |
| `timestamp` | datetime | Window timestamp |
| `sector` | string | Market sector |
| `market_cap_band` | string | Market cap classification |
| `timeframe` | string | Data timeframe (1d, 1h, etc.) |

### 2.2 Structural Inputs

| Field | Type | Description |
|-------|------|-------------|
| `quantra_score` | float | QuantraScore (0-100) |
| `regime` | string | Market regime classification |
| `risk_tier` | string | Risk level (low, medium, high, extreme) |
| `entropy_state` | string | Entropy classification |
| `suppression_state` | string | Suppression detection |
| `drift_state` | string | Drift classification |
| `protocol_vector` | list[float] | 115-dimensional protocol encoding |

### 2.3 Future Outcomes

| Field | Type | Description |
|-------|------|-------------|
| `ret_5d` | float | 5-day forward return |
| `ret_10d` | float | 10-day forward return |
| `max_drawdown_10d` | float | Maximum drawdown over 10 days |
| `max_gain_10d` | float | Maximum gain over 10 days |

### 2.4 Quality Labels

| Field | Type | Description |
|-------|------|-------------|
| `future_quality_tier` | string | A, B, C, D, A_PLUS based on returns |
| `regime_label` | string | chop, trend_up, trend_down, squeeze, crash |

### 2.5 Runner Flags

| Field | Type | Description |
|-------|------|-------------|
| `hit_runner_threshold` | bool | True if 15%+ gain achieved |
| `hit_monster_threshold` | bool | True if 25%+ gain achieved |

### 2.6 Safety Labels

| Field | Type | Description |
|-------|------|-------------|
| `future_crash` | bool | True if crash detected |
| `high_chop` | bool | True if high chop environment |
| `avoid_trade` | bool | Composite safety flag |

---

## 3. ApexLabV2Builder

The builder creates labeled rows from OHLCV windows:

```python
from src.quantracore_apex.apexlab.apexlab_v2 import ApexLabV2Builder

builder = ApexLabV2Builder(
    runner_threshold=0.15,    # 15% for runner
    monster_threshold=0.25,   # 25% for monster
    enable_logging=True
)

row = builder.build_row(
    window=ohlcv_window,
    future_prices=future_prices,
    timeframe="1d",
    sector="Technology",
    market_cap_band="mega"
)
```

---

## 4. ApexLabV2DatasetBuilder

For batch dataset construction:

```python
from src.quantracore_apex.apexlab.apexlab_v2 import ApexLabV2DatasetBuilder

dataset_builder = ApexLabV2DatasetBuilder(
    output_dir="./training_data",
    enable_logging=True
)

# Build and save dataset
dataset_builder.build_and_save(
    windows=windows,
    future_prices_list=future_prices_list,
    output_name="training_v2"
)
```

---

## 5. Protocol Vector Encoding

The 115-dimensional protocol vector encodes:

| Range | Protocols | Description |
|-------|-----------|-------------|
| 0-79 | T01-T80 | Tier protocol fire states |
| 80-104 | LP01-LP25 | Learning protocol states |
| 105-109 | MR01-MR05 | MonsterRunner states |
| 110-114 | Ω1-Ω5 | Omega directive states |

---

## 6. Quality Tier Buckets

| Tier | Return Range | Description |
|------|--------------|-------------|
| A_PLUS | > 15% | Exceptional performance |
| A | 8-15% | Strong performance |
| B | 3-8% | Good performance |
| C | 0-3% | Neutral performance |
| D | < 0% | Negative performance |

---

## 7. Integration with ApexCore V2

ApexLab V2 rows are used to train ApexCore V2 models:

```
ApexLabV2Row → Training Features → ApexCoreV2Model
                                 ├─ quantra_score head
                                 ├─ runner_prob head
                                 ├─ quality_tier head
                                 ├─ avoid_trade head
                                 └─ regime head
```

---

## 8. Test Coverage

ApexLab V2 includes comprehensive tests:

| Test File | Tests | Description |
|-----------|-------|-------------|
| test_apexlab_v2_schema.py | 38 | Schema validation |
| test_apexlab_v2_dataset_shapes.py | 17 | Output shapes |
| test_apexlab_v2_eval_metrics.py | 17 | Evaluation metrics |

---

## 9. Related Documentation

- [ApexLab Overview](APEXLAB_OVERVIEW.md)
- [ApexCore V2](APEXCORE_V2.md)
- [ApexCore Models](APEXCORE_MODELS.md)
- [PredictiveAdvisor](PREDICTIVE_ADVISOR.md)

---

**QuantraCore Apex v9.0-A** — Lamont Labs | November 2025
