# Estimated Move Module Specification

**QuantraCore Apex v9.0-A | Lamont Labs | November 2025**

---

## Overview

The Estimated Move Module provides deterministic + model-assisted statistical move range analysis for research purposes. This is **NOT** a prediction engine, price target generator, or trading signal provider.

**Core Principle:** Statistical distributions derived from historical patterns, regime analysis, and model outputs — never guarantees or forecasts.

---

## Module Classification

| Attribute | Value |
|-----------|-------|
| Module Type | Research-Only Statistical Engine |
| Output Type | Move Range Distributions (Percentiles) |
| Compliance Mode | Permanent Research-Only |
| Safety Gates | 4 Mandatory Fail-Closed Gates |
| Integration | ApexEngine, ApexCore, MonsterRunner |

---

## Input Features

### Deterministic Inputs (from ApexEngine)

| Field | Type | Description |
|-------|------|-------------|
| `quantra_score` | float | QuantraScore 0-100 from engine |
| `risk_tier` | string | Risk tier classification |
| `volatility_band` | string | Current volatility band |
| `entropy_band` | string | Entropy state band |
| `regime_type` | string | Market regime classification |
| `suppression_state` | string | Suppression gate state |
| `protocol_vector` | List[float] | 105-element protocol firing vector |
| `float_pressure` | float | Float pressure metric |
| `liquidity_score` | float | Liquidity assessment |
| `market_cap_band` | string | Market capitalization band |

### Model Inputs (from ApexCore)

| Field | Type | Description |
|-------|------|-------------|
| `runner_prob` | float | MonsterRunner probability 0-1 |
| `quality_tier_logits` | List[float] | Quality tier predictions |
| `avoid_trade_prob` | float | Avoid trade probability 0-1 |
| `model_quantra_score` | float | Model-predicted QuantraScore |
| `ensemble_disagreement` | float | Model ensemble disagreement 0-1 |

### Vision Inputs (Optional, from ApexVision)

| Field | Type | Description |
|-------|------|-------------|
| `visual_runner_score` | float | Visual pattern runner score |
| `visual_pattern_logits` | List[float] | Visual pattern classifications |
| `visual_uncertainty` | float | Visual analysis uncertainty |

---

## Output Structure

### EstimatedMoveOutput

```python
@dataclass
class EstimatedMoveOutput:
    symbol: str
    timestamp: datetime
    ranges: Dict[str, MoveRange]  # Per-horizon ranges
    overall_uncertainty: float    # 0-1
    runner_boost_applied: bool
    quality_modifier: float
    safety_clamped: bool
    confidence: MoveConfidence
    computation_mode: str         # "deterministic", "model", "hybrid"
    compliance_note: str
```

### MoveRange (Per Horizon)

```python
@dataclass
class MoveRange:
    horizon: HorizonWindow
    min_move_pct: float      # 5th percentile
    low_move_pct: float      # 20th percentile
    median_move_pct: float   # 50th percentile
    high_move_pct: float     # 80th percentile
    max_move_pct: float      # 95th percentile
    uncertainty_score: float
    sample_count: int
```

---

## Horizon Windows

| Horizon ID | Name | Days | Use Case |
|------------|------|------|----------|
| `1d` | Short Term | 1 | Intraday research |
| `3d` | Medium Term | 3 | Near-term analysis |
| `5d` | Extended Term | 5 | Weekly research |
| `10d` | Research Term | 10 | Extended analysis |

---

## Safety Gates

The module implements four mandatory fail-closed safety gates:

### Gate 1: Avoid Trade Threshold

```python
if input.avoid_trade_prob > 0.35:
    safety_clamped = True
    # Reduces expected move and upside bias
```

**Trigger:** ApexCore model signals high avoid-trade probability.

### Gate 2: Suppression Block

```python
if input.suppression_state == "blocked":
    safety_clamped = True
    # Blocks expansion estimates
```

**Trigger:** Engine suppression gate is in blocked state.

### Gate 3: Uncertainty Threshold

```python
if uncertainty > 0.7:
    return neutral_output()
    # Returns zero ranges with high uncertainty flag
```

**Trigger:** Overall uncertainty exceeds acceptable threshold.

### Gate 4: Ensemble Disagreement

```python
if input.ensemble_disagreement > 0.3:
    computation_mode = "deterministic"
    # Ignores model outputs, uses only deterministic analysis
```

**Trigger:** Model ensemble has high disagreement.

---

## Computation Modes

### Hybrid Mode (Default)

Combines deterministic engine analysis with model predictions:

1. Base volatility from market cap band
2. Regime and volatility band modifiers
3. Runner boost from MonsterRunner probability
4. Quality modifier from QuantraScore and model outputs

### Deterministic Mode

Falls back when model disagreement is high:

1. Uses only engine-derived metrics
2. Ignores ApexCore model outputs
3. Conservative volatility estimates

### Neutral Mode

Activated when uncertainty is too high:

1. Returns zero move ranges
2. Sets all uncertainty scores to maximum
3. Compliance note indicates suppression

---

## Base Volatility by Market Cap

Annualized volatility assumptions by market capitalization:

| Band | Base Volatility (Annual %) |
|------|---------------------------|
| Mega | 15% |
| Large | 20% |
| Mid | 28% |
| Small | 38% |
| Micro | 50% |
| Nano | 65% |
| Penny | 80% |

---

## Regime Multipliers

Volatility scaling by detected regime:

| Regime | Multiplier |
|--------|-----------|
| Trending | 1.2x |
| Ranging | 0.8x |
| Volatile | 1.5x |
| Quiet | 0.6x |
| Breakout | 1.4x |
| Breakdown | 1.3x |
| Consolidation | 0.7x |

---

## API Endpoints

### GET `/estimated_move/{symbol}`

Single symbol estimated move analysis.

**Parameters:**
- `symbol` (path): Stock symbol
- `timeframe` (query): Analysis timeframe (default: "1d")
- `lookback_days` (query): Historical lookback (default: 150)

**Response:**
```json
{
  "symbol": "TEST",
  "estimated_move": {
    "ranges": {
      "1d": {"median_move_pct": 0.45, "max_move_pct": 1.82},
      "3d": {"median_move_pct": 0.78, "max_move_pct": 3.15},
      "5d": {"median_move_pct": 1.01, "max_move_pct": 4.07},
      "10d": {"median_move_pct": 1.43, "max_move_pct": 5.75}
    },
    "confidence": "moderate",
    "safety_clamped": false
  },
  "compliance_note": "Structural research output only..."
}
```

### POST `/estimated_move/batch`

Batch estimated move analysis for multiple symbols.

**Request Body:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "timeframe": "1d",
  "lookback_days": 150,
  "max_results": 50
}
```

### GET `/estimated_move/horizons`

Returns available horizon windows.

---

## Integration Points

### ApexDesk Integration

- Displays move ranges in symbol detail panels
- Color-coded confidence indicators
- Safety clamp warnings when applicable

### QuantraVision Integration

- Real-time move range overlays
- Visual confidence bands
- Android copilot estimated move widget

### PredictiveAdvisor Integration

- Estimated move used in candidate ranking
- Runner boost correlation with uprank signals
- Safety clamps reduce advisory confidence

---

## Compliance Statements

Every output includes mandatory compliance language:

> "Structural research output only - NOT a price target or trading signal"

This is **NOT**:
- A prediction of future price movement
- A guarantee of any expected return
- A trading signal or recommendation
- Financial advice of any kind

This **IS**:
- Statistical distribution based on historical patterns
- Research tool for structural analysis
- Risk-aware range estimation with safety gates

---

## Testing Coverage

| Test Category | Count | Coverage |
|--------------|-------|----------|
| Core Computation | 5 | Engine initialization, output structure |
| Safety Gates | 4 | All four mandatory gates |
| Runner Boost | 2 | Boost application logic |
| Market Cap Bands | 2 | Volatility scaling |
| Regime Effects | 1 | Regime multiplier validation |
| Batch Processing | 1 | Multi-symbol processing |
| Serialization | 2 | Output dict conversion |
| **Total** | **19** | **Full module coverage** |

---

## File Structure

```
src/quantracore_apex/estimated_move/
├── __init__.py          # Module exports
├── schemas.py           # Input/output dataclasses
└── engine.py            # Core computation engine

tests/estimated_move/
├── __init__.py
└── test_estimated_move.py  # 19 comprehensive tests
```

---

## Future Enhancements

### ApexVision Integration (v10.x)

- Visual pattern overlay on move ranges
- Chart-based confidence adjustment
- Pattern-specific volatility modifiers

### Historical Calibration

- Backtested accuracy tracking
- Calibration curve generation
- Confidence interval refinement

---

*QuantraCore Apex v9.0-A | Estimated Move Module | Research-Only*
