# Entry/Exit Optimization Engine (EEO Engine)

**QuantraCore Apex v9.0-A | November 2025**

## Overview

The Entry/Exit Optimization Engine (EEO Engine) sits between the Apex signal engine and the Execution Engine. For every structural signal, it calculates:

- **Best entry price zone** - Optimal entry range based on volatility, spreads, and market conditions
- **Best entry style** - MARKET, LIMIT, STOP, or STOP_LIMIT orders
- **Best exit plan** - Targets, stops, trailing stops, and time-based exits
- **Position sizing** - Risk-based sizing using fixed-fraction methodology

All within deterministic, risk-aware, and compliance-friendly constraints.

## Integration Points

### Upstream

- Apex deterministic engine (signals, QuantraScore, ZDE, protocols, regimes)
- ApexCore v2 predictive heads (runner_prob, quality_tier, avoid_trade, etc.)
- Estimated Move module (if enabled)

### Downstream

- ExecutionEngine (converts plans to OrderTickets)
- BrokerRouter (Alpaca PAPER now, LIVE later)
- ApexLab v2 (uses realized PnL to refine future policies)
- ApexDesk (displays entry/exit plans to user)

## Core Components

### 1. Data Models

#### EntryExitPlan

The primary output structure containing:

```python
@dataclass
class EntryExitPlan:
    plan_id: str
    symbol: str
    direction: SignalDirection  # LONG or SHORT
    entry_mode: EntryMode      # SINGLE or SCALED_IN
    exit_mode: ExitMode        # SINGLE or SCALED_OUT
    size_notional: float       # USD position size
    
    base_entry: BaseEntry              # Primary entry configuration
    scaled_entries: List[ScaledEntry]  # Optional scaled entries
    
    protective_stop: ProtectiveStop    # Stop-loss configuration
    profit_targets: List[ProfitTarget] # T1, T2, etc.
    trailing_stop: TrailingStop        # Trailing stop rules
    time_based_exit: TimeBasedExit     # Time exit rules
    
    abort_conditions: List[AbortCondition]  # Pre-fill abort rules
    metadata: PlanMetadata                   # Quality, confidence, etc.
```

### 2. Context Inputs

#### SignalContext

From the Apex signal engine:

- symbol, timeframe, signal_time
- direction (LONG/SHORT)
- quantra_score (0-100)
- protocol_trace
- regime_type, volatility_band, liquidity_band
- suppression_state, zde_label

#### PredictiveContext

From ApexCore v2:

- runner_prob
- future_quality_tier (A_PLUS, A, B, C, D)
- avoid_trade_prob
- ensemble_disagreement
- estimated_move_range (min, median, max)

#### MarketMicrostructure

Real-time market data:

- current_bid, current_ask, mid_price
- spread, spread_pct
- atr_14, recent_bar_ranges
- depth/liquidity approximations

#### RiskContext

Account information:

- account_equity
- per_trade_risk_fraction
- max_notional_per_symbol
- current_exposure, open_position_count

### 3. Policy Profiles

Three configurable profiles control optimization behavior:

| Profile | Risk Fraction | Aggressiveness | Trailing Stop | Time Exit |
|---------|---------------|----------------|---------------|-----------|
| Conservative | 0.5% | LOW | OFF | 30 bars |
| Balanced | 1.0% | MEDIUM | ON | 50 bars |
| Aggressive Research | 2.0% | HIGH | ON | 80 bars |

### 4. Entry Optimizer

Strategies based on market conditions:

| Strategy | Trigger | Behavior |
|----------|---------|----------|
| Baseline Long/Short | Default | ATR-based entry zone, prefer LIMIT |
| High Volatility | volatility_band=HIGH or regime=crash/squeeze | Conservative LIMITs, scaled entries |
| Low Liquidity | liquidity_band=LOW or wide spread | Deeper inside spread, reduced size |
| Runner Anticipation | runner_prob high + quality A/A+ | Slightly more aggressive entry |
| ZDE Aware | zde_label=TRUE | Tightened entry zone |

### 5. Exit Optimizer

Components:

#### Protective Stop
- Based on ATR multiple (configurable per profile)
- Adjusted for volatility band
- Tightened if ZDE research applies

#### Profit Targets
- T1: Median estimated move (default 50% exit)
- T2: Upper estimated move band (remaining 50%)
- Extended targets for runner candidates

#### Trailing Stop
- ATR mode: Trail by K * ATR
- Percent mode: Trail by X% from MFE
- Structural mode: Trail under swing structures

#### Time-Based Exit
- Max bars in trade (profile-dependent)
- End-of-day exit for intraday strategies

## API Endpoints

### Build Plan

```http
POST /eeo/plan
Content-Type: application/json

{
  "symbol": "AAPL",
  "direction": "LONG",
  "quantra_score": 75.0,
  "current_price": 150.0,
  "atr": 3.0,
  "account_equity": 100000.0,
  "runner_prob": 0.65,
  "quality_tier": "A",
  "estimated_move_median": 5.0,
  "estimated_move_max": 8.0,
  "profile": "balanced"
}
```

Response:

```json
{
  "success": true,
  "plan": {
    "plan_id": "abc123",
    "symbol": "AAPL",
    "direction": "LONG",
    "size_notional": 10000.0,
    "base_entry": {
      "order_type": "MARKET",
      "entry_price": 150.01,
      "entry_zone": {"lower": 148.8, "upper": 151.1}
    },
    "protective_stop": {
      "enabled": true,
      "stop_price": 144.0,
      "rationale": "Standard ATR-based stop"
    },
    "profit_targets": [
      {"target_price": 152.5, "fraction_to_exit": 0.5},
      {"target_price": 158.0, "fraction_to_exit": 0.5}
    ],
    "risk_reward_ratio": 0.42
  }
}
```

### Execute Plan

```http
POST /eeo/plan/execute
```

Builds a plan and immediately executes the entry using paper trading.

### List Profiles

```http
GET /eeo/profiles
```

### Get Profile Details

```http
GET /eeo/profiles/{profile_name}
```

## Integration with Execution Engine

The ExecutionEngine has an `execute_plan` method that:

1. Validates the plan
2. Converts base_entry to an OrderTicket
3. Runs risk checks via RiskEngine
4. Routes order via BrokerRouter
5. Returns execution results

```python
from src.quantracore_apex.broker.execution_engine import ExecutionEngine
from src.quantracore_apex.eeo_engine import EntryExitOptimizer

optimizer = EntryExitOptimizer(profile_type=ProfileType.BALANCED)
plan = optimizer.build_plan(context)

engine = ExecutionEngine(config=broker_config)
result = engine.execute_plan(plan)
```

## Safety and Compliance

### Guarantees

- Entry/exit calculations are policy-based, not hype or naked prediction
- All behavior is transparent and logged for auditors
- PAPER mode only for now; LIVE mode requires institutional enablement
- System never bypasses RiskEngine, even for "best entry" ideas
- Retail-facing views stay research-only; no guaranteed price promises

### Abort Conditions

Plans include pre-fill abort conditions:

- If avoid_trade_prob spikes beyond threshold before fill
- If volatility regime flips to crash before fill
- If spread explodes beyond tolerance

## Test Coverage

42 tests covering:

- Data models and structures
- Entry optimization strategies
- Exit optimization (stops, targets, trailing)
- Profile system
- Full plan building
- Integration with execution engine

Run tests:

```bash
python -m pytest tests/eeo_engine/ -v
```

## Usage Example

```python
from src.quantracore_apex.eeo_engine import (
    EntryExitOptimizer,
    SignalContext,
    PredictiveContext,
    MarketMicrostructure,
    RiskContext,
    EEOContext,
    SignalDirection,
    QualityTier,
    ProfileType,
)

# Create signal context
signal = SignalContext(
    symbol='AAPL',
    direction=SignalDirection.LONG,
    quantra_score=75.0,
)

# Create predictive context
predictive = PredictiveContext(
    runner_prob=0.65,
    future_quality_tier=QualityTier.A,
    estimated_move_median=5.0,
    estimated_move_max=8.0,
)

# Create market data
micro = MarketMicrostructure.from_price(150.0, atr=3.0)

# Create risk context
risk = RiskContext(
    account_equity=100000.0,
    per_trade_risk_fraction=0.01,
)

# Build complete context
context = EEOContext(
    signal=signal,
    predictive=predictive,
    microstructure=micro,
    risk=risk,
)

# Create optimizer and build plan
optimizer = EntryExitOptimizer(profile_type=ProfileType.BALANCED)
plan = optimizer.build_plan(context)

# Access plan details
print(f"Entry: ${plan.base_entry.entry_price:.2f}")
print(f"Stop: ${plan.protective_stop.stop_price:.2f}")
print(f"R:R Ratio: {plan.risk_reward_ratio():.2f}")
```

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
