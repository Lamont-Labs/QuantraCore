# QuantraCore Apex — Broker Layer v1 Specification

## Overview

The Broker Layer v1 transforms QuantraCore Apex from a pure signal/intelligence engine into a fully structured institutional execution engine with strict modes, pluggable broker adapters, and comprehensive risk controls.

**SAFETY:** Live trading is DISABLED by default. Paper trading only.

## Version

- Version: 1.0
- Date: November 2025
- Author: Lamont Labs

## Architecture

### Directory Structure

```
src/quantracore_apex/broker/
├── __init__.py           # Package exports
├── enums.py              # ExecutionMode, OrderSide, TimeInForce, etc.
├── models.py             # OrderTicket, ExecutionResult, BrokerPosition
├── config.py             # Configuration loading from YAML/env
├── router.py             # BrokerRouter (adapter selection)
├── risk_engine.py        # Deterministic risk checks
├── execution_engine.py   # Signal → Order mapping and execution
├── execution_logger.py   # Structured audit logging
├── oms.py                # Legacy OMS (backward compatibility)
└── adapters/
    ├── __init__.py
    ├── base_adapter.py       # Abstract BrokerAdapter base class
    ├── null_adapter.py       # No-op adapter for RESEARCH mode
    ├── paper_sim_adapter.py  # Offline simulation adapter
    └── alpaca_adapter.py     # Alpaca paper trading adapter

config/
└── broker.yaml           # Broker configuration

scripts/
└── apex_paper_trader.py  # CLI for paper trading

logs/
├── execution/            # Execution logs
├── execution_audit/      # Audit logs with risk decisions
└── execution_broker_raw/ # Raw broker API logs
```

## Execution Modes

| Mode | Description | Adapter Used |
|------|-------------|--------------|
| RESEARCH | No orders, signals only (default) | NullAdapter |
| PAPER | Orders to paper simulator/Alpaca paper | PaperSimAdapter or AlpacaPaperAdapter |
| LIVE | Real broker (DISABLED) | Raises error |

## Configuration

### broker.yaml

```yaml
execution:
  mode: "RESEARCH"  # RESEARCH | PAPER | LIVE
  default_account: "primary"

accounts:
  primary:
    broker: "ALPACA"
    mode: "PAPER"
    risk_profile: "standard"

brokers:
  alpaca:
    paper:
      base_url: "https://paper-api.alpaca.markets"
      api_key_env: "ALPACA_PAPER_API_KEY"
      api_secret_env: "ALPACA_PAPER_API_SECRET"
      enabled: true
    live:
      enabled: false  # MUST remain false

risk:
  max_notional_exposure_usd: 50000
  max_position_notional_per_symbol_usd: 5000
  max_positions: 30
  max_daily_turnover_usd: 100000
  max_order_notional_usd: 3000
  max_leverage: 2.0
  block_short_selling: true
  block_margin: true
  require_positive_equity: true
```

### Environment Variables

For Alpaca paper trading:
```bash
export ALPACA_PAPER_API_KEY="your-paper-key"
export ALPACA_PAPER_API_SECRET="your-paper-secret"
```

## Data Models

### OrderTicket

```python
@dataclass
class OrderTicket:
    symbol: str
    side: OrderSide           # BUY | SELL
    qty: float
    order_type: OrderType     # MARKET | LIMIT | STOP | STOP_LIMIT
    limit_price: Optional[float]
    stop_price: Optional[float]
    time_in_force: TimeInForce  # DAY | GTC | IOC | FOK
    intent: OrderIntent         # OPEN_LONG | CLOSE_LONG | etc.
    source_signal_id: str
    strategy_id: str
    metadata: OrderMetadata
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    order_id: str
    broker: str
    status: OrderStatus       # NEW | FILLED | REJECTED | etc.
    filled_qty: float
    avg_fill_price: float
    timestamp_utc: str
    raw_broker_payload: dict
```

### ApexSignal

```python
@dataclass
class ApexSignal:
    signal_id: str
    symbol: str
    direction: SignalDirection  # LONG | SHORT | EXIT | HOLD
    quantra_score: float
    runner_prob: float
    regime: str
    volatility_band: str
    estimated_move: Optional[dict]
    size_hint: Optional[float]
```

## Risk Engine

The RiskEngine performs 9 deterministic safety checks:

1. **max_order_notional** - Single order size limit
2. **positive_equity** - Account must have positive equity
3. **max_positions** - Position count limit
4. **max_position_notional_per_symbol** - Per-symbol exposure limit
5. **max_notional_exposure** - Total portfolio exposure limit
6. **max_daily_turnover** - Daily trading volume limit
7. **max_leverage** - Leverage ratio limit
8. **block_short_selling** - Prevents short positions
9. **block_margin** - Prevents margin usage

**Fail-closed behavior:** If ANY check fails, the order is REJECTED.

## Execution Pipeline

```
1. Receive ApexSignal
2. Build OrderTicket from signal
3. Run RiskEngine.check(order)
4. If REJECT → log and return None
5. If APPROVE → BrokerRouter.place_order(ticket)
6. Log ExecutionResult in execution log
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/broker/status` | GET | Get broker layer status |
| `/broker/positions` | GET | Get current positions |
| `/broker/orders` | GET | Get open orders |
| `/broker/equity` | GET | Get account equity |
| `/broker/execute` | POST | Execute a trading signal |
| `/broker/config` | GET | Get sanitized config |

### Execute Signal Request

```json
{
  "symbol": "AAPL",
  "direction": "LONG",
  "quantra_score": 75.0,
  "runner_prob": 0.25,
  "size_hint": 0.01
}
```

## CLI Usage

### Paper Trader

```bash
# Set environment variables
export ALPACA_PAPER_API_KEY="your-key"
export ALPACA_PAPER_API_SECRET="your-secret"

# Run paper trader
python scripts/apex_paper_trader.py --config config/broker.yaml --mode PAPER

# Test with a signal
python scripts/apex_paper_trader.py --symbol AAPL --direction LONG

# Status only
python scripts/apex_paper_trader.py --status-only
```

## Logging

### Execution Log

Location: `logs/execution/execution_YYYYMMDD.log`

Contains:
- Timestamp
- Mode (RESEARCH/PAPER/LIVE)
- Signal ID
- Symbol, side, qty
- Order type, intent
- Risk decision
- Broker status
- Fill details

### Audit Log

Location: `logs/execution_audit/audit_YYYYMMDD.log`

Contains:
- Detailed risk checks
- Pre/post account equity
- Checks passed/failed
- Ticket hash for integrity

### Broker Raw Log

Location: `logs/execution_broker_raw/`

Contains raw API request/response JSON for full audit trail.

## Safety Guarantees

1. **ExecutionMode defaults to RESEARCH** - No execution unless explicitly set
2. **LIVE mode is DISABLED** - Raises RuntimeError if attempted
3. **RiskEngine approves every order** - Fail-closed design
4. **No hidden auto-trading** - Explicit scripts and config required
5. **Full audit trail** - Every order logged, including paper trades
6. **Short selling blocked** - By default
7. **Margin blocked** - By default

## Integration with Apex Engine

The broker layer integrates with the existing Apex engine:

```python
from src.quantracore_apex.broker import ExecutionEngine, ApexSignal, SignalDirection

# Initialize (loads config from broker.yaml)
engine = ExecutionEngine()

# Create signal from Apex analysis
signal = ApexSignal(
    signal_id="sig_001",
    symbol="AAPL",
    direction=SignalDirection.LONG,
    quantra_score=75.0,
    runner_prob=0.25,
    size_hint=0.01,
)

# Execute
result = engine.execute_signal(signal)

if result and result.is_success:
    print(f"Order placed: {result.order_id}")
```

## Test Coverage

28 tests covering:
- Order model serialization
- Risk engine rejection paths
- Paper sim adapter fills
- Router mode selection
- Execution engine pipeline

Run tests:
```bash
python -m pytest tests/broker/test_broker_layer.py -v
```

## Future Extensions

### Live Trading (Institution-Only)

When ready for live trading:

1. Extend `AlpacaPaperAdapter` with `AlpacaLiveAdapter`
2. Update `config/broker.yaml`:
   ```yaml
   brokers:
     alpaca:
       live:
         enabled: true
   execution:
     mode: "LIVE"
   ```
3. Update compliance documentation
4. Implement additional safeguards

### Additional Brokers

The adapter pattern supports adding new brokers:

1. Create `new_broker_adapter.py` extending `BrokerAdapter`
2. Implement all abstract methods
3. Add configuration in `broker.yaml`
4. Update `BrokerRouter` selection logic

---

*QuantraCore Apex v9.0-A | Broker Layer v1 | November 2025*
