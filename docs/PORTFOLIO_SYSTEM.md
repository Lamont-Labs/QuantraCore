# QuantraCore Apex™ — Portfolio System

**Version:** 8.0  
**Component:** Position & Exposure Management  
**Status:** Active

---

## Overview

The Portfolio System is an optional data structure for tracking account state. It provides position management, exposure analysis, and PnL tracking for research and visualization purposes.

---

## Compliance Statement

- Used for research and visualization only
- May not produce recommendations
- No trade signals generated from portfolio state

---

## Tracked Data

### Positions

| Field | Description |
|-------|-------------|
| Symbol | Ticker symbol |
| Quantity | Position size |
| Entry price | Average entry |
| Current price | Market price |
| Unrealized PnL | Open profit/loss |
| Entry timestamp | When opened |

### Orders

| Field | Description |
|-------|-------------|
| Order ID | Unique identifier |
| Status | pending \| filled \| cancelled |
| Type | market \| limit \| stop |
| Side | buy \| sell |
| Quantity | Order size |
| Fill price | Execution price |

### Fills

| Field | Description |
|-------|-------------|
| Fill ID | Unique identifier |
| Order ID | Associated order |
| Price | Execution price |
| Quantity | Filled amount |
| Timestamp | Execution time |
| Fees | Transaction costs |

---

## PnL Tracking

### Unrealized PnL

Calculated in real-time for all open positions:

```
unrealized_pnl = (current_price - entry_price) * quantity
```

### Realized PnL

Calculated when positions are closed:

```
realized_pnl = (exit_price - entry_price) * quantity - fees
```

---

## Exposure Analysis

### By Sector

| Metric | Description |
|--------|-------------|
| Sector allocation | Percentage per sector |
| Sector concentration | Largest sector exposure |
| Sector correlation | Cross-sector correlation |

### By Volatility Band

| Metric | Description |
|--------|-------------|
| Low vol exposure | Positions in low volatility |
| Medium vol exposure | Positions in medium volatility |
| High vol exposure | Positions in high volatility |

---

## Metrics

### Position Sizing

Position sizing is calculated deterministically:

- Based on risk parameters
- Volatility-adjusted
- Sector-aware
- Never exceeds max exposure rules

### Exposure Rules

| Rule | Default |
|------|---------|
| Max single position | 10% of portfolio |
| Max sector exposure | 25% of portfolio |
| Max total exposure | 50% of portfolio |
| Max correlation group | 30% of portfolio |

### Risk Metrics

| Metric | Description |
|--------|-------------|
| Delta exposure | Net directional exposure |
| Portfolio heat | Total capital at risk |
| Sector correlation risk | Correlated sector exposure |

---

## Data Structure

```json
{
  "account": {
    "total_value": 100000.00,
    "cash": 50000.00,
    "buying_power": 75000.00
  },
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": 100,
      "entry_price": 150.00,
      "current_price": 155.00,
      "unrealized_pnl": 500.00,
      "sector": "Technology",
      "volatility_band": "medium"
    }
  ],
  "exposure": {
    "total": 0.155,
    "by_sector": {
      "Technology": 0.155
    },
    "by_volatility": {
      "low": 0.00,
      "medium": 0.155,
      "high": 0.00
    }
  },
  "pnl": {
    "unrealized": 500.00,
    "realized_today": 0.00,
    "realized_total": 1250.00
  }
}
```

---

## Integration

### With Risk Engine

Portfolio state feeds into Risk Engine checks:
- Exposure limits
- Concentration checks
- Correlation analysis

### With Dashboard

Portfolio data displayed in dashboard:
- Position table
- Exposure charts
- PnL timeline

---

## Related Documentation

- [Broker/OMS](BROKER_OMS.md)
- [Risk Engine](RISK_ENGINE.md)
- [Architecture](ARCHITECTURE.md)
