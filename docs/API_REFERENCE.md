# API Reference — QuantraCore Apex v8.0

## Overview

QuantraCore Apex exposes a RESTful API via FastAPI for integration with external systems. All endpoints produce deterministic, reproducible outputs.

---

## Base URL

```
http://localhost:5000
```

---

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "8.0",
  "time": "2025-10-15T14:30:00Z"
}
```

---

### GET /score

Compute deterministic QuantraScore for a ticker.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ticker` | string | Yes | Stock ticker symbol (e.g., "AAPL") |
| `seed` | integer | No | Random seed for determinism (default: 42) |

**Example:**
```
GET /score?ticker=AAPL&seed=42
```

**Response:**
```json
{
  "ticker": "AAPL",
  "seed": 42,
  "quantrascore": 0.78,
  "regime": "trending",
  "risk_tier": "medium",
  "entropy_band": "low",
  "timestamp": "2025-10-15T14:30:00Z",
  "hash": "sha256:abc123..."
}
```

---

### GET /risk/hud

Retrieve Risk HUD (Heads-Up Display) data.

**Response:**
```json
{
  "ticker": "AAPL",
  "score": 0.78,
  "risk_tier": "medium",
  "filters": [
    {"id": "T01", "status": "pass"},
    {"id": "T02", "status": "pass"},
    {"id": "T03", "status": "warning"}
  ],
  "omega_status": {
    "Ω1": "active",
    "Ω2": "standby",
    "Ω3": "active",
    "Ω4": "active"
  }
}
```

---

### GET /audit/export

Export audit data to file.

**Response:**
```json
{
  "status": "exported",
  "path": "dist/golden_demo_outputs/audit_export.json",
  "timestamp": "2025-10-15T14:30:00Z",
  "entries": 1234
}
```

**Side Effect:** Creates or updates `dist/golden_demo_outputs/audit_export.json`

---

### GET /trading/account

Get Alpaca paper trading account status.

**Response:**
```json
{
  "account": {
    "equity": 100015.21,
    "cash": 67347.43,
    "buying_power": 167362.64,
    "positions_count": 4,
    "positions": [
      {"symbol": "NET", "qty": 50, "market_value": 9943.50, "unrealized_pl": -6.00}
    ]
  },
  "timestamp": "2025-12-01T19:05:00Z"
}
```

---

### GET /trading/setups

Get ranked swing trade candidates.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `top_n` | integer | No | Number of setups to return (default: 10) |
| `min_score` | float | No | Minimum QuantraScore (default: 60.0) |

**Response:**
```json
{
  "setups": [
    {
      "symbol": "NET",
      "quantrascore": 83.0,
      "current_price": 198.60,
      "entry": 198.80,
      "stop": 196.29,
      "target": 203.81,
      "shares": 50,
      "position_value": 9930.00,
      "risk_amount": 125.50,
      "reward_amount": 250.50,
      "risk_reward": 2.0,
      "conviction": "low",
      "regime": "range_bound",
      "timing": "none"
    }
  ],
  "count": 9,
  "timestamp": "2025-12-01T19:03:00Z"
}
```

---

### POST /trading/execute

Execute top swing trades on Alpaca paper trading.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `count` | integer | No | Number of trades to execute (default: 3) |

**Response:**
```json
{
  "execution_result": {
    "success": true,
    "timestamp": "2025-12-01T19:03:18Z",
    "account_before": {"equity": 100027.00, "cash": 96938.11},
    "account_after": {"equity": 100017.42, "cash": 72543.11},
    "setups_scanned": 9,
    "trades_selected": [
      {"symbol": "NET", "quantrascore": 83.0, "shares": 50, "entry": 198.80}
    ],
    "trades_executed": [
      {"symbol": "NET", "order_id": "a10b4734-9d31-...", "shares": 50, "status": "new"}
    ]
  },
  "timestamp": "2025-12-01T19:03:19Z"
}
```

**Side Effect:** Executes market orders on Alpaca paper trading; logs trades to `investor_logs/auto_trades/`

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Missing required parameter: ticker"
}
```

### 500 Internal Server Error

```json
{
  "detail": "Internal processing error",
  "trace_id": "uuid-v4"
}
```

---

## Determinism Guarantee

All API responses include:
- Reproducible outputs for identical inputs
- Hash verification of response content
- Timestamp for audit purposes

Using the same `seed` parameter ensures identical results across calls.

---

## Rate Limiting

Default configuration:
- 100 requests per minute per client
- Configurable via environment variables

---

## Authentication

Demo mode: No authentication required.

Production mode: Configure via environment variables (see [Security](SECURITY.md)).
