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
