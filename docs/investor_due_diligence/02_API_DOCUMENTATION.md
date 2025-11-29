# API Documentation

**Document Classification:** Investor Due Diligence  
**Version:** 9.0-A  
**Date:** November 2025  
**Status:** Production  
**Owner:** Lamont Labs

---

## Executive Summary

QuantraCore Apex exposes a RESTful API via FastAPI with 36 endpoints providing complete access to the analysis engine, risk assessment, compliance monitoring, and portfolio simulation capabilities. All endpoints produce deterministic, reproducible outputs with cryptographic verification.

---

## API Overview

| Attribute | Value |
|-----------|-------|
| Protocol | REST (HTTP/HTTPS) |
| Format | JSON |
| Authentication | Configurable (disabled in demo) |
| Rate Limiting | 100 requests/minute (configurable) |
| Documentation | Auto-generated OpenAPI/Swagger |
| Base URL | `http://localhost:8000` |

---

## Endpoint Inventory (36 Total)

### Core Analysis Endpoints (8)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API root information |
| `/health` | GET | System health check |
| `/api/stats` | GET | System statistics (13 modules) |
| `/scan_symbol` | POST | Single symbol Apex analysis |
| `/scan_universe` | POST | Multi-symbol batch scanning |
| `/scan_universe_mode` | POST | Mode-specific universe scan |
| `/trace/{hash}` | GET | Full protocol trace retrieval |
| `/symbol_info/{symbol}` | GET | Symbol information |

### Risk & Signal Endpoints (3)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/risk/assess/{symbol}` | POST | Comprehensive risk assessment |
| `/signal/generate/{symbol}` | POST | Unified trade signal generation |
| `/monster_runner/{symbol}` | POST | Rare event precursor detection |

### Compliance Endpoints (5)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/compliance/score` | GET | Regulatory excellence score |
| `/compliance/excellence` | GET | Excellence multipliers |
| `/compliance/report` | GET | Full compliance report |
| `/compliance/standards` | GET | Regulatory standards list |
| `/compliance/omega` | GET | Omega directive status |

### Predictive Advisory Endpoints (4)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictive/status` | GET | Advisory layer status |
| `/predictive/model_info` | GET | Model manifest information |
| `/predictive/advise` | POST | Get predictive advisory |
| `/predictive/batch_advise` | POST | Batch predictive advisory |

### Engine & System Endpoints (7)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/desk` | GET | Desk configuration |
| `/engine/health_extended` | GET | Extended health check |
| `/drift/status` | GET | Drift status |
| `/replay/run_demo` | POST | Run replay demo |
| `/score/consistency/{symbol}` | GET | Score consistency check |
| `/modes` | GET | Available modes |
| `/universe_stats` | GET | Universe statistics |

### Portfolio & OMS Endpoints (9)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/portfolio/status` | GET | Portfolio snapshot |
| `/portfolio/heat_map` | GET | Sector heat map |
| `/oms/orders` | GET | Order list |
| `/oms/positions` | GET | Position list |
| `/oms/place` | POST | Place simulated order |
| `/oms/submit/{id}` | POST | Submit order |
| `/oms/fill` | POST | Simulate order fill |
| `/oms/cancel/{id}` | POST | Cancel order |
| `/oms/reset` | POST | Reset OMS state |

---

## Core Endpoint Specifications

### GET /health

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T12:00:00Z",
  "engine": "operational",
  "data_layer": "operational"
}
```

---

### POST /scan_symbol

Perform comprehensive Apex analysis on a single symbol.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "lookback_days": 150,
  "seed": 42
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "quantrascore": 72.5,
  "score_bucket": "high",
  "regime": "trending",
  "risk_tier": "medium",
  "entropy_state": "stable",
  "suppression_state": "none",
  "drift_state": "neutral",
  "verdict_action": "hold",
  "verdict_confidence": 0.85,
  "omega_alerts": [],
  "protocol_fired_count": 23,
  "window_hash": "sha256:abc123...",
  "timestamp": "2025-11-29T12:00:00Z"
}
```

---

### GET /compliance/score

Get current regulatory compliance excellence score.

**Response:**
```json
{
  "overall_score": 99.25,
  "excellence_level": "EXCEPTIONAL",
  "timestamp": "2025-11-29T12:00:00Z",
  "metrics": {
    "determinism_score": 100.0,
    "latency_score": 98.5,
    "stress_test_score": 99.0,
    "audit_completeness": 100.0
  },
  "standards_met": [
    "SEC 15c3-5",
    "FINRA 15-09",
    "MiFID II RTS 6",
    "Basel BCBS 239"
  ],
  "standards_exceeded": [
    "FINRA 15-09 §3 (3x)",
    "MiFID II RTS 6 Article 17 (5x)",
    "Basel BCBS 239 (3x)"
  ],
  "compliance_mode": "RESEARCH_ONLY"
}
```

---

### POST /risk/assess/{symbol}

Comprehensive multi-factor risk assessment.

**Response:**
```json
{
  "symbol": "AAPL",
  "risk_assessment": {
    "risk_tier": "medium",
    "risk_score": 45.2,
    "permission": "allow",
    "denial_reasons": [],
    "volatility_check": "pass",
    "regime_check": "pass",
    "entropy_check": "pass"
  },
  "underlying_analysis": {
    "quantrascore": 72.5,
    "regime": "trending",
    "entropy_state": "stable"
  },
  "timestamp": "2025-11-29T12:00:00Z"
}
```

---

### POST /monster_runner/{symbol}

Rare event precursor detection (MonsterRunner analysis).

**Response:**
```json
{
  "symbol": "AAPL",
  "runner_probability": 0.15,
  "runner_state": "dormant",
  "rare_event_class": "none",
  "metrics": {
    "compression_trace": 0.23,
    "entropy_floor": 0.12,
    "volume_pulse": 0.45,
    "range_contraction": 0.34,
    "primed_confidence": 0.18
  },
  "compliance_note": "Structural detection only - not a trade signal"
}
```

---

### GET /predictive/status

Advisory layer operational status.

**Response:**
```json
{
  "status": "NO_VALID_MANIFEST",
  "message": "Predictive advisory layer ready, awaiting model training",
  "fail_closed": true,
  "model_loaded": false,
  "advisory_enabled": false,
  "compliance_note": "Advisory layer operates in fail-closed mode"
}
```

---

## Request/Response Standards

### Request Headers

| Header | Required | Description |
|--------|----------|-------------|
| Content-Type | Yes | application/json |
| Accept | No | application/json |
| X-Request-ID | No | Client tracking ID |

### Response Headers

| Header | Description |
|--------|-------------|
| X-Response-Hash | SHA-256 of response body |
| X-Request-ID | Echo of client tracking ID |
| X-Timestamp | Server timestamp |

### Error Response Format

```json
{
  "detail": "Error description",
  "trace_id": "uuid-v4",
  "timestamp": "2025-11-29T12:00:00Z"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found (resource missing) |
| 500 | Internal Server Error |

---

## Determinism Guarantee

All API responses include:

1. **Reproducible outputs** for identical inputs when using same seed
2. **Hash verification** via `window_hash` field
3. **Timestamp** for audit trail
4. **Trace capability** via `/trace/{hash}` endpoint

**Determinism Contract:**
- Same `symbol` + `seed` + `lookback_days` = identical `quantrascore`
- Hash chain for complete auditability
- 100% reproducible across restarts

---

## Rate Limiting

| Tier | Limit | Window |
|------|-------|--------|
| Default | 100 requests | 1 minute |
| Burst | 20 requests | 1 second |

Rate limit headers returned:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

---

## Authentication (Production Mode)

| Method | Description |
|--------|-------------|
| API Key | Header-based authentication |
| JWT | Token-based for session management |
| mTLS | Client certificate authentication |

Demo mode runs without authentication for evaluation purposes.

---

## Interactive Documentation

| URL | Description |
|-----|-------------|
| `/docs` | Swagger UI (interactive) |
| `/redoc` | ReDoc (reference) |
| `/openapi.json` | OpenAPI 3.0 specification |

---

## SDK Support

### Python Example

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000")

# Scan a symbol
response = client.post("/scan_symbol", json={
    "symbol": "AAPL",
    "timeframe": "1d",
    "lookback_days": 150,
    "seed": 42
})
result = response.json()
print(f"QuantraScore: {result['quantrascore']}")
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/scan_symbol', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'AAPL',
    timeframe: '1d',
    lookback_days: 150,
    seed: 42
  })
});
const result = await response.json();
console.log(`QuantraScore: ${result.quantrascore}`);
```

---

## Versioning

| Version | Status | Notes |
|---------|--------|-------|
| v9.0-A | Current | Institutional Hardening |
| v8.0 | Legacy | Deprecated |

API versioning via URL path (`/v9/`) planned for future releases.

---

*Document prepared for investor due diligence. Confidential.*
