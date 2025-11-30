# QuantraCore Apex™ — Security & Compliance

**Version:** 9.0-A  
**Component:** System Security & Regulatory Compliance  
**Status:** Active  
**Last Updated:** November 2025

---

## Overview

This document covers the comprehensive security and compliance framework for QuantraCore Apex, including authentication, data integrity, anti-tamper measures, and regulatory compliance.

---

## API Security (v9.0-A)

### Authentication

All protected API endpoints require the `X-API-Key` header for authentication.

| Configuration | Description |
|--------------|-------------|
| `APEX_API_KEY` | Primary API key (environment variable) |
| `APEX_API_KEY_2` | Secondary API key (optional) |
| `APEX_AUTH_DISABLED=true` | Bypass authentication (development only) |

### CORS Policy

Allowed origins are restricted to:
- `http://localhost:*` and `http://127.0.0.1:*`
- `https://*.replit.dev` and `https://*.repl.co`

### Rate Limiting

Data adapters implement non-blocking rate limiting to prevent API quota exhaustion:
- **Polygon.io:** 12.5 second delay (5 calls/min on free tier)
- **Binance:** 0.1 second delay (async-compatible)

### Cache Security

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max Entries | 1000 | Prevents memory exhaustion |
| TTL | 300 seconds | 5-minute expiration |
| Eviction | LRU | Least recently used removal |

---

## Compliance Center

### Role

Ensure all logic and outputs are legally safe and compliant with financial regulations.

---

## Safe Language Filter

### Behavior

1. Scan all outputs for prohibited verbs
2. Rewrite into structural phrasing
3. Trigger Omega-4 on violation

### Prohibited Terms

| Term | Category |
|------|----------|
| buy | Trade action |
| sell | Trade action |
| long | Position direction |
| short | Position direction |
| entry | Timing instruction |
| exit | Timing instruction |
| target | Price target |
| stop | Risk instruction |
| scalp | Trading style |

### Replacement Examples

| Prohibited | Compliant Replacement |
|------------|----------------------|
| "Strong buy signal" | "Structural analysis: trend=upward, strength=high" |
| "Enter at 150" | "Key structural level: 150" |
| "Target 160" | "Resistance zone: 160" |

---

## Compliance Reporter

### Outputs

- `compliance_log.json` — All compliance checks
- `compliance_violations` — Detected violations
- `auto-corrections` — Applied corrections

### Log Format

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "check_type": "output_scan",
  "input_text": "Strong buy signal detected",
  "violation_detected": true,
  "violation_type": "prohibited_term",
  "term": "buy",
  "action": "rewrite",
  "output_text": "Structural analysis shows upward trend with high strength",
  "omega_triggered": false
}
```

---

## Broker Safeguards

### Rules

| Rule | Description |
|------|-------------|
| Live trading disabled | Unless manually unlocked |
| Risk engine approval | Required for all orders |
| Compliance config signing | Required for live mode |
| Invalid trade blocking | Automatic rejection |

---

## User Data Protection

### Guarantees

| Guarantee | Description |
|-----------|-------------|
| No personal data stored | No user-identifiable information |
| Encrypted caching only | All cache encrypted |
| No cloud uploads | User data never sent externally |

---

## ApexCore Output Filters

### Rules

- ApexCore predictions must be structural
- No model may imply trade action
- Any unsafe output triggers Omega-4

---

## Security Layer

### Authentication

| Requirement | Implementation |
|-------------|----------------|
| No hard-coded API keys | Environment variables only |
| Encrypted token storage | Secure credential store |
| Key rotation support | Periodic rotation capability |

### Dataset Integrity

| Measure | Implementation |
|---------|----------------|
| SHA-256 hashes | All dataset blocks |
| Hash mismatch action | Training abort |
| Hash embedding | In MODEL_MANIFEST |

### Anti-Tamper

| Measure | Implementation |
|---------|----------------|
| Protocol version map | Stored and locked |
| Model hash check | Runtime verification |
| Mismatch action | Fail-closed |

---

## Repository Security

### SBOM Requirements

- Package list
- Dependency versions
- Licenses
- Build metadata
- Hash of protocol map
- Model hashes

### Security Rules

- No dev/test keys in repo
- No unvetted binaries
- Dependency audit required
- Regular vulnerability scanning

---

## Device Security

| Measure | Implementation |
|---------|----------------|
| Local-only storage | No cloud sync |
| Encrypted logs | Optional encryption |
| No telemetry | No usage tracking |

---

## Compliance Modes

### Research Mode (Default)

- All trading disabled
- Analysis only
- Full logging
- No compliance concerns

### Simulation Mode

- Simulated execution
- No real orders
- Full compliance checks
- Training-safe

### Paper Mode

- Paper trading with broker
- No real money
- Full compliance
- Risk engine active

### Live-Ready Mode

- Real trading capability
- Full compliance required
- Risk engine armed
- Omega directives active

---

## Audit Trail

All system activity is logged for audit:

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "event_type": "compliance_check",
  "component": "output_filter",
  "input_hash": "sha256:...",
  "output_hash": "sha256:...",
  "compliance_status": "passed",
  "omega_status": "clear"
}
```

---

## Related Documentation

- [Omega Directives](OMEGA_DIRECTIVES.md)
- [Determinism Tests](DETERMINISM_TESTS.md)
- [SBOM & Provenance](SBOM_PROVENANCE.md)
- [Risk Engine](RISK_ENGINE.md)
