# Security and Hardening

**Version:** 8.0  
**Component:** Hardening Stack  
**Role:** System security, integrity protection, and defense in depth

---

## 1. Overview

QuantraCore Apex implements multiple layers of security to protect system integrity, data confidentiality, and operational safety. The system is designed for institutional environments where security is paramount.

---

## 2. Core Security Principles

From the master spec:

> "Determinism first. Fail-closed always. No cloud dependencies. Local-only learning."

These principles directly support security:
- **Determinism** — Predictable, auditable behavior
- **Fail-closed** — Safe defaults under uncertainty
- **No cloud** — Reduced attack surface
- **Local learning** — Data sovereignty

---

## 3. Omega Directives (Security Perspective)

The Omega directives provide system-level security controls:

### 3.1 Ω1 — Hard Safety Lock

Primary security enforcement:
- Blocks system operation when integrity is compromised
- Triggers on tampering detection
- Requires manual intervention to reset
- Cannot be bypassed by normal operations

### 3.2 Ω2 — Entropy Override

Security response to market chaos:
- Detects abnormal market entropy
- Triggers protective mode
- Reduces exposure during uncertainty
- Logs all activations

### 3.3 Ω3 — Drift Override

Security response to regime instability:
- Detects high drift conditions
- Triggers conservative mode
- Prevents decisions during instability
- Logs all activations

### 3.4 Ω4 — Compliance Override

Regulatory security enforcement:
- Blocks forbidden actions
- Enforces mode transitions
- Validates data usage
- Filters outputs

---

## 4. Data Security

### 4.1 Local Storage Only

From the data_ingest specification:
- **Local only** — All data stored locally
- **Encrypted store** — Data at rest encryption

### 4.2 No Cloud Dependencies

Core logic never requires cloud connectivity:
- All analysis runs locally
- Training happens offline in ApexLab
- No external data leakage paths

### 4.3 Hardware Target Security

The GMKtec NucBox K6 workstation provides:
- Physical security (dedicated hardware)
- Air-gapped operation capability
- Local storage for logs and models

---

## 5. Broker Layer Security

### 5.1 Fail-Closed Conditions

The broker layer fails closed when:

| Condition | Response |
|-----------|----------|
| Risk model rejects | Block execution |
| Drift is high | Block execution |
| Entropy is high | Block execution |
| Compliance fails | Block execution |

### 5.2 Mode Restrictions

| Mode | Security Level |
|------|----------------|
| Paper | Safe (no real orders) |
| Sim | Safe (simulation only) |
| Live | Locked behind compliance gates |

### 5.3 Comprehensive Logging

All broker activity is logged:
- All orders
- All rejections
- All risk overrides

---

## 6. Risk Engine Security

The risk engine enforces security through:

| Check | Security Purpose |
|-------|------------------|
| Drawdown threshold | Capital protection |
| Volatility band | Exposure control |
| Slippage estimate | Execution safety |
| Spread constraints | Cost protection |
| Regime mismatch | Strategy safety |
| Sector instability | Diversification safety |
| Microtrait imbalance | Pattern safety |

---

## 7. Proof Logging Security

### 7.1 Immutable Audit Trail

Proof logs provide:
- Timestamp verification
- All indicators recorded
- All protocols fired
- QuantraScore tracking
- Final verdict logging
- Omega override recording
- Drift state tracking
- Entropy signature capture

### 7.2 Log Integrity

Logs are protected by:
- Append-only storage
- Cryptographic hashing
- Local-only storage

---

## 8. Model Security

### 8.1 ApexCore Principles

From the master spec:
- "Apex is always the teacher"
- "ApexCore never overrides Apex"
- "Fails closed if uncertain"

### 8.2 Model Manifest

Every model includes:
- `MODEL_MANIFEST.json` — Complete documentation
- Deterministic hash — Integrity verification
- Training metrics — Validation proof

---

## 9. SBOM Security

The Software Bill of Materials ensures:

- Full package manifest — Known dependencies
- Dependency lockfiles — Version pinning
- Model hashes — Integrity verification
- Protocol version list — Configuration tracking

---

## 10. Test Coverage

Security-relevant test categories:

| Category | Tests |
|----------|-------|
| Engine | Protocol firing, regression baseline |
| ApexLab | Dataset integrity, label reproducibility |
| ApexCore | Inference speed, consistency with Apex, fail-closed paths |
| Risk | Drift/entropy detection, failure rate |

---

## 11. Defense in Depth

QuantraCore Apex implements defense in depth:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Hardware Security (K6 workstation, local storage)         │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2: Data Security (encrypted store, local only)               │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3: Omega Directives (Ω1–Ω4 safety overrides)                 │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4: Risk Engine (multiple safety checks)                       │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5: Broker Controls (fail-closed, mode restrictions)           │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6: Proof Logging (immutable audit trail)                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Summary

QuantraCore Apex's security architecture provides comprehensive protection through the Omega directives (Ω1 hard safety lock, Ω2 entropy override, Ω3 drift override, Ω4 compliance override), local-only data handling, encrypted storage, fail-closed broker controls, and immutable proof logging. The defense-in-depth approach ensures that no single point of failure can compromise system integrity.
