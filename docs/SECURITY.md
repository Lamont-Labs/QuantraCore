# Security — QuantraCore Apex v8.0

## Overview

QuantraCore Apex implements comprehensive security measures suitable for institutional environments. This document provides a summary; see [Security and Hardening](SECURITY_AND_HARDENING.md) for complete details.

---

## Key Security Features

### Hash Verification

- All system components verified via SHA-256 hashes
- Tampering detected before affecting system behavior
- Integrated with Ω1 (Integrity Lock) for automatic halt on failure

### Encryption

- **At Rest:** AES-256-GCM for all sensitive data
- **In Transit:** TLS 1.3 required, certificate pinning
- **Archives:** Envelope encryption with key rotation

### Outbound Allowlists

- Network communication restricted to approved destinations
- Firewall enforcement with application-level verification
- All blocked attempts logged

### Config Guard (Ω3)

- Prevents unauthorized configuration changes
- Dual approval for critical settings
- Change audit logging

---

## Omega Directives

| Directive | Purpose |
|-----------|---------|
| Ω1 — Integrity Lock | Halts system if integrity compromised |
| Ω2 — Risk Kill Switch | Stops activity on risk limit breach |
| Ω3 — Config Guard | Protects configuration from unauthorized changes |
| Ω4 — Compliance Gate | Enforces regulatory constraints |

---

## Demo Environment Notes

In the demo configuration:
- Synthetic data only (no real financial data)
- No secrets or API keys included
- Sandbox/IAM controls simplified for demo purposes
- Full security stack documented for production deployment

---

## Resources

- [Security and Hardening](SECURITY_AND_HARDENING.md) — Complete security documentation
- [Compliance Policy](COMPLIANCE_POLICY.md) — Regulatory compliance details
- [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml) — Security section in specification
