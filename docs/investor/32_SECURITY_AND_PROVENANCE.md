# Security and Provenance

**Document Classification:** Investor Due Diligence — Risk/Compliance  
**Version:** 9.0-A  
**Date:** November 2025  

---

## Is This Supply-Chain Safe? How Do We Know We're Running What You Say?

This document explains the security measures and provenance guarantees that ensure system integrity.

---

## SBOM (Software Bill of Materials)

### What Is SBOM?

The Software Bill of Materials is a comprehensive inventory of all components used in the system.

**Location:** `SBOM.json`

### SBOM Contents

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "version": 1,
  "metadata": {
    "timestamp": "2025-11-29T00:00:00Z",
    "component": {
      "name": "quantracore-apex",
      "version": "9.0.0-A"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "numpy",
      "version": "1.24.0",
      "purl": "pkg:pypi/numpy@1.24.0",
      "hashes": [
        {"alg": "SHA-256", "content": "abc123..."}
      ]
    },
    ...
  ]
}
```

### SBOM Coverage

| Category | Examples |
|----------|----------|
| Python packages | numpy, pandas, scikit-learn, fastapi |
| Node.js packages | react, vite, tailwindcss |
| System dependencies | Nix packages |
| Internal modules | Engine, lab, models |

---

## PROVENANCE.manifest

### What Is Provenance?

The provenance manifest provides cryptographic verification of critical files.

**Location:** `PROVENANCE.manifest`

### Manifest Contents

```
# QuantraCore Apex Provenance Manifest
# Generated: 2025-11-29T00:00:00Z
# SHA256 hashes for critical files

# Core Engine
sha256:abc123def456...  src/quantracore_apex/engine/core.py
sha256:789ghi012jkl...  src/quantracore_apex/engine/protocols.py

# Models
sha256:mno345pqr678...  models/apexcore_v2/big/model.pkl
sha256:stu901vwx234...  models/apexcore_v2/mini/model.pkl

# Configuration
sha256:yza567bcd890...  config/defaults.yaml

# Tests
sha256:efg123hij456...  tests/test_determinism.py
```

### Verification Process

```python
def verify_provenance(manifest_path):
    with open(manifest_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            expected_hash, file_path = line.strip().split()
            actual_hash = compute_sha256(file_path)
            if actual_hash != expected_hash:
                raise IntegrityError(f"Hash mismatch: {file_path}")
    return True
```

---

## Model Manifest Hashes

### Per-Model Verification

Each model has its own manifest with detailed hashes:

```json
{
  "model_family": "apexcore_v2",
  "variant": "big",
  "version": "2.0.0",
  "hashes": {
    "model": "sha256:abc123...",
    "scaler": "sha256:def456...",
    "calibrator": "sha256:789ghi...",
    "config": "sha256:jkl012..."
  },
  "created": "2025-11-15T10:30:00Z",
  "created_by": "training_pipeline_v2"
}
```

### Integrity Check at Load Time

```python
def load_model_with_verification(model_dir):
    manifest = load_manifest(model_dir / "manifest.json")
    
    for component, expected_hash in manifest.hashes.items():
        file_path = model_dir / f"{component}.pkl"
        actual_hash = compute_sha256(file_path)
        
        if actual_hash != expected_hash:
            raise ModelIntegrityError(
                f"Hash mismatch for {component}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    
    return load_model(model_dir)
```

---

## Network Allowlist Concept

### Principle

The system is designed to operate with minimal network access. When network access is needed, it should be limited to known-good endpoints.

### Allowed Endpoints

| Endpoint | Purpose |
|----------|---------|
| api.polygon.io | Market data (Polygon) |
| www.alphavantage.co | Market data (Alpha Vantage) |
| query1.finance.yahoo.com | Market data (Yahoo) |

### Blocked by Default

- Arbitrary outbound connections
- Telemetry endpoints
- Update servers (updates are manual)
- Any endpoint not on allowlist

### Implementation

This would be enforced at the network/firewall level in production deployments, not within the application.

---

## Key Storage

### Principle

> Secrets never in code. Secrets never in version control.

### Storage Methods

| Secret | Storage Method |
|--------|----------------|
| `POLYGON_API_KEY` | Environment variable |
| `ALPHA_VANTAGE_API_KEY` | Environment variable |
| API credentials | Environment variables |
| Database passwords | Environment variables |

### What's NOT in the Repo

```
❌ API keys
❌ Passwords
❌ Private keys
❌ Database credentials
❌ Any secret or credential
```

### Environment Variable Access

```python
import os

# Correct: Read from environment
api_key = os.environ.get("POLYGON_API_KEY")

# NEVER: Hardcode secrets
# api_key = "abc123xyz789"  # FORBIDDEN
```

---

## Dependency Security

### Vulnerability Scanning

Dependencies are scanned for known vulnerabilities:

| Tool | Purpose |
|------|---------|
| pip-audit | Python vulnerability scanning |
| npm audit | Node.js vulnerability scanning |
| Snyk | Cross-platform scanning |

### Update Policy

| Severity | Response Time |
|----------|---------------|
| Critical | 24 hours |
| High | 7 days |
| Medium | 30 days |
| Low | Next release |

### Pinned Versions

All dependencies are pinned to specific versions:

```
# requirements.txt
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
```

This ensures reproducibility and prevents supply-chain attacks through version substitution.

---

## Code Signing (Future)

For production deployments, code signing would provide:
- Verification of code origin
- Tamper detection
- Trust chain from source to deployment

This is not currently implemented but would be part of production hardening.

---

## Security Checklist

| Item | Status |
|------|--------|
| No secrets in code | ✓ |
| SBOM generated | ✓ |
| Provenance manifest | ✓ |
| Model manifests | ✓ |
| Dependency pinning | ✓ |
| Hash verification | ✓ |
| Network allowlist | Design only |
| Code signing | Future |
| Vulnerability scanning | CI pipeline |

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
