# QuantraCore Apex™ — SBOM & Provenance

**Version:** 8.0  
**Component:** Software Bill of Materials & Build Provenance  
**Status:** Active

---

## Overview

This document describes the Software Bill of Materials (SBOM) and provenance tracking for QuantraCore Apex, ensuring full traceability, reproducibility, and security compliance.

---

## SBOM (Software Bill of Materials)

### Purpose

The SBOM provides a complete inventory of all software components, dependencies, and their relationships within QuantraCore Apex.

### Contents

| Component | Description |
|-----------|-------------|
| Package list | All installed packages |
| Dependency versions | Exact version numbers |
| Licenses | License for each dependency |
| Build metadata | Build environment details |
| Protocol map hash | Hash of protocol version map |
| Model hashes | Hashes of all models |

### SBOM Format

```json
{
  "sbom_version": "1.0",
  "generated": "2025-11-27T00:00:00Z",
  "apex_version": "8.0.0",
  "packages": [
    {
      "name": "numpy",
      "version": "1.24.0",
      "license": "BSD-3-Clause",
      "hash": "sha256:..."
    },
    {
      "name": "tensorflow-lite",
      "version": "2.14.0",
      "license": "Apache-2.0",
      "hash": "sha256:..."
    }
  ],
  "protocol_map_hash": "sha256:...",
  "model_hashes": {
    "ApexCore_Full": "sha256:...",
    "ApexCore_Mini": "sha256:..."
  }
}
```

---

## Provenance Tracking

### Purpose

Provenance provides a complete audit trail of how the software was built, by whom, and from what source.

### Tracked Information

| Field | Description |
|-------|-------------|
| Commit hash | Git commit SHA |
| Author | Commit author |
| Timestamp | Build timestamp |
| Build log | Deterministic build log |
| Protocol version map | Version of each protocol |
| Dataset hashes | Training data hashes |

### Provenance Format

```json
{
  "provenance_version": "1.0",
  "build_id": "BUILD-2025-001234",
  "timestamp": "2025-11-27T00:00:00Z",
  "source": {
    "repository": "https://github.com/Lamont-Labs/QuantraCore",
    "commit_hash": "abc123def456...",
    "branch": "main",
    "tag": "v8.0.0"
  },
  "author": {
    "name": "Jesse J. Lamont",
    "email": "lamontlabs@proton.me"
  },
  "build_environment": {
    "os": "Linux",
    "python_version": "3.11",
    "build_tool": "make"
  },
  "protocol_version_map": {
    "T01": "1.0.0",
    "T02": "1.0.0",
    "...": "..."
  },
  "dataset_hashes": {
    "training_set": "sha256:...",
    "validation_set": "sha256:...",
    "golden_set": "sha256:..."
  }
}
```

---

## Storage Structure

```
sbom/
├── SBOM_v8.0.0.json
├── SBOM_v8.0.1.json
└── ...

provenance/
├── BUILD-2025-001234.json
├── BUILD-2025-001235.json
└── ...
```

---

## Verification

### SBOM Verification

```bash
# Verify package hashes
python verify_sbom.py --sbom sbom/SBOM_v8.0.0.json

# Output
Verifying 45 packages...
numpy: VERIFIED
tensorflow-lite: VERIFIED
...
All packages verified successfully.
```

### Provenance Verification

```bash
# Verify build provenance
python verify_provenance.py --build BUILD-2025-001234

# Output
Verifying build provenance...
Commit hash: VERIFIED
Protocol map: VERIFIED
Dataset hashes: VERIFIED
Model hashes: VERIFIED
Build provenance verified successfully.
```

---

## Security Requirements

### Repository Security

| Requirement | Status |
|-------------|--------|
| No dev/test keys in repo | Enforced |
| No unvetted binaries | Enforced |
| Dependency audit | Regular |
| Vulnerability scanning | Automated |

### Dependency Management

- All dependencies version-locked
- License compatibility verified
- Security advisories monitored
- Regular update cycles

---

## Model Manifest Integration

The MODEL_MANIFEST.json links to SBOM and provenance:

```json
{
  "model_id": "ApexCore_Full_v8.0.0",
  "model_hash": "sha256:...",
  "sbom_reference": "sbom/SBOM_v8.0.0.json",
  "provenance_reference": "provenance/BUILD-2025-001234.json",
  "training_metadata": {
    "dataset_hash": "sha256:...",
    "protocol_version_map_hash": "sha256:...",
    "training_timestamp": "2025-11-27T00:00:00Z"
  }
}
```

---

## Audit Trail

Every significant operation is logged:

```json
{
  "timestamp": "2025-11-27T12:00:00Z",
  "event_type": "sbom_generation",
  "apex_version": "8.0.0",
  "packages_counted": 45,
  "sbom_hash": "sha256:...",
  "stored_at": "sbom/SBOM_v8.0.0.json"
}
```

---

## Related Documentation

- [Security & Compliance](SECURITY_COMPLIANCE.md)
- [Determinism Tests](DETERMINISM_TESTS.md)
- [ApexLab Training](APEXLAB_TRAINING.md)
- [Architecture](ARCHITECTURE.md)
