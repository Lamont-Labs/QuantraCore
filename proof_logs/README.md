# Proof Logs

**Document Classification:** Technical Operations  
**Version:** 9.0-A  
**Date:** November 2025  

---

## What Are Proof Logs?

Proof logs are cryptographic audit records that document every analysis performed by QuantraCore Apex. They enable verification, replay, and regulatory audit of system outputs.

---

## Purpose

| Use Case | Description |
|----------|-------------|
| **Audit** | Prove what analysis was performed and when |
| **Replay** | Reproduce past analyses for verification |
| **Compliance** | Meet regulatory record-keeping requirements |
| **Debugging** | Investigate specific decisions |
| **Determinism** | Verify outputs are reproducible |

---

## Log Format

Each proof log entry contains:

```json
{
  "timestamp": "2025-11-29T10:15:00.000Z",
  "version": "9.0.0-A",
  "event_type": "ANALYSIS_COMPLETE",
  "symbol": "XYZ",
  "input": {
    "data_hash": "sha256:abc123...",
    "config_hash": "sha256:def456...",
    "model_manifest_hash": "sha256:789ghi..."
  },
  "output": {
    "quantra_score": 72.4,
    "quality_tier": "B",
    "runner_prob": 0.35,
    "output_hash": "sha256:jkl012..."
  },
  "protocol_trace": {
    "T01": 2.3,
    "T05": 1.8,
    "T15": 4.2
  },
  "omega_status": {
    "Ω1": "INACTIVE",
    "Ω2": "INACTIVE",
    "Ω3": "INACTIVE",
    "Ω4": "ACTIVE",
    "Ω5": "INACTIVE"
  },
  "compliance_mode": "RESEARCH_ONLY"
}
```

---

## Hash Verification

### Input Hash

The input hash captures:
- OHLCV data used for analysis
- Configuration settings active
- Model manifest in use

```
input_hash = SHA256(data + config + manifest)
```

### Output Hash

The output hash captures:
- All analysis outputs
- Protocol contributions
- Final scores and classifications

```
output_hash = SHA256(all_outputs)
```

### Verification

```bash
# Verify a specific analysis
python -m quantracore_apex.verify_proof proof_logs/2025-11-29/XYZ.json

# Expected output:
# Input hash: VERIFIED
# Output hash: VERIFIED
# Determinism: CONFIRMED
```

---

## Replay Capability

Proof logs enable replay:

```python
from quantracore_apex.replay import replay_analysis

# Load proof log
proof = load_proof_log("proof_logs/2025-11-29/XYZ.json")

# Replay analysis
replay_result = replay_analysis(proof)

# Verify determinism
assert replay_result.output_hash == proof.output.output_hash
```

---

## Directory Structure

```
proof_logs/
├── README.md           # This file
├── 2025-11-29/         # Logs by date
│   ├── XYZ.json
│   ├── ABC.json
│   └── ...
├── 2025-11-28/
│   └── ...
└── index.json          # Log index
```

---

## Retention Policy

| Log Type | Retention |
|----------|-----------|
| Full proof logs | 7 years |
| Index files | 7 years |
| Summary logs | 90 days |

This meets regulatory requirements for:
- SEC Rule 17a-4 (7 years)
- FINRA Rule 4511 (6 years)
- MiFID II RTS 6 (5 years)

---

## Export and Archive

### Export Single Log

```bash
python -m quantracore_apex.export_proof \
    --symbol XYZ \
    --date 2025-11-29 \
    --output export/XYZ_2025-11-29.json
```

### Archive Period

```bash
python -m quantracore_apex.archive_proofs \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --output archive/2025_proofs.tar.gz
```

---

## Integrity Verification

### Verify All Logs

```bash
python -m quantracore_apex.verify_all_proofs proof_logs/

# Expected output:
# Verified: 1,234 logs
# Failed: 0 logs
# Integrity: 100%
```

### Detect Tampering

If a log has been modified:

```
WARNING: Hash mismatch detected
  File: proof_logs/2025-11-29/XYZ.json
  Expected: sha256:abc123...
  Actual: sha256:xyz789...
  Status: TAMPERED
```

---

## API Access

Proof logs are accessible via API:

```bash
# Get specific proof
curl http://localhost:8000/proof/export?symbol=XYZ&date=2025-11-29

# Search proofs
curl http://localhost:8000/proof/search?start=2025-11-01&end=2025-11-30

# Verify proof
curl -X POST http://localhost:8000/proof/verify \
    -d @proof_logs/2025-11-29/XYZ.json
```

---

## Compliance Notes

- Proof logs are append-only (no modification after creation)
- Deletion requires explicit operator action with audit trail
- All access is logged
- Encryption at rest recommended for production

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
