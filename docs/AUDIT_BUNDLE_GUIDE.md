# Audit Bundle Guide

This guide explains how to create, verify, and use audit bundles for QuantraCore Apex v9.0-A.

---

## 1. What is an Audit Bundle?

An audit bundle is a self-contained package of documentation, test results, configuration snapshots, and verification artifacts that enables:

- **Reproducibility** - Replay any analysis with identical results
- **Verification** - Confirm system integrity via hashes and manifests
- **Compliance** - Demonstrate research-only operation to auditors
- **Acquisition Due Diligence** - Provide complete system documentation

---

## 2. Bundle Contents

Location: `dist/quantracore_apex_proof_bundle_v9_0a/`

| File | Description |
|------|-------------|
| `QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md` | Complete system specification |
| `ENGINE_TEST_SUMMARY.json` | Test results with pass/fail counts |
| `DRIFT_BASELINES/*.json` | Drift detection baseline snapshots |
| `MODEL_MANIFEST_ACTIVE.json` | Active model metadata and hashes |
| `SAMPLE_REPLAY_REPORT.md` | Example replay execution report |
| `COMPLIANCE_AND_SAFETY.md` | Compliance documentation |
| `AUDIT_BUNDLE_GUIDE.md` | This guide |
| `config_snapshot.yaml` | Configuration at bundle creation |

---

## 3. Creating an Audit Bundle

### Automatic Creation

```bash
python scripts/create_audit_bundle.py
```

### Manual Creation

```bash
mkdir -p dist/quantracore_apex_proof_bundle_v9_0a

cp docs/QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md dist/quantracore_apex_proof_bundle_v9_0a/
cp provenance/ENGINE_TEST_SUMMARY.json dist/quantracore_apex_proof_bundle_v9_0a/
cp -r provenance/drift_baselines dist/quantracore_apex_proof_bundle_v9_0a/
cp docs/COMPLIANCE_AND_SAFETY.md dist/quantracore_apex_proof_bundle_v9_0a/
cp docs/AUDIT_BUNDLE_GUIDE.md dist/quantracore_apex_proof_bundle_v9_0a/

python -c "import yaml, json; print(yaml.dump({'config': 'snapshot'}))" > dist/quantracore_apex_proof_bundle_v9_0a/config_snapshot.yaml
```

---

## 4. Verifying an Audit Bundle

### Hash Verification

Each bundle includes SHA-256 hashes for key files:

```bash
cd dist/quantracore_apex_proof_bundle_v9_0a
sha256sum -c CHECKSUMS.sha256
```

### Model Manifest Verification

The model manifest includes:
- Model version and training date
- Input feature dimensions
- Output head configurations
- SHA-256 hash of model weights

```python
import json
with open('MODEL_MANIFEST_ACTIVE.json') as f:
    manifest = json.load(f)
    
print(f"Model: {manifest['model_id']}")
print(f"Hash: {manifest['sha256']}")
print(f"Version: {manifest['version']}")
```

### Test Summary Verification

```python
import json
with open('ENGINE_TEST_SUMMARY.json') as f:
    summary = json.load(f)
    
assert summary['tests_passed'] >= 355
assert summary['tests_failed'] == 0
print(f"Tests: {summary['tests_passed']} passed, {summary['tests_skipped']} skipped")
```

---

## 5. Replaying a Specific Period

### Setup

1. Ensure cached data exists for the target period
2. Load the symbol universe from config
3. Configure replay parameters

### Execution

```bash
qapex replay-demo --universe demo --start 2024-01-01 --end 2024-03-31
```

### Programmatic Replay

```python
from src.quantracore_apex.replay.replay_engine import ReplayEngine

engine = ReplayEngine()
result = engine.run_replay(
    universe="demo",
    start_date="2024-01-01",
    end_date="2024-03-31",
    timeframe="1d"
)

print(f"Signals generated: {result['signal_count']}")
print(f"Equity curve final: {result['equity_curve'][-1]}")
print(f"Drift flags: {result['drift_flags']}")
```

---

## 6. Drift Baseline Comparison

### Loading Baselines

```python
import json
from pathlib import Path

baselines_dir = Path('provenance/drift_baselines')
for baseline_file in baselines_dir.glob('*.json'):
    with open(baseline_file) as f:
        baseline = json.load(f)
    print(f"{baseline_file.stem}: {baseline['metric']}")
```

### Comparing to Current

```python
from src.quantracore_apex.core.drift_detector import DriftDetector

detector = DriftDetector()
detector.load_baselines('provenance/drift_baselines')

current_metrics = detector.compute_current_metrics(universe_results)
comparison = detector.compare_to_baseline(current_metrics)

if comparison['drift_detected']:
    print(f"DRIFT DETECTED: {comparison['reason']}")
else:
    print("No drift detected - within baseline tolerances")
```

---

## 7. Log Export

### Score Consistency Log

```bash
cat provenance/score_consistency_log.jsonl | jq '.'
```

### Drift Events Log

```bash
cat provenance/drift_events_log.jsonl | jq 'select(.severity == "warning")'
```

### Model Promotion Log

```bash
cat provenance/model_promotion_log.jsonl | jq 'select(.action == "promoted")'
```

---

## 8. Bundle for External Review

When preparing a bundle for external auditors:

1. **Sanitize** - Remove any API keys or secrets (should not be present)
2. **Include** - All provenance files and test summaries
3. **Document** - Add reviewer-specific notes if needed
4. **Compress** - Create a zip archive with checksums

```bash
cd dist
zip -r quantracore_apex_audit_bundle_v9_0a.zip quantracore_apex_proof_bundle_v9_0a/
sha256sum quantracore_apex_audit_bundle_v9_0a.zip > quantracore_apex_audit_bundle_v9_0a.zip.sha256
```

---

## 9. Compliance Checklist

Before submitting an audit bundle, verify:

- [ ] All tests passing (ENGINE_TEST_SUMMARY.json)
- [ ] No secrets in any files
- [ ] mode.yaml shows research mode
- [ ] Compliance note present in API responses
- [ ] Simulation mode enforced in OMS
- [ ] Model manifest includes valid hashes
- [ ] Drift baselines are current

---

*Last Updated: November 2025*
