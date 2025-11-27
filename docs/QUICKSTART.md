# Quickstart — QuantraCore Apex v8.0

## Prerequisites

- Python 3.10+
- pip or uv package manager
- Git

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Lamont-Labs/QuantraCore.git
cd QuantraCore
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Demo

### CLI Demo

Run the deterministic demo via the command-line interface:

```bash
python -m cli.main
```

This executes the Apex engine and outputs results to `dist/golden_demo_outputs/`.

### API Server

Start the FastAPI server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 5000
```

Available endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with status, version, and time |
| `GET /score?ticker=AAPL&seed=42` | Deterministic score computation |
| `GET /risk/hud` | Risk HUD data |
| `GET /audit/export` | Export audit data to file |

---

## Verifying Determinism

Run the verification script to confirm reproducibility:

```bash
bash verify.sh
```

All outputs should match expected checksums.

---

## Running Tests

Execute the test suite:

```bash
pytest -q
```

For verbose output:

```bash
pytest -v
```

---

## Next Steps

- Read the [System Overview](OVERVIEW_QUANTRACORE_APEX.md) for architecture details
- Review the [Master Spec v8.0](QUANTRACORE_APEX_MASTER_SPEC_v8_0.yml) for complete specifications
- See the [Developer Guide](DEVELOPER_GUIDE.md) for contribution guidelines

---

**QuantraCore Apex v8.0** — Deterministic. Reproducible. Acquisition-Ready.
