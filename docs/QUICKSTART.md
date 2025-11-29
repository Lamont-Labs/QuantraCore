# Quickstart — QuantraCore Apex v9.0-A

## Prerequisites

- Python 3.11+
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
- Review the [Master Spec v9.0-A](QUANTRACORE_APEX_MASTER_SPEC_v9.0-A.md) for complete specifications
- See the [Developer Guide](DEVELOPER_GUIDE.md) for contribution guidelines
- Explore the [ApexLab V2](APEXLAB_V2.md) and [ApexCore V2](APEXCORE_V2.md) documentation

---

**QuantraCore Apex v9.0-A** — Deterministic. Reproducible. Research-Ready.
