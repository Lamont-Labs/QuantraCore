# QuantraCore Apex — Developer Quickstart Guide

**Get up and running in 30 minutes or less.**

## Prerequisites

- Python 3.11+
- Node.js 18+ (for ApexDesk frontend)
- Git

## Step 1: Clone and Setup (5 minutes)

```bash
# Clone the repository
git clone <repo-url>
cd quantracore-apex

# Create Python virtual environment (if running locally)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
# or use make:
make dev-install
```

## Step 2: Verify Installation (2 minutes)

```bash
# Run smoke tests
make test-smoke

# Or directly:
python -m pytest tests/hardening/ -v --tb=short
```

Expected output: All tests should pass.

## Step 3: Generate Protocol Manifest (1 minute)

```bash
python scripts/generate_protocol_manifest.py
```

This creates `config/protocol_manifest.yaml` with the canonical protocol execution order.

## Step 4: Run a Demo Scan (5 minutes)

```bash
# Start the FastAPI backend
uvicorn src.quantracore_apex.server.app:app --host 0.0.0.0 --port 8000

# In another terminal, test the API
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

## Step 5: Explore the Codebase (10 minutes)

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/quantracore_apex/core/` | Deterministic engine core |
| `src/quantracore_apex/protocols/` | T01-T80, LP01-LP25, MR01-MR05, Omega |
| `src/quantracore_apex/apexlab/` | Offline training environment |
| `src/quantracore_apex/apexcore/` | Neural model integration |
| `src/quantracore_apex/broker/` | Execution engine, risk, adapters |
| `src/quantracore_apex/eeo_engine/` | Entry/Exit Optimization |
| `src/quantracore_apex/hardening/` | Safety and compliance infrastructure |
| `dashboard/` | React frontend (ApexDesk) |
| `tests/` | Comprehensive test suite |
| `docs/` | Technical documentation |

### Key Files

| File | Purpose |
|------|---------|
| `config/mode.yaml` | System mode (RESEARCH/PAPER/LIVE) |
| `config/broker.yaml` | Broker and risk configuration |
| `config/protocol_manifest.yaml` | Protocol execution order |
| `replit.md` | Project documentation |

## Step 6: Run the Full Test Suite (10 minutes)

```bash
# All tests
make test

# Specific test categories
python -m pytest tests/hardening/ -v          # Hardening tests
python -m pytest tests/broker/ -v             # Broker tests
python -m pytest tests/eeo_engine/ -v         # EEO tests
```

## Development Workflows

### Starting the Development Environment

```bash
# Backend only
uvicorn src.quantracore_apex.server.app:app --reload --port 8000

# Frontend (in dashboard/ directory)
cd dashboard && npm run dev

# Both (using workflows in Replit)
# Backend runs on port 8000
# Frontend runs on port 5000
```

### Running Tests

```bash
make test           # Full test suite
make test-smoke     # Fast smoke tests
make lint           # Linting
make typecheck      # Type checking
```

### Code Style

- Python: Black, Ruff (PEP8 compliant)
- TypeScript: ESLint, Prettier
- Docstrings: Google style
- Type hints: Required for all public functions

## Understanding the System

### Execution Modes

| Mode | Description | Orders Allowed |
|------|-------------|----------------|
| RESEARCH | Default safe mode | None |
| PAPER | Paper trading | Paper only |
| LIVE | Institutional live | All (disabled by default) |

### Safety Guarantees

1. **Fail-closed**: System refuses to act on uncertainty
2. **Deterministic**: All outputs reproducible given inputs
3. **No silent paths**: All decisions logged
4. **Mode boundaries**: RESEARCH cannot place orders

### Key Documentation

- [Master Spec](docs/QUANTRACORE_APEX_MASTER_SPEC.md)
- [Broker Layer](docs/BROKER_LAYER_SPEC.md)
- [EEO Engine](docs/ENTRY_EXIT_OPTIMIZATION_ENGINE_SPEC.md)
- [Hardening Blueprint](docs/SECURITY_COMPLIANCE/hardening_blueprint.md)

## Troubleshooting

### Tests failing?

```bash
# Check for missing dependencies
pip install -e ".[dev]"

# Regenerate manifest
python scripts/generate_protocol_manifest.py

# Run with verbose output
python -m pytest tests/ -v --tb=long
```

### Backend not starting?

```bash
# Check config validation
python -c "from src.quantracore_apex.hardening import ConfigValidator; v = ConfigValidator(); v.validate_all()"
```

### Mode violations?

The system defaults to RESEARCH mode. Check `config/mode.yaml` and `config/broker.yaml`.

## Next Steps

1. Read the [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
2. Explore the [Protocol Documentation](docs/ENGINE_SPEC/protocols.md)
3. Try the [Example Scripts](examples/)
4. Run the [Nuclear Tests](docs/TESTING/nuclear_suite.md) for determinism verification

---

*QuantraCore Apex v9.0-A | Lamont Labs | November 2025*
